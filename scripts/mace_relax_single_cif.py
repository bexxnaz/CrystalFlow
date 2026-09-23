
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

from ase.optimize import FIRE
from ase.filters import FrechetCellFilter
from ase.constraints import FixSymmetry
from ase.io.trajectory import Trajectory

from relax_wbm import build_mace_calculator
# NOTE: this used to be a separate make_mace_calc() here, deliberately
# mirroring the project's relax_structures.py so the two matched exactly.
# Importing relax_wbm.py's version instead is a tradeoff in the other
# direction: it now matches relax_wbm.py's --mace_energy_check calculator
# exactly (single source of truth -- the two can't silently drift apart on
# dtype/dispersion again, which is exactly what happened before), at the
# cost of no longer being a byte-for-byte mirror of relax_structures.py's
# own make_mace_calc. If you specifically need this script to match
# relax_structures.py instead, revert to a local make_mace_calc().


def relax_and_log(atoms, fmax=0.05, steps=500, relax_cell=True,
                   fixsymmetry=False, traj_path=None):
    """Relax `atoms` in place via FIRE, logging energy/positions/cell/
    forces at EVERY step (not just initial/final).

    NOTE: when relax_cell=True, FIRE's fmax convergence check runs against
    FrechetCellFilter's COMBINED atomic-forces-plus-stress quantity, not
    the plain atomic forces logged here as max_atomic_force -- so the last
    logged max_atomic_force can sit slightly above fmax even at genuine
    convergence. That's expected, not truncation.

    Returns a dict: steps, energy_total_eV, energy_per_atom_eV, a, b, c,
    alpha, beta, gamma, volume, max_atomic_force, frac_coords (list of
    [n_atoms, 3] arrays, one per step).
    """
    if fixsymmetry:
        try:
            atoms.set_constraint(FixSymmetry(atoms))
        except Exception as exc:
            print(f"WARNING: could not fix symmetry: {exc}")

    target = FrechetCellFilter(atoms) if relax_cell else atoms

    history = {k: [] for k in (
        'steps', 'energy_total_eV', 'energy_per_atom_eV',
        'a', 'b', 'c', 'alpha', 'beta', 'gamma', 'volume',
        'max_atomic_force', 'frac_coords',
    )}
    n_atoms = len(atoms)

    ase_traj = Trajectory(str(traj_path), 'w', atoms) if traj_path else None

    def _log():
        e = atoms.get_potential_energy()
        forces = atoms.get_forces()
        cell = atoms.get_cell()
        a, b, c, alpha, beta, gamma = cell.cellpar()

        history['steps'].append(len(history['steps']))
        history['energy_total_eV'].append(float(e))
        history['energy_per_atom_eV'].append(float(e) / n_atoms)
        history['a'].append(a); history['b'].append(b); history['c'].append(c)
        history['alpha'].append(alpha); history['beta'].append(beta); history['gamma'].append(gamma)
        history['volume'].append(float(cell.volume))
        history['max_atomic_force'].append(float(np.abs(forces).max()))
        history['frac_coords'].append(atoms.get_scaled_positions().copy())
        if ase_traj is not None:
            ase_traj.write()

    dyn = FIRE(target, logfile=None)
    dyn.attach(_log, interval=1)
    _log()  # step 0, before any optimizer step
    dyn.run(fmax=fmax, steps=steps)

    if ase_traj is not None:
        ase_traj.close()

    return history


def save_energy_csv(history, out_path):
    cols = ['steps', 'energy_total_eV', 'energy_per_atom_eV',
            'a', 'b', 'c', 'alpha', 'beta', 'gamma', 'volume', 'max_atomic_force']
    df = pd.DataFrame({c: history[c] for c in cols}).rename(columns={'steps': 'step'})
    df.to_csv(out_path, index=False)
    print(f'Saved {out_path}')
    return df


def save_positions_csv(history, species, out_path):
    rows = []
    for step, frac in zip(history['steps'], history['frac_coords']):
        for atom_index, (sp, xyz) in enumerate(zip(species, frac)):
            rows.append({'step': step, 'atom_index': atom_index, 'species': sp,
                         'frac_x': xyz[0], 'frac_y': xyz[1], 'frac_z': xyz[2]})
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f'Saved {out_path} ({len(df)} rows)')
    return df


def compute_rmsd_trajectory(history, species, gt_cif, out_path,
                             stol=0.5, angle_tol=10, ltol=0.3):
    """RMSD, via pymatgen StructureMatcher, between each FIRE step and a
    ground-truth structure -- reuses compute_metrics.py's Crystal,
    get_rms_dist, and get_gt_crys_ori directly (the same primitives
    relax_wbm.py's --structure_match_check uses for the diffusion
    trajectory), not reimplemented here. Lazy import so a normal run
    (without --gt_cif_file/--gt_file) never pays compute_metrics.py's
    import-time cost (matminer/pyxtal featurizer setup).

    history: relax_and_log's own output -- frac_coords and lattice params
    are already logged per step, nothing needs re-reading from disk.
    gt_cif: ground-truth structure as a CIF string.

    compute_valid=False deliberately -- compute_valid=True triggers
    smact_validity() (composition sanity check), which on some cluster
    environments crashes with a UnicodeDecodeError reading one of smact's
    own data files (a locale/encoding issue, not a real invalidity). The
    proper fix is environment-level (set PYTHONUTF8=1 or a UTF-8 locale
    before running Python); this default just avoids depending on that
    being fixed. StructureMatcher itself still returns None on a genuine
    geometric mismatch regardless of this flag.
    """
    from compute_metrics import Crystal, get_rms_dist, get_gt_crys_ori
    from pymatgen.core.periodic_table import Element
    from pymatgen.analysis.structure_matcher import StructureMatcher

    atom_types = np.array([Element(s).Z for s in species])
    gt = get_gt_crys_ori(gt_cif, compute_valid=False, compute_fp=False)
    matcher = StructureMatcher(stol=stol, angle_tol=angle_tol, ltol=ltol)

    rows = []
    for i, step in enumerate(history['steps']):
        crys_array_dict = {
            'frac_coords': history['frac_coords'][i],
            'atom_types': atom_types,
            'lengths': np.array([history['a'][i], history['b'][i], history['c'][i]]),
            'angles': np.array([history['alpha'][i], history['beta'][i], history['gamma'][i]]),
        }
        pred = Crystal(crys_array_dict, compute_valid=False, compute_fp=False)
        rmsd = get_rms_dist(pred, gt, pred.valid, matcher)
        rows.append({'step': step, 'rmsd': rmsd})

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f'Saved {out_path}')
    return df


def plot_energy(df, out_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df['step'], df['energy_per_atom_eV'], marker='o', markersize=3)
    ax.set_xlabel('step')
    ax.set_ylabel('energy per atom (eV)')
    ax.set_title('MACE relaxation: energy per atom vs step')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'Saved {out_path}')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cif_file', required=True, help='path to the input .cif structure')
    ap.add_argument('--model_path', default='medium',
                     help="MACE model checkpoint path, OR a size keyword "
                          "('small'/'medium'/'large') for the pretrained MACE-MP-0 "
                          "foundation model. mace_mp() already distinguishes a local "
                          "path from a keyword internally -- passing a keyword "
                          "downloads the model automatically on first use (needs "
                          "network access once, then caches locally). Default: "
                          "'medium'.")
    ap.add_argument('--device', default='cuda', help="'cuda' or 'cpu'")
    ap.add_argument('--default_dtype', default='float64',
                     help="default 'float64' here, vs relax_wbm.py's default 'float32' "
                          "-- if comparing energies against a relax_wbm.py "
                          "--mace_energy_check run, pass --default_dtype float32 to match.")
    ap.add_argument('--dispersion', action='store_true',
                     help='include a Grimme D3-style dispersion correction. Default OFF, '
                          "matching relax_structures.py's convention.")
    ap.add_argument('--fmax', type=float, default=0.05,
                     help='force-convergence threshold, eV/A (default 0.05)')
    ap.add_argument('--steps', type=int, default=500, help='max relaxation steps')
    ap.add_argument('--no-relax-cell', dest='relax_cell', action='store_false', default=True,
                     help='fix the cell (relax atomic positions only)')
    ap.add_argument('--fixsymmetry', action='store_true',
                     help='fix the space-group symmetry during relaxation')
    ap.add_argument('--label', default=None, help='output filename prefix (default: cif stem)')
    ap.add_argument('--outdir', default=None, help='output directory (default: next to the cif)')
    ap.add_argument('--gt_cif_file', default=None,
                     help='path to a single ground-truth .cif -- if given (or --gt_file '
                          '+ --material_id), also computes RMSD vs. this structure at '
                          'every FIRE step, saved to <label>_rmsd_trajectory.csv')
    ap.add_argument('--gt_file', default=None,
                     help="CSV with 'material_id' and 'cif' columns (same convention as "
                          "relax_wbm.py/compute_metrics.py's --gt_file) -- alternative to "
                          "--gt_cif_file. Needs --material_id too.")
    ap.add_argument('--material_id', default=None,
                     help='which row to use from --gt_file (ignored with --gt_cif_file)')
    args = ap.parse_args()

    cif_path = Path(args.cif_file)
    label = args.label or cif_path.stem
    outdir = Path(args.outdir) if args.outdir else cif_path.parent
    outdir.mkdir(parents=True, exist_ok=True)

    struct = Structure.from_file(str(cif_path))
    species = [str(s) for s in struct.species]
    atoms = AseAtomsAdaptor().get_atoms(struct)
    atoms.calc = build_mace_calculator(args.model_path, args.device,
                                        default_dtype=args.default_dtype,
                                        dispersion=args.dispersion)

    print(f'Relaxing {cif_path.name} ({len(atoms)} atoms), '
          f'fmax={args.fmax}, max steps={args.steps}, relax_cell={args.relax_cell}...')
    history = relax_and_log(
        atoms, fmax=args.fmax, steps=args.steps, relax_cell=args.relax_cell,
        fixsymmetry=args.fixsymmetry, traj_path=outdir / f'{label}.traj',
    )

    n_steps = len(history['steps'])
    print(f'Done: {n_steps} steps logged. '
          f'{history["energy_per_atom_eV"][0]:.4f} -> {history["energy_per_atom_eV"][-1]:.4f} eV/atom')

    energy_df = save_energy_csv(history, outdir / f'{label}_energy_trajectory.csv')
    save_positions_csv(history, species, outdir / f'{label}_positions_trajectory.csv')
    plot_energy(energy_df, outdir / f'{label}_energy_trajectory.png')

    gt_cif = None
    if args.gt_cif_file:
        gt_cif = Path(args.gt_cif_file).read_text()
    elif args.gt_file:
        if not args.material_id:
            raise SystemExit("--gt_file needs --material_id too.")
        gt_csv = pd.read_csv(args.gt_file)
        row = gt_csv[gt_csv['material_id'] == args.material_id]
        if len(row) == 0:
            raise SystemExit(f"material_id {args.material_id!r} not found in {args.gt_file}")
        gt_cif = row['cif'].iloc[0]

    if gt_cif is not None:
        compute_rmsd_trajectory(history, species, gt_cif,
                                 outdir / f'{label}_rmsd_trajectory.csv')


if __name__ == '__main__':
    main()