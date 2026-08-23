import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from p_tqdm import p_map

# ---------------------------------------------------------------------------
# VERIFY these two imports against your actual repo before running.
#   - build_crystal: grep -n "def build_crystal" diffcsp/common/data_utils.py
#     Expected signature roughly: build_crystal(crystal_str, niggli=True,
#     primitive=False) -> pymatgen Structure (already reduced/canonicalized).
#     If the real signature/name differs, adjust build_crystal_wrapper() below
#     -- everything else in this script only depends on getting back a
#     pymatgen Structure with the SAME reduction the training pipeline uses.
#   - lattice_polar_decompose_torch: same import used elsewhere in your
#     flow.py / sample() code.
# ---------------------------------------------------------------------------
from diffcsp.common.data_utils import build_crystal
from diffcsp.common.data_utils import lattice_polar_decompose_torch


def build_crystal_wrapper(cif_str, niggli=True, primitive=False, tol=0.01):
    """Thin wrapper so the niggli/primitive/tol settings are set in ONE place
    and can be checked against whatever your data config actually uses
    (see niggli / primitive / tolerance in conf/data/*.yaml)."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            structure = build_crystal(cif_str, niggli=niggli, primitive=primitive)
        return structure
    except Exception:
        return None


def parse_row(row, niggli, primitive, tol):
    """Parse one (material_id, cif) row through the REPO'S OWN preprocessing,
    not a raw pymatgen parse. Returns (material_id, Structure_or_None)."""
    structure = build_crystal_wrapper(row['cif'], niggli=niggli, primitive=primitive, tol=tol)
    return row['material_id'], structure


# ---------------------------------------------------------------------------
# Diagnostic (mean-of-norm) metrics -- kept for reference / plotting, but NOT
# used for the sigma inversion below, since E[norm] requires an awkward
# chi-distribution correction factor (sqrt(8/pi) for 3D, a Gamma-function
# ratio for 6D). See mic_coord_sq_dists / lattice_polar_sq_error for the
# metrics actually used to recover coord_noise / lattice_noise.
# ---------------------------------------------------------------------------
def mic_coord_rmsd(frac_a, frac_b):
    """Mean per-atom MIC displacement between two fractional coordinate arrays.
    Assumes frac_a, frac_b are aligned by atom index (same order)."""
    diff = (frac_a - frac_b - 0.5) % 1.0 - 0.5
    dists = np.sqrt((diff ** 2).sum(axis=-1))  # per-atom distance
    return dists.mean()


def lattice_realspace_error(struct_a, struct_b):
    """Relative length error and absolute angle error (degrees) -- real-space,
    NOT comparable to lattice_noise directly, kept only as a secondary diagnostic."""
    a_abc = np.array(struct_a.lattice.abc)
    b_abc = np.array(struct_b.lattice.abc)
    a_ang = np.array(struct_a.lattice.angles)
    b_ang = np.array(struct_b.lattice.angles)

    length_rel_err = np.abs(a_abc - b_abc) / b_abc  # relative to relaxed (ground truth)
    angle_abs_err = np.abs(a_ang - b_ang)
    return length_rel_err.mean(), angle_abs_err.mean()


def lattice_polar_error(struct_a, struct_b):
    """L2 distance in lattice_polar space -- same space as lattice_noise, but
    a mean-of-norm summary, so it needs a Gamma-function correction to invert
    exactly. Kept as a secondary diagnostic; see lattice_polar_sq_error for
    the version actually used to recover lattice_noise."""
    mat_a = torch.tensor(struct_a.lattice.matrix, dtype=torch.float32).unsqueeze(0)
    mat_b = torch.tensor(struct_b.lattice.matrix, dtype=torch.float32).unsqueeze(0)
    polar_a = lattice_polar_decompose_torch(mat_a)
    polar_b = lattice_polar_decompose_torch(mat_b)
    return torch.norm(polar_a - polar_b, dim=-1).item()


# ---------------------------------------------------------------------------
# Squared-distance metrics -- these are what we actually invert to recover
# coord_noise / lattice_noise, since E[sum of squares] = d * sigma^2 exactly,
# with no distributional correction factor needed (d = 3 for coords, d = 6
# for lattice_polar). This matches exactly how perturb_batch() draws noise:
# i.i.d. N(0, sigma^2) per component (per atom per axis for coords, per
# lattice_polar component for the lattice).
# ---------------------------------------------------------------------------
def mic_coord_sq_dists(frac_a, frac_b):
    """Per-atom SQUARED MIC displacement: dx^2 + dy^2 + dz^2 per atom.
    Returns an array of shape (n_atoms,) -- pool these across ALL atoms in
    ALL structures (not per-structure averages first) before inverting,
    since perturb_batch draws noise independently per atom."""
    diff = (frac_a - frac_b - 0.5) % 1.0 - 0.5
    return (diff ** 2).sum(axis=-1)


def lattice_polar_sq_error(struct_a, struct_b):
    """Squared L2 distance in lattice_polar (6D) space, for ONE structure
    pair. Pool these across structures (there is one lattice per structure,
    unlike coordinates which have one value per atom)."""
    mat_a = torch.tensor(struct_a.lattice.matrix, dtype=torch.float32).unsqueeze(0)
    mat_b = torch.tensor(struct_b.lattice.matrix, dtype=torch.float32).unsqueeze(0)
    polar_a = lattice_polar_decompose_torch(mat_a)
    polar_b = lattice_polar_decompose_torch(mat_b)
    return (polar_a - polar_b).pow(2).sum().item()


def main(args):
    df_init = pd.read_csv(args.initial_csv)
    df_relaxed = pd.read_csv(args.relaxed_csv)

    print(f'Loaded {len(df_init)} initial / {len(df_relaxed)} relaxed rows.')

    merged = pd.merge(df_init, df_relaxed, on='material_id', suffixes=('_init', '_relaxed'))
    print(f'{len(merged)} matched pairs after merge on material_id.')

    if args.max_structures is not None:
        merged = merged.sample(n=min(args.max_structures, len(merged)), random_state=0)
        print(f'Subsampled to {len(merged)} pairs.')

    init_rows = merged[['material_id']].assign(cif=merged['cif_init']).to_dict('records')
    relaxed_rows = merged[['material_id']].assign(cif=merged['cif_relaxed']).to_dict('records')

    print(f'Parsing initial structures (niggli={args.niggli}, primitive={args.primitive})...')
    init_parsed = dict(p_map(
        lambda r: parse_row(r, args.niggli, args.primitive, args.tolerance),
        init_rows, num_cpus=args.njobs,
    ))
    print(f'Parsing relaxed structures (niggli={args.niggli}, primitive={args.primitive})...')
    relaxed_parsed = dict(p_map(
        lambda r: parse_row(r, args.niggli, args.primitive, args.tolerance),
        relaxed_rows, num_cpus=args.njobs,
    ))

    # secondary / diagnostic (mean-of-norm) arrays, one entry per structure
    coord_rmsds = []
    length_errs = []
    angle_errs = []
    lattice_polar_errs = []
    n_sites_list = []

    # arrays used for the actual sigma inversion
    coord_sq_dists_all = []   # pooled across every atom, every structure
    lattice_sq_errs = []      # one value per structure pair

    skipped_parse_fail = 0
    skipped_site_mismatch = 0
    skipped_species_mismatch = 0

    for mat_id in merged['material_id']:
        s_init = init_parsed.get(mat_id)
        s_relaxed = relaxed_parsed.get(mat_id)

        if s_init is None or s_relaxed is None:
            skipped_parse_fail += 1
            continue

        if len(s_init) != len(s_relaxed):
            skipped_site_mismatch += 1
            continue

        # Sanity check: same species in same order. NOTE this is a NECESSARY
        # but not SUFFICIENT check -- niggli/primitive reduction run
        # independently on two related-but-different structures can in
        # principle produce a matching species sequence with atoms actually
        # swapped in identity. This check catches gross mismatches only.
        species_init = [site.specie.symbol for site in s_init]
        species_relaxed = [site.specie.symbol for site in s_relaxed]
        if species_init != species_relaxed:
            skipped_species_mismatch += 1
            continue

        # diagnostics
        rmsd = mic_coord_rmsd(s_init.frac_coords, s_relaxed.frac_coords)
        len_err, ang_err = lattice_realspace_error(s_init, s_relaxed)
        polar_err = lattice_polar_error(s_init, s_relaxed)

        coord_rmsds.append(rmsd)
        length_errs.append(len_err)
        angle_errs.append(ang_err)
        lattice_polar_errs.append(polar_err)
        n_sites_list.append(len(s_init))

        # squared-distance quantities used for sigma inversion
        sq_dists = mic_coord_sq_dists(s_init.frac_coords, s_relaxed.frac_coords)
        coord_sq_dists_all.extend(sq_dists.tolist())

        lattice_sq_errs.append(
            lattice_polar_sq_error(s_init, s_relaxed)
        )

    coord_rmsds = np.array(coord_rmsds)
    length_errs = np.array(length_errs)
    angle_errs = np.array(angle_errs)
    lattice_polar_errs = np.array(lattice_polar_errs)
    coord_sq_dists_all = np.array(coord_sq_dists_all)
    lattice_sq_errs = np.array(lattice_sq_errs)

    def stats(arr):
        return {
            'mean': float(arr.mean()),
            'median': float(np.median(arr)),
            'std': float(arr.std()),
            'p10': float(np.percentile(arr, 10)),
            'p25': float(np.percentile(arr, 25)),
            'p75': float(np.percentile(arr, 75)),
            'p90': float(np.percentile(arr, 90)),
        }

    # --- exact sigma inversion ---------------------------------------------
    # E[coord_sq_dist]   = 3 * sigma_coord^2    -> sigma_coord   = sqrt(mean/3)
    # E[lattice_sq_err]  = 6 * sigma_lattice^2  -> sigma_lattice = sqrt(mean/6)
    coord_noise_estimate = float(np.sqrt(coord_sq_dists_all.mean() / 3.0))
    lattice_noise_estimate = float(np.sqrt(lattice_sq_errs.mean() / 6.0))

    # optional: bootstrap a rough confidence interval on the two estimates
    def bootstrap_sigma(sq_arr, dof, n_boot=1000, seed=0):
        rng = np.random.default_rng(seed)
        n = len(sq_arr)
        boot_means = rng.choice(sq_arr, size=(n_boot, n), replace=True).mean(axis=1)
        boot_sigmas = np.sqrt(boot_means / dof)
        return float(np.percentile(boot_sigmas, 2.5)), float(np.percentile(boot_sigmas, 97.5))

    coord_noise_ci = bootstrap_sigma(coord_sq_dists_all, dof=3.0)
    lattice_noise_ci = bootstrap_sigma(lattice_sq_errs, dof=6.0)

    summary = {
        'niggli': args.niggli,
        'primitive': args.primitive,
        'tolerance': args.tolerance,
        'n_pairs_total': len(merged),
        'n_pairs_used': len(coord_rmsds),
        'n_atoms_pooled_for_coord_estimate': int(len(coord_sq_dists_all)),
        'skipped_parse_fail': skipped_parse_fail,
        'skipped_site_mismatch': skipped_site_mismatch,
        'skipped_species_mismatch': skipped_species_mismatch,

        # *** these are the numbers to plug into perturb_batch() ***
        'coord_noise_estimate': coord_noise_estimate,
        'coord_noise_estimate_95ci': coord_noise_ci,
        'lattice_noise_estimate': lattice_noise_estimate,
        'lattice_noise_estimate_95ci': lattice_noise_ci,

        # diagnostics (mean-of-norm, NOT directly the noise sigma)
        'coord_rmsd_meanofnorm': stats(coord_rmsds),
        'lattice_polar_err_meanofnorm': stats(lattice_polar_errs),

        # secondary diagnostics, real-space units, NOT comparable to lattice_noise
        'length_rel_err': stats(length_errs),
        'angle_abs_err_deg': stats(angle_errs),
    }

    print('\n=== WBM initial -> relaxed noise calibration (repo preprocessing) ===')
    print(json.dumps(summary, indent=2))

    print(f'\n>>> Use coord_noise ~= {coord_noise_estimate:.4f} '
          f'(95% CI [{coord_noise_ci[0]:.4f}, {coord_noise_ci[1]:.4f}]) in perturb_batch().')
    print(f'>>> Use lattice_noise ~= {lattice_noise_estimate:.4f} '
          f'(95% CI [{lattice_noise_ci[0]:.4f}, {lattice_noise_ci[1]:.4f}]) in perturb_batch().')
    print('\nThese are exact inversions (sigma = sqrt(mean_sq / dof)), not chi-distribution')
    print('approximations, and match how perturb_batch() actually draws noise: i.i.d. per-atom')
    print('per-axis Gaussian for coordinates, i.i.d. per-component Gaussian for lattice_polar.')
    print('\n(Note: synthetic noise is iid per-atom/per-dim Gaussian; WBM distortion is')
    print('structured/correlated, so this is a magnitude match, not a shape match.)')
    print(f'\nskipped_species_mismatch={skipped_species_mismatch} -- if this is nontrivial relative to')
    print('n_pairs_used, independent niggli reduction may be producing inconsistent atom ordering')
    print('between initial/relaxed pairs; inspect a few of those cases before trusting the rest.')

    np.savez(
        Path(args.out).with_suffix('.npz'),
        coord_rmsd=coord_rmsds, length_err=length_errs, angle_err=angle_errs,
        lattice_polar_err=lattice_polar_errs, n_sites=np.array(n_sites_list),
        coord_sq_dists_all=coord_sq_dists_all, lattice_sq_errs=lattice_sq_errs,
    )
    with open(args.out, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\nSaved summary to {args.out}')
    print(f'Saved per-structure/per-atom arrays to {Path(args.out).with_suffix(".npz")}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--initial_csv', required=True)
    parser.add_argument('--relaxed_csv', required=True)
    parser.add_argument('--out', default='wbm_distance_summary.json')
    parser.add_argument('--max_structures', type=int, default=None,
                         help='subsample for a quick first pass; None = use all')
    parser.add_argument('-j', '--njobs', type=int, default=16)

    # MUST match whatever your training data config (conf/data/*.yaml) uses,
    # e.g. flow_polar.yaml's niggli/primitive/tolerance settings, so the
    # representation here matches training exactly.
    parser.add_argument('--niggli', type=lambda s: s.lower() != 'false', default=True,
                         help='match your data config niggli setting (default True)')
    parser.add_argument('--primitive', type=lambda s: s.lower() != 'false', default=False,
                         help='match your data config primitive setting (default False)')
    parser.add_argument('--tolerance', type=float, default=0.01,
                         help='match your data config tolerance setting')

    args = parser.parse_args()
    main(args)