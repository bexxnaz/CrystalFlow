"""
Measure the real unrelaxed -> relaxed distance in WBM, using the same
MIC-RMSD (coords) and lattice_polar-space distance (lattice) definitions
your model actually trains/perturbs with, so the result is directly
comparable to the coord_noise / lattice_noise sweep in relax_eval.py.

No model involved -- this purely characterizes how far WBM's initial
(post-substitution, pre-DFT-relaxation) structures sit from their DFT-relaxed
targets, to calibrate which part of the synthetic noise sweep is realistic.

Usage:
    python wbm_distance.py \
        --initial_csv wbm_initial_crystalflow_le20atoms.csv \
        --relaxed_csv wbm_test_crystalflow_le20atoms.csv \
        --out wbm_distance_summary.json
"""


import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from p_tqdm import p_map
from pymatgen.core.structure import Structure
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# adjust this import to match your actual repo layout
from diffcsp.common.data_utils import (
    EPSILON,
    cart_to_frac_coords,
    frac_to_cart_coords,
    lattice_params_to_matrix_torch,
    lattice_polar_build_torch,
    lattice_polar_decompose_torch,
    lengths_angles_to_volume,
    mard,
    min_distance_sqr_pbc,
)


def parse_row(row):
    """Parse one (material_id, cif) row into a pymatgen Structure, or None on failure."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            structure = Structure.from_str(row['cif'], fmt='cif')
        return row['material_id'], structure
    except Exception:
        return row['material_id'], None


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
    """L2 distance in lattice_polar space -- DIRECTLY comparable to lattice_noise,
    since this is the exact representation the model trains and is perturbed in."""
    mat_a = torch.tensor(struct_a.lattice.matrix, dtype=torch.float32).unsqueeze(0)
    mat_b = torch.tensor(struct_b.lattice.matrix, dtype=torch.float32).unsqueeze(0)
    polar_a = lattice_polar_decompose_torch(mat_a)
    polar_b = lattice_polar_decompose_torch(mat_b)
    return torch.norm(polar_a - polar_b, dim=-1).item()


def main(args):
    df_init = pd.read_csv(args.initial_csv)
    df_relaxed = pd.read_csv(args.relaxed_csv)

    print(f'Loaded {len(df_init)} initial / {len(df_relaxed)} relaxed rows.')

    merged = pd.merge(df_init, df_relaxed, on='material_id', suffixes=('_init', '_relaxed'))
    print(f'{len(merged)} matched pairs after merge on material_id.')

    if args.max_structures is not None:
        merged = merged.sample(n=min(args.max_structures, len(merged)), random_state=0)
        print(f'Subsampled to {len(merged)} pairs.')

    init_rows = merged[['material_id']].assign(cif=merged['cif_init'])
    relaxed_rows = merged[['material_id']].assign(cif=merged['cif_relaxed'])

    print('Parsing initial structures...')
    init_parsed = dict(p_map(parse_row, init_rows.to_dict('records'), num_cpus=args.njobs))
    print('Parsing relaxed structures...')
    relaxed_parsed = dict(p_map(parse_row, relaxed_rows.to_dict('records'), num_cpus=args.njobs))

    coord_rmsds = []
    length_errs = []
    angle_errs = []
    lattice_polar_errs = []
    n_sites_list = []
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

        species_init = [site.specie.symbol for site in s_init]
        species_relaxed = [site.specie.symbol for site in s_relaxed]
        if species_init != species_relaxed:
            skipped_species_mismatch += 1
            continue

        rmsd = mic_coord_rmsd(s_init.frac_coords, s_relaxed.frac_coords)
        len_err, ang_err = lattice_realspace_error(s_init, s_relaxed)
        polar_err = lattice_polar_error(s_init, s_relaxed)

        coord_rmsds.append(rmsd)
        length_errs.append(len_err)
        angle_errs.append(ang_err)
        lattice_polar_errs.append(polar_err)
        n_sites_list.append(len(s_init))

    coord_rmsds = np.array(coord_rmsds)
    length_errs = np.array(length_errs)
    angle_errs = np.array(angle_errs)
    lattice_polar_errs = np.array(lattice_polar_errs)

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

    summary = {
        'n_pairs_total': len(merged),
        'n_pairs_used': len(coord_rmsds),
        'skipped_parse_fail': skipped_parse_fail,
        'skipped_site_mismatch': skipped_site_mismatch,
        'skipped_species_mismatch': skipped_species_mismatch,

        'coord_rmsd': stats(coord_rmsds),                # compare directly to coord_noise
        'lattice_polar_err': stats(lattice_polar_errs),  # compare directly to lattice_noise

        # secondary diagnostics, real-space units, NOT directly comparable to lattice_noise
        'length_rel_err': stats(length_errs),
        'angle_abs_err_deg': stats(angle_errs),
    }

    print('\n=== WBM initial -> relaxed distance summary ===')
    print(json.dumps(summary, indent=2))

    print('\ncoord_rmsd is directly comparable to your coord_noise sweep (0.02 / 0.05 / 0.1 / 0.2 / 0.4).')
    print('lattice_polar_err is directly comparable to your lattice_noise sweep -- both live in the')
    print('same lattice_polar representation the model was trained and perturbed in.')
    print('length_rel_err / angle_abs_err_deg are real-space diagnostics only; do NOT compare them')
    print('directly to lattice_noise, the mapping between the two spaces is nonlinear.')
    print('\n(Note: synthetic noise is iid per-atom/per-dim Gaussian; WBM distortion is')
    print('structured/correlated, so this is a magnitude comparison, not a shape one.)')

    np.savez(
        Path(args.out).with_suffix('.npz'),
        coord_rmsd=coord_rmsds, length_err=length_errs, angle_err=angle_errs,
        lattice_polar_err=lattice_polar_errs, n_sites=np.array(n_sites_list),
    )
    with open(args.out, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\nSaved summary to {args.out}')
    print(f'Saved per-structure arrays to {Path(args.out).with_suffix(".npz")}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--initial_csv', required=True)
    parser.add_argument('--relaxed_csv', required=True)
    parser.add_argument('--out', default='wbm_distance_summary.json')
    parser.add_argument('--max_structures', type=int, default=None,
                         help='subsample for a quick first pass; None = use all')
    parser.add_argument('-j', '--njobs', type=int, default=16)
    args = parser.parse_args()
    main(args)