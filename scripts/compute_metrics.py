import argparse
import json
import os
import pickle
import sys
import warnings
from collections import Counter
from pathlib import Path
from functools import partial

import numpy as np
import pandas as pd
from matminer.featurizers.composition.composite import ElementProperty
from matminer.featurizers.site.fingerprint import CrystalNNFingerprint
from p_tqdm import p_map
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core.composition import Composition
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pyxtal import pyxtal
from scipy.stats import wasserstein_distance
from tqdm import tqdm
# from joblib import Parallel, delayed

sys.path.append('.')

from eval_utils import (
    CompScaler,
    compute_cov,
    get_crystals_list,
    get_fp_pdist,
    load_config,
    load_data,
    prop_model_eval,
    smact_validity,
    structure_validity,
)

CrystalNNFP = CrystalNNFingerprint.from_preset("ops")
CompFP = ElementProperty.from_preset('magpie')

Percentiles = {
    'mp20': np.array([-3.17562208, -2.82196882, -2.52814761]),
    'carbon': np.array([-154.527093, -154.45865733, -154.44206825]),
    'perovskite': np.array([0.43924842, 0.61202443, 0.7364607]),
}

COV_Cutoffs = {
    'mp20': {'struc': 0.4, 'comp': 10.},
    'carbon': {'struc': 0.2, 'comp': 4.},
    'perovskite': {'struc': 0.2, 'comp': 4},
}


class Crystal(object):

    def __init__(self, crys_array_dict, compute_valid=True, compute_fp=False, ignore_smact=False):
        self.frac_coords = crys_array_dict['frac_coords']
        self.atom_types = crys_array_dict['atom_types']
        self.lengths = crys_array_dict['lengths']
        self.angles = crys_array_dict['angles']
        self.dict = crys_array_dict
        if len(self.atom_types.shape) > 1:
            self.dict['atom_types'] = (np.argmax(self.atom_types, axis=-1) + 1)
            self.atom_types = (np.argmax(self.atom_types, axis=-1) + 1)

        self.get_structure()
        self.get_composition()

        self.ignore_smact = ignore_smact
        if compute_valid:
            self.get_validity()
        else:
            self.valid = self.comp_valid = self.struct_valid = True

        if compute_fp:
            self.get_fingerprints()
        else:
            self.comp_fp = self.struct_fp = None


    def get_structure(self):
        if min(self.lengths.tolist()) < 0:
            self.constructed = False
            self.invalid_reason = 'non_positive_lattice'
        if np.isnan(self.lengths).any() or np.isnan(self.angles).any() or  np.isnan(self.frac_coords).any():
            self.constructed = False
            self.invalid_reason = 'nan_value'
        else:
            try:
                self.structure = Structure(
                    lattice=Lattice.from_parameters(
                        *(self.lengths.tolist() + self.angles.tolist())),
                    species=self.atom_types, coords=self.frac_coords, coords_are_cartesian=False)
                self.constructed = True
            except Exception:
                self.constructed = False
                self.invalid_reason = 'construction_raises_exception'
            if self.structure.volume < 0.1:
                self.constructed = False
                self.invalid_reason = 'unrealistically_small_lattice'

    def get_composition(self):
        elem_counter = Counter(self.atom_types)
        composition = [(elem, elem_counter[elem])
                       for elem in sorted(elem_counter.keys())]
        elems, counts = list(zip(*composition))
        counts = np.array(counts)
        counts = counts / np.gcd.reduce(counts)
        self.elems = elems
        self.comps = tuple(counts.astype('int').tolist())

    def get_validity(self):
        if len(self.elems) >= 8:
            self.comp_valid = False
        else:
            self.comp_valid = smact_validity(self.elems, self.comps) if not self.ignore_smact else True
        if self.constructed:
            if _is_odd(self.structure):
                self.struct_valid = False
            else:
                self.struct_valid = structure_validity(self.structure)
        else:
            self.struct_valid = False
        self.valid = self.comp_valid and self.struct_valid

    def get_fingerprints(self):
        elem_counter = Counter(self.atom_types)
        comp = Composition(elem_counter)
        self.comp_fp = CompFP.featurize(comp)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                site_fps = [CrystalNNFP.featurize(self.structure, i) for i in range(len(self.structure))]
        except Exception:
            # counts crystal as invalid if fingerprint cannot be constructed.
            self.valid = False
            self.comp_fp = None
            self.struct_fp = None
            return
        self.struct_fp = np.array(site_fps).mean(axis=0)


def _is_odd(structure: Structure):
    lengths = np.array(structure.lattice.abc)
    angles = np.array(structure.lattice.angles)
    if any(angles < 10) or any(angles > 170):
        return True
    elif any(lengths / np.power(len(structure), 1 / 3) > 20):
        return True
    elif any(lengths / np.power(len(structure), 1 / 3) < 0.1):
        return True
    elif not (0.1 < (structure.volume / len(structure)) < 100):
        return True
    return False

def _exact_composition_match(pred, gt):
    """Exact stoichiometry match (not just reduced ratio) -- CSP is a
    fixed-composition task, so pred and gt should have the IDENTICAL
    multiset of atom types, not just the same reduced formula."""
    return Counter(pred.atom_types.tolist()) == Counter(gt.atom_types.tolist())


def get_rms_dist(pred: Crystal, gt, is_valid, matcher):
    """
    Returns {'rms_dist': float or None, 'category': str, 'detail': str}
    instead of a bare rmsd/None, so a failure can be attributed to a specific
    stage instead of collapsing into an undifferentiated None. `is_valid` is
    kept in the signature (unused) only so the existing p_map(..., validity, ...)
    call site doesn't need to change -- the checks below recompute pred/gt
    validity at finer granularity than the single boolean it used to be.
    """
    if not pred.constructed:
        return {'rms_dist': None, 'category': 'pred_not_constructed',
                'detail': getattr(pred, 'invalid_reason', 'unknown')}

    if _is_odd(pred.structure):
        lengths = pred.structure.lattice.abc
        angles = pred.structure.lattice.angles
        return {'rms_dist': None, 'category': 'pred_odd_geometry',
                'detail': f'lengths={tuple(round(x, 2) for x in lengths)} '
                          f'angles={tuple(round(x, 1) for x in angles)}'}

    if not pred.comp_valid:
        return {'rms_dist': None, 'category': 'pred_invalid_composition',
                'detail': f'formula={pred.structure.composition.reduced_formula}'}

    if not pred.struct_valid:
        dist_mat = pred.structure.distance_matrix
        np.fill_diagonal(dist_mat, np.inf)
        return {'rms_dist': None, 'category': 'pred_invalid_structure',
                'detail': f'min_interatomic_dist={dist_mat.min():.3f} A'}

    if not gt.valid:
        return {'rms_dist': None, 'category': 'gt_invalid',
                'detail': getattr(gt, 'invalid_reason', 'unknown')}

    if not _exact_composition_match(pred, gt):
        return {'rms_dist': None, 'category': 'composition_mismatch',
                'detail': (f'pred={pred.structure.composition.reduced_formula} '
                           f'gt={gt.structure.composition.reduced_formula} '
                           f'pred_natoms={len(pred.atom_types)} gt_natoms={len(gt.atom_types)}')}

    try:
        result = matcher.get_rms_dist(pred.structure, gt.structure)
    except Exception as e:
        return {'rms_dist': None, 'category': 'geometry_too_different',
                'detail': f'matcher exception: {e}'}

    if result is None:
        return {'rms_dist': None, 'category': 'geometry_too_different', 'detail': ''}

    return {'rms_dist': result[0], 'category': 'matched', 'detail': ''}


# Module-level matcher, built once -- NOT per-call, since StructureMatcher
# construction has nontrivial overhead and this runs over the full test set.
# stol=1.0, scale=False matches pred_vs_ref_struct_symmetry() in
# matbench_discovery/metrics/geo_opt.py (ltol=0.2, angle_tol=5 stay at
# pymatgen defaults -- MBD only overrides stol and scale).
_MBD_MATCHER = StructureMatcher(stol=1.0, scale=False)


def get_rms_dist_mbd(pred: Crystal, gt: Crystal):
    """Per-structure RMSD under the Matbench Discovery protocol.

    Mirrors pred_vs_ref_struct_symmetry()'s per-material computation:
    returns raw rmsd on success, None on any failure (unconstructed
    prediction, matcher returning no fit, or an exception). No fill-value
    substitution happens here -- that's a separate aggregation-time step,
    done in RecEvalMBD.get_metrics() to mirror calc_geo_opt_metrics().
    """
    if not pred.constructed:
        return None
    try:
        result = _MBD_MATCHER.get_rms_dist(pred.structure, gt.structure)
    except Exception:
        result = None
    if result is None:
        return None
    rmsd, max_dist = result
    return rmsd


class RecEvalMBD(object):
    """Matbench Discovery-faithful RMSD evaluator.

    Two-stage, matching their actual code split across two functions:
      1. get_rms_dist_mbd()   -- per-structure RMSD, NaN on failure
                                  (mirrors pred_vs_ref_struct_symmetry)
      2. get_metrics() below  -- fillna(1.0).mean() aggregation
                                  (mirrors calc_geo_opt_metrics)
    """

    def __init__(self, pred_crys, gt_crys, njobs=1):
        assert len(pred_crys) == len(gt_crys)
        self.preds = pred_crys
        self.gts = gt_crys
        self.njobs = njobs

    def get_metrics(self):
        if self.njobs > 1:
            raw = p_map(get_rms_dist_mbd, self.preds, self.gts,
                        num_cpus=self.njobs, ncols=79)
        else:
            raw = [get_rms_dist_mbd(p, g) for p, g in zip(self.preds, self.gts)]

        self.raw_rmsds = raw  # keep around for inspection, mirrors self.rms_dists in RecEval

        raw_series = pd.Series(raw, dtype=float)  # None -> NaN automatically

        mean_rmsd = raw_series.fillna(1.0).mean()      # THE leaderboard-comparable number
        median_rmsd = raw_series.fillna(1.0).median()
        n_matched = int(raw_series.notna().sum())

        return {
            'mbd_rmsd_mean': float(mean_rmsd),
            'mbd_rmsd_median': float(median_rmsd),
            'raw_match_rate': n_matched / len(raw_series),
            'raw_rmsd_matched_only_mean': float(raw_series.mean()),  # for your own transparency, not for direct leaderboard comparison
            'n_structures': len(raw_series),
        }

# ---------------------------------------------------------------------------
# Geometry-optimization evaluation: mirrors matbench-discovery's own
# two-stage split (matbench_discovery/structure/symmetry.py):
#   1. get_sym_info_from_structs()   -- symmetry computed ONCE per structure
#   2. pred_vs_ref_struct_symmetry() -- diffs + RMSD, matched by material_id
# then calc_geo_opt_metrics() (matbench_discovery/metrics/geo_opt.py) is
# imported and called directly for aggregation -- their exact code, not a
# reimplementation.
#
# NOTE on matching: the official pipeline matches pred/gt by material_id
# (set intersection of two structure dicts) and raises on duplicate/missing
# IDs. This repo's Crystal objects don't currently carry a material_id, so
# the code below matches pred_crys[i] to gt_crys[i] BY POSITION instead --
# only correct if crys_array_list and your gt_file/true_crystal_array_list
# were built in the same order. If you want the official ID-based matching
# (safer against silent misalignment), thread material_id through
# get_gt_crys_ori()/Crystal and switch the zip() below to dict lookups on
# shared IDs.
# ---------------------------------------------------------------------------
 
def get_sym_info_moyopy(structure: Structure, symprec: float, angle_tolerance=None):
    """Space group number and symmetry-operation count via moyopy, matching
    matbench_discovery/structure/symmetry.py::get_sym_info_from_structs()
    exactly: MoyoAdapter.from_py_obj() (NOT from_structure -- that method
    doesn't exist), and sym_data.operations.num_operations (NOT
    len(operations) -- operations is an object with this attribute, not a
    plain sequence). This is the default, faithful path."""
    try:
        import moyopy
        from moyopy.interface import MoyoAdapter
        moyo_cell = MoyoAdapter.from_py_obj(structure)
        sym_data = moyopy.MoyoDataset(
            moyo_cell, symprec=symprec, angle_tolerance=angle_tolerance
        )
        return sym_data.number, sym_data.operations.num_operations
    except Exception:
        return None, None
 
 
def get_sym_info_pymatgen(structure: Structure, symprec: float):
    """Fallback/cross-check path via pymatgen's SpacegroupAnalyzer (spglib).
    NOT what the official leaderboard uses -- get_sym_info_moyopy above is
    the faithful default. Useful only to sanity-check moyopy results
    against a second, independently-verified implementation."""
    try:
        sga = SpacegroupAnalyzer(structure, symprec=symprec)
        return sga.get_space_group_number(), len(sga.get_symmetry_operations())
    except Exception:
        return None, None
 
 
def _precompute_sym_info(crys_list, symprec, use_moyopy=True, njobs=1):
    """Stage 1: symmetry info computed ONCE per structure, mirroring
    get_sym_info_from_structs(). Returns a list of (spg_num, n_sym_ops)
    tuples, same length/order as crys_list; (None, None) for unconstructed
    or symmetry-finder-failed structures. Computing this once per unique
    structure (rather than once per pred/gt PAIR) avoids redundant moyopy
    calls -- important at WBM scale where recomputation adds up fast."""
    sym_fn = get_sym_info_moyopy if use_moyopy else get_sym_info_pymatgen
 
    def _one(crys):
        if not getattr(crys, 'constructed', False):
            return (None, None)
        return sym_fn(crys.structure, symprec)
 
    if njobs > 1:
        return p_map(_one, crys_list, num_cpus=njobs, ncols=79)
    return [_one(c) for c in crys_list]
 
 
def get_geo_opt_row(pred: Crystal, gt: Crystal, pred_sym, gt_sym):
    """Stage 2: diff + RMSD for one pred/gt pair, mirroring
    pred_vs_ref_struct_symmetry(). pred_sym / gt_sym are the PRECOMPUTED
    (spg_num, n_sym_ops) tuples from _precompute_sym_info() -- symmetry is
    NOT recomputed here."""
    row = {
        'structure_rmsd_vs_dft': np.nan,
        'n_sym_ops_diff': np.nan,
        'spg_num_diff': np.nan,
        'max_pair_dist': np.nan,
    }
 
    pred_spg, pred_nops = pred_sym
    gt_spg, gt_nops = gt_sym
    if pred_spg is not None and gt_spg is not None:
        row['spg_num_diff'] = pred_spg - gt_spg
    if pred_nops is not None and gt_nops is not None:
        row['n_sym_ops_diff'] = pred_nops - gt_nops
 
    if not pred.constructed:
        return row
 
    try:
        # order matches pred_vs_ref_struct_symmetry(): pred first, then ref/gt
        result = _MBD_MATCHER.get_rms_dist(pred.structure, gt.structure)
    except Exception:
        result = None
    if result is not None:
        rmsd, max_dist = result
        row['structure_rmsd_vs_dft'] = rmsd
        row['max_pair_dist'] = max_dist
 
    return row
 
 
def _calc_geo_opt_metrics_vendored(df_model_analysis: pd.DataFrame) -> dict:
    """Vendored copy of matbench_discovery.metrics.geo_opt.calc_geo_opt_metrics(),
    logic verified verbatim against the actual upstream source. Reimplemented
    locally because the matbench-discovery package now requires Python
    >=3.14, incompatible with this project's Python 3.11 environment.
 
    Uses plain string column/key names instead of the pymatviz.Key /
    matbench_discovery.enums.MbdKey enums -- since this function both
    consumes and produces those names itself (GeoOptEvalMBD builds the
    input DataFrame with exactly these column names), internal consistency
    is all that's required; the arithmetic below is copied unchanged from
    the real source, which is the part that actually needs to match
    upstream."""
    spg_diff = df_model_analysis['spg_num_diff']
    n_sym_ops_diff = df_model_analysis['n_sym_ops_diff']
    rmsd_vals = df_model_analysis['structure_rmsd_vs_dft']
 
    valid_sym_mask = spg_diff.notna()
    n_valid_sym = valid_sym_mask.sum()
 
    mean_rmsd = pd.to_numeric(rmsd_vals, errors='coerce').fillna(1.0).mean()
 
    sym_ops_mae = n_sym_ops_diff[valid_sym_mask].abs().mean()
 
    changed_mask = (spg_diff != 0) & valid_sym_mask
    sym_decreased = (n_sym_ops_diff < 0) & changed_mask
    sym_increased = (n_sym_ops_diff > 0) & changed_mask
    sym_matched = ~changed_mask & valid_sym_mask
 
    return {
        'structure_rmsd_vs_dft': float(mean_rmsd),
        'n_sym_ops_mae': float(sym_ops_mae),
        'symmetry_decrease': float(sym_decreased.sum() / n_valid_sym) if n_valid_sym > 0 else float('nan'),
        'symmetry_match': float(sym_matched.sum() / n_valid_sym) if n_valid_sym > 0 else float('nan'),
        'symmetry_increase': float(sym_increased.sum() / n_valid_sym) if n_valid_sym > 0 else float('nan'),
        'n_structures': int(n_valid_sym),
    }
 
 
class GeoOptEvalMBD(object):
    """Matbench Discovery-faithful geometry-optimization evaluator.
 
    Two-stage, matching the real symmetry.py split:
      1. _precompute_sym_info() on preds and on gts SEPARATELY (each
         structure's symmetry computed exactly once)
      2. get_geo_opt_row() diffs the precomputed results + RMSD, per pair
 
    then aggregates via matbench_discovery's REAL calc_geo_opt_metrics().
    """
 
    def __init__(self, pred_crys, gt_crys, njobs=1, symprec=1e-2, use_moyopy=True):
        assert len(pred_crys) == len(gt_crys)
        self.preds = pred_crys
        self.gts = gt_crys
        self.njobs = njobs
        self.symprec = symprec
        self.use_moyopy = use_moyopy
 
    def get_metrics(self):
        pred_sym = _precompute_sym_info(self.preds, self.symprec, self.use_moyopy, self.njobs)
        gt_sym = _precompute_sym_info(self.gts, self.symprec, self.use_moyopy, self.njobs)
 
        rows = [
            get_geo_opt_row(p, g, ps, gs)
            for p, g, ps, gs in zip(self.preds, self.gts, pred_sym, gt_sym)
        ]
        self.df_model_analysis = pd.DataFrame(rows)  # kept for inspection / CSV dump
 
        try:
            from matbench_discovery.metrics.geo_opt import calc_geo_opt_metrics
            self.used_vendored_metrics_fn = False
        except Exception:
            # Broad except deliberately: matbench-discovery's internals have
            # been seen to fail in ways beyond a plain ImportError (e.g. a
            # SyntaxError inside a transitively-imported submodule from a
            # corrupted/mismatched install). Any failure to import the real
            # function should fall back to the verified vendored copy rather
            # than crash the whole evaluation.
            calc_geo_opt_metrics = _calc_geo_opt_metrics_vendored
            self.used_vendored_metrics_fn = True
        mbd_metrics = calc_geo_opt_metrics(self.df_model_analysis)
        # mbd_metrics's own "n_structures"-type key means count of structures
        # with VALID SYMMETRY data (their docstring: "total number of
        # structures is counted based on valid symmetry data") -- NOT total
        # submitted and NOT count with valid RMSD. Don't confuse it with
        # n_structures_submitted below; that bug (silently overwriting their
        # real key with a different definition) is exactly what this
        # rewrite fixes.
 
        n_submitted = len(self.df_model_analysis)
        n_rmsd_valid = int(self.df_model_analysis['structure_rmsd_vs_dft'].notna().sum())
 
        return {
            **mbd_metrics,
            'n_structures_submitted': n_submitted,
            'n_rmsd_valid': n_rmsd_valid,
            'used_vendored_metrics_fn': self.used_vendored_metrics_fn,
        }

class RecEval(object):

    def __init__(self, pred_crys, gt_crys, stol=0.5, angle_tol=10, ltol=0.3, njobs=1):
        assert len(pred_crys) == len(gt_crys)
        self.matcher = StructureMatcher(
            stol=stol, angle_tol=angle_tol, ltol=ltol)
        self.preds = pred_crys
        self.gts = gt_crys
        self.njobs = njobs

    def get_match_rate_and_rms(self):
        validity = [c1.valid and c2.valid for c1, c2 in zip(self.preds, self.gts)]

        results = p_map(
            partial(get_rms_dist, matcher=self.matcher),
            self.preds,
            self.gts,
            validity,
            num_cpus=self.njobs,
            ncols=79,
        )
        self.results = results

        rms_dists = np.array([r['rms_dist'] for r in results])
        categories = [r['category'] for r in results]
        self.categories = categories

        match_rate = sum(rms_dists != None) / len(self.preds)
        matched_mask = rms_dists != None
        mean_rms_dist = rms_dists[matched_mask].mean() if matched_mask.any() else float('nan')

        category_counts = dict(Counter(categories))
        category_pcts = {k: v / len(self.preds) for k, v in category_counts.items()}

        return {
            'match_rate': match_rate,
            'rms_dist': mean_rms_dist,
            'category_counts': category_counts,
            'category_pcts': category_pcts,
        }

    def get_metrics(self):
        metrics = {}
        metrics.update(self.get_match_rate_and_rms())
        return metrics


class RecEvalBatch(object):

    def __init__(self, pred_crys, gt_crys, stol=0.5, angle_tol=10, ltol=0.3):
        self.matcher = StructureMatcher(
            stol=stol, angle_tol=angle_tol, ltol=ltol)
        self.preds = pred_crys
        self.gts = gt_crys
        self.batch_size = len(self.preds)

    def get_match_rate_and_rms(self):
        def process_one(pred, gt, is_valid):
            if not is_valid:
                return None
            try:
                rms_dist = self.matcher.get_rms_dist(
                    pred.structure, gt.structure)
                rms_dist = None if rms_dist is None else rms_dist[0]
                return rms_dist
            except Exception:
                return None

        rms_dists = []
        self.all_rms_dis = np.zeros((self.batch_size, len(self.gts)))
        for i in tqdm(range(len(self.preds[0]))):
            tmp_rms_dists = []
            for j in range(self.batch_size):
                rmsd = process_one(self.preds[j][i], self.gts[i], self.preds[j][i].valid)
                self.all_rms_dis[j][i] = rmsd
                if rmsd is not None:
                    tmp_rms_dists.append(rmsd)
            if len(tmp_rms_dists) == 0:
                rms_dists.append(None)
            else:
                rms_dists.append(np.min(tmp_rms_dists))

        rms_dists = np.array(rms_dists)
        match_rate = sum(rms_dists != None) / len(self.preds[0])
        mean_rms_dist = rms_dists[rms_dists != None].mean()
        return {
            'match_rate': match_rate,
            'rms_dist': mean_rms_dist
        }

    def get_metrics(self):
        metrics = {}
        metrics.update(self.get_match_rate_and_rms())
        return metrics



class GenEval(object):

    def __init__(self, pred_crys, gt_crys, n_samples=1000, eval_model_name=None):
        self.crys = pred_crys
        self.gt_crys = gt_crys
        self.n_samples = n_samples
        self.eval_model_name = eval_model_name

        valid_crys = [c for c in pred_crys if c.valid]
        if len(valid_crys) >= n_samples:
            sampled_indices = np.random.choice(
                len(valid_crys), n_samples, replace=False)
            self.valid_samples = [valid_crys[i] for i in sampled_indices]
        else:
            raise Exception(
                f'not enough valid crystals in the predicted set: {len(valid_crys)}/{n_samples}')

    def get_validity(self):
        comp_valid = np.array([c.comp_valid for c in self.crys]).mean()
        struct_valid = np.array([c.struct_valid for c in self.crys]).mean()
        valid = np.array([c.valid for c in self.crys]).mean()
        return {'comp_valid': comp_valid,
                'struct_valid': struct_valid,
                'valid': valid}


    def get_density_wdist(self):
        pred_densities = [c.structure.density for c in self.valid_samples]
        gt_densities = [c.structure.density for c in self.gt_crys]
        wdist_density = wasserstein_distance(pred_densities, gt_densities)
        return {'wdist_density': wdist_density}


    def get_num_elem_wdist(self):
        pred_nelems = [len(set(c.structure.species))
                       for c in self.valid_samples]
        gt_nelems = [len(set(c.structure.species)) for c in self.gt_crys]
        wdist_num_elems = wasserstein_distance(pred_nelems, gt_nelems)
        return {'wdist_num_elems': wdist_num_elems}

    def get_prop_wdist(self):
        if self.eval_model_name is not None:
            pred_props = prop_model_eval(self.eval_model_name, [
                                         c.dict for c in self.valid_samples])
            gt_props = prop_model_eval(self.eval_model_name, [
                                       c.dict for c in self.gt_crys])
            wdist_prop = wasserstein_distance(pred_props, gt_props)
            return {'wdist_prop': wdist_prop}
        else:
            return {'wdist_prop': None}

    def get_coverage(self):
        cutoff_dict = COV_Cutoffs[self.eval_model_name]
        (cov_metrics_dict, combined_dist_dict) = compute_cov(
            self.crys, self.gt_crys,
            struc_cutoff=cutoff_dict['struc'],
            comp_cutoff=cutoff_dict['comp'])
        return cov_metrics_dict

    def get_metrics(self):
        metrics = {}
        metrics.update(self.get_validity())
        metrics.update(self.get_density_wdist())
        # metrics.update(self.get_prop_wdist())
        metrics.update(self.get_num_elem_wdist())
        metrics.update(self.get_coverage())
        return metrics

class OptEval(object):

    def __init__(self, crys, num_opt=100, eval_model_name=None):
        """
        crys is a list of length (<step_opt> * <num_opt>),
        where <num_opt> is the number of different initialization for optimizing crystals,
        and <step_opt> is the number of saved crystals for each intialzation.
        default to minimize the property.
        """
        step_opt = int(len(crys) / num_opt)
        self.crys = crys
        self.step_opt = step_opt
        self.num_opt = num_opt
        self.eval_model_name = eval_model_name

    def get_success_rate(self):
        valid_indices = np.array([c.valid for c in self.crys])
        valid_indices = valid_indices.reshape(self.step_opt, self.num_opt)
        valid_x, valid_y = valid_indices.nonzero()
        props = np.ones([self.step_opt, self.num_opt]) * np.inf
        valid_crys = [c for c in self.crys if c.valid]
        if len(valid_crys) == 0:
            sr_5, sr_10, sr_15 = 0, 0, 0
        else:
            pred_props = prop_model_eval(self.eval_model_name, [
                                         c.dict for c in valid_crys])
            percentiles = Percentiles[self.eval_model_name]
            props[valid_x, valid_y] = pred_props
            best_props = props.min(axis=0)
            sr_5 = (best_props <= percentiles[0]).mean()
            sr_10 = (best_props <= percentiles[1]).mean()
            sr_15 = (best_props <= percentiles[2]).mean()
        return {'SR5': sr_5, 'SR10': sr_10, 'SR15': sr_15}

    def get_metrics(self):
        return self.get_success_rate()


def get_file_paths(root_path, task, label='', suffix='pt'):
    if label == '':
        out_name = f'eval_{task}.{suffix}'
    else:
        out_name = f'eval_{task}_{label}.{suffix}'
    out_name = os.path.join(root_path, out_name)
    return out_name


def get_crystal_array_list(file_path, batch_idx=0):
    data = load_data(file_path)
    if batch_idx == -1:
        batch_size = data['frac_coords'].shape[0]
        crys_array_list = []
        for i in range(batch_size):
            tmp_crys_array_list = get_crystals_list(
                data['frac_coords'][i],
                data['atom_types'][i],
                data['lengths'][i],
                data['angles'][i],
                data['num_atoms'][i])
            crys_array_list.append(tmp_crys_array_list)
    elif batch_idx == -2:
        crys_array_list = get_crystals_list(
            data['frac_coords'],
            data['atom_types'],
            data['lengths'],
            data['angles'],
            data['num_atoms'])
    else:
        crys_array_list = get_crystals_list(
            data['frac_coords'][batch_idx],
            data['atom_types'][batch_idx],
            data['lengths'][batch_idx],
            data['angles'][batch_idx],
            data['num_atoms'][batch_idx])

    if 'input_data_batch' in data:
        batch = data['input_data_batch']
        if isinstance(batch, dict):
            true_crystal_array_list = get_crystals_list(
                batch['frac_coords'], batch['atom_types'], batch['lengths'],
                batch['angles'], batch['num_atoms'])
        else:
            true_crystal_array_list = get_crystals_list(
                batch.frac_coords, batch.atom_types, batch.lengths,
                batch.angles, batch.num_atoms)
    else:
        true_crystal_array_list = None

    return crys_array_list, true_crystal_array_list


def get_gt_crys_ori(cif):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        structure = Structure.from_str(cif,fmt='cif')
    lattice = structure.lattice
    crys_array_dict = {
        'frac_coords':structure.frac_coords,
        'atom_types':np.array([_.Z for _ in structure.species]),
        'lengths': np.array(lattice.abc),
        'angles': np.array(lattice.angles)
    }
    return Crystal(crys_array_dict)

def main(args):
    all_metrics = {}

    cfg = load_config(args.root_path)
    eval_model_name = cfg.data.eval_model_name

    if 'opt' in args.tasks:
        opt_file_path = get_file_paths(args.root_path, 'opt', args.label)
        crys_array_list, _ = get_crystal_array_list(opt_file_path)
        opt_crys = p_map(lambda x: Crystal(x), crys_array_list, num_cpus=args.njobs)

        opt_evaluator = OptEval(opt_crys, eval_model_name=eval_model_name)
        opt_metrics = opt_evaluator.get_metrics()
        all_metrics.update(opt_metrics)

    elif 'gen' in args.tasks:

        gen_file_path = get_file_paths(args.root_path, 'gen', args.label)
        recon_file_path = get_file_paths(args.root_path, 'recon', args.label)
        crys_array_list, _ = get_crystal_array_list(gen_file_path, batch_idx = -2)
        gen_crys = p_map(lambda x: Crystal(x), crys_array_list, num_cpus=args.njobs)
        if args.gt_file != '':
            csv = pd.read_csv(args.gt_file)
            gt_crys = p_map(get_gt_crys_ori, csv['cif'], num_cpus=args.njobs)
        else:
            _, true_crystal_array_list = get_crystal_array_list(
                recon_file_path)
            gt_crys = p_map(lambda x: Crystal(x), true_crystal_array_list, num_cpus=args.njobs)
        gen_evaluator = GenEval(
            gen_crys, gt_crys, eval_model_name=eval_model_name)
        gen_metrics = gen_evaluator.get_metrics()
        all_metrics.update(gen_metrics)

    elif 'csp_mbd' in args.tasks:
        recon_file_path = get_file_paths(args.root_path, 'diff', args.label)
        crys_array_list, true_crystal_array_list = get_crystal_array_list(recon_file_path, batch_idx=0)
        if args.gt_file != '':
            csv = pd.read_csv(args.gt_file)
            gt_crys = p_map(get_gt_crys_ori, csv['cif'])
        else:
            gt_crys = p_map(lambda x: Crystal(x), true_crystal_array_list, num_cpus=args.njobs)
        pred_crys = p_map(lambda x: Crystal(x), crys_array_list, num_cpus=args.njobs)

        rec_evaluator = RecEvalMBD(pred_crys, gt_crys, njobs=args.njobs)
        all_metrics.update(rec_evaluator.get_metrics())

    elif 'geo_opt_mbd' in args.tasks:
        recon_file_path = get_file_paths(args.root_path, 'diff', args.label)
        crys_array_list, true_crystal_array_list = get_crystal_array_list(recon_file_path, batch_idx=0)
        if args.gt_file != '':
            csv = pd.read_csv(args.gt_file)
            gt_crys = p_map(get_gt_crys_ori, csv['cif'])
        else:
            gt_crys = p_map(lambda x: Crystal(x), true_crystal_array_list, num_cpus=args.njobs)
        pred_crys = p_map(lambda x: Crystal(x), crys_array_list, num_cpus=args.njobs)

        geo_opt_evaluator = GeoOptEvalMBD(
            pred_crys, gt_crys, njobs=args.njobs,
            symprec=args.symprec, use_moyopy=not args.use_pymatgen_symmetry,
        )
        all_metrics.update(geo_opt_evaluator.get_metrics())

        # dump the raw per-structure DataFrame alongside the aggregated metrics,
        # so you can inspect exactly which rows are NaN and why before trusting
        # the aggregated numbers at WBM scale.
        analysis_out_name = 'geo_opt_analysis.csv' if args.label == '' else f'geo_opt_analysis_{args.label}.csv'
        geo_opt_evaluator.df_model_analysis.to_csv(
            os.path.join(args.root_path, analysis_out_name), index=False
        )
        print(f'Saved per-structure geo-opt analysis to {analysis_out_name}')

    else:

        recon_file_path = get_file_paths(args.root_path, 'diff', args.label)
        batch_idx = -1 if args.multi_eval else 0
        crys_array_list, true_crystal_array_list = get_crystal_array_list(
            recon_file_path, batch_idx = batch_idx)
        if args.gt_file != '':
            csv = pd.read_csv(args.gt_file)
            gt_crys = p_map(get_gt_crys_ori, csv['cif'])
        else:
            gt_crys = p_map(lambda x: Crystal(x), true_crystal_array_list, num_cpus=args.njobs)

        if not args.multi_eval:
            pred_crys = p_map(lambda x: Crystal(x), crys_array_list, num_cpus=args.njobs)
        else:
            pred_crys = []
            for i in range(len(crys_array_list)):
                if args.multi_idx is not None and i != args.multi_idx:
                    continue
                print(f"Processing batch {i}")
                pred_crys.append(p_map(lambda x: Crystal(x), crys_array_list[i], num_cpus=args.njobs))

        if args.multi_eval:
            rec_evaluator = RecEvalBatch(pred_crys, gt_crys)
        else:
            rec_evaluator = RecEval(pred_crys, gt_crys)

        recon_metrics = rec_evaluator.get_metrics()

        if hasattr(rec_evaluator, "all_rms_dis"):
            all_metrics["all_rms_dis"] = rec_evaluator.all_rms_dis.tolist()

        all_metrics.update(recon_metrics)



    print(all_metrics)

    if args.label == '':
        metrics_out_file = 'eval_metrics.json'
    else:
        metrics_out_file = f'eval_metrics_{args.label}.json'
    if args.multi_idx is not None:
        metrics_out_file = str(Path(metrics_out_file).stem + f"_{args.multi_idx}.json")
    metrics_out_file = os.path.join(args.root_path, metrics_out_file)

    # only overwrite metrics computed in the new run.
    if Path(metrics_out_file).exists():
        with open(metrics_out_file, 'r') as f:
            written_metrics = json.load(f)
            if isinstance(written_metrics, dict):
                written_metrics.update(all_metrics)
            else:
                with open(metrics_out_file, 'w') as f:
                    json.dump(all_metrics, f)
        if isinstance(written_metrics, dict):
            with open(metrics_out_file, 'w') as f:
                json.dump(written_metrics, f)
    else:
        with open(metrics_out_file, 'w') as f:
            json.dump(all_metrics, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', required=True)
    parser.add_argument('--label', default='')
    parser.add_argument('--tasks', nargs='+', default=['csp'])
    parser.add_argument('--gt_file',default='')
    parser.add_argument('--multi_eval',action='store_true')
    parser.add_argument('--multi_idx', type=int, default=None, help="index for multi_eval (special case)")
    parser.add_argument('-j', '--njobs', default=8, type=int)
    parser.add_argument('--symprec', type=float, default=1e-2,
                         help="symmetry-finder tolerance for geo_opt_mbd task "
                              "(matbench-discovery reports both 1e-2 and 1e-5)")
    parser.add_argument('--use_pymatgen_symmetry', action='store_true',
                         help="use pymatgen/spglib instead of moyopy for the geo_opt_mbd "
                              "symmetry pass. moyopy (the default) is what the official "
                              "matbench-discovery leaderboard actually uses; this flag is "
                              "only for cross-checking moyopy results against an "
                              "independent implementation, not for matching official numbers.")
    args = parser.parse_args()
    main(args)