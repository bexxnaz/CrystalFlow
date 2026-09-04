"""Compact field-norm report for `relax_wbm.py` runs started from RELAXED
structures.

The question this answers: if the model is evaluated on a structure that is
already relaxed, how large is the velocity field it predicts, and does that
field stay put over sampling steps?

For an EqM model a relaxed structure should be a fixed point, so
  * step-1 field norm  = the model's residual / noise floor on relaxed inputs
  * step1 / plateau    = how far the model drags the structure off that input
                         before settling (1.0 = input already is the fixed point)
  * tail slope         = whether it has settled at all

Pass two CSVs to compare runs (e.g. trained vs. untrained checkpoint).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

MODALITIES = ("coord_field_norm", "lattice_field_norm")
TRAJ_KEYS = ["material_id", "eval_idx"]


def load(csv_path):
    df = pd.read_csv(csv_path, sep=None, engine="python")
    df = df.dropna(subset=["step"]).dropna(subset=list(MODALITIES), how="all")
    df["step"] = df["step"].astype(int)
    return df.drop_duplicates(TRAJ_KEYS + ["step"], keep="first")


def stats(df, col, plateau_steps):
    """Per-structure first value / plateau / tail slope for one modality."""
    wide = df.pivot_table(index=TRAJ_KEYS, columns="step", values=col,
                          aggfunc="first").sort_index(axis=1)
    M = wide.to_numpy(dtype=float)
    steps = wide.columns.to_numpy(dtype=float)
    if np.all(np.isnan(M)):
        return None

    n_traj, n_steps = M.shape
    Wp = min(max(2, plateau_steps), n_steps)

    first = np.full(n_traj, np.nan)
    plateau = np.full(n_traj, np.nan)
    slope = np.full(n_traj, np.nan)
    for i in range(n_traj):
        m = ~np.isnan(M[i])
        if not m.any():
            continue
        v, s = M[i][m], steps[m]
        first[i] = v[0]
        w = min(Wp, v.size)
        plateau[i] = v[-w:].mean()
        if w >= 2:
            slope[i] = np.polyfit(s[-w:] - s[-w], v[-w:], 1)[0]

    return dict(n_traj=n_traj, n_steps=n_steps, Wp=Wp, first=first,
                plateau=plateau, slope_rel=slope / plateau,
                ratio=first / plateau)


def q(x, p):
    return np.nanpercentile(x, p)


def report(name, st):
    print(f"\n--- {name} ---")
    if st is None:
        print("  all-NaN (modality held fixed)")
        return
    f, pl, r = st["first"], st["plateau"], st["ratio"]
    sr = st["slope_rel"] * 100

    print(f"  ON THE RELAXED INPUT (step 1)")
    print(f"    median {np.nanmedian(f):.4g}"
          f"   [p10 {q(f,10):.4g}  p90 {q(f,90):.4g}]"
          f"   spread p90/p10 {q(f,90)/q(f,10):.1f}x")
    print(f"  AFTER SETTLING (mean of last {st['Wp']} of {st['n_steps']} steps)")
    print(f"    median {np.nanmedian(pl):.4g}"
          f"   [p10 {q(pl,10):.4g}  p90 {q(pl,90):.4g}]")
    print(f"  step1 / plateau              : median {np.nanmedian(r):.2f}x"
          f"   [p10 {q(r,10):.2f}  p90 {q(r,90):.2f}]")
    print(f"    -> {'input IS ~the fixed point' if np.nanmedian(r) < 1.2 else 'model moves away from the input, then settles'}")
    print(f"  tail slope (% of plateau/step): median {np.nanmedian(sr):+.3f}"
          f"   [p10 {q(sr,10):+.3f}  p90 {q(sr,90):+.3f}]")
    print(f"    -> {'PLATEAUED' if abs(np.nanmedian(sr)) < 0.1 else 'STILL DRIFTING'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", type=Path, nargs="+",
                    help="eval_field_norms*.csv (pass 2 to compare runs)")
    ap.add_argument("--plateau-steps", type=int, default=10,
                    help="steps averaged at the end of each trajectory (default 10)")
    args = ap.parse_args()

    runs = []
    for path in args.csv:
        df = load(path)
        print(f"\n=== {path} ===")
        print(f"  {df.groupby(TRAJ_KEYS).ngroups} trajectories "
              f"({df['material_id'].nunique()} materials x {df['eval_idx'].nunique()} evals)")
        run = {m: stats(df, m, args.plateau_steps) for m in MODALITIES}
        for m in MODALITIES:
            report(m, run[m])
        runs.append((path, run))

    if len(runs) == 2:
        (pa, a), (pb, b) = runs
        print(f"\n=== {pa.stem}  vs  {pb.stem} ===")
        for m in MODALITIES:
            if a[m] is None or b[m] is None:
                continue
            fa, fb = np.nanmedian(a[m]["first"]), np.nanmedian(b[m]["first"])
            pla, plb = np.nanmedian(a[m]["plateau"]), np.nanmedian(b[m]["plateau"])
            print(f"  {m}:")
            print(f"    step-1  {fa:.4g}  ->  {fb:.4g}   ({fb/fa:.2f}x)")
            print(f"    plateau {pla:.4g}  ->  {plb:.4g}   ({plb/pla:.2f}x)")


if __name__ == "__main__":
    main()
