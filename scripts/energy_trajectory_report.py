
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

TRAJ_KEYS = ["material_id", "eval_idx"]

# Real cohesive/formation energies are roughly -10 to 0 eV/atom. Anything
# past this is a numerical blow-up (e.g. overlapping/collapsed atoms giving
# MACE's repulsive term a near-singularity), not a real energy -- flagged
# and excluded rather than silently classified or left to wreck a shared
# plot axis.
PHYSICAL_BOUND_EV_PER_ATOM = 100.0


def load(csv_path):
    df = pd.read_csv(csv_path, sep=None, engine="python")
    df = df.dropna(subset=["step"])
    df["step"] = df["step"].astype(int)
    return df.drop_duplicates(TRAJ_KEYS + ["step"], keep="first")


def classify_trajectory(v, s, rel_range_tol, irregular_rho=-0.7):
    if np.any(np.abs(v) > PHYSICAL_BOUND_EV_PER_ATOM):
        # Don't compute range/rho on a corrupted trajectory -- a spike from
        # e.g. 1e18 down to a normal value would trivially read as
        # "decreasing", which is true but meaningless: the story here is
        # "structural blow-up", not "converging".
        return dict(range=np.nan, rel_range=np.nan, rho=np.nan,
                     trend="invalid_structure", irregular=False)
    if v.size < 4:
        return dict(range=np.nan, rel_range=np.nan, rho=np.nan,
                     trend="unknown", irregular=False)

    # abs() here on purpose -- energy is typically negative, unlike field
    # norms; without abs(), a negative scale flips rel_range's sign and
    # rel_range < rel_range_tol becomes true almost always, misclassifying
    # nearly everything as "floored" regardless of what's really happening.
    scale = np.median(np.abs(v))
    rng = np.percentile(v, 95) - np.percentile(v, 5)
    rel_range = rng / scale if scale > 0 else np.nan
    rho, _ = spearmanr(s, v)

    if np.isnan(rel_range) or rel_range < rel_range_tol:
        trend = "floored"
    elif rho >= 0:
        trend = "increasing"
    else:
        trend = "decreasing"

    irregular = (trend == "decreasing") and (not np.isnan(rho)) and (rho > irregular_rho)

    return dict(range=rng, rel_range=rel_range, rho=rho, trend=trend, irregular=irregular)


def summarize(df, col, rel_range_tol, irregular_rho):
    wide = df.pivot_table(index=TRAJ_KEYS, columns="step", values=col,
                           aggfunc="first").sort_index(axis=1)
    if wide.isna().all(axis=None):
        return None, wide

    steps = wide.columns.to_numpy(dtype=float)
    rows = []
    for i, key in enumerate(wide.index):
        v = wide.iloc[i].to_numpy(dtype=float)
        m = ~np.isnan(v)
        vv, ss = v[m], steps[m]
        c = classify_trajectory(vv, ss, rel_range_tol, irregular_rho)
        c['first_step_value'] = float(vv[0]) if vv.size else np.nan
        c['last_step_value'] = float(vv[-1]) if vv.size else np.nan
        rows.append({**dict(zip(TRAJ_KEYS, key)), **c})
    return pd.DataFrame(rows), wide


def report(col, summary):
    print(f"\n--- {col} ---")
    if summary is None or len(summary) == 0:
        print("  no data")
        return

    print(f"  {len(summary)} trajectories")
    print(f"  range : median {summary['range'].median():.4g}"
          f"   [p10 {summary['range'].quantile(.1):.4g}  p90 {summary['range'].quantile(.9):.4g}]")
    print(f"  rho   : median {summary['rho'].median():.3f}"
          f"   [p10 {summary['rho'].quantile(.1):.3f}  p90 {summary['rho'].quantile(.9):.3f}]")
    print(f"  trend:")
    for t, frac in summary["trend"].value_counts(normalize=True).items():
        print(f"    {t:<12}: {frac*100:5.1f}%")
    irr_frac = summary["irregular"].mean()
    if irr_frac > 0:
        n_dec = (summary["trend"] == "decreasing").sum()
        print(f"  irregular (labeled 'decreasing' but rho isn't strongly negative "
              f"-- possible overshoot/oscillation): {irr_frac*100:.1f}% of all "
              f"trajectories ({int(summary['irregular'].sum())}/{n_dec} of 'decreasing')")

    if "delta_last_minus_gt" in summary.columns:
        d = summary["delta_last_minus_gt"].dropna()
        if len(d):
            print(f"\n  LAST STEP vs GROUND TRUTH ({col}):")
            print(f"    median {d.median():+.4f}   [p10 {d.quantile(.1):+.4f}  "
                  f"p90 {d.quantile(.9):+.4f}]   ({len(d)}/{len(summary)} with a GT match)")
            frac_below = (d < 0).mean()
            print(f"    {frac_below*100:.1f}% ended BELOW the ground-truth energy "
                  f"(more stable than MACE's own scoring of the reference structure -- "
                  f"worth a second look, could be a real improvement or a scoring "
                  f"artifact); {100-frac_below*100:.1f}% ended above (the typical case "
                  f"for an imperfect relaxation).")

    # small-N -- print every row, not just the aggregate
    cols = TRAJ_KEYS + ["trend", "rho", "range", "first_step_value", "last_step_value"]
    if "gt_value" in summary.columns:
        cols += ["gt_value", "delta_last_minus_gt"]
    print(f"\n  per-material detail:")
    print(summary[cols].to_string(index=False))


def plot_trajectories_with_gt(wide, gt_by_id, outdir, col, max_labeled=15):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = wide.columns.to_numpy(dtype=float)
    all_vals = wide.to_numpy(dtype=float)
    n_traj = len(wide.index)

    # Flag blown-up trajectories FIRST -- exclude them from both the axis
    # range and the drawn lines, and report them separately. A silently
    # clipped outlier is itself a finding (a broken structure), not
    # something to just hide by zooming in.
    blown_up = []
    ok_rows = []
    for i, key in enumerate(wide.index):
        mat_id = key[0] if isinstance(key, tuple) else key
        v = wide.iloc[i].to_numpy(dtype=float)
        finite = v[np.isfinite(v)]
        if finite.size and np.max(np.abs(finite)) > PHYSICAL_BOUND_EV_PER_ATOM:
            blown_up.append(mat_id)
        else:
            ok_rows.append(i)

    fig, ax = plt.subplots(figsize=(8, 5))

    if len(ok_rows) == 0:
        print("  WARNING: every trajectory is flagged as a structural blow-up -- "
              "nothing plottable.")
        plt.close(fig)
        return blown_up

    ok_vals = all_vals[ok_rows]
    finite = ok_vals[np.isfinite(ok_vals)]
    lo, hi = np.percentile(finite, [1, 99])
    pad = 0.1 * (hi - lo) if hi > lo else max(abs(hi), 1.0) * 0.1

    if n_traj <= max_labeled:
        cmap = plt.get_cmap("tab10")
        for i in ok_rows:
            key = wide.index[i]
            mat_id = key[0] if isinstance(key, tuple) else key
            v = wide.iloc[i].to_numpy(dtype=float)
            m = np.isfinite(v)
            color = cmap(i % 10)
            ax.plot(steps[m], v[m], marker="o", markersize=2.5, color=color,
                    alpha=0.85, label=str(mat_id))
            if mat_id in gt_by_id and not np.isnan(gt_by_id[mat_id]):
                ax.axhline(gt_by_id[mat_id], color=color, linestyle="--",
                           alpha=0.5, linewidth=1.2)
        ax.legend(fontsize=7, ncol=2)
        title_suffix = "  (dashed = ground truth)" if gt_by_id else ""
    else:
        # Too many materials for individual colors/legend to mean anything
        # (tab10 only has 10 colors -- they'd repeat every 10 materials, and
        # a 60-entry legend doesn't fit on the figure anyway). Thin,
        # unlabeled lines + a highlighted median instead. Per-material GT
        # dashed lines dropped here too -- use the delta histogram for the
        # population GT comparison at this scale.
        for i in ok_rows:
            v = wide.iloc[i].to_numpy(dtype=float)
            m = np.isfinite(v)
            ax.plot(steps[m], v[m], color="C0", alpha=0.12, linewidth=0.8)
        median_traj = np.nanmedian(ok_vals, axis=0)
        ax.plot(steps, median_traj, color="C3", linewidth=2,
                label=f"median (n={len(ok_rows)})")
        ax.legend(fontsize=9)
        title_suffix = ("  -- see the delta histogram for ground-truth "
                         "comparison at this N" if gt_by_id else "")

    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel("step")
    ax.set_ylabel(col)
    ax.set_title(f"{col} vs step{title_suffix}")
    fig.tight_layout()
    out = Path(outdir) / f"{col}_trajectories.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")

    if blown_up:
        print(f"  WARNING: {len(blown_up)} material(s) have |{col}| > "
              f"{PHYSICAL_BOUND_EV_PER_ATOM} at some step -- almost certainly a "
              f"structural blow-up (e.g. overlapping atoms), not a real energy. "
              f"Excluded from this plot; investigate directly: {blown_up}")

    return blown_up


def _zero_aligned_bins(values, n_bins):
    """Uniform-width bin edges that land exactly on 0 -- so no bar straddles
    zero and obscures whether its materials are actually above or below GT."""
    vmin, vmax = min(float(np.min(values)), 0.0), max(float(np.max(values)), 0.0)
    width = (vmax - vmin) / n_bins if vmax > vmin else 1.0
    n_neg = int(np.ceil(-vmin / width)) if vmin < 0 else 0
    n_pos = int(np.ceil(vmax / width)) if vmax > 0 else 0
    return np.arange(-n_neg, n_pos + 1) * width


def plot_delta_bar(summary, outdir, col):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = summary.dropna(subset=["delta_last_minus_gt"])
    n_before = len(d)
    d = d[d["delta_last_minus_gt"].abs() <= PHYSICAL_BOUND_EV_PER_ATOM]
    n_excluded = n_before - len(d)
    if len(d) == 0:
        print("  (no valid rows with a ground-truth match -- skipping delta distribution plot)")
        return

    values = d["delta_last_minus_gt"].to_numpy()
    n_bins = max(3, min(30, int(np.ceil(np.sqrt(len(values))))))
    bins = _zero_aligned_bins(values, n_bins)

    fig, ax = plt.subplots(figsize=(7, 5))
    counts, bins, patches = ax.hist(values, bins=bins, edgecolor="black", alpha=0.85)
    for patch, left, right in zip(patches, bins[:-1], bins[1:]):
        patch.set_facecolor("C0" if (left + right) / 2 < 0 else "C3")

    ax.axvline(0, color="black", linewidth=1.2, linestyle="--")
    ax.set_xlabel(f"last-step {col} minus ground truth")
    ax.set_ylabel("number of materials")
    ax.set_title("Distribution of final-structure energy vs ground truth\n"
                  "(blue = below GT / more stable, red = above GT / less stable)")
    fig.tight_layout()
    out = Path(outdir) / f"{col}_delta_vs_gt.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")
    if n_excluded:
        print(f"  WARNING: excluded {n_excluded} material(s) from this histogram -- "
              f"|delta| > {PHYSICAL_BOUND_EV_PER_ATOM} eV/atom, almost certainly a "
              f"structural blow-up rather than a real energy difference.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", type=Path, help="eval_mace_energy*.csv (per-step trajectory)")
    ap.add_argument("--gt-csv", type=Path, default=None,
                     help="eval_gt_energy*.csv (one row per material) -- if given, also "
                          "computes last-step-vs-ground-truth delta and plots it")
    ap.add_argument("--column", choices=["energy_per_atom_eV", "energy_total_eV"],
                     default="energy_per_atom_eV",
                     help="which energy column to analyze (default: per-atom -- the "
                          "size-invariant one, comparable across materials with "
                          "different atom counts; energy_total_eV is not)")
    ap.add_argument("--rel-range-tol", type=float, default=0.15,
                     help="range/median(|v|) below this -> floored (default 0.15)")
    ap.add_argument("--irregular-rho", type=float, default=-0.7,
                     help="a 'decreasing' trajectory with rho above this gets flagged "
                          "as irregular -- informational only (default -0.7)")
    ap.add_argument("--max-labeled", type=int, default=15,
                     help="above this many trajectories, switch the trajectory plot "
                          "from per-material colored lines + legend (unreadable past "
                          "~10-15, since tab10 only has 10 colors) to thin unlabeled "
                          "lines + a highlighted median (default 15)")
    ap.add_argument("--outdir", type=Path, default=Path("."))
    ap.add_argument("--out-csv", type=Path, default=None,
                     help="save the per-material summary table here")
    args = ap.parse_args()

    df = load(args.csv)
    print(f"=== {args.csv} ===")
    print(f"{df.groupby(TRAJ_KEYS).ngroups} trajectories")

    summary, wide = summarize(df, args.column, args.rel_range_tol, args.irregular_rho)
    if summary is None:
        print(f"no data for column {args.column!r}")
        return

    gt_by_id = {}
    if args.gt_csv is not None:
        gt_df = pd.read_csv(args.gt_csv)
        gt_col = "gt_" + args.column   # matches relax_wbm.py's naming convention
        if gt_col not in gt_df.columns:
            raise SystemExit(f"{args.gt_csv} has no column {gt_col!r} -- expected "
                              f"relax_wbm.py's eval_gt_energy*.csv (from --gt_file).")
        gt_by_id = dict(zip(gt_df["material_id"], gt_df[gt_col]))
        summary["gt_value"] = summary["material_id"].map(gt_by_id)
        summary["delta_last_minus_gt"] = summary["last_step_value"] - summary["gt_value"]

    report(args.column, summary)

    args.outdir.mkdir(parents=True, exist_ok=True)
    plot_trajectories_with_gt(wide, gt_by_id, args.outdir, args.column, max_labeled=args.max_labeled)
    if gt_by_id:
        plot_delta_bar(summary, args.outdir, args.column)

    if args.out_csv:
        summary.to_csv(args.out_csv, index=False)
        print(f"\nSaved summary table ({len(summary)} rows) to {args.out_csv}")


if __name__ == "__main__":
    main()