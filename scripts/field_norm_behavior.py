import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

MODALITIES = ("coord_field_norm", "lattice_field_norm")
TRAJ_KEYS = ["material_id", "eval_idx"]


def load(csv_path):
    df = pd.read_csv(csv_path, sep=None, engine="python")
    df = df.dropna(subset=["step"]).dropna(subset=list(MODALITIES), how="all")
    df["step"] = df["step"].astype(int)
    return df.drop_duplicates(TRAJ_KEYS + ["step"], keep="first")


def classify_trajectory(v, s, rel_range_tol, irregular_rho=-0.7):
    if v.size < 4:
        return dict(range=np.nan, rel_range=np.nan, rho=np.nan,
                     trend="unknown", irregular=False)

    scale = np.median(v)
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


def summarize(df, col, rel_range_tol, irregular_rho=-0.7):
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
        rows.append({**dict(zip(TRAJ_KEYS, key)), **c})
    return pd.DataFrame(rows), wide


def fan_chart(wide, modality, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = wide.columns.to_numpy(dtype=float)
    med, p10, p25 = wide.median(axis=0), wide.quantile(0.10, axis=0), wide.quantile(0.25, axis=0)
    p75, p90 = wide.quantile(0.75, axis=0), wide.quantile(0.90, axis=0)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.fill_between(steps, p10, p90, alpha=0.15, color="C0", label="p10-p90")
    ax.fill_between(steps, p25, p75, alpha=0.3, color="C0", label="p25-p75")
    ax.plot(steps, med, color="C0", label="median")
    ax.set_xlabel("step")
    ax.set_ylabel(modality)
    ax.set_title(f"{modality}: spread across all materials, over steps")
    ax.legend()
    fig.tight_layout()
    out = Path(outdir) / f"{modality}_fan_chart.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


def joint_scatter(coord_summary, lattice_summary, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    merged = coord_summary.merge(lattice_summary, on=TRAJ_KEYS, suffixes=("_coord", "_lattice"))
    merged["coord_floored"] = merged["trend_coord"] == "floored"
    merged["lattice_floored"] = merged["trend_lattice"] == "floored"

    def joint_cat(row):
        if row["coord_floored"] and row["lattice_floored"]:
            return "both floored"
        if row["coord_floored"]:
            return "only coord floored"
        if row["lattice_floored"]:
            return "only lattice floored"
        return "neither floored"

    merged["joint"] = merged.apply(joint_cat, axis=1)

    print("\n--- joint coord/lattice categorization ---")
    print((merged["joint"].value_counts(normalize=True) * 100).round(1))

    color = {"both floored": "C2", "only coord floored": "C0",
             "only lattice floored": "C1", "neither floored": "C3"}
    fig, ax = plt.subplots(figsize=(6, 5))
    for cat, c in color.items():
        sub = merged[merged["joint"] == cat]
        if len(sub) == 0:
            continue
        ax.scatter(sub["range_coord"] + 1e-6, sub["range_lattice"] + 1e-6,
                   s=12, alpha=0.6, color=c, label=f"{cat} ({len(sub)})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("coord_field_norm range (per material)")
    ax.set_ylabel("lattice_field_norm range (per material)")
    ax.set_title("Materials categorized by coord/lattice range")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = Path(outdir) / "joint_range_scatter.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")
    return merged


def report(name, summary):
    print(f"\n--- {name} ---")
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
        print(f"  irregular (labeled 'decreasing' but rho isn't strongly "
              f"negative -- possible overshoot/oscillation): "
              f"{irr_frac*100:.1f}% of all trajectories "
              f"({int(summary['irregular'].sum())}/{n_dec} of the 'decreasing' bucket)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", type=Path)
    ap.add_argument("--rel-range-tol", type=float, default=0.15,
                     help="range/median(v) below this -> floored (default 0.15). "
                          "This is the only threshold that actually matters now -- "
                          "see calibrate_thresholds.py for a data-derived suggestion, "
                          "keeping in mind it may not be a sharp natural cutoff.")
    ap.add_argument("--irregular-rho", type=float, default=-0.7,
                     help="a 'decreasing' trajectory with rho above this (i.e. not "
                          "strongly negative) gets flagged as irregular -- informational "
                          "only, does not change the trend label (default -0.7)")
    ap.add_argument("--outdir", type=Path, default=Path("."))
    ap.add_argument("--out-csv", type=Path, default=None,
                     help="save the per-material summary table here (one row per "
                          "material/eval_idx, with range/rho/trend/irregular for "
                          "each modality, plus the joint category if both are present)")
    args = ap.parse_args()

    df = load(args.csv)
    print(f"=== {args.csv} ===")
    print(f"{df.groupby(TRAJ_KEYS).ngroups} trajectories")

    summaries = {}
    for m in MODALITIES:
        summary, wide = summarize(df, m, args.rel_range_tol, args.irregular_rho)
        report(m, summary)
        if summary is not None:
            fan_chart(wide, m, args.outdir)
            summaries[m] = summary

    out_table = None
    if len(summaries) == 2:
        out_table = joint_scatter(summaries[MODALITIES[0]], summaries[MODALITIES[1]], args.outdir)
    elif len(summaries) == 1:
        out_table = next(iter(summaries.values()))

    if args.out_csv and out_table is not None:
        out_table.to_csv(args.out_csv, index=False)
        print(f"\nSaved summary table ({len(out_table)} rows) to {args.out_csv}")


if __name__ == "__main__":
    main()