
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


def compute_range_rho(df, col):
    """Raw range / rel_range / rho per trajectory -- no thresholds applied."""
    wide = df.pivot_table(index=TRAJ_KEYS, columns="step", values=col,
                           aggfunc="first").sort_index(axis=1)
    if wide.isna().all(axis=None):
        return None

    steps = wide.columns.to_numpy(dtype=float)
    rows = []
    for i, key in enumerate(wide.index):
        v = wide.iloc[i].to_numpy(dtype=float)
        m = ~np.isnan(v)
        vv, ss = v[m], steps[m]
        if vv.size < 4:
            rng, rel_range, rho = np.nan, np.nan, np.nan
        else:
            scale = np.median(vv)
            rng = np.percentile(vv, 95) - np.percentile(vv, 5)
            rel_range = rng / scale if scale > 0 else np.nan
            rho, _ = spearmanr(ss, vv)
        rows.append({**dict(zip(TRAJ_KEYS, key)), "range": rng,
                     "rel_range": rel_range, "rho": rho})
    return pd.DataFrame(rows)


def otsu_threshold(x, bins=50):
    """Standard Otsu's method: the threshold maximizing between-class
    variance for a 1D distribution with (assumed) two populations."""
    x = x[np.isfinite(x)]
    if x.size < 10 or np.ptp(x) == 0:
        return np.nan

    hist, edges = np.histogram(x, bins=bins)
    hist = hist.astype(float)
    centers = (edges[:-1] + edges[1:]) / 2
    total = hist.sum()
    sum_all = np.sum(hist * centers)

    sumB, wB = 0.0, 0.0
    best_var, best_thresh = -1.0, centers[0]
    for i in range(len(hist)):
        wB += hist[i]
        if wB == 0:
            continue
        wF = total - wB
        if wF == 0:
            break
        sumB += centers[i] * hist[i]
        mB = sumB / wB
        mF = (sum_all - sumB) / wF
        var_between = wB * wF * (mB - mF) ** 2
        if var_between > best_var:
            best_var = var_between
            best_thresh = centers[i]
    return best_thresh


def calibrate(name, rr):
    print(f"\n--- {name} ---")
    rel_range = rr["rel_range"].to_numpy(dtype=float)
    rho = rr["rho"].to_numpy(dtype=float)

    zero_range = int(np.sum(rel_range == 0))
    if zero_range:
        print(f"  {zero_range} trajectories ({100*zero_range/len(rel_range):.1f}%) "
              f"have EXACTLY zero range -- trivially floored regardless of any "
              f"threshold, excluded from the calibration below.")

    nonzero = rel_range[(rel_range > 0) & np.isfinite(rel_range)]
    if nonzero.size < 10:
        print("  not enough non-trivial trajectories to calibrate a threshold.")
        return None, None

    log_tol = otsu_threshold(np.log10(nonzero))
    suggested_rel_range_tol = 10 ** log_tol if np.isfinite(log_tol) else np.nan
    print(f"  Otsu-suggested --rel-range-tol : {suggested_rel_range_tol:.4g}")

    suggested_rho_tol = None
    if np.isfinite(suggested_rel_range_tol):
        low_mask = rel_range < suggested_rel_range_tol
        low_rho = rho[low_mask & np.isfinite(rho)]
        if low_rho.size >= 10:
            suggested_rho_tol = float(np.percentile(np.abs(low_rho), 95))
            print(f"  Suggested --rho-tol            : {suggested_rho_tol:.3f}"
                  f"   (95th pct of |rho| among the {int(low_mask.sum())} "
                  f"trajectories below the range threshold -- i.e. what pure "
                  f"noise correlation looks like here)")
        else:
            print("  Not enough low-range trajectories to calibrate --rho-tol "
                  "from noise -- read it off the scatter plot instead.")

        print("\n  sensitivity: 'floored' fraction for nearby --rel-range-tol choices")
        for factor in (0.5, 0.75, 1.0, 1.5, 2.0):
            tol = suggested_rel_range_tol * factor
            frac = np.mean(rel_range[np.isfinite(rel_range)] < tol)
            print(f"    {tol:.4g} ({factor}x suggested): floored = {frac*100:.1f}%")

    return suggested_rel_range_tol, suggested_rho_tol


def plot_diagnostic(name, rr, rel_range_tol, rho_tol, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rel_range = rr["rel_range"].to_numpy(dtype=float)
    rho = rr["rho"].to_numpy(dtype=float)
    m = np.isfinite(rel_range) & np.isfinite(rho) & (rel_range > 0)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(rel_range[m], rho[m], s=8, alpha=0.4)
    ax.set_xscale("log")
    ax.set_xlabel("rel_range (log scale)")
    ax.set_ylabel("rho (spearman, step vs value)")
    ax.set_title(f"{name}: rho vs range -- look for where the cloud "
                 f"stops being scattered around rho=0")
    if rel_range_tol and np.isfinite(rel_range_tol):
        ax.axvline(rel_range_tol, color="C3", linestyle="--", label="suggested rel-range-tol")
    if rho_tol and np.isfinite(rho_tol):
        ax.axhline(-rho_tol, color="C1", linestyle="--", label="suggested rho-tol")
        ax.axhline(rho_tol, color="C1", linestyle="--")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = Path(outdir) / f"{name}_calibration_scatter.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", type=Path)
    ap.add_argument("--outdir", type=Path, default=Path("."))
    args = ap.parse_args()

    df = load(args.csv)
    print(f"=== {args.csv} ===")

    for m in MODALITIES:
        rr = compute_range_rho(df, m)
        if rr is None:
            print(f"\n--- {m} ---\n  all-NaN, skipping")
            continue
        rel_range_tol, rho_tol = calibrate(m, rr)
        plot_diagnostic(m, rr, rel_range_tol, rho_tol, args.outdir)


if __name__ == "__main__":
    main()