

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


def window_median(v, s, centre, half_width):
    """Median of v over steps within +/- half_width of `centre`, so the
    readout is a local average rather than one noisy sample."""
    m = np.abs(s - centre) <= half_width
    return np.median(v[m]) if m.any() else np.nan


def stats(df, col, win_frac, early_k):
    wide = df.pivot_table(index=TRAJ_KEYS, columns="step", values=col,
                          aggfunc="first").sort_index(axis=1)
    M = wide.to_numpy(dtype=float)
    steps = wide.columns.to_numpy(dtype=float)
    if np.all(np.isnan(M)):
        return None

    n_traj, n_steps = M.shape
    first = np.full(n_traj, np.nan)
    y_half = np.full(n_traj, np.nan)     # value around step N/2
    y_end = np.full(n_traj, np.nan)      # value around step N
    halflife = np.full(n_traj, np.nan)   # steps to halve, over the 2nd half
    peak_step = np.full(n_traj, np.nan)  # step of max field (1 = monotone decay)
    peak_ratio = np.full(n_traj, np.nan) # y_max / y_1  (>1 = rises before falling)
    early = np.full(n_traj, np.nan)      # y(early_k) / y(1)
    path_all = np.full(n_traj, np.nan)   # (1/N) * sum y  = upper bound on |dx|
    path_early = np.full(n_traj, np.nan) # same, first early_k steps only

    for i in range(n_traj):
        m = ~np.isnan(M[i])
        if m.sum() < 4:
            continue
        v, s = M[i][m], steps[m]
        smax = s[-1]
        hw = max(1.0, win_frac * smax)

        first[i] = v[0]
        y_half[i] = window_median(v, s, smax / 2.0, hw)
        y_end[i] = window_median(v, s, smax, hw)

        # --- early dynamics: is the DFT structure even a local minimum? ---
        j = int(np.argmax(v))
        peak_step[i] = s[j]
        peak_ratio[i] = v[j] / v[0] if v[0] > 0 else np.nan
        k = min(early_k, v.size) - 1
        early[i] = v[k] / v[0] if v[0] > 0 else np.nan

        # path length = eta * sum|v| with eta = 1/n_steps; an upper bound on
        # the net displacement, and independent of N since eta*N == 1.
        path_all[i] = v.sum() / n_steps
        path_early[i] = v[:k + 1].sum() / n_steps

        # exponential rate over the whole second half: d ln(y) / d step
        sec = (s >= smax / 2.0) & (v > 0)
        if sec.sum() >= 4:
            k = np.polyfit(s[sec] - s[sec][0], np.log(v[sec]), 1)[0]
            if k < 0:
                halflife[i] = np.log(2.0) / (-k)

    return dict(n_traj=n_traj, n_steps=n_steps, first=first,
                y_half=y_half, y_end=y_end,
                doubling=y_end / y_half, halflife=halflife,
                peak_step=peak_step, peak_ratio=peak_ratio, early=early,
                path_all=path_all, path_early=path_early, early_k=early_k)


def q(x, p):
    return np.nanpercentile(x, p)


def report(name, st):
    print(f"\n--- {name} ---")
    if st is None:
        print("  all-NaN (modality held fixed)")
        return
    f, d, hl = st["first"], st["doubling"], st["halflife"]
    N = st["n_steps"]

    print(f"  ON THE RELAXED INPUT (step 1)   <- the primary number")
    print(f"    median {np.nanmedian(f):.4g}"
          f"   [p10 {q(f,10):.4g}  p90 {q(f,90):.4g}]"
          f"   spread p90/p10 {q(f,90)/q(f,10):.1f}x")

    k = st["early_k"]
    ps, pr = st["peak_step"], st["peak_ratio"]
    frac_rise = np.nanmean(ps > 1)
    print(f"  EARLY DYNAMICS")
    print(f"    field peaks at step 1 for {(1-frac_rise)*100:.1f}% of structures"
          f"  (median peak step {np.nanmedian(ps):.0f})")
    if frac_rise > 0.02:
        up = pr[ps > 1]
        print(f"    -> {frac_rise*100:.1f}% RISE before falling, by median "
              f"{np.nanmedian(up):.2f}x -- the relaxed input is NOT a local "
              f"minimum of the field for these")
    print(f"    y({k})/y(1)                  : median {np.nanmedian(st['early']):.3f}"
          f"   [p10 {q(st['early'],10):.3f}  p90 {q(st['early'],90):.3f}]")
    pa, pe = st["path_all"], st["path_early"]
    print(f"    path length (upper bound on displacement, eta=1/N):")
    print(f"      first {k} steps            : median {np.nanmedian(pe):.4g}")
    print(f"      full trajectory           : median {np.nanmedian(pa):.4g}"
          f"   ({np.nanmedian(pe)/np.nanmedian(pa)*100:.1f}% incurred in the "
          f"first {k} steps)")

    print(f"  DOUBLING TEST  y(N)/y(N/2)   (N={N})")
    md = np.nanmedian(d)
    print(f"    median {md:.3f}   [p10 {q(d,10):.3f}  p90 {q(d,90):.3f}]")
    print(f"    y(N/2) median {np.nanmedian(st['y_half']):.4g}"
          f"  ->  y(N) median {np.nanmedian(st['y_end']):.4g}")
    if md > 0.9:
        print(f"    -> FLOOR: value stopped changing; a plateau exists")
    elif md > 0.7:
        print(f"    -> SLOW DECAY: no clean floor yet, extend N to separate")
    else:
        print(f"    -> NO FLOOR: still ~halving per doubling of steps "
              f"(decaying toward zero)")

    frac_conv = np.nanmean(d > 0.9)
    print(f"    fraction of structures with a floor (ratio>0.9): {frac_conv*100:.1f}%")

    print(f"  DECAY RATE (ln-fit over steps {N//2}-{N})")
    n_dec = np.isfinite(hl).sum()
    if n_dec:
        print(f"    half-life median {np.nanmedian(hl):.0f} steps"
              f"   [p10 {q(hl,10):.0f}  p90 {q(hl,90):.0f}]"
              f"   ({n_dec}/{st['n_traj']} still decaying)")
        print(f"    -> N would need to be ~{np.nanmedian(hl)*5:.0f} steps "
              f"for a 32x further drop")
    else:
        print(f"    no structure is decaying over the second half (flat)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", type=Path, nargs="+",
                    help="eval_field_norms*.csv (pass 2 to compare runs)")
    ap.add_argument("--win-frac", type=float, default=0.04,
                    help="half-width of the y(N/2) / y(N) readout windows, as a "
                         "fraction of N (default 0.04)")
    ap.add_argument("--early-k", type=int, default=5,
                    help="how many leading steps count as the early transient; "
                         "set this to your --min-steps (default 5)")
    args = ap.parse_args()

    runs = []
    for path in args.csv:
        df = load(path)
        print(f"\n=== {path} ===")
        print(f"  {df.groupby(TRAJ_KEYS).ngroups} trajectories "
              f"({df['material_id'].nunique()} materials x "
              f"{df['eval_idx'].nunique()} evals)")
        run = {m: stats(df, m, args.win_frac, args.early_k) for m in MODALITIES}
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
            print(f"  {m}: step-1  {fa:.4g}  ->  {fb:.4g}   ({fb/fa:.2f}x)")


if __name__ == "__main__":
    main()
