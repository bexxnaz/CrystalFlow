"""Analyze the per-step field-norm trajectories written by
scripts/relax_wbm.py -> save_field_norm_csv().


The script answers, per modality (coord / lattice):

  1. Does the norm plateau, or keep dropping for the full trajectory?
     - aggregate (median-across-structures) curve
     - step at which 90 / 95 / 99 % of the total decay has happened
     - step from which the curve stays within tol of its plateau value
     - tail slope vs. initial slope, + linear extrapolation of further drop

  2. What is the plateau value per modality, and how much does it vary
     structure-to-structure?
     - per-structure plateau = mean of the last W steps of that trajectory
     - median / mean / CV / percentiles of that distribution
     - per-structure tail slope distribution ("what fraction are still
       creeping down?")
     - correlation of plateau with n_atoms
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

MODALITIES = ("coord_field_norm", "lattice_field_norm")
TRAJ_KEYS = ["material_id", "eval_idx"]


def load_csv(csv_path):
    # sep=None -> sniff , or \t (the file relax_wbm writes is comma-separated)
    df = pd.read_csv(csv_path, sep=None, engine="python")
    missing = {"step", *MODALITIES, *TRAJ_KEYS} - set(df.columns)
    if missing:
        raise SystemExit(f"CSV is missing expected columns: {sorted(missing)}")

    n_raw = len(df)
    # null-baseline rows have step == NaN and NaN norms -- drop them
    df = df.dropna(subset=["step"])
    df = df.dropna(subset=list(MODALITIES), how="all")
    n_dropped = n_raw - len(df)

    df["step"] = df["step"].astype(int)
    if df.duplicated(TRAJ_KEYS + ["step"]).any():
        n_dup = int(df.duplicated(TRAJ_KEYS + ["step"]).sum())
        print(f"WARNING: {n_dup} duplicate (material_id, eval_idx, step) rows "
              f"-- keeping the first of each.")
        df = df.drop_duplicates(TRAJ_KEYS + ["step"], keep="first")

    df = df.sort_values(TRAJ_KEYS + ["step"]).reset_index(drop=True)
    return df, n_dropped


def to_matrix(df, col):
    """[n_traj, n_steps] float array, NaN-padded; also returns step axis,
    the trajectory index, and per-trajectory n_atoms."""
    wide = (df.pivot_table(index=TRAJ_KEYS, columns="step", values=col,
                           aggfunc="first")
              .sort_index(axis=1))
    steps = wide.columns.to_numpy(dtype=int)
    M = wide.to_numpy(dtype=float)
    natoms = (df.groupby(TRAJ_KEYS)["n_atoms"].first()
                .reindex(wide.index).to_numpy(dtype=float))
    return M, steps, wide.index, natoms


# --------------------------------------------------------------------------- #
# per-trajectory tail statistics
# --------------------------------------------------------------------------- #
def per_traj_stats(M, steps, W):
    """For each row: first value, last value, plateau (mean of last W valid
    steps), and OLS slope over those last W steps (raw units per step)."""
    n = M.shape[0]
    first = np.full(n, np.nan)
    last = np.full(n, np.nan)
    plateau = np.full(n, np.nan)
    slope = np.full(n, np.nan)
    length = np.zeros(n, dtype=int)

    for i in range(n):
        v = M[i]
        m = ~np.isnan(v)
        if not m.any():
            continue
        v = v[m]
        s = steps[m].astype(float)
        length[i] = v.size
        first[i] = v[0]
        last[i] = v[-1]
        w = min(W, v.size)
        tv, ts = v[-w:], s[-w:]
        plateau[i] = tv.mean()
        if w >= 2:
            slope[i] = np.polyfit(ts - ts[0], tv, 1)[0]
    return dict(first=first, last=last, plateau=plateau, slope=slope, length=length)


def curve_stats(M):
    """Across-structure percentile curves over the step axis."""
    q = np.nanpercentile(M, [10, 25, 50, 75, 90], axis=0)
    return dict(p10=q[0], p25=q[1], med=q[2], p75=q[3], p90=q[4],
                mean=np.nanmean(M, axis=0), count=np.sum(~np.isnan(M), axis=0))


def first_stable_step(curve, steps, plateau, tol):
    """First step from which |curve - plateau| / plateau <= tol for all
    subsequent steps. None if never."""
    rel = np.abs(curve - plateau) / plateau
    below = rel <= tol
    for i in range(len(below)):
        if below[i:].all():
            return int(steps[i])
    return None


def decay_step(curve, steps, plateau, frac):
    """First step at which `frac` of the total (start -> plateau) decay has
    been achieved."""
    total = curve[0] - plateau
    if total <= 0:
        return None
    achieved = (curve[0] - curve) / total
    hit = np.where(achieved >= frac)[0]
    return int(steps[hit[0]]) if len(hit) else None


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def pct(x, p):
    return np.nanpercentile(x, p)


def analyze_modality(name, df, args):
    M, steps, idx, natoms = to_matrix(df, name)
    n_traj, n_steps = M.shape

    if np.all(np.isnan(M)):
        print(f"\n=== {name} ===\n  all-NaN (modality was held fixed) -- skipping.")
        return None
    if n_steps < 3:
        print(f"\n=== {name} ===\n  only {n_steps} step(s) per trajectory -- "
              f"not enough for a plateau/slope analysis.")
        return None

    W = args.tail_steps or max(2, round(args.tail_frac * n_steps))
    Wp = args.plateau_steps or max(2, round(args.plateau_frac * n_steps))
    W = min(max(2, W), n_steps)
    Wp = min(max(2, Wp), n_steps)

    cs = curve_stats(M)
    med = cs["med"]
    agg_plateau = med[-Wp:].mean()

    # per-trajectory
    pt = per_traj_stats(M, steps, W)
    ptp = per_traj_stats(M, steps, Wp)["plateau"]  # plateau over last Wp
    plateau = ptp
    slope = pt["slope"]
    slope_rel = slope / plateau                    # fractional change per step
    total_decay = pt["first"] / plateau            # x-fold drop start->plateau

    # aggregate-curve tail vs head slope
    head_n = max(2, min(W, n_steps // 2))
    slope_tail = np.polyfit(steps[-W:] - steps[-W], med[-W:], 1)[0]
    slope_head = np.polyfit(steps[:head_n] - steps[0], med[:head_n], 1)[0]
    extrap = slope_tail * n_steps                   # linear "next N steps" drop

    lengths = pt["length"]
    ragged = lengths.min() != lengths.max()

    print(f"\n=== {name} ===")
    print(f"  trajectories                 : {n_traj}")
    if ragged:
        print(f"  steps per trajectory         : min {lengths.min()}, "
              f"median {int(np.median(lengths))}, max {lengths.max()}  (ragged)")
    else:
        print(f"  steps per trajectory         : {n_steps} (all equal)")
    print(f"  tail window W                 : last {W} steps  (slope / 'still dropping')")
    print(f"  plateau window               : last {Wp} steps  (plateau value)")

    print("\n  -- aggregate curve (median across trajectories) --")
    print(f"  step 1                       : {med[0]:.5g}")
    print(f"  step {int(steps[-1]):<4d}                   : {med[-1]:.5g}"
          f"   (plateau = mean last {Wp}: {agg_plateau:.5g})")
    print(f"  total decay factor           : {med[0] / agg_plateau:.2f}x")
    for f in (0.90, 0.95, 0.99):
        s = decay_step(med, steps, agg_plateau, f)
        print(f"  step reaching {int(f*100)}% of decay   : "
              f"{s if s is not None else 'never'}")
    for tol in args.tol:
        s = first_stable_step(med, steps, agg_plateau, tol)
        print(f"  within {tol*100:>4.1f}% of plateau from  : "
              f"{s if s is not None else 'never'}")
    print(f"  initial slope (first {head_n})      : {slope_head:+.3g} /step  "
          f"({slope_head / agg_plateau * 100:+.2f}% of plateau/step)")
    print(f"  tail slope (last {W})           : {slope_tail:+.3g} /step  "
          f"({slope_tail / agg_plateau * 100:+.3f}% of plateau/step)")
    if slope_head != 0:
        print(f"  tail / initial slope ratio   : {slope_tail / slope_head:.2e}")
    print(f"  linear extrapolation: {n_steps} more steps -> "
          f"{abs(extrap) / agg_plateau * 100:.2f}% further change")
    verdict = ("PLATEAUED" if abs(slope_tail / agg_plateau) < 1e-3
               else "STILL DRIFTING")
    print(f"  => {verdict} (|tail slope| "
          f"{'<' if verdict == 'PLATEAUED' else '>='} 0.1% of plateau/step)")

    print(f"\n  -- per-structure plateau (mean of last {Wp} steps) --")
    print(f"  median {np.nanmedian(plateau):.5g} | mean {np.nanmean(plateau):.5g}"
          f" | std {np.nanstd(plateau):.3g}")
    print(f"  CV (std/mean)                : {np.nanstd(plateau)/np.nanmean(plateau):.3f}")
    print(f"  p10 {pct(plateau,10):.4g} | p25 {pct(plateau,25):.4g} | "
          f"p75 {pct(plateau,75):.4g} | p90 {pct(plateau,90):.4g}")
    print(f"  min {np.nanmin(plateau):.4g} | max {np.nanmax(plateau):.4g}")
    print(f"  total decay factor (start/plateau): "
          f"median {np.nanmedian(total_decay):.2f}x, "
          f"p10 {pct(total_decay,10):.2f}x, p90 {pct(total_decay,90):.2f}x")

    print(f"\n  -- per-structure tail slope (OLS over last {W} steps) --")
    print(f"  as % of that structure's plateau, per step:")
    print(f"  median {np.nanmedian(slope_rel)*100:+.3f}%/step | "
          f"p10 {pct(slope_rel,10)*100:+.3f} | p90 {pct(slope_rel,90)*100:+.3f}")
    for thr in (0.001, 0.002, 0.005):
        frac = np.nanmean(slope_rel < -thr)
        print(f"  fraction still dropping faster than -{thr*100:.1f}%/step : {frac*100:.1f}%")

    if np.isfinite(natoms).any():
        good = np.isfinite(plateau) & np.isfinite(natoms)
        if good.sum() > 2 and np.ptp(natoms[good]) > 0:
            r = np.corrcoef(plateau[good], natoms[good])[0, 1]
            print(f"\n  corr(plateau, n_atoms)       : r = {r:+.3f}")

    return dict(
        name=name, idx=idx, steps=steps, M=M, curve=cs,
        plateau=plateau, slope=slope, slope_rel=slope_rel,
        total_decay=total_decay, natoms=natoms,
        W=W, Wp=Wp, agg_plateau=agg_plateau,
    )


# --------------------------------------------------------------------------- #
# plots
# --------------------------------------------------------------------------- #
def make_plots(results, stem, outdir, n_sample_traj, seed):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib not available -- skipping --plot)")
        return

    rng = np.random.default_rng(seed)

    for res in results:
        if res is None:
            continue
        name = res["name"]
        steps = res["steps"]
        M = res["M"]
        cs = res["curve"]

        # --- curve with spread band + sample of individual trajectories ---
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        for ax, logy in zip(axes, (False, True)):
            k = min(n_sample_traj, M.shape[0])
            for i in rng.choice(M.shape[0], size=k, replace=False):
                ax.plot(steps, M[i], color="0.7", lw=0.4, alpha=0.5)
            ax.fill_between(steps, cs["p10"], cs["p90"], alpha=0.18,
                            color="C0", label="p10–p90")
            ax.fill_between(steps, cs["p25"], cs["p75"], alpha=0.30,
                            color="C0", label="p25–p75")
            ax.plot(steps, cs["med"], color="C0", lw=2.0, label="median")
            ax.axhline(res["agg_plateau"], color="C3", ls="--", lw=1,
                       label=f"plateau {res['agg_plateau']:.3g}")
            ax.set_xlabel("sampling step")
            ax.set_ylabel(name)
            if logy:
                ax.set_yscale("log")
                ax.set_title("log y")
            else:
                ax.set_title("linear y")
            ax.legend(fontsize=8)
        fig.suptitle(f"{name}: {M.shape[0]} trajectories")
        fig.tight_layout()
        p = outdir / f"{stem}_{name}_curve.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        print(f"  wrote {p}")

        # --- normalized-to-own-plateau median curve (fractional approach) ---
        norm = M / res["plateau"][:, None]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.fill_between(steps, np.nanpercentile(norm, 25, axis=0),
                        np.nanpercentile(norm, 75, axis=0), alpha=0.3, color="C1")
        ax.plot(steps, np.nanmedian(norm, axis=0), color="C1", lw=2)
        ax.axhline(1.0, color="0.4", ls="--", lw=1)
        ax.set_xlabel("sampling step")
        ax.set_ylabel(f"{name} / own plateau")
        hi = float(np.nanpercentile(norm, 95))
        ax.set_ylim(0.0, max(1.5, min(5.0, hi)))
        ax.set_title("approach to each structure's own plateau (median, IQR)")
        fig.tight_layout()
        p = outdir / f"{stem}_{name}_normalized.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        print(f"  wrote {p}")

        # --- histograms: plateau value + tail slope ---
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].hist(res["plateau"][np.isfinite(res["plateau"])], bins=60,
                     color="C0")
        axes[0].axvline(np.nanmedian(res["plateau"]), color="C3", ls="--")
        axes[0].set_xlabel(f"per-structure plateau ({name})")
        axes[0].set_ylabel("structures")
        sr = res["slope_rel"][np.isfinite(res["slope_rel"])] * 100
        axes[1].hist(sr, bins=60, color="C0")
        axes[1].axvline(0, color="0.4", ls="--")
        axes[1].axvline(np.median(sr), color="C3", ls="--")
        axes[1].set_xlabel(f"tail slope (% of plateau / step), last {res['W']} steps")
        axes[1].set_ylabel("structures")
        fig.suptitle(name)
        fig.tight_layout()
        p = outdir / f"{stem}_{name}_hist.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        print(f"  wrote {p}")


# --------------------------------------------------------------------------- #
def write_summary_csv(results, path):
    base = None
    for res in results:
        if res is None:
            continue
        d = pd.DataFrame(index=res["idx"])
        d[f"{res['name']}_plateau"] = res["plateau"]
        d[f"{res['name']}_tail_slope"] = res["slope"]
        d[f"{res['name']}_tail_slope_rel_pct_per_step"] = res["slope_rel"] * 100
        d[f"{res['name']}_total_decay_factor"] = res["total_decay"]
        base = d if base is None else base.join(d)
    if base is not None:
        base.insert(0, "n_atoms",
                    next(r for r in results if r is not None)["natoms"])
        base = base.reset_index()
        base.to_csv(path, index=False)
        print(f"\nper-structure summary -> {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", type=Path, help="eval_field_norms*.csv from relax_wbm.py")
    ap.add_argument("--tail-frac", type=float, default=0.2,
                    help="fraction of steps used as the tail window for slope / "
                         "'still dropping' (default 0.2)")
    ap.add_argument("--tail-steps", type=int, default=None,
                    help="absolute tail window in steps (overrides --tail-frac)")
    ap.add_argument("--plateau-frac", type=float, default=0.1,
                    help="fraction of steps averaged for the plateau value "
                         "(default 0.1)")
    ap.add_argument("--plateau-steps", type=int, default=None,
                    help="absolute plateau window in steps (overrides --plateau-frac)")
    ap.add_argument("--tol", type=float, nargs="+", default=[0.05, 0.02, 0.01],
                    help="tolerances for the 'within X%% of plateau' step")
    ap.add_argument("--plot", action="store_true", help="write figures next to the CSV")
    ap.add_argument("--outdir", type=Path, default=None,
                    help="directory for figures / summary CSV (default: CSV's dir)")
    ap.add_argument("--n-sample-traj", type=int, default=120,
                    help="individual trajectories to overlay on the curve plot")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df, n_dropped = load_csv(args.csv)
    outdir = args.outdir or args.csv.parent
    outdir.mkdir(parents=True, exist_ok=True)
    stem = args.csv.stem

    n_traj = df.groupby(TRAJ_KEYS).ngroups
    n_mid = df["material_id"].nunique()
    n_eval = df["eval_idx"].nunique()
    print(f"=== field-norm trajectory analysis: {args.csv} ===")
    print(f"trajectories : {n_traj}  ({n_mid} material_ids x {n_eval} eval_idx)")
    print(f"rows dropped (null-baseline / NaN): {n_dropped}")
    nsu = df["n_steps_used"].dropna().unique()
    print(f"n_steps_used values: {sorted(nsu.tolist())[:8]}"
          f"{' ...' if len(nsu) > 8 else ''}")

    results = [analyze_modality(m, df, args) for m in MODALITIES]

    write_summary_csv(results, outdir / f"{stem}_summary.csv")
    if args.plot:
        print("\nplots:")
        make_plots(results, stem, outdir, args.n_sample_traj, args.seed)


if __name__ == "__main__":
    main()
