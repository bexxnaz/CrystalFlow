


import argparse
import json
from pathlib import Path
 
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
 
 
PHYSICAL_BOUND_EV_PER_ATOM = 100  # matches energy_trajectory_report.py's
                                   # blow-up-exclusion convention from
                                   # earlier in this investigation
 
 
def load_run(outdir, label):
    outdir = Path(outdir)
    suffix = f'_{label}' if label else ''
 
    field_df = pd.read_csv(outdir / f'eval_field_norms{suffix}.csv')
    mace_df = pd.read_csv(outdir / f'eval_mace_energy{suffix}.csv')
    with open(outdir / f'eval_metrics{suffix}.json') as f:
        metrics = json.load(f)
    gt_df = pd.read_csv(outdir / f'eval_gt_energy{suffix}.csv')
 
    return field_df, mace_df, metrics, gt_df
 
 
def load_mace_relax_trajectory(outdir, material_id, eval_idx=0, explicit_path=None,
                                suffix='energy_trajectory'):
    """Optional: MACE's own FIRE-optimizer relaxation of the saved initial
    CIF (mace_relax_single_cif.py's output) -- energy by default, or RMSD
    (suffix='rmsd_trajectory') if mace_relax_single_cif.py was run with
    --gt_cif_file/--gt_file. Returns None, not an error, if nothing is
    found -- this is an opt-in extra, not something every run will have.
    Tries, in order: an explicit path if given, then the two most likely
    default locations (mace_relax/ subfolder, or directly in outdir --
    depends on what --outdir was passed to mace_relax_single_cif.py when
    it ran). Prints exactly what it checked either way, so a genuine miss
    is easy to diagnose instead of silently dropping a row.
    """
    outdir = Path(outdir)
    filename = f'{material_id}_eval{eval_idx}_{suffix}.csv'
    candidates = [Path(explicit_path)] if explicit_path else [
        outdir / 'mace_relax' / filename,
        outdir / filename,
    ]
    for path in candidates:
        if path.exists():
            print(f'Found MACE native-relax {suffix} at {path}')
            return pd.read_csv(path)
    print(f'No mace_relax {suffix} found for {material_id} eval{eval_idx} -- checked: '
          f'{", ".join(str(p) for p in candidates)}. Pass an explicit path to point at '
          f'it directly if it lives somewhere else.')
    return None
 
 
def build_combined_df(field_df, mace_df, metrics, material_id, eval_idx=0):
    """One row per step, all three signals aligned on the SAME step number.
 
    NOTE on step alignment: eval_field_norms.csv labels its steps 1..N (N
    completed integration steps -- there's no "field norm at step 0" since
    the field is a per-transition quantity, not a per-state one).
    eval_mace_energy.csv and all_rms_dis both index/label steps 0..N (0 =
    the untouched starting structure). These are treated here as the SAME
    step-number convention (field_norms' step=k is taken to mean the same
    "k completed steps" as mace_energy's step=k and all_rms_dis[k]) -- this
    is the most self-consistent reading of both CSV writers' own labeling,
    but I could not confirm it directly against model.sample()'s internal
    indexing. Worth a one-time sanity check: field_norms' last step number
    should equal n_steps_used, and should line up with mace_energy's step
    range for the same material.
    """
    f = field_df[(field_df.material_id == material_id) & (field_df.eval_idx == eval_idx)]
    m = mace_df[(mace_df.material_id == material_id) & (mace_df.eval_idx == eval_idx)]
 
    rmsd_list = [r[0] if r else np.nan for r in metrics['all_rms_dis']]
    rmsd_df = pd.DataFrame({'step': range(len(rmsd_list)), 'rmsd': rmsd_list})
 
    df = m[['step', 'energy_per_atom_eV']].merge(
        f[['step', 'coord_field_norm', 'lattice_field_norm']], on='step', how='outer'
    ).merge(rmsd_df, on='step', how='outer').sort_values('step').reset_index(drop=True)
    return df
 
 
def plot_combined(df, gt_energy_per_atom, material_id, out_path, zoom_from_step=None,
                   mace_relax_df=None, mace_relax_rmsd_df=None, label=None):
    """Two columns per signal: LEFT = full trajectory (linear, shows the
    overall trend at low resolution -- fine, since the job there is just
    'it's dropping fast'), RIGHT = zoomed into the tail (linear, y-axis
    auto-fit to ONLY that subset of data), so once the numbers get close
    the actual values are directly readable -- not a log transform, which
    trades exact-value readability for equal resolution everywhere.
    """
    if zoom_from_step is None:
        # Default: the LATER of (a) halfway through the trajectory or (b)
        # whenever RMSD first becomes defined -- so the zoom starts
        # roughly where things have stabilized, not mid-transition.
        # Override with --zoom_from_step if this doesn't land well on
        # your actual data.
        halfway = df['step'].max() / 2
        valid_rmsd = df.loc[df['rmsd'].notna(), 'step']
        first_valid_rmsd_step = valid_rmsd.min() if len(valid_rmsd) else halfway
        zoom_from_step = max(halfway, first_valid_rmsd_step)
 
    df_zoom = df[df['step'] >= zoom_from_step]
 
    # Rows: field norm, energy-solo, [energy-joint], rmsd-solo, [rmsd-joint]
    # -- built with a running row counter rather than hardcoded indices, so
    # adding/removing an optional row can't silently shift another row's
    # index out from under it.
    # Rows: field norm, energy-solo, [energy-joint], rmsd-solo,
    # [rmsd-mace-relax-solo, rmsd-joint] -- built with a running row
    # counter rather than hardcoded indices, so adding/removing an
    # optional row can't silently shift another row's index out from
    # under it. rmsd-mace-relax-solo and rmsd-joint are gated on the SAME
    # condition (mace_relax_rmsd_df present) since the joint view needs
    # that same data, so that condition contributes 2 rows, not 1.
    n_rows = 3 + 2 * (mace_relax_df is not None) + 2 * (mace_relax_rmsd_df is not None)
    fig, axes = plt.subplots(n_rows, 2, figsize=(13, 3.3 * n_rows))
    row = 0
 
    def _mark_zoom_region(ax):
        ax.axvline(zoom_from_step, color='grey', linestyle=':', linewidth=1)
 
    def _annotate_first_value(ax, x, y, color='black'):
        """Text-labels the first non-NaN (x, y) point directly on the
        plot with its exact numeric value -- axis ticks alone often
        can't be read precisely, especially at the extreme scales some
        of these panels span (e.g. billions of eV at the very start of
        the energy panels). Silently does nothing if the series is empty
        or entirely NaN."""
        x = np.asarray(x)
        y = np.asarray(y, dtype=float)
        if len(y) == 0:
            return
        valid = ~np.isnan(y)
        if not valid.any():
            return
        idx = int(np.argmax(valid))  # index of the first True
        ax.annotate(f'{y[idx]:.3g}', xy=(x[idx], y[idx]), xytext=(4, 4),
                    textcoords='offset points', fontsize=7, color=color)
 
    def _mark_range(ax, y, color):
        """Two thin horizontal reference lines at the min and max of a
        series, values labeled -- for the 'flattened' (zoomed) columns
        specifically: a plateau can still be wiggling within some range
        even though it looks visually flat, and this makes that range
        directly readable instead of requiring the reader to infer it
        from the y-axis scale alone. Does nothing if there's no real
        range (empty, all-NaN, or a single repeated value)."""
        y = np.asarray(y, dtype=float)
        y = y[~np.isnan(y)]
        if len(y) == 0:
            return
        y_min, y_max = y.min(), y.max()
        if y_min == y_max:
            return
        for val, va in [(y_min, 'top'), (y_max, 'bottom')]:
            ax.axhline(val, color=color, linestyle=':', linewidth=0.8, alpha=0.5)
            ax.annotate(f'{val:.3g}', xy=(1, val), xycoords=('axes fraction', 'data'),
                        xytext=(-3, 3 if va == 'top' else -3), textcoords='offset points',
                        fontsize=6, color=color, ha='right', va=va)
 
    # --- field norm ---
    for col, (data, title) in enumerate([
        (df, 'full trajectory'), (df_zoom, f'zoomed: step \u2265 {zoom_from_step:.0f}')
    ]):
        ax = axes[row, col]
        ax.plot(data['step'], data['coord_field_norm'], label='coord field norm', color='tab:blue')
        ax.plot(data['step'], data['lattice_field_norm'], label='lattice field norm', color='tab:orange')
        _annotate_first_value(ax, data['step'], data['coord_field_norm'], color='tab:blue')
        _annotate_first_value(ax, data['step'], data['lattice_field_norm'], color='tab:orange')
        if col == 1:
            _mark_range(ax, data['coord_field_norm'], color='tab:blue')
            _mark_range(ax, data['lattice_field_norm'], color='tab:orange')
        ax.set_ylabel('field norm')
        ax.set_title(title, fontsize=10)
        if col == 0:
            ax.legend(fontsize=8)
            _mark_zoom_region(ax)
    row += 1
 
    # --- energy, EQM model ONLY (clean, no overlay) ---
    for col, data in enumerate([df, df_zoom]):
        ax = axes[row, col]
        ax.plot(data['step'], data['energy_per_atom_eV'], color='tab:green', label='EQM model')
        _annotate_first_value(ax, data['step'], data['energy_per_atom_eV'], color='tab:green')
        if col == 1:
            _mark_range(ax, data['energy_per_atom_eV'], color='tab:green')
        ax.axhline(gt_energy_per_atom, color='black', linestyle='--', linewidth=1,
                   label=f'ground truth ({gt_energy_per_atom:.4f} eV/atom)')
        ax.set_ylabel('MACE energy (eV/atom)')
        ax.set_title('EQM model only' if col == 0 else None, fontsize=9)
        ax.legend(fontsize=7)
        if col == 0:
            _mark_zoom_region(ax)
    row += 1
 
    # --- energy, MACE native relax ONLY -- its own step axis/scale, not
    # overlaid with the EQM model. Zoomed column here uses ITS OWN
    # halfway point (matching the RMSD solo-mace-relax row's approach),
    # not the shared zoom_from_step -- reusing that would likely show an
    # empty panel, since MACE's own FIRE relaxation is usually much
    # shorter than the diffusion trajectory. ---
    if mace_relax_df is not None:
        mr_e_zoom_from = mace_relax_df['step'].max() / 2 if len(mace_relax_df) else 0
        mr_e_df_zoom = mace_relax_df[mace_relax_df['step'] >= mr_e_zoom_from]
        for col, (data, title) in enumerate([
            (mace_relax_df, 'MACE native relax only: full'),
            (mr_e_df_zoom, f'MACE native relax only: zoomed (own step \u2265 {mr_e_zoom_from:.0f})'),
        ]):
            ax = axes[row, col]
            ax.plot(data['step'], data['energy_per_atom_eV'], color='tab:purple',
                     marker='s', markersize=3, label='MACE native relax')
            _annotate_first_value(ax, data['step'], data['energy_per_atom_eV'], color='tab:purple')
            if col == 1:
                _mark_range(ax, data['energy_per_atom_eV'], color='tab:purple')
            ax.axhline(gt_energy_per_atom, color='black', linestyle='--', linewidth=1,
                       label=f'ground truth ({gt_energy_per_atom:.4f} eV/atom)')
            ax.set_ylabel('MACE energy (eV/atom)')
            ax.set_title(title, fontsize=9)
            ax.legend(fontsize=7)
            ax.set_xlabel('step (FIRE steps, own axis)')
        row += 1
 
    # --- energy, JOINT with MACE's own FIRE relaxation (only if available) ---
    if mace_relax_df is not None:
        # LEFT column: cropped to min(both models' max step), not the
        # full EQM range. EQM often runs to hundreds/thousands of steps
        # while MACE's own FIRE relaxation converges in a few dozen --
        # plotted on the full EQM range, the shorter trajectory was
        # squashed into an unreadable sliver at the left edge.
        joint_e_min_step = min(df['step'].max(), mace_relax_df['step'].max())
        df_joint_e_left = df[df['step'] <= joint_e_min_step]
        mr_joint_e_left = mace_relax_df[mace_relax_df['step'] <= joint_e_min_step]
        for col, (data, mr) in enumerate([
            (df_joint_e_left, mr_joint_e_left),
            (df_zoom, mace_relax_df[mace_relax_df['step'] >= zoom_from_step]),
        ]):
            ax = axes[row, col]
            ax.plot(data['step'], data['energy_per_atom_eV'], color='tab:green', label='EQM model')
            _annotate_first_value(ax, data['step'], data['energy_per_atom_eV'], color='tab:green')
            if col == 1:
                _mark_range(ax, data['energy_per_atom_eV'], color='tab:green')
            ax.axhline(gt_energy_per_atom, color='black', linestyle='--', linewidth=1,
                       label=f'ground truth ({gt_energy_per_atom:.4f} eV/atom)')
            if len(mr):
                ax.plot(mr['step'], mr['energy_per_atom_eV'], color='tab:purple', linestyle='--',
                         label='MACE native relax\n(own FIRE steps -- not the same axis as diffusion steps)')
                _annotate_first_value(ax, mr['step'], mr['energy_per_atom_eV'], color='tab:purple')
                if col == 1:
                    _mark_range(ax, mr['energy_per_atom_eV'], color='tab:purple')
            ax.set_ylabel('MACE energy (eV/atom)')
            ax.set_title(f'joint, step \u2264 {joint_e_min_step:.0f}' if col == 0 else None, fontsize=9)
            ax.legend(fontsize=7)
        row += 1
 
    def _shade_not_constructible(ax, data, valid):
        """Grey-shade EVERY contiguous region where RMSD is undefined --
        not just a leading prefix. A structure can become invalid again
        after being valid (a mid-trajectory gap), or stay invalid all the
        way to the final step (a trailing gap that never recovers) --
        this walks the whole valid/invalid sequence and shades each
        invalid run on its own, rather than only handling "NaN at the
        start" or "NaN everywhere"."""
        steps = data['step'].to_numpy()
        valid_arr = valid.to_numpy()
        n = len(steps)
        if n == 0:
            return
 
        if not valid_arr.any():
            ax.axvspan(steps.min(), steps.max(), color='grey', alpha=0.15,
                       label='structure not constructible (entire range)')
            ax.text(0.5, 0.5, 'structure never constructible\n(RMSD undefined at every step)',
                    ha='center', va='center', transform=ax.transAxes, fontsize=8, color='dimgrey')
            return
 
        labeled = False
        i = 0
        while i < n:
            if not valid_arr[i]:
                j = i
                while j < n and not valid_arr[j]:
                    j += 1
                # invalid run covers steps[i:j] -- shades from the first
                # to the last NaN step in this specific run (prefix,
                # mid-trajectory, or trailing all shaded the same way).
                ax.axvspan(steps[i], steps[j - 1], color='grey', alpha=0.15,
                           label=None if labeled else 'structure not constructible')
                labeled = True
                i = j
            else:
                i += 1
 
    is_last_rmsd_row = mace_relax_rmsd_df is None
 
    # --- RMSD, EQM model ONLY (clean, no overlay) ---
    for col, data in enumerate([df, df_zoom]):
        ax = axes[row, col]
        valid = data['rmsd'].notna()
        ax.plot(data['step'], data['rmsd'], color='tab:red', marker='o',
                 markersize=3, label='EQM model')
        _annotate_first_value(ax, data['step'], data['rmsd'], color='tab:red')
        if col == 1:
            _mark_range(ax, data['rmsd'], color='tab:red')
        _shade_not_constructible(ax, data, valid)
        ax.set_ylabel('RMSD vs. ground truth (\u00c5)')
        ax.set_title('EQM model only' if (col == 0 and mace_relax_rmsd_df is not None) else None,
                     fontsize=9)
        ax.legend(fontsize=7)
        if col == 0:
            _mark_zoom_region(ax)
        if is_last_rmsd_row:
            ax.set_xlabel('step')
    row += 1
 
    # --- RMSD, MACE native relax ONLY -- its own step axis/scale, not
    # overlaid with the EQM model (whose step count and meaning
    # differ). Zoomed column here uses ITS OWN halfway point, not the
    # shared zoom_from_step above -- reusing that would likely show an
    # empty panel, since MACE's own FIRE relaxation is usually much
    # shorter than the diffusion trajectory. ---
    if mace_relax_rmsd_df is not None:
        mr_valid_full = mace_relax_rmsd_df[mace_relax_rmsd_df['rmsd'].notna()]
        mr_zoom_from = mr_valid_full['step'].max() / 2 if len(mr_valid_full) else 0
        mr_df_zoom = mace_relax_rmsd_df[mace_relax_rmsd_df['step'] >= mr_zoom_from]
        for col, (data, title) in enumerate([
            (mace_relax_rmsd_df, 'MACE native relax only: full'),
            (mr_df_zoom, f'MACE native relax only: zoomed (own step \u2265 {mr_zoom_from:.0f})'),
        ]):
            ax = axes[row, col]
            valid = data['rmsd'].notna()
            ax.plot(data['step'], data['rmsd'], color='tab:purple',
                     marker='s', markersize=3, label='MACE native relax')
            _annotate_first_value(ax, data['step'], data['rmsd'], color='tab:purple')
            if col == 1:
                _mark_range(ax, data['rmsd'], color='tab:purple')
            _shade_not_constructible(ax, data, valid)
            ax.set_ylabel('RMSD vs. ground truth (\u00c5)')
            ax.set_title(title, fontsize=9)
            ax.legend(fontsize=7)
            ax.set_xlabel('step (FIRE steps, own axis)')
        row += 1
 
        # --- RMSD, JOINT: both curves on the EQM model's step axis,
        # cropped to min(both models' max step) on the left column for
        # the same reason as the energy-joint row above -- otherwise a
        # short MACE-relax trajectory gets squashed into an unreadable
        # sliver against EQM's full range. If MACE-relax has no valid
        # points in a given column (e.g. every value is NaN, or its
        # trajectory doesn't reach that far), its line is simply omitted
        # from that column rather than shown empty. ---
        joint_r_min_step = min(df['step'].max(), mace_relax_rmsd_df['step'].max())
        df_joint_r_left = df[df['step'] <= joint_r_min_step]
        mr_joint_r_left = mace_relax_rmsd_df[mace_relax_rmsd_df['step'] <= joint_r_min_step]
        for col, (data, mr) in enumerate([
            (df_joint_r_left, mr_joint_r_left),
            (df_zoom, mace_relax_rmsd_df[mace_relax_rmsd_df['step'] >= zoom_from_step]),
        ]):
            ax = axes[row, col]
            valid = data['rmsd'].notna()
            ax.plot(data['step'], data['rmsd'], color='tab:red', marker='o',
                     markersize=3, label='EQM model')
            _annotate_first_value(ax, data['step'], data['rmsd'], color='tab:red')
            if col == 1:
                _mark_range(ax, data['rmsd'], color='tab:red')
            _shade_not_constructible(ax, data, valid)
            mr_valid = mr[mr['rmsd'].notna()] if len(mr) else mr
            if len(mr_valid):
                ax.plot(mr['step'], mr['rmsd'], color='tab:purple', linestyle='--',
                         marker='s', markersize=3,
                         label='MACE native relax\n(own FIRE steps -- not the same axis as diffusion steps)')
                _annotate_first_value(ax, mr['step'], mr['rmsd'], color='tab:purple')
                if col == 1:
                    _mark_range(ax, mr['rmsd'], color='tab:purple')
            ax.set_ylabel('RMSD vs. ground truth (\u00c5)')
            ax.set_title(f'joint, step \u2264 {joint_r_min_step:.0f}' if col == 0 else None, fontsize=9)
            ax.legend(fontsize=7)
            ax.set_xlabel('step')
        row += 1
 
    for ax in axes.flat:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
 
    fig.suptitle(f'{material_id}' f'{label}', fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'Saved {out_path}')
 
 
def summarize(df, gt_energy_per_atom):
    plausible = df['energy_per_atom_eV'].abs() < PHYSICAL_BOUND_EV_PER_ATOM
    first_plausible_step = df.loc[plausible, 'step'].min() if plausible.any() else None
 
    valid_rmsd = df['rmsd'].notna()
    first_valid_step = df.loc[valid_rmsd, 'step'].min() if valid_rmsd.any() else None
    final_rmsd = df.loc[valid_rmsd, 'rmsd'].iloc[-1] if valid_rmsd.any() else None
    final_energy = df['energy_per_atom_eV'].dropna().iloc[-1] if df['energy_per_atom_eV'].notna().any() else None
 
    rmsd_series = df.loc[valid_rmsd, 'rmsd'].reset_index(drop=True)
    is_monotonic_improving = bool(rmsd_series.is_monotonic_decreasing) if len(rmsd_series) > 1 else None
    n_increases = int((rmsd_series.diff() > 0).sum()) if len(rmsd_series) > 1 else None
 
    print('--- Summary ---')
    print(f'Energy first physically plausible (|E| < {PHYSICAL_BOUND_EV_PER_ATOM} eV/atom): '
          f'step {first_plausible_step}')
    print(f'Structure first constructible (RMSD defined): step {first_valid_step}')
    if first_plausible_step is not None and first_valid_step is not None:
        lag = first_valid_step - first_plausible_step
        if lag > 0:
            print(f'  -> structural validity lagged physical plausibility by {lag} step(s) '
                  f'(energy was a leading indicator here)')
        elif lag < 0:
            print(f'  -> structural validity arrived {-lag} step(s) BEFORE energy became '
                  f'plausible (energy was NOT a leading indicator here)')
        else:
            print(f'  -> both arrived at the same step')
    print(f'Final RMSD vs. ground truth: {final_rmsd}')
    if final_energy is not None:
        print(f'Final energy: {final_energy:.4f} eV/atom (ground truth: {gt_energy_per_atom:.4f}, '
              f'delta: {final_energy - gt_energy_per_atom:.4f})')
    print(f'RMSD monotonically improving over its valid range: {is_monotonic_improving} '
          f'({n_increases} step(s) where it got worse)')
 
 
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--outdir', required=True, help='the eval_runs/<label>_<jobid> folder')
    ap.add_argument('--label', default='', help="matches relax_wbm.py/compute_metrics.py's --label")
    ap.add_argument('--material_id', default=None,
                     help='which material to plot (default: the only one present)')
    ap.add_argument('--eval_idx', type=int, default=0)
    ap.add_argument('--zoom_from_step', type=float, default=None,
                     help='step at which the zoomed (right-column) panels start. '
                          'Default: automatically the later of halfway through the '
                          'trajectory or whenever RMSD first becomes defined.')
    ap.add_argument('--mace_relax_csv', default=None,
                     help='explicit path to mace_relax_single_cif.py\'s '
                          '<label>_energy_trajectory.csv, if it isn\'t at the default '
                          'location (outdir/mace_relax/<material>_eval<n>_energy_trajectory.csv '
                          'or outdir/<material>_eval<n>_energy_trajectory.csv)')
    ap.add_argument('--mace_relax_rmsd_csv', default=None,
                     help='explicit path to mace_relax_single_cif.py\'s '
                          '<label>_rmsd_trajectory.csv (only produced if it was run with '
                          '--gt_cif_file/--gt_file), if it isn\'t at the default location')
    args = ap.parse_args()
 
    field_df, mace_df, metrics, gt_df = load_run(args.outdir, args.label)
 
    material_id = args.material_id or field_df['material_id'].iloc[0]
    gt_row = gt_df[gt_df.material_id == material_id]
    gt_energy_per_atom = float(gt_row['gt_energy_per_atom_eV'].iloc[0])
 
    df = build_combined_df(field_df, mace_df, metrics, material_id, args.eval_idx)
    mace_relax_df = load_mace_relax_trajectory(args.outdir, material_id, args.eval_idx,
                                                explicit_path=args.mace_relax_csv,
                                                suffix='energy_trajectory')
    mace_relax_rmsd_df = load_mace_relax_trajectory(args.outdir, material_id, args.eval_idx,
                                                     explicit_path=args.mace_relax_rmsd_csv,
                                                     suffix='rmsd_trajectory')
    outdir = Path(args.outdir)
    plot_combined(df, gt_energy_per_atom, material_id,
                  outdir / f'{material_id}_combined_trajectory.png',
                  zoom_from_step=args.zoom_from_step, mace_relax_df=mace_relax_df,
                  mace_relax_rmsd_df=mace_relax_rmsd_df,
                  label=args.label)
    summarize(df, gt_energy_per_atom)
    csv_out = outdir / f'{material_id}_combined_trajectory.csv'
    df.to_csv(csv_out, index=False)
    print(f'Saved {csv_out}')
 
 
if __name__ == '__main__':
    main()
 