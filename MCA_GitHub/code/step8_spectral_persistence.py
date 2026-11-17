#!/usr/bin/env python3
"""Step 8: time-weighted spectral persistence and PDE amplification (no training).

Put this file in Numerics and run:
    python step8_spectral_persistence.py

Reads step7_dynamic_spectral_bias_results/mode_error_trajectories.csv,
or ./mode_error_trajectories.csv. Outputs to step8_spectral_persistence_results.
Uses trapezoidal time integration over recorded epochs, not an unweighted
average. Results concern the measured, truncated Fourier spectrum only.
"""
import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


def read_csv(path):
    with path.open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def save(path, rows):
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def trap_mean(series):
    """Time average of y over increasing (t,y) samples."""
    if len(series) < 2:
        return float('nan')
    span = series[-1][0] - series[0][0]
    if span <= 0:
        return float('nan')
    return sum((b[0]-a[0])*(a[1]+b[1])*0.5
               for a,b in zip(series, series[1:]))/span


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input', type=Path, help='Step 7 mode_error_trajectories.csv')
    ap.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument('--max-mode', type=int, default=32)
    args = ap.parse_args()
    if args.max_mode < 1:
        ap.error('--max-mode must be positive')
    root = args.root.resolve()
    candidates = [root/'step7_dynamic_spectral_bias_results'/'mode_error_trajectories.csv',
                  root/'mode_error_trajectories.csv']
    src = args.input or next((p for p in candidates if p.is_file()), None)
    if src is None or not src.is_file():
        ap.error('Cannot find mode_error_trajectories.csv. Supply --input PATH.')
    out = root/'step8_spectral_persistence_results'
    out.mkdir(exist_ok=True, parents=True)
    rows = read_csv(src)
    needed = {'k','alpha','loss','seed','epoch','mode','abs_error','residual_coeff'}
    if not rows or not needed.issubset(rows[0]):
        ap.error('Input CSV lacks required Step 7 columns: '+str(sorted(needed)))
    by_run = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        try:
            k, alpha, seed = int(row['k']),float(row['alpha']),int(row['seed'])
            mode, epoch = int(row['mode']),int(row['epoch'])
            err, residual = float(row['abs_error']),float(row['residual_coeff'])
            if mode < 1 or mode > args.max_mode: continue
            if epoch < 0 or not all(map(math.isfinite,[alpha,err,residual])):
                raise ValueError('Nonfinite values or negative epoch')
        except (ValueError, KeyError) as exc:
            ap.error('Invalid row: '+str(exc))
        key = (k,alpha,row['loss'],seed)
        if epoch in by_run[key][mode]:
            ap.error(f'Duplicate epoch {epoch} for run {key}, mode {mode}')
        by_run[key][mode][epoch] = (err,residual)

    mode_results = []
    run_results = []
    for (k,alpha,loss,seed), modes in sorted(by_run.items()):
        # Require common time grid across modes to make modal shares comparable.
        grids = [set(x) for x in modes.values()]
        common = sorted(set.intersection(*grids))
        if len(common) < 2:
            print('SKIP: insufficient common epochs',k,alpha,seed)
            continue
        per_mode = []
        for j, history in sorted(modes.items()):
            amp = 1/((j*math.pi)**2+alpha)
            error_energy = [(t,0.5*history[t][0]**2) for t in common]
            residual_energy = [(t,0.5*history[t][1]**2) for t in common]
            p_error = trap_mean(error_energy)
            p_residual = trap_mean(residual_energy)
            predicted = amp**2*p_residual
            final = error_energy[-1][1]
            per_mode.append(dict(k=k,alpha=alpha,loss=loss,seed=seed,mode=j,
                is_target=int(j==k),n_epochs=len(common),first_epoch=common[0],last_epoch=common[-1],
                inverse_amplification=amp,inverse_amplification_squared=amp**2,
                mean_solution_error_energy=p_error,mean_residual_energy=p_residual,
                predicted_solution_error_energy=predicted,
                absolute_identity_defect=abs(p_error-predicted),
                relative_identity_defect=abs(p_error-predicted)/max(p_error,predicted,1e-30),
                final_solution_error_energy=final))
        total = sum(x['mean_solution_error_energy'] for x in per_mode)
        final_total = sum(x['final_solution_error_energy'] for x in per_mode)
        for x in per_mode:
            x['persistence_share'] = x['mean_solution_error_energy']/total if total else 0
            x['final_error_share'] = x['final_solution_error_energy']/final_total if final_total else 0
        per_mode.sort(key=lambda x:-x['mean_solution_error_energy'])
        for rank,x in enumerate(per_mode,1):
            x['persistence_rank'] = rank
            mode_results.append(x)
        low = sum(x['mean_solution_error_energy'] for x in per_mode if x['mode']<=4)
        target = sum(x['mean_solution_error_energy'] for x in per_mode if x['mode']==k)
        first = sum(x['mean_solution_error_energy'] for x in per_mode if x['mode']==1)
        off_target = total-target
        run_results.append(dict(k=k,alpha=alpha,loss=loss,seed=seed,
            n_modes=len(per_mode),n_epochs=len(common),first_epoch=common[0],last_epoch=common[-1],
            time_averaged_modal_squared_L2=total,
            time_averaged_modal_L2_rms=math.sqrt(total),
            low_modes_1_to_4_share=low/total if total else 0,
            first_mode_share=first/total if total else 0,
            target_mode_share=target/total if total else 0,
            off_target_share=off_target/total if total else 0,
            dominant_mode=per_mode[0]['mode'],
            dominant_mode_share=per_mode[0]['persistence_share'],
            identity_defect_sum=sum(x['absolute_identity_defect'] for x in per_mode),
            top_5_modes=';'.join(str(x['mode']) for x in per_mode[:5])))
    if not run_results:
        ap.error('No valid runs with at least two recorded epochs')
    save(out/'mode_persistence_metrics.csv',mode_results)
    save(out/'run_persistence_summary.csv',run_results)
    grouped=defaultdict(list)
    for r in run_results:
        grouped[(r['k'],r['alpha'])].append(r)
    aggregates=[]
    for (k,alpha),runs in sorted(grouped.items()):
        def avg(field): return sum(r[field] for r in runs)/len(runs)
        aggregates.append(dict(k=k,alpha=alpha,n_seeds=len(runs),
            mean_time_averaged_modal_squared_L2=avg('time_averaged_modal_squared_L2'),
            mean_low_modes_share=avg('low_modes_1_to_4_share'),
            mean_first_mode_share=avg('first_mode_share'),
            mean_target_mode_share=avg('target_mode_share'),
            mean_off_target_share=avg('off_target_share')))
    save(out/'configuration_persistence_summary.csv',aggregates)
    print(f'Processed {len(run_results)} runs and {len(mode_results)} mode-run records.')
    print('Saved:',out)
    print('IMPORTANT: time-averaged SQUARED modal L2, using recorded epochs and common modes.')
    print('Check relative_identity_defect before interpreting the amplification identity.')
    return 0

if __name__=='__main__':
    raise SystemExit(main())
