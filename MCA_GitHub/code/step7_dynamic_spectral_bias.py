#!/usr/bin/env python3
"""Step 7: Mode-wise learning, persistence, rebounds, and PDE amplification.

Place this script in Numerics and run:
    python step7_dynamic_spectral_bias.py

Reads dynamic_modes.csv from Step 3/4 run folders. Writes to a NEW
step7_dynamic_spectral_bias_results folder. No model training or extra packages.

IMPORTANT: e_j is a signed *error* coefficient, not a target coefficient.
The ratio |e_j(t)|/|e_j(0)| is not meaningful for near-zero initial modes.
Modes can grow before shrinking. Sparse epoch sampling bounds crossing times.
"""
import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parent
PAT = re.compile(r'^k(\d+)_alpha_([pm])(\d+(?:d\d+)?)_(normalized|raw)_seed_(\d+)$')

def read(path):
    with path.open(newline='',encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))

def write(path, rows, columns):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore')
        w.writeheader(); w.writerows(rows)

def parse(path):
    m=PAT.fullmatch(path.parent.name)
    if not m: return None
    return (int(m[1]), float(m[3].replace('d','.')) * (-1 if m[2]=='m' else 1),m[4],int(m[5]))

def first_sustained(series, threshold):
    """First observed time at/below threshold, remaining so thereafter."""
    for i,(t,x) in enumerate(series):
        if x <= threshold and all(v <= threshold for _,v in series[i:]):
            return t
    return None

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=BASE)
    ap.add_argument('--max-mode',type=int,default=32)
    ap.add_argument('--relative-threshold',type=float,default=0.1)
    ap.add_argument('--initial-floor-fraction',type=float,default=0.01,
                    help='Initial |e_j| must exceed this fraction of max initial |e_j| to use relative crossing')
    ap.add_argument('--rebound-factor',type=float,default=2.,
                    help='Detect a rebound if a later modal magnitude exceeds this factor times a prior minimum')
    args=ap.parse_args()
    if not (0<args.relative_threshold<1): ap.error('relative-threshold must be between 0 and 1')
    if not (0<=args.initial_floor_fraction<1): ap.error('initial-floor-fraction must be in [0,1)')
    if args.max_mode<1 or args.rebound_factor<=1: ap.error('invalid mode count or rebound factor')
    root=args.root.resolve(); output=root/'step7_dynamic_spectral_bias_results'
    paths=[]
    for folder in ('step3_shifted_operator_results','step4_multiseed_results'):
        d=root/folder
        if d.exists(): paths.extend(d.rglob('dynamic_modes.csv'))
    runs={}; skipped=[]
    for path in sorted(paths):
        key=parse(path)
        if key is None:
            skipped.append({'file':str(path),'reason':'Unrecognized run directory'});continue
        rows=read(path)
        if not rows or not {'epoch','mode','error_coeff','residual_coeff'}.issubset(rows[0]):
            skipped.append({'file':str(path),'reason':'Missing required columns'});continue
        hist=defaultdict(dict); bad=False
        for r in rows:
            try:
                t=int(r['epoch']); j=int(r['mode']); e=float(r['error_coeff']); res=float(r['residual_coeff'])
                if t<0 or j<1 or not(math.isfinite(e) and math.isfinite(res)): raise ValueError
                if j>args.max_mode:continue
                if j in hist[t]: raise ValueError
                hist[t][j]=(e,res)
            except (KeyError,ValueError,TypeError): bad=True;break
        if bad or not hist:
            skipped.append({'file':str(path),'reason':'Invalid or empty modal history'});continue
        # If the same run is copied to Step 4, prefer the longer history; flag conflicts.
        if key in runs:
            prev=runs[key]
            overlap=set(prev['hist']) & set(hist)
            inconsistent=any(abs(prev['hist'][t][j][0]-hist[t][j][0])>1e-7
                             for t in overlap for j in set(prev['hist'][t])&set(hist[t]))
            if inconsistent:
                skipped.append({'file':str(path),'reason':'Conflicting duplicate run'});continue
            if len(hist)<=len(prev['hist']):continue
        runs[key]={'hist':dict(hist),'file':str(path)}
    if not runs:
        print('No valid dynamic_modes.csv found. Expected Step 3/4 run subfolders under Numerics.')
        return 1
    metrics=[]; curves=[]; summaries=[]
    for (k,alpha,loss,seed), run in sorted(runs.items()):
        hist=run['hist']; times=sorted(hist); t0=times[0]; tlast=times[-1]
        initial=hist[t0]; initial_scale=max((abs(v[0]) for v in initial.values()),default=0)
        floor=args.initial_floor_fraction*initial_scale
        js=sorted(set().union(*(set(x) for x in hist.values())))
        total_initial=sum(.5*v[0]**2 for v in initial.values())
        total_final=sum(.5*v[0]**2 for v in hist[tlast].values())
        target_energy_final=.5*hist[tlast].get(k,(0,0))[0]**2
        per_run=[]
        for j in js:
            series=[(t,abs(hist[t][j][0])) for t in times if j in hist[t]]
            if len(series)<2: continue
            initial_mag=series[0][1]; final_mag=series[-1][1]
            reliable=initial_mag>floor and initial_mag>1e-14
            threshold=args.relative_threshold*initial_mag if reliable else None
            first_cross=next((t for t,v in series if threshold is not None and v<=threshold),None)
            sustained=first_sustained(series,threshold) if threshold is not None else None
            min_time,min_val=min(series,key=lambda pair:pair[1])
            rebound=(final_mag/min_val if min_val>1e-14 else float('inf') if final_mag>1e-14 else 1.)
            later_max=max((v for t,v in series if t>min_time),default=min_val)
            late_rebound=bool(later_max>args.rebound_factor*max(min_val,1e-14))
            amplification=1/((j*math.pi)**2+alpha)
            row=dict(k=k,alpha=alpha,loss=loss,seed=seed,mode=j,target_mode=int(j==k),
                inverse_amplification=amplification,n_epochs=len(series),initial_abs_error=initial_mag,
                final_abs_error=final_mag,final_error_energy=.5*final_mag**2,
                relative_crossing_eligible=int(reliable),threshold=threshold if reliable else '',
                first_crossing_epoch=first_cross if first_cross is not None else '',
                sustained_crossing_epoch=sustained if sustained is not None else '',
                first_crossing_rebounded=int(first_cross is not None and sustained!=first_cross),
                min_error_epoch=min_time,min_abs_error=min_val,
                final_to_min_ratio=rebound,late_rebound=int(late_rebound),
                final_over_initial=final_mag/initial_mag if reliable else '',
                final_energy_share=.5*final_mag**2/total_final if total_final else '',
                source_file=run['file'])
            metrics.append(row);per_run.append(row)
            for t,v in series:
                curves.append(dict(k=k,alpha=alpha,loss=loss,seed=seed,epoch=t,mode=j,
                    abs_error=v,relative_error=v/initial_mag if reliable else '',
                    inverse_amplification=amplification,error_energy=.5*v*v,
                    residual_coeff=hist[t][j][1]))
        eligible=[r for r in per_run if r['relative_crossing_eligible']]
        summaries.append(dict(k=k,alpha=alpha,loss=loss,seed=seed,first_epoch=t0,last_epoch=tlast,
            n_modes=len(per_run),eligible_modes=len(eligible),
            first_crossing_modes=sum(r['first_crossing_epoch']!='' for r in eligible),
            sustained_crossing_modes=sum(r['sustained_crossing_epoch']!='' for r in eligible),
            rebounding_modes=sum(r['late_rebound'] for r in per_run),
            initial_modal_l2=math.sqrt(total_initial),final_modal_l2=math.sqrt(total_final),
            final_first_mode_energy_share=next((r['final_energy_share'] for r in per_run if r['mode']==1),''),
            final_target_mode_energy_share=(target_energy_final/total_final if total_final else ''),
            target_mode_sustained_crossing=next((r['sustained_crossing_epoch'] for r in per_run if r['mode']==k),'')))
    output.mkdir(parents=True,exist_ok=True)
    write(output/'mode_learning_metrics.csv',metrics,list(metrics[0]))
    write(output/'mode_error_trajectories.csv',curves,list(curves[0]))
    write(output/'run_learning_summary.csv',summaries,list(summaries[0]))
    if skipped:write(output/'skipped_files.csv',skipped,['file','reason'])
    print(f'Loaded {len(runs)} unique run histories, {len(metrics)} mode summaries.')
    print('Output:',output)
    print('IMPORTANT: a missing learning time is not zero; near-zero initial modes are ineligible.')
    print('IMPORTANT: first/sustained crossing times are sampled-epoch estimates, not continuous times.')
    print('IMPORTANT: inverse amplification is fixed by operator; it does not establish causal learning rates.')
    if len(runs)<18:print(f'WARNING: {18-len(runs)} of the 18 expected histories are missing.')
    return 0

if __name__=='__main__':
    raise SystemExit(main())
