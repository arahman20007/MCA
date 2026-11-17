#!/usr/bin/env python3
"""Step 5: counterfactual spectral amplification vs residual redistribution.

NO TRAINING. Requires per-run final_spectrum.csv from Step 3/4.
Run from Numerics:
    python step5_counterfactual_spectral_analysis.py

Searches both step3_shifted_operator_results and step4_multiseed_results.
Writes ONLY to Numerics/step5_counterfactual_results.

The analysis uses the first J sine modes, so results are truncated modal L2
norms, not necessarily the full solution error. Compares only matching k,
seed, loss, epochs, and common available modes.
"""
import argparse
import csv
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REQUIRED = {'mode', 'residual_coeff'}

def readcsv(p):
    with p.open('r', newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))

def writecsv(p, rows, cols):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)

def parse_run(path):
    # folder e.g. k16_alpha_p1000d0_normalized_seed_3
    m = re.fullmatch(r'k(\d+)_alpha_([pm])(\d+(?:d\d+)?)_(normalized|raw)_seed_(\d+)', path.parent.name)
    if not m:
        return None
    k = int(m.group(1))
    alpha = float(m.group(3).replace('d','.')) * (-1 if m.group(2)=='m' else 1)
    return k, alpha, m.group(4), int(m.group(5))

def load_run(path):
    ident = parse_run(path)
    if ident is None:
        return None, 'Unrecognized run directory'
    rows = readcsv(path)
    if not rows or not REQUIRED.issubset(rows[0]):
        return None, 'Missing required mode or residual_coeff column'
    spec = {}
    for row in rows:
        try:
            j = int(row['mode'])
            r = float(row['residual_coeff'])
            e = float(row['error_coeff']) if row.get('error_coeff','') != '' else None
            if j <= 0 or not math.isfinite(r) or (e is not None and not math.isfinite(e)):
                raise ValueError()
            if j in spec:
                return None, 'Duplicate Fourier mode'
            spec[j] = (r, e)
        except (ValueError, KeyError):
            return None, 'Invalid spectral coefficient'
    summary_path = path.parent/'summary.csv'
    summary = readcsv(summary_path)[0] if summary_path.is_file() and readcsv(summary_path) else {}
    try:
        epochs = int(summary.get('epochs', 50000))
        full_l2 = float(summary.get('final_l2_error','nan'))
    except ValueError:
        epochs, full_l2 = 50000, float('nan')
    return dict(k=ident[0], alpha=ident[1], loss=ident[2], seed=ident[3],
                epochs=epochs, spec=spec, full_l2=full_l2, path=path), None

def modal_norm(spec, modes, alpha, use_error=False):
    acc = 0.
    for j in modes:
        r,e = spec[j]
        coeff = e if use_error and e is not None else r / ((j*math.pi)**2 + alpha)
        acc += coeff**2
    return math.sqrt(0.5*acc)

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=ROOT)
    ap.add_argument('--output', type=Path)
    ap.add_argument('--max-mode', type=int, default=32)
    args = ap.parse_args()
    root = args.root.resolve()
    out = (args.output or root/'step5_counterfactual_results').resolve()
    if args.max_mode < 1:
        ap.error('--max-mode must be positive')
    candidates = []
    for dirname in ('step3_shifted_operator_results', 'step4_multiseed_results'):
        directory = root/dirname
        if directory.exists():
            candidates.extend(directory.rglob('final_spectrum.csv'))
    runs, skipped = {}, []
    for path in sorted(candidates):
        run, err = load_run(path)
        if err:
            skipped.append(dict(path=str(path), reason=err))
            continue
        key = (run['k'],run['alpha'],run['loss'],run['seed'],run['epochs'])
        # Prefer Step 4 copies when duplicate runs exist; numerical values should match.
        if key in runs:
            prev = runs[key]
            if any(abs(prev['spec'][j][0] - run['spec'][j][0]) > 1e-8
                   for j in prev['spec'].keys() & run['spec'].keys()):
                skipped.append(dict(path=str(path),reason='Duplicate configuration with differing spectra; review manually'))
                continue
        runs[key] = run
    if not runs:
        print('No final_spectrum.csv files found. Place Step 3/4 run folders under Numerics.')
        print('Expected e.g. Numerics/step4_multiseed_results/k8_alpha_p100d0_normalized_seed_2/final_spectrum.csv')
        return 1

    diag = []
    for r in sorted(runs.values(), key=lambda x:(x['k'],x['seed'],x['alpha'])):
        modes = sorted(j for j in r['spec'] if j<=args.max_mode)
        pred = modal_norm(r['spec'], modes, r['alpha'])
        measured = modal_norm(r['spec'], modes, r['alpha'], use_error=True) if all(r['spec'][j][1] is not None for j in modes) else float('nan')
        diag.append(dict(k=r['k'], alpha=r['alpha'], seed=r['seed'],loss=r['loss'],
                         epochs=r['epochs'], n_modes=len(modes), modal_l2_from_residual=pred,
                         modal_l2_from_error=measured, full_l2=r['full_l2'],
                         modal_identity_gap=abs(pred-measured), file=str(r['path'])))

    # Counterfactual: residual from alpha_source mapped through inverse of alpha_target.
    matrix = []
    comparisons = []
    groups = {}
    for r in runs.values():
        groups.setdefault((r['k'],r['seed'],r['loss'],r['epochs']),[]).append(r)
    for (k,seed,loss,epochs), group in sorted(groups.items()):
        group.sort(key=lambda r:r['alpha'])
        for source in group:
            for target in group:
                common = sorted(j for j in source['spec'] if j in target['spec'] and j<=args.max_mode)
                if not common:
                    continue
                val = modal_norm(source['spec'],common,target['alpha'])
                matrix.append(dict(k=k,seed=seed,loss=loss,epochs=epochs,
                                   alpha_residual_source=source['alpha'],alpha_inverse_target=target['alpha'],
                                   common_modes=len(common),counterfactual_modal_l2=val))
        # Ordered pair decomposition: total ratio = fixed residual (amplification) * residual redistribution.
        for a in group:
            for b in group:
                if a['alpha']>=b['alpha']:
                    continue
                common = sorted(j for j in a['spec'] if j in b['spec'] and j<=args.max_mode)
                if not common:
                    continue
                ea = modal_norm(a['spec'],common,a['alpha'])
                cf = modal_norm(a['spec'],common,b['alpha'])
                eb = modal_norm(b['spec'],common,b['alpha'])
                comparisons.append(dict(k=k,seed=seed,loss=loss,epochs=epochs,
                    alpha_a=a['alpha'],alpha_b=b['alpha'],common_modes=len(common),
                    actual_a_modal_l2=ea,counterfactual_a_residual_b_inverse=cf,
                    actual_b_modal_l2=eb,
                    amplification_only_ratio=cf/ea if ea else '',
                    residual_redistribution_ratio=eb/cf if cf else '',
                    total_ratio=eb/ea if ea else '',
                    identity_a_gap=abs(ea-modal_norm(a['spec'],common,a['alpha'],True)),
                    identity_b_gap=abs(eb-modal_norm(b['spec'],common,b['alpha'],True))))

    out.mkdir(parents=True, exist_ok=True)
    writecsv(out/'run_diagnostics.csv',diag,list(diag[0]))
    if matrix: writecsv(out/'counterfactual_matrix.csv',matrix,list(matrix[0]))
    if comparisons: writecsv(out/'amplification_vs_optimization.csv',comparisons,list(comparisons[0]))
    if skipped: writecsv(out/'skipped_files.csv',skipped,['path','reason'])
    print(f'Loaded {len(runs)} unique run spectra.')
    print(f'Produced {len(matrix)} counterfactuals and {len(comparisons)} paired decompositions.')
    print('Output:',out)
    print('NOTE: These are TRUNCATED modal norms, not full L2 norms.')
    print('NOTE: Changing alpha also changes training; the fixed-residual counterfactual isolates only inverse weighting.')
    if len(runs)<18:
        print(f'WARNING: Only {len(runs)} of the 18 planned spectra found. Copy remaining final_spectrum.csv files into their run folders.')
    return 0

if __name__=='__main__':
    raise SystemExit(main())
