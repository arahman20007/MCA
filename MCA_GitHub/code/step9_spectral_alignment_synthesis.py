#!/usr/bin/env python3
"""Step 9: spectral alignment synthesis, no retraining.

Save this script in Numerics and run:
    python step9_spectral_alignment_synthesis.py

Reads step8_spectral_persistence_results/mode_persistence_metrics.csv.
Writes CSV tables and PNG figures to step9_spectral_alignment_results.
Requires matplotlib for figures; the CSV analyses still work without it.
"""
import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

HERE = Path(__file__).resolve().parent

def read_rows(path):
    with path.open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))

def write_rows(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)

def safe_ratio(a,b):
    return a/b if b>0 else float('nan')

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=HERE)
    ap.add_argument('--input',type=Path)
    ap.add_argument('--output',type=Path)
    args=ap.parse_args()
    root=args.root.resolve()
    inp=(args.input or root/'step8_spectral_persistence_results'/'mode_persistence_metrics.csv').resolve()
    out=(args.output or root/'step9_spectral_alignment_results').resolve()
    if not inp.exists():
        print('Missing:',inp)
        print('Copy the Step 8 mode_persistence_metrics.csv into the expected Step 8 folder, or use --input PATH')
        return 1
    raw=read_rows(inp)
    required={'k','alpha','seed','mode','mean_residual_energy','mean_solution_error_energy','inverse_amplification_squared'}
    if not raw or not required.issubset(raw[0]):
        print('Input CSV missing required columns:',required-set(raw[0] if raw else []))
        return 1
    groups=defaultdict(dict)
    for r in raw:
        try:
            k=int(r['k']); alpha=float(r['alpha']); seed=int(r['seed']); j=int(r['mode'])
            loss=r.get('loss','normalized')
            R=float(r['mean_residual_energy']); P=float(r['mean_solution_error_energy'])
            W=float(r['inverse_amplification_squared'])
            if j<1 or min(R,P,W)<0 or not all(map(math.isfinite,[R,P,W])): raise ValueError()
            key=(k,alpha,loss,seed)
            if j in groups[key]: raise ValueError('duplicate mode')
            groups[key][j]=(R,P,W)
        except (ValueError,TypeError) as exc:
            print('Invalid input row:',r,'error:',exc)
            return 1
    rows=[]; spectra=[]
    for (k,alpha,loss,seed), modes in sorted(groups.items()):
        J=len(modes)
        rsum=sum(v[0] for v in modes.values())
        wsum=sum(v[2] for v in modes.values())
        psum=sum(v[1] for v in modes.values())
        predicted=sum(v[0]*v[2] for v in modes.values())
        if not rsum or not wsum:
            print('Skipping zero residual or amplification:',k,alpha,seed)
            continue
        q={j:v[0]/rsum for j,v in modes.items()}
        w={j:v[2]/wsum for j,v in modes.items()}
        alignment=J*sum(q[j]*w[j] for j in modes)
        cosine=sum(q[j]*w[j] for j in modes)/math.sqrt(sum(x*x for x in q.values())*sum(x*x for x in w.values()))
        low=[j for j in modes if j<=4]
        dominant=max(modes,key=lambda j:modes[j][1])
        rows.append(dict(k=k,alpha=alpha,loss=loss,seed=seed,n_modes=J,
            alignment_vs_uniform=alignment,cosine_alignment=cosine,
            total_residual_persistence=rsum,total_solution_persistence=psum,
            predicted_solution_persistence=predicted,
            absolute_identity_gap=abs(psum-predicted),
            effective_amplification_squared=predicted/rsum,
            uniform_residual_baseline=wsum/J,
            low_mode_residual_share=sum(q[j] for j in low),
            low_mode_solution_share=safe_ratio(sum(modes[j][1] for j in low),psum),
            target_residual_share=q.get(k,0),
            target_solution_share=safe_ratio(modes[k][1],psum) if k in modes else float('nan'),
            dominant_solution_error_mode=dominant))
        for j,(R,P,W) in sorted(modes.items()):
            spectra.append(dict(k=k,alpha=alpha,loss=loss,seed=seed,mode=j,
                residual_persistence=R,solution_persistence=P,
                inverse_amplification_squared=W,residual_share=q[j],
                amplification_weight_share=w[j],solution_error_share=safe_ratio(P,psum),
                amplified_residual_energy=R*W))
    if not rows:
        print('No valid runs.')
        return 1
    out.mkdir(parents=True,exist_ok=True)
    write_rows(out/'run_alignment_indices.csv',rows,list(rows[0]))
    write_rows(out/'mode_alignment_profiles.csv',spectra,list(spectra[0]))
    configs=defaultdict(list)
    for r in rows: configs[(r['k'],r['alpha'],r['loss'])].append(r)
    fields=['k','alpha','loss','n_seeds']
    metrics=['alignment_vs_uniform','cosine_alignment','total_residual_persistence',
             'total_solution_persistence','effective_amplification_squared',
             'low_mode_residual_share','low_mode_solution_share',
             'target_residual_share','target_solution_share']
    fields += [part+'_'+m for m in metrics for part in ('mean','sd')]
    summaries=[]
    for (k,alpha,loss),g in sorted(configs.items()):
        s=dict(k=k,alpha=alpha,loss=loss,n_seeds=len(g))
        for m in metrics:
            values=[r[m] for r in g if math.isfinite(r[m])]
            s['mean_'+m]=mean(values) if values else ''
            s['sd_'+m]=stdev(values) if len(values)>1 else ''
        summaries.append(s)
    write_rows(out/'configuration_alignment_summary.csv',summaries,fields)
    # Counterfactual using time-integrated residual spectrum held fixed.
    cf=[]
    byseed=defaultdict(list)
    for key,modes in groups.items():
        byseed[(key[0],key[2],key[3])].append((key[1],modes))
    for (k,loss,seed),group in sorted(byseed.items()):
        group.sort(key=lambda x:x[0])
        for alpha_a,source in group:
            for alpha_b,target in group:
                common=sorted(set(source)&set(target))
                if not common: continue
                ea=sum(source[j][0]/((j*math.pi)**2+alpha_a)**2 for j in common)
                cf_b=sum(source[j][0]/((j*math.pi)**2+alpha_b)**2 for j in common)
                eb=sum(target[j][0]/((j*math.pi)**2+alpha_b)**2 for j in common)
                cf.append(dict(k=k,seed=seed,loss=loss,alpha_residual_source=alpha_a,
                    alpha_inverse_target=alpha_b,n_common_modes=len(common),
                    source_time_averaged_squared_error=ea,
                    counterfactual_time_averaged_squared_error=cf_b,
                    target_time_averaged_squared_error=eb,
                    amplification_only_ratio=safe_ratio(cf_b,ea),
                    residual_redistribution_ratio=safe_ratio(eb,cf_b),
                    total_ratio=safe_ratio(eb,ea)))
    if cf: write_rows(out/'persistence_counterfactual_matrix.csv',cf,list(cf[0]))
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('Matplotlib not available; CSV files generated without PNG figures.')
    else:
        # Figure 1: residual and solution persistence shares by mode (per k,alpha; mean of seed shares).
        for k in sorted(set(r['k'] for r in rows)):
            fig,axes=plt.subplots(1,2,figsize=(12,4.5),sharex=True)
            for alpha in sorted(set(r['alpha'] for r in rows if r['k']==k)):
                selected=[r for r in spectra if r['k']==k and r['alpha']==alpha]
                bymode=defaultdict(list)
                for r in selected: bymode[r['mode']].append(r)
                modes=sorted(bymode)
                for ax,col,label in ((axes[0],'residual_share','Residual persistence share'),
                                     (axes[1],'solution_error_share','Solution-error persistence share')):
                    ax.plot(modes,[mean(z[col] for z in bymode[j]) for j in modes],marker='.',label=f'alpha={alpha:g}')
                    ax.set_title(label); ax.set_xlabel('Fourier mode j');ax.set_yscale('symlog',linthresh=1e-6)
                    ax.grid(True,alpha=.2)
            axes[0].set_ylabel('Mean share across seeds (symlog)')
            for ax in axes: ax.legend()
            fig.suptitle(f'k={k}: spectral persistence profiles')
            fig.tight_layout();fig.savefig(out/f'figure1_persistence_spectra_k{k}.png',dpi=200)
            plt.close(fig)
        # Figure 2: amplification profile
        fig,ax=plt.subplots(figsize=(8,5))
        for alpha in sorted(set(r['alpha'] for r in rows)):
            j=range(1,max(r['n_modes'] for r in rows)+1)
            ax.plot(list(j),[1/((m*math.pi)**2+alpha)**2 for m in j],label=f'alpha={alpha:g}')
        ax.set_yscale('log');ax.set_xlabel('Fourier mode j');ax.set_ylabel('Inverse amplification squared')
        ax.set_title('Operator spectral amplification profiles');ax.legend();ax.grid(True,alpha=.2)
        fig.tight_layout();fig.savefig(out/'figure2_operator_amplification.png',dpi=200);plt.close(fig)
        # Figure 3: alignment by seed and configuration
        fig,ax=plt.subplots(figsize=(9,5))
        for k in sorted(set(r['k'] for r in rows)):
            for alpha in sorted(set(r['alpha'] for r in rows if r['k']==k)):
                subset=[r for r in rows if r['k']==k and r['alpha']==alpha]
                ax.scatter([f'k{k} / a{alpha:g}']*len(subset),[r['alignment_vs_uniform'] for r in subset],label=f'k={k}, a={alpha:g}')
        ax.axhline(1,linestyle='--',linewidth=1,label='Uniform-residual reference')
        ax.set_ylabel('Spectral alignment index (uniform reference = 1)')
        ax.set_title('Residual-amplification alignment across seeds');ax.tick_params(axis='x',rotation=25)
        ax.grid(True,axis='y',alpha=.2);fig.tight_layout()
        fig.savefig(out/'figure3_alignment_across_seeds.png',dpi=200);plt.close(fig)
    print('Processed',len(rows),'runs and',len(spectra),'modal records.')
    print('Saved output to:',out)
    print('CAUTION: alignment is descriptive and dependent on the retained Fourier modes; it is not proof of a causal spectral-bias mechanism.')
    return 0

if __name__=='__main__':
    raise SystemExit(main())
