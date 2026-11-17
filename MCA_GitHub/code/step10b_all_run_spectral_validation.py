#!/usr/bin/env python3
"""Step 10b: high-resolution eigenmode validation for all saved Step 10 runs.

Place beside step10_variable_coefficient_pinn.py in Numerics.
Run: python step10b_all_run_spectral_validation.py

No retraining. Loads best.pt and final.pt from all 18 run folders, constructs
N=1024 discrete flux-form eigenbasis, compares 32/64/128 modal projections,
and writes only into step10b_spectral_validation_results.

Continuous modal identity is approximate due to finite differences/quadrature.
Discrete modal identity is algebraic; its small defect is NOT independent
validation of the continuous PDE residual.
"""
import argparse
import csv
import importlib.util
import math
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent

def load_original():
    p = ROOT / 'step10_variable_coefficient_pinn.py'
    if not p.exists():
        raise FileNotFoundError(f'Missing original Step 10 script: {p}')
    spec = importlib.util.spec_from_file_location('step10_original', p)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def read_summary(path):
    with path.open(newline='', encoding='utf-8-sig') as f:
        return next(csv.DictReader(f))

def write_csv(path, rows):
    if not rows: return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def l2(v):
    return math.sqrt(float(torch.sum(v*v)))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT/'step10_variable_coefficient_results')
    parser.add_argument('--output', type=Path, default=ROOT/'step10b_spectral_validation_results')
    parser.add_argument('--grid', type=int, default=1024)
    parser.add_argument('--modes', nargs='+', type=int, default=[32,64,128])
    parser.add_argument('--ks', nargs='+', type=int, default=[8,16])
    parser.add_argument('--alphas', nargs='+', type=float, default=[0,100,1000])
    parser.add_argument('--seeds', nargs='+', type=int, default=[1,2,3])
    parser.add_argument('--checkpoints', nargs='+', choices=['best','final'], default=['best','final'])
    parser.add_argument('--device', choices=['cpu','cuda'], default='cpu')
    args = parser.parse_args()
    if args.grid < max(args.modes) or min(args.modes) < 1:
        parser.error('Grid must be >= largest mode count, and modes must be positive')
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA unavailable')
    torch.set_default_dtype(torch.float64)
    original = load_original()
    device = torch.device(args.device)
    n = args.grid
    h = 1.0/(n+1)
    x = (torch.arange(1,n+1,device=device,dtype=torch.float64)/(n+1)).reshape(-1,1)
    exact_cache = {k:torch.sin(k*math.pi*x).flatten() for k in args.ks}
    faces = (torch.arange(n+1,device=device,dtype=torch.float64)+0.5)*h
    af = original.coefficient(faces)
    summary_rows, mode_rows, skipped = [], [], []
    eig_cache = {}
    for alpha in args.alphas:
        print(f'Computing eigenbasis: N={n}, alpha={alpha:g} ...', flush=True)
        eig_cache[alpha] = original.discrete_eigensystem(n,alpha,device)[:2]
    for k in args.ks:
        for alpha in args.alphas:
            vals, vecs = eig_cache[alpha]
            phi = vecs[:,:max(args.modes)]
            for seed in args.seeds:
                folder = args.input/f'k{k}_alpha_{alpha:g}_seed_{seed}'
                summary_file = folder/'summary.csv'
                if not summary_file.is_file():
                    skipped.append(dict(k=k,alpha=alpha,seed=seed,checkpoint='',reason='Missing summary.csv',path=str(folder)))
                    continue
                try:
                    metadata = read_summary(summary_file)
                    net = original.Net(int(metadata['width']),int(metadata['depth'])).to(device).double()
                except Exception as exc:
                    skipped.append(dict(k=k,alpha=alpha,seed=seed,checkpoint='',reason=f'Could not construct network: {exc}',path=str(folder)))
                    continue
                for checkpoint in args.checkpoints:
                    path = folder/f'{checkpoint}.pt'
                    if not path.is_file():
                        skipped.append(dict(k=k,alpha=alpha,seed=seed,checkpoint=checkpoint,reason='Missing checkpoint',path=str(path)))
                        continue
                    try:
                        state = torch.load(path,map_location=device,weights_only=True)
                        net.load_state_dict(state)
                        net.eval()
                        with torch.enable_grad():
                            residual, output = original.residual(net,x,k,alpha)
                        error = output.detach().flatten()-exact_cache[k]
                        rc = residual.detach().flatten()
                        ec = h*(phi.T@error)
                        rcont = h*(phi.T@rc)
                        # Construct the discrete operator applied to the sampled ERROR.
                        left = torch.cat((torch.zeros(1,device=device),error[:-1]))
                        right = torch.cat((error[1:],torch.zeros(1,device=device)))
                        rdisc = ((af[:-1]+af[1:])*error-af[:-1]*left-af[1:]*right)/(h*h)+alpha*error
                        rd = h*(phi.T@rdisc)
                        pred_cont = rcont/vals[:phi.shape[1]]
                        pred_disc = rd/vals[:phi.shape[1]]
                        full_error = math.sqrt(h*float(torch.sum(error**2)))
                        for J in sorted(set(args.modes)):
                            observed = ec[:J]
                            cont = pred_cont[:J]
                            disc = pred_disc[:J]
                            captured = l2(observed)
                            unresolved = math.sqrt(max(0.,full_error**2-captured**2))
                            defect_cont = l2(observed-cont)
                            defect_disc = l2(observed-disc)
                            unresolved_fraction = (unresolved/full_error)**2 if full_error>0 else 0.
                            summary_rows.append(dict(k=k,alpha=alpha,seed=seed,checkpoint=checkpoint,
                                grid=n,modes=J,full_l2_error=full_error,modal_l2_error=captured,
                                unresolved_l2_error=unresolved,unresolved_energy_fraction=unresolved_fraction,
                                continuous_identity_defect=defect_cont,
                                continuous_identity_relative=defect_cont/max(captured,1e-14),
                                discrete_identity_defect=defect_disc,
                                max_continuous_mode_defect=float(torch.max(torch.abs(observed-cont))),
                                lambda_1=float(vals[0]),lambda_J=float(vals[J-1]),
                                flag_identity_relative_gt_1pct=int(defect_cont>0.01*max(captured,1e-14)),
                                flag_unresolved_energy_gt_1pct=int(unresolved_fraction>0.01)))
                        for j in range(phi.shape[1]):
                            mode_rows.append(dict(k=k,alpha=alpha,seed=seed,checkpoint=checkpoint,
                                grid=n,mode=j+1,eigenvalue=float(vals[j]),
                                error_coefficient=float(ec[j]),continuous_residual_coefficient=float(rcont[j]),
                                predicted_continuous=float(pred_cont[j]),predicted_discrete=float(pred_disc[j]),
                                continuous_defect=float(ec[j]-pred_cont[j]),
                                discrete_defect=float(ec[j]-pred_disc[j])))
                        print(f'k={k} alpha={alpha:g} seed={seed} {checkpoint}: '
                              f'L2={full_error:.5g}; J=128 defect={defect_cont:.3g}; '
                              f'unresolved={unresolved:.3g}',flush=True)
                    except Exception as exc:
                        skipped.append(dict(k=k,alpha=alpha,seed=seed,checkpoint=checkpoint,
                            reason=f'Evaluation failed: {type(exc).__name__}: {exc}',path=str(path)))
                # Save incrementally in case long-running process is interrupted.
                write_csv(args.output/'validation_summary.csv',summary_rows)
                write_csv(args.output/'mode_validation_details.csv',mode_rows)
                write_csv(args.output/'skipped_checkpoints.csv',skipped)
    print(f'Finished: {len(summary_rows)} validation rows, {len(skipped)} skipped entries.')
    print('Output:',args.output.resolve())
    print('WARNING: A single grid is not a grid-convergence study. Step 10a provides convergence evidence for two cases only.')
    print('WARNING: Best checkpoints selected using exact-solution L2; report as oracle diagnostics, not deployable model selection.')

if __name__=='__main__':
    main()
