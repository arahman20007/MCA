#!/usr/bin/env python3
"""Step 10a: validate variable-coefficient eigenbasis WITHOUT retraining.

Place next to step10_variable_coefficient_pinn.py in Numerics.
Run: python step10a_eigenbasis_convergence.py
Option: --grids 128 256 512 1024 --cases 16:0:1 8:100:1

Requires PyTorch; imports the original Step 10 script to preserve the exact
architecture, PDE, and forcing. Reads saved final.pt/best.pt checkpoints.
Writes ONLY to step10a_eigenbasis_convergence_results.
"""
import argparse,csv,importlib.util,math
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parent

def load_step10():
    path=ROOT/'step10_variable_coefficient_pinn.py'
    if not path.exists(): raise FileNotFoundError(f'Missing original Step 10 script: {path}')
    spec=importlib.util.spec_from_file_location('step10_original',path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod

def read_summary(path):
    with path.open(newline='',encoding='utf-8-sig') as f:
        return next(csv.DictReader(f))

def write_csv(path,rows):
    if not rows:return
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--grids',nargs='+',type=int,default=[128,256,512,1024])
    p.add_argument('--cases',nargs='+',default=['16:0:1','8:100:1'],help='k:alpha:seed')
    p.add_argument('--checkpoints',nargs='+',choices=['final','best'],default=['final','best'])
    p.add_argument('--modes',type=int,default=32)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--input',type=Path,default=ROOT/'step10_variable_coefficient_results')
    p.add_argument('--output',type=Path,default=ROOT/'step10a_eigenbasis_convergence_results')
    args=p.parse_args()
    if any(n<args.modes or n<2 for n in args.grids):p.error('Every grid size must be >= modes and >= 2')
    if args.device=='cuda' and not torch.cuda.is_available():p.error('CUDA not available')
    step10=load_step10();device=torch.device(args.device)
    allrows=[];moderows=[]
    for label in args.cases:
        try:k_s,a_s,seed_s=label.split(':');k=int(k_s);alpha=float(a_s);seed=int(seed_s)
        except ValueError:p.error(f'Invalid case {label!r}; expected k:alpha:seed')
        folder=args.input/f'k{k}_alpha_{alpha:g}_seed_{seed}'
        if not folder.exists():
            print('MISSING RUN FOLDER:',folder);continue
        summary=read_summary(folder/'summary.csv')
        net=step10.Net(int(summary['width']),int(summary['depth'])).to(device).double()
        for ck in args.checkpoints:
            file=folder/(ck+'.pt')
            if not file.exists():print('MISSING CHECKPOINT:',file);continue
            state=torch.load(file,map_location=device,weights_only=True)
            net.load_state_dict(state);net.eval()
            print(f'Case k={k}, alpha={alpha:g}, seed={seed}, checkpoint={ck}',flush=True)
            for n in args.grids:
                vals,vecs,h=step10.discrete_eigensystem(n,alpha,device)
                jmax=min(args.modes,n)
                x=(torch.arange(1,n+1,device=device,dtype=torch.float64)/(n+1)).reshape(-1,1)
                with torch.enable_grad():rc,uc=step10.residual(net,x,k,alpha)
                err=(uc.detach().flatten()-torch.sin(k*math.pi*x).flatten())
                rv=rc.detach().flatten()
                phi=vecs[:,:jmax]
                ec=h*(phi.T@err); rcont=h*(phi.T@rv)
                # Discrete residual computed from the SAME discrete operator as its eigenbasis.
                # This is an algebraic identity up to floating-point rounding.
                v=err
                a_faces=step10.coefficient((torch.arange(n+1,dtype=torch.float64,device=device)+0.5)*h)
                left=torch.cat((torch.zeros(1,dtype=torch.float64,device=device),v[:-1]))
                right=torch.cat((v[1:],torch.zeros(1,dtype=torch.float64,device=device)))
                rdisc=((a_faces[:-1]+a_faces[1:])*v-a_faces[:-1]*left-a_faces[1:]*right)/(h*h)+alpha*v
                rd=h*(phi.T@rdisc)
                predcont=rcont/vals[:jmax];preddisc=rd/vals[:jmax]
                diffcont=ec-predcont;diffdisc=ec-preddisc
                norm=lambda z:math.sqrt(float(torch.sum(z*z)))
                total=math.sqrt(h*float(torch.sum(err*err)))
                row=dict(k=k,alpha=alpha,seed=seed,checkpoint=ck,grid=n,modes=jmax,
                    l2_error_grid=total,modal_l2=norm(ec),unresolved_l2=math.sqrt(max(0,total**2-norm(ec)**2)),
                    continuous_identity_defect=norm(diffcont),discrete_identity_defect=norm(diffdisc),
                    continuous_identity_relative=norm(diffcont)/max(norm(ec),1e-14),
                    max_mode_defect_continuous=float(torch.max(torch.abs(diffcont))),
                    lambda_1=float(vals[0]),lambda_last=float(vals[jmax-1]))
                allrows.append(row)
                for j in range(jmax):
                    moderows.append(dict(k=k,alpha=alpha,seed=seed,checkpoint=ck,grid=n,mode=j+1,
                        eigenvalue=float(vals[j]),error_coefficient=float(ec[j]),
                        predicted_continuous=float(predcont[j]),predicted_discrete=float(preddisc[j]),
                        continuous_defect=float(diffcont[j]),discrete_defect=float(diffdisc[j])))
                print(f'  N={n:4d} continuous defect={row["continuous_identity_defect"]:.6g}, '
                      f'discrete defect={row["discrete_identity_defect"]:.3g}, '
                      f'unresolved L2={row["unresolved_l2"]:.6g}',flush=True)
                write_csv(args.output/'grid_convergence_summary.csv',allrows)
                write_csv(args.output/'mode_convergence_details.csv',moderows)
    if not allrows:raise SystemExit('No checkpoints found. Check --input path and run folders.')
    print('Saved results to:',args.output)
    print('Interpretation: discrete identity should be near machine precision; continuous identity should converge with N.')
    print('If continuous defect does not decrease, inspect continuous vs discrete residual consistency and resolution.')
if __name__=='__main__':main()
