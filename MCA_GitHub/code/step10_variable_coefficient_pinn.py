#!/usr/bin/env python3
"""Step 10: variable-coefficient elliptic PINN and operator-eigenmode diagnostics.

Run from Numerics:
  python step10_variable_coefficient_pinn.py --smoke-test
  python step10_variable_coefficient_pinn.py --epochs 50000 --seeds 1 2 3 --ks 8 16 --alphas 0 100 1000

Only requires PyTorch and Python standard library. Results in a NEW
step10_variable_coefficient_results folder. No changes to Steps 1-9.

Operator: L u=-(a u')'+alpha u; a=1+0.5 sin(2*pi*x).
Exact solution sin(k*pi*x), hard homogeneous Dirichlet boundaries.
Discrete operator eigenbasis is computed using symmetric tridiagonal
flux-form finite differences and torch.linalg.eigh. Diagnostics are
approximate discrete projections; convergence checks are required.
"""
import argparse, csv, math, random, time
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parent
PI=math.pi

def save_csv(path, rows):
    if not rows: return
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def coefficient(x): return 1.0+0.5*torch.sin(2*PI*x)
def coefficient_prime(x): return PI*torch.cos(2*PI*x)
def forcing(x,k,alpha):
    q=k*PI
    return (coefficient(x)*q*q+alpha)*torch.sin(q*x)-coefficient_prime(x)*q*torch.cos(q*x)

def discrete_eigensystem(n,alpha,device):
    # n interior points, spacing h=1/(n+1). SPD flux-form discretization.
    # eigenvectors are orthonormal in discrete h-weighted inner product.
    h=1.0/(n+1)
    faces=(torch.arange(n+1,dtype=torch.float64,device=device)+0.5)*h
    a=coefficient(faces)
    diagonal=(a[:-1]+a[1:])/(h*h)+alpha
    off=-a[1:-1]/(h*h)
    M=torch.diag(diagonal)+torch.diag(off,diagonal=1)+torch.diag(off,diagonal=-1)
    vals,vecs=torch.linalg.eigh(M)
    # columns normalized so h*sum phi_j**2 =1
    return vals,vecs/math.sqrt(h),h

class Net(torch.nn.Module):
    def __init__(self,width=64,depth=4):
        super().__init__()
        layers=[];d=1
        for _ in range(depth):
            layers.extend([torch.nn.Linear(d,width),torch.nn.Tanh()]);d=width
        layers.append(torch.nn.Linear(d,1))
        self.net=torch.nn.Sequential(*layers)
    def forward(self,x):return x*(1-x)*self.net(x)

def residual(net,x,k,alpha):
    x=x.detach().clone().requires_grad_(True)
    u=net(x)
    ux=torch.autograd.grad(u,x,torch.ones_like(u),create_graph=True)[0]
    uxx=torch.autograd.grad(ux,x,torch.ones_like(ux),create_graph=True)[0]
    r=-coefficient(x)*uxx-coefficient_prime(x)*ux+alpha*u-forcing(x,k,alpha)
    return r,u

def diagnostics(net,k,alpha,eigvals,eigvecs,h,device,jmax):
    n=eigvecs.shape[0]
    x=(torch.arange(1,n+1,dtype=torch.float64,device=device)/(n+1)).reshape(-1,1)
    with torch.enable_grad():
        r,u=residual(net,x,k,alpha)
    exact=torch.sin(k*PI*x)
    err=(u-exact).detach().flatten();rv=r.detach().flatten()
    phi=eigvecs[:,:jmax]
    ec=h*(phi.T@err);rc=h*(phi.T@rv)
    predicted=rc/eigvals[:jmax]
    # The continuous residual and discrete eigenbasis do not obey an exact
    # modal identity at finite grid resolution: report discrepancy explicitly.
    return {
      'l2_error_grid':math.sqrt(h*float(torch.sum(err*err))),
      'l2_residual_grid':math.sqrt(h*float(torch.sum(rv*rv))),
      'l2_modal_error':math.sqrt(float(torch.sum(ec*ec))),
      'l2_modal_predicted':math.sqrt(float(torch.sum(predicted*predicted))),
      'modal_identity_defect':math.sqrt(float(torch.sum((ec-predicted)**2))),
      'mode_rows':[
        dict(mode=j+1,eigenvalue=float(eigvals[j]),inverse_amplification=float(1/eigvals[j]),
             error_coeff=float(ec[j]),residual_coeff=float(rc[j]),
             predicted_error_coeff=float(predicted[j]),
             identity_defect=float(ec[j]-predicted[j]))
        for j in range(jmax)]}

def train(args,k,alpha,seed,device,output):
    torch.manual_seed(seed);random.seed(seed)
    net=Net(args.width,args.depth).to(device).double()
    opt=torch.optim.Adam(net.parameters(),lr=args.lr)
    x=((torch.arange(args.collocation,dtype=torch.float64,device=device)+0.5)/args.collocation).reshape(-1,1)
    scale=max(1.0,(k*PI)**2+alpha)
    vals,vecs,h=discrete_eigensystem(args.eigen_grid,alpha,device)
    run=output/f'k{k}_alpha_{alpha:g}_seed_{seed}'
    run.mkdir(parents=True,exist_ok=True)
    history=[];start=time.time();best=float('inf');best_epoch=-1
    checkpoints=set([0,args.epochs]);checkpoints.update(range(args.record_every,args.epochs+1,args.record_every))
    for epoch in range(args.epochs+1):
        if epoch in checkpoints:
            d=diagnostics(net,k,alpha,vals,vecs,h,device,args.modes)
            history.append(dict(epoch=epoch,loss_normalized=float('nan') if epoch==0 else last_loss,
                                l2_error_grid=d['l2_error_grid'],l2_residual_grid=d['l2_residual_grid'],
                                l2_modal_error=d['l2_modal_error'],l2_modal_predicted=d['l2_modal_predicted'],
                                modal_identity_defect=d['modal_identity_defect']))
            for row in d['mode_rows']:row.update(epoch=epoch)
            save_csv(run/f'eigenmodes_epoch_{epoch:06d}.csv',d['mode_rows'])
            if d['l2_error_grid']<best:
                best=d['l2_error_grid'];best_epoch=epoch
                torch.save(net.state_dict(),run/'best.pt')
            print(f'k={k} alpha={alpha:g} seed={seed} epoch={epoch} L2={d["l2_error_grid"]:.5g} '
                  f'defect={d["modal_identity_defect"]:.3g}',flush=True)
        if epoch==args.epochs:break
        opt.zero_grad(set_to_none=True)
        r,_=residual(net,x,k,alpha)
        loss=torch.mean((r/scale)**2)
        loss.backward();opt.step()
        last_loss=float(loss.detach())
    save_csv(run/'training_history.csv',history)
    summary=dict(k=k,alpha=alpha,seed=seed,epochs=args.epochs,collocation=args.collocation,
        eigen_grid=args.eigen_grid,modes=args.modes,width=args.width,depth=args.depth,lr=args.lr,
        best_l2_error=best,best_epoch=best_epoch,final_l2_error=history[-1]['l2_error_grid'],
        final_modal_identity_defect=history[-1]['modal_identity_defect'],elapsed_seconds=time.time()-start)
    save_csv(run/'summary.csv',[summary]);torch.save(net.state_dict(),run/'final.pt')
    return summary

def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--smoke-test',action='store_true',help='Quick plumbing test, not scientific evidence')
    p.add_argument('--ks',type=int,nargs='+',default=[8,16]);p.add_argument('--alphas',type=float,nargs='+',default=[0,100,1000]);p.add_argument('--seeds',type=int,nargs='+',default=[1,2,3])
    p.add_argument('--epochs',type=int,default=50000);p.add_argument('--record-every',type=int,default=1000)
    p.add_argument('--collocation',type=int,default=256);p.add_argument('--eigen-grid',type=int,default=256)
    p.add_argument('--modes',type=int,default=32);p.add_argument('--width',type=int,default=64)
    p.add_argument('--depth',type=int,default=4);p.add_argument('--lr',type=float,default=0.001)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--output',type=Path,default=ROOT/'step10_variable_coefficient_results')
    args=p.parse_args()
    if args.smoke_test:
        args.ks=[8];args.alphas=[0.0];args.seeds=[1];args.epochs=10;args.record_every=5
        args.collocation=32;args.eigen_grid=48;args.modes=12;args.width=16;args.depth=2
        args.output=ROOT/'step10_variable_coefficient_smoke_test'
    if args.modes>args.eigen_grid or min(args.collocation,args.eigen_grid,args.modes)<1:
        p.error('Require positive sizes and modes <= eigen-grid')
    if args.epochs<1 or args.record_every<1:p.error('epochs and record-every must be positive')
    device=torch.device(args.device)
    if args.device=='cuda' and not torch.cuda.is_available():p.error('CUDA not available')
    args.output.mkdir(parents=True,exist_ok=True)
    all_summaries=[]
    for k in args.ks:
        for alpha in args.alphas:
            for seed in args.seeds:
                all_summaries.append(train(args,k,alpha,seed,device,args.output))
                save_csv(args.output/'all_run_summaries.csv',all_summaries)
    print('Finished',len(all_summaries),'runs. Results:',args.output)
    print('IMPORTANT: check modal_identity_defect convergence by increasing --eigen-grid.')
if __name__=='__main__':main()
