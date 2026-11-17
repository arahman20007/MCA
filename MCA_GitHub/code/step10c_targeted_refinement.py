#!/usr/bin/env python3
"""Step 10c: targeted grid and modal-truncation convergence, without retraining.

Requires the original step10_variable_coefficient_pinn.py and saved checkpoints.
Run: python step10c_targeted_refinement.py
Optional: --grids 1024 1536 2048 --modes 128 192 256

Uses SciPy's symmetric tridiagonal eigensolver when available (recommended).
Falls back to PyTorch's dense symmetric eigensolver, which can be slow at N=2048.
"""
import argparse
import csv
import importlib.util
import math
from pathlib import Path
import time
import torch

ROOT = Path(__file__).resolve().parent
CASES = [(16, 100.0, 3, 'best'), (8, 1000.0, 3, 'best')]

def load_original():
    p = ROOT / 'step10_variable_coefficient_pinn.py'
    if not p.is_file():
        raise FileNotFoundError(f'Original Step 10 Python file not found: {p}')
    spec = importlib.util.spec_from_file_location('step10_original', p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def save_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

def eigensystem(mod, n, alpha, jmax, backend):
    h = 1.0 / (n + 1)
    faces = (torch.arange(n + 1, dtype=torch.float64) + 0.5) * h
    a = mod.coefficient(faces)
    diag = (a[:-1] + a[1:]) / (h * h) + alpha
    off = -a[1:-1] / (h * h)
    if backend in ('auto', 'scipy'):
        try:
            from scipy.linalg import eigh_tridiagonal
            values, vectors = eigh_tridiagonal(diag.numpy(), off.numpy(), select='i',
                                                 select_range=(0, jmax-1), check_finite=False)
            return torch.from_numpy(values.copy()), torch.from_numpy(vectors.copy()) / math.sqrt(h), h, 'scipy'
        except ImportError:
            if backend == 'scipy':
                raise RuntimeError('SciPy unavailable. Install scipy or use --backend torch.')
    matrix = torch.diag(diag) + torch.diag(off, 1) + torch.diag(off, -1)
    values, vectors = torch.linalg.eigh(matrix)
    return values[:jmax], vectors[:, :jmax] / math.sqrt(h), h, 'torch'

def evaluate(mod, model, k, alpha, n, values, phi, h, modes):
    x = (torch.arange(1, n + 1, dtype=torch.float64) / (n + 1)).reshape(-1, 1)
    with torch.enable_grad():
        residual, u = mod.residual(model, x, k, alpha)
    error = u.detach().flatten() - torch.sin(k * math.pi * x.flatten())
    residual = residual.detach().flatten()
    e_coeff = h * (phi.T @ error)
    r_coeff = h * (phi.T @ residual)
    pred = r_coeff / values
    face_x = (torch.arange(n + 1, dtype=torch.float64) + 0.5) * h
    a = mod.coefficient(face_x)
    left = torch.cat([torch.zeros(1, dtype=torch.float64), error[:-1]])
    right = torch.cat([error[1:], torch.zeros(1, dtype=torch.float64)])
    discrete_r = ((a[:-1] + a[1:]) * error - a[:-1] * left - a[1:] * right) / h**2 + alpha * error
    pred_discrete = (h * (phi.T @ discrete_r)) / values
    full_sq = h * float(error @ error)
    records = []
    for J in modes:
        observed = e_coeff[:J]
        predicted = pred[:J]
        modal_sq = float(observed @ observed)
        defect = float(torch.linalg.vector_norm(observed-predicted))
        captured = math.sqrt(max(0., modal_sq))
        unresolved_fraction = max(0., full_sq - modal_sq) / max(full_sq, 1e-30)
        records.append(dict(grid=n, modes=J, full_l2_error=math.sqrt(full_sq),
                            captured_l2_error=captured,
                            unresolved_l2_error=math.sqrt(max(0., full_sq-modal_sq)),
                            unresolved_energy_fraction=unresolved_fraction,
                            continuous_identity_defect=defect,
                            continuous_identity_relative=defect/max(captured, 1e-14),
                            discrete_identity_defect=float(torch.linalg.vector_norm(observed-pred_discrete[:J])),
                            lambda_1=float(values[0]), lambda_J=float(values[J-1]),
                            identity_pass_1pct=int(defect <= .01 * max(captured,1e-14)),
                            unresolved_pass_1pct=int(unresolved_fraction <= .01)))
    return records, e_coeff, pred, values

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input', type=Path, default=ROOT/'step10_variable_coefficient_results')
    ap.add_argument('--output', type=Path, default=ROOT/'step10c_targeted_refinement_results')
    ap.add_argument('--grids', type=int, nargs='+', default=[1024,1536,2048])
    ap.add_argument('--modes', type=int, nargs='+', default=[128,192,256])
    ap.add_argument('--backend', choices=['auto','scipy','torch'], default='auto')
    ap.add_argument('--smoke-test', action='store_true', help='Small analytic test without checkpoints')
    args = ap.parse_args()
    torch.set_default_dtype(torch.float64)
    mod = load_original()
    modes = sorted(set(args.modes))
    grids = sorted(set(args.grids))
    if min(grids) < max(modes) or min(modes) < 1:
        ap.error('Every grid must have at least as many interior nodes as retained modes.')
    if args.smoke_test:
        # Analytic error: model returns zero, so e=-sin(k*pi*x); forcing is exact.
        # No trained checkpoint needed, tests numerical spectral pipeline.
        class Zero(torch.nn.Module):
            def forward(self,x): return (x**3)*0
        model = Zero()
        cases = [(8,1000.,0,'analytic_zero')]
    else:
        cases = CASES
    summary = []
    modal = []
    for k, alpha, seed, checkpoint in cases:
        if not args.smoke_test:
            folder = args.input / f'k{k}_alpha_{alpha:g}_seed_{seed}'
            summary_path = folder/'summary.csv'
            path = folder/f'{checkpoint}.pt'
            if not summary_path.is_file() or not path.is_file():
                raise FileNotFoundError(f'Missing Step 10 run files in {folder}. Expected summary.csv and {checkpoint}.pt')
            with summary_path.open(newline='', encoding='utf-8-sig') as f:
                metadata = next(csv.DictReader(f))
            model = mod.Net(int(metadata['width']), int(metadata['depth'])).double()
            model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
            model.eval()
        print(f'\nCASE k={k} alpha={alpha:g} seed={seed} checkpoint={checkpoint}', flush=True)
        for n in grids:
            t0 = time.perf_counter()
            vals, phi, h, backend = eigensystem(mod, n, alpha, max(modes), args.backend)
            recs, ec, pred, eigenvalues = evaluate(mod, model, k, alpha, n, vals, phi, h, modes)
            elapsed = time.perf_counter()-t0
            for row in recs:
                row.update(k=k,alpha=alpha,seed=seed,checkpoint=checkpoint,
                           backend=backend,seconds=elapsed)
                summary.append(row)
                print(f' N={n:4d} J={row["modes"]:3d} '
                      f'defect={row["continuous_identity_defect"]:.4g} '
                      f'rel={row["continuous_identity_relative"]:.3%} '
                      f'unresolved={row["unresolved_energy_fraction"]:.3%} '
                      f'pass={bool(row["identity_pass_1pct"] and row["unresolved_pass_1pct"])}',flush=True)
            for j in range(max(modes)):
                modal.append(dict(k=k,alpha=alpha,seed=seed,checkpoint=checkpoint,
                                  grid=n,mode=j+1,eigenvalue=float(eigenvalues[j]),
                                  error_coefficient=float(ec[j]),
                                  predicted_continuous=float(pred[j]),
                                  continuous_defect=float(ec[j]-pred[j])))
            save_csv(args.output/'refinement_summary.csv', summary)
            save_csv(args.output/'refinement_mode_details.csv', modal)
    print('\nFinished. Results in:',args.output.resolve())
    print('Note: 1% thresholds are numerical quality criteria, not mathematical proofs.')
    print('Note: high-resolution grid refinement of the trained continuous residual remains empirical.')

if __name__ == '__main__':
    main()
