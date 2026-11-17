"""Reproduce post-hoc Fourier norm comparison from saved final-checkpoint error spectra.
This is NOT an implementation or training run of the DFR method.
"""
from pathlib import Path
import numpy as np
import pandas as pd

base = Path(__file__).resolve().parent
rows=[]
for k in (8,16):
    for objective in ('raw','normalized'):
        filename=f'k{k}_{objective}_seed_1_final_spectrum.csv'
        d=pd.read_csv(base/filename).sort_values('frequency')
        j=d['frequency'].to_numpy(dtype=int)
        assert np.array_equal(j,np.arange(1,65)), filename
        c=d['error_coefficient'].to_numpy(dtype=float)
        mu=(np.pi*j)**2
        # Dirichlet Poisson L=-d^2/dx^2; c_j are unnormalized sine coefficients.
        # Residual coefficients reconstructed algebraically; not independently measured.
        r=mu*c
        l2_energy=.5*c**2
        hminus1_energy=.5*r**2/mu
        rows.append({'k':k,'objective':objective,'checkpoint':'final','seed':1,'J':64,
                     'L2_norm':np.sqrt(l2_energy.sum()),
                     'DFR_Hminus1_norm':np.sqrt(hminus1_energy.sum()),
                     'L2_low4_share':l2_energy[j<=4].sum()/l2_energy.sum(),
                     'DFR_low4_share':hminus1_energy[j<=4].sum()/hminus1_energy.sum(),
                     'L2_target_share':l2_energy[j==k].sum()/l2_energy.sum(),
                     'DFR_target_share':hminus1_energy[j==k].sum()/hminus1_energy.sum(),
                     'residual_coeff_source':'reconstructed as lambda_j*error_coefficient; not measured independently'})
summary=pd.DataFrame(rows)
summary.to_csv(base/'DFR_Posthoc_Comparison_Results.csv',index=False)
print(summary.to_string(index=False))
