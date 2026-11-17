# PINN spectral bias: reproducibility archive (curated)

This archive accompanies the manuscript **When Does Spectral Bias Matter?**

## Verified reproduction

From the archive root, run:

```bash
python code/reproduce_pinn_summary.py
```

The original script expects `Spectral_Bias_GitHub_Staging/data` next to itself. For convenience the packaged script has been adjusted to read `../data` and write to `../figures`. It recreates Figure 10 (PDF, PNG, SVG) from the included CSVs. Python requirements: numpy, pandas, matplotlib.

## Scope and limitations

- Figure 10 is regenerated from archived numerical CSVs; its panel (d) is training-integrated, while (a)--(c) are final-checkpoint results from a separate objective comparison.
- The remaining figure files are provided as previously generated assets; exact regeneration of all nine other figures has **not** been independently established.
- Included experiment scripts are research records, not a verified one-command end-to-end training pipeline. Full model checkpoints and every original raw run are not included.
- This is a **curated archive**, not a claim of complete experimental reproducibility.
- SHA256SUMS.csv lists the packaged file hashes.

No repository URL or DOI is assigned. The author should confirm declarations and repository availability before submission.
