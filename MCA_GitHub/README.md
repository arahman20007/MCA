# PINN spectral bias — curated research archive

Companion research materials for **“When Does Spectral Bias Matter? Residual
Persistence and PDE Error Amplification in Physics-Informed Neural Networks.”**

This folder is a **curated archive**, not a fully verified end-to-end training
reproduction package. The numerical CSVs and archived figures are retained from
the author's uploaded GitHub repository; they have not been replaced by
synthetic or reconstructed raw experiments.

## Repository layout

- `code/`: experimental and analysis scripts
- `data/`: selected saved numerical results
- `figures/`: saved manuscript figure assets
- `FIGURE_SOURCES.md`: figure-by-figure reproduction status
- `SHA256SUMS.csv`: file-integrity manifest
- `requirements.txt`: Python dependencies
- `verify_repository.py`: integrity and input-availability checks
- `generate_checksums.py`: regenerate the SHA-256 manifest after changes

## Installation

Use a Python 3.10+ environment. From **this folder**:

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

Install an appropriate PyTorch build for your platform; CPU/GPU builds may
require instructions from the official PyTorch installation guide. The included
scripts use the Python standard library, NumPy, pandas, Matplotlib, and PyTorch.

## Reproduce Figure 10

```bash
python code/reproduce_pinn_summary.py
```

This script reads three archived CSVs from `data/` and writes Figure 10 in
PDF, PNG, and SVG formats to `figures/`. The numerical factor table is
**training-integrated**, whereas the plotted spectra and objective comparison
are **final-checkpoint** quantities from a separate experiment.

## DFR post-hoc comparison — inputs included

```bash
python code/DFR_Posthoc_Comparison_Reproduce.py
```

This script requires the following **original** input files in `data/`:

- `k8_raw_seed_1_final_spectrum.csv`
- `k8_normalized_seed_1_final_spectrum.csv`
- `k16_raw_seed_1_final_spectrum.csv`
- `k16_normalized_seed_1_final_spectrum.csv`

Each file contains `frequency` and `error_coefficient` columns for
frequencies 1–64. **All four source CSVs are now included in `data/`.**
The DFR post-hoc script has been executed successfully with these inputs
and produced four result rows. The reconstructed final-checkpoint Fourier
error norms were:

| Target mode | Raw | Normalized |
|---|---:|---:|
| `k=8` | 3.794044 | 3.794189 |
| `k=16` | 7.413037 | 7.400297 |

These are **post-hoc Fourier calculations**, not newly trained DFR models.
The archived `data/DFR_Posthoc_Comparison_Results.csv` should be compared
with the regenerated output before claiming numerical agreement.

## Other experiments and figures

The `step*.py` scripts are research records. Some require additional
checkpoints, run histories, and configurations not included here.
Do not interpret their presence as a verified one-command reproduction
pipeline. See `FIGURE_SOURCES.md` for figure-level status.

## Verify archive integrity

```bash
python verify_repository.py
```

This verifies the recorded file hashes and reports any missing DFR inputs.
After the four spectral CSVs were added, the local repository integrity
check reported **39/39 listed files matched**. This count applies to the
verified local version; any subsequent edits require regenerating the manifest.
Run verification after downloading a fresh ZIP from GitHub. If you edit any
file, regenerate the manifest with:

```bash
python generate_checksums.py
```

## Limitations and provenance

- Figure 10 regeneration and the four-input DFR post-hoc calculation have been tested with archived data.
- The remaining figure assets are included, but exact regeneration has
  not been independently established.
- Full training reruns have not been performed in this repository audit.
- Original model checkpoints and some raw experimental outputs are absent.
- No fabricated replacement datasets or experiment results were added.
- Verify that any manuscript Data Availability Statement describes this
  limited scope accurately.

Repository: https://github.com/arahman20007/MCA
