# SDT-Net: anatomy-rendered spectral digital twin for kidney stone burden and composition in CT

Code accompanying the manuscript *"Anatomy-Rendered Spectral Digital Twin for Partial-Volume-Aware Burden
Estimation and Calibrated Composition Assessment of Kidney Stones in CT"*.

All files sit in one folder. Python 3.11, CPU only. Everything (data, models, tables, figures) is reproduced
from fixed seeds.

## Install

```
pip install -r requirements.txt
```

## Files

| File | Purpose |
|---|---|
| `physics.py` | Polychromatic spectra, detector weighting, NIST/Elam attenuation (xraydb), Hounsfield conversion, site protocols A-E |
| `simulator.py` | RenalStone-DT-Sim: patient-specific renal phantom, stones, distractor calcifications, acquisition chain, stone-free counterfactual |
| `make_dataset.py` | Generates the 3,374 examinations and the patient-wise splits (`SDT_DATA`, default `/home/claude/data`) |
| `data.py` | Loading, scaling, protocol code, memory-lean training store |
| `model.py` | 3D U-Net backbone, collecting-system gate, auxiliary counterfactual head, SSL reconstruction head |
| `train.py` | Masked spectral pretraining and supervised multi-task training (variants `full`, `nossl`, `noanat`, `nobg`, `mtunet`, `agunet`) |
| `twin.py` | Anatomy-rendered counterfactual, blur-invariant moments, PSF calibration, differentiable renderer, Laplace covariance, material coordinates |
| `infer.py` | Runs a trained backbone + twin on every cohort and stores per-stone records |
| `baselines_classic.py` | 200 HU threshold detection with LDA / random-forest composition |
| `heads.py` | Propagated ensemble head, Mondrian conformal sets (APS and threshold scores), material-manifold distance, burden fusion |
| `evaluate.py` | Metrics, conformal volume intervals, variance model, treatment-level referral |
| `run_eval.py` | All analyses of Section V; writes `results/*.json` |
| `runner2.py`, `runner3.py`, `runner4.py` | Job queues used for training and inference (one thread per process) |
| `cohort_stats.py` | Cohort table |
| `fig_arch.py`, `fig_data.py`, `fig_case.py`, `fig_results.py` | Figures (grayscale) |

## Reproduce

```
python make_dataset.py                      # ~1.5 h on 2 cores
python runner2.py T & python runner2.py I   # training lane and inference lane
python runner3.py                           # ablation / PSF-sensitivity inference
python baselines_classic.py
python run_eval.py                          # all tables as JSON in results/
python fig_results.py                       # result figures
```

Environment variables: `SDT_DATA` (cases), `SDT_RUNS` (checkpoints and records), `SDT_RES` (results), `SDT_FIG` (figures).

## Notes

* Splits are by patient; repeat examinations follow their patient and are used only for repeatability.
* Temperatures, conformal thresholds, the variance model and the OOD threshold are fitted on validation or
  calibration splits only.
* The cohort is simulated. It provides exact ground truth for volume and composition but does not replace
  validation on phantom-mounted ex vivo stones or clinical data.

## Bundled results

The `*.json` files in this folder are the exact outputs of `run_eval.py`, `cohort_stats.py`, `volcal.py`,
`failure.py` and `timing.py` used for every number in the manuscript (set `SDT_RES` to this folder to rebuild the
figures from them with `python fig_results.py`).
