# FS-PREM: Few-Shot Port Risk Event Mining

<p align="center">
  <a href="#"><img src="https://img.shields.io/badge/Conference-IEEE%20ICDM-blue?style=flat-square"/></a>
  <a href="#"><img src="https://img.shields.io/badge/Python-3.9%2B-green?style=flat-square&logo=python"/></a>
  <a href="#"><img src="https://img.shields.io/badge/Framework-TensorFlow%20%7C%20Scikit--Learn-orange?style=flat-square"/></a>
  <a href="#"><img src="https://img.shields.io/badge/Data-IBTrACS%20%7C%20AIS-lightgrey?style=flat-square"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow?style=flat-square"/></a>
</p>

> **FS-PREM** is a physics-aware framework for predicting rare port disruption events from streaming hurricane advisories. It combines a novel PORTEX disruption index, rolling-window conformal prediction, and lightweight sequential models to deliver calibrated risk estimates in real time — even under extreme data scarcity.

---

## Overview

U.S. Gulf Coast ports (Houston, New Orleans, Mobile, Tampa, Corpus Christi) are critical nodes in global energy and agricultural supply chains. Post-COVID, storm-induced disruptions interact with labor shortages and supply chain fragility to amplify losses. Yet predicting *when* a port will be disrupted — from raw hurricane advisories alone — remains an unsolved few-shot learning challenge.

FS-PREM addresses this with three integrated components:

```
Hurricane Advisories (IBTrACS)          AIS Vessel Traffic (MarineCadastre)
         │                                          │
         └──────────────────┬───────────────────────┘
                            ▼
              ┌─────────────────────────┐
              │   PORTEX Index          │
              │  H · P · U · E · V      │
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │  Physics-Weighted GLM   │  ◄── L1/Group Lasso
              │  Compact GRU / Conv1D   │  ◄── Temporal modeling
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │  Rolling Conformal CP   │  ◄── Calibrated uncertainty
              └────────────┬────────────┘
                           │
                    Risk Score + Uncertainty Band
```

---

## ML Contributions

**PORTEX Disruption Index** — a physics-grounded, port-centric risk score that replaces storm-centric indices (Saffir-Simpson, CMEHI). PORTEX gates hurricane hazard through proximity and fuses it with operational exposure and infrastructure vulnerability, producing a single interpretable scalar per advisory–port pair.

**Physics-Weighted GLM with Sparsity Penalties** — a logistic regression over the five PORTEX dimensions fitted with L1 and group-lasso penalties. Constraining the model to domain-defined features rather than raw tabular columns yields stable, interpretable coefficients across LOSO folds — something tree ensembles fail to provide. Coefficient analysis confirms Vulnerability (+17.61) and Hazard (+3.21) as the dominant drivers, with Proximity showing a counterintuitive negative sign (−0.62), reflecting that storms very close to ports follow more predictable tracks.

**Streaming Mondrian Conformal Prediction (SMCP)** — a rolling-window conformal wrapper applied at each advisory update. Unlike standard CP, the rolling calibration queue discards stale advisories, maintaining valid coverage under inter-storm distributional shift without assuming exchangeability across the full history. At α=0.05 the framework achieves 99.4% empirical coverage while keeping average prediction set size at ~1.001 — effectively single-point predictions with guaranteed uncertainty bounds.

**Leave-One-Storm-Out (LOSO) Evaluation** — a cross-validation strategy that withholds all advisories from a single storm at test time, directly measuring generalization to unseen storm dynamics. This exposes severe overfitting in standard ensembles (Gradient Boosting: train AUC 0.99 → test AUC 0.73) and validates that physics-informed features are the key to closing the gap.

**Compact Sequential Architectures** — GRU and Conv1D models with hidden dimensions ≤ 16, designed to capture temporal advisory evolution (intensification trajectories, track shifts) while remaining fast enough for streaming inference. Compact GRU achieves AUC 0.797 (+6.3% over baselines) with negligible runtime per advisory.

---

## Key Results

| Model | AUC | Coverage (α=0.05) |
|---|---|---|
| Logistic Regression (baseline) | 0.750 | — |
| Random Forest (baseline) | 0.748 | — |
| Gradient Boosting (baseline) | 0.730† | — |
| **PORTEX-Weighted GLM** | **0.833** | — |
| PORTEX + SMCP | 0.740–0.750 | **99.4%** |
| Compact GRU | 0.797 | — |
| Compact Conv1D | 0.757 | — |

†Gradient Boosting overfits: train AUC ≈ 0.99, test AUC 0.73 under LOSO CV.

**PORTEX coefficient analysis:**

| Feature | Coefficient |
|---|---|
| Vulnerability (V) | +17.61 |
| Hazard (H) | +3.21 |
| Uncertainty (U) | +0.43 |
| Exposure (E) | 0.00 |
| Proximity (P) | −0.62 |

---

## The PORTEX Index

$$\text{PORTEX}_{p,t} = 100\Big[0.5(H_{p,t} \cdot P_{p,t}) + 0.2\,U_{\text{norm}} + 0.2\,E_{p,t} + 0.1\,V_{p,t}\Big]$$

| Dimension | Symbol | Description |
|---|---|---|
| Hazard | H | Nonlinear wind-speed + gale-radius intensity proxy |
| Proximity | P | Distance-modulated hazard gate |
| Uncertainty | U | Normalized advisory forecast error |
| Exposure | E | Port traffic throughput & vessel mix |
| Vulnerability | V | Infrastructure fragility vs. efficiency baseline |

---

## Repository Structure

```
fs-prem/
├── notebooks/                  # Reproducible experiment notebooks
│   ├── 01_data_setup.ipynb     # Data loading, PORTEX recomputation, labeling
│   ├── 02_baselines.ipynb      # LOSO CV baselines (LR, RF, GBT, SVM)
│   ├── 03_portex_glm.ipynb     # Physics-aware weighted GLM + interpretability
│   ├── 04_conformal.ipynb      # Streaming conformal prediction pipeline
│   ├── 05_sequential.ipynb     # Compact GRU + Conv1D temporal models
│   └── 06_interpretability.ipynb  # Case studies & conformal bands
│
├── src/
│   ├── data/
│   │   ├── loader.py           # IBTrACS + AIS loading & merging
│   │   └── preprocessing.py    # PORTEX computation, NaN imputation, labeling
│   ├── models/
│   │   ├── portex_glm.py       # Physics-weighted logistic model (L1/group lasso)
│   │   ├── sequential.py       # Compact GRU and Conv1D architectures
│   │   └── baselines.py        # LR, RF, GBT, SVM wrappers
│   ├── evaluation/
│   │   ├── loso_cv.py          # Leave-One-Storm-Out cross-validation
│   │   └── conformal.py        # Rolling-window conformal prediction
│   └── utils/
│       ├── metrics.py          # AUC, PR-AUC, F1, confusion matrix helpers
│       └── visualization.py    # Publication-quality plot generation
│
├── configs/
│   └── default.yaml            # Hyperparameters, thresholds, paths
├── results/figures/            # Generated plots (populated after runs)
├── requirements.txt
└── README.md
```

---

## Quickstart

```bash
git clone https://github.com/your-username/fs-prem.git
cd fs-prem
pip install -r requirements.txt
```

Update `configs/default.yaml` with your data paths, then run notebooks `01` → `06` in order, or use the src API directly:

```python
from src.data.preprocessing import compute_portex, create_event_labels
from src.models.portex_glm import PORTEXWeightedGLM
from src.evaluation.loso_cv import run_loso_cv
from src.evaluation.conformal import RollingConformalWrapper

df = compute_portex(raw_df)
df = create_event_labels(df, threshold_percentile=0.95)

model = PORTEXWeightedGLM(penalty='l1')
results = run_loso_cv(model, df)

cp_model = RollingConformalWrapper(model, alpha=0.05, window=100)
```

**Data sources:**
- IBTrACS: [ncei.noaa.gov/products/international-best-track-archive](https://www.ncei.noaa.gov/products/international-best-track-archive)
- AIS vessel records: [hub.marinecadastre.gov](https://hub.marinecadastre.gov/)

---

## Citation

```bibtex
@inproceedings{ashok2026fsprem,
  title     = {FS-PREM: A Physics-Aware Framework for Predicting Port Disruption},
  author    = {Ashok, Shriraghav},
  booktitle = {IEEE International Conference on Data Mining (ICDM)},
  year      = {2026},
  note      = {Horizon Labs, Future Impact Initiative}
}
```
