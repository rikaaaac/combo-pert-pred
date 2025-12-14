# Combinatorial Perturbation Prediction with Ensemble Learning

A framework for predicting gene expression changes from combinatorial genetic perturbations using an ensemble of additive baseline and VAE models, with active learning for optimal experimental design.

---

## Table of Contents

1. [Overview](#overview)
2. [Installation & Setup](#installation--setup)
3. [Quick Start](#quick-start)
4. [Detailed Workflow](#detailed-workflow)
5. [Data Format](#data-format)
6. [Evaluation Metrics](#evaluation-metrics)
7. [Output Files](#output-files)
8. [Best Practices](#best-practices)
9. [References](#references)

---

## Overview

This project implements an **additive + VAE ensemble** for predicting combinatorial gene perturbation effects. The ensemble combines:

* **Additive Baseline**: Linear superposition of single-gene effects (gene_AB = gene_A + gene_B)
* **VAE (Variational Autoencoder)**: Captures nonlinear epistatic interactions
* **Weighted Ensemble**: Optimally combines both models with learned weights

### Key Features

* **Two-Model Ensemble**: Interpretable additive baseline + flexible VAE
* **Epistemic Uncertainty**: Quantifies model disagreement to identify knowledge gaps
* **Active Learning Strategies**:
  * Random: uniform sampling baseline
  * Uncertainty: select where models disagree most
  * Synergy: prioritize non-additive interactions (GEARS-style GI scores)
  * Diversity: sample from underrepresented expression-space regions
  * Weighted: α*uncertainty + β*synergy + γ*diversity
* **Conformal Prediction**: Calibrated prediction intervals with 90% coverage
* **Experiment Optimization**: Recommend experiments that maximize information gained

### Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                   INPUT DATA                            │
│  Binary Perturbation Matrix (X) + Expression (y)        │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│              ENSEMBLE MODELS                            │
│  ┌───────────────────┐         ┌────────────────────┐   │
│  │ Additive Baseline │         │        VAE         │   │
│  │f(x_AB) = Δ_A + Δ_B│         |  Nonlinear model   │   │
│  └───────────────────┘         └────────────────────┘   │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│         WEIGHTED ENSEMBLE                               │
│  Prediction: w1*f_add + w2*f_VAE                        │
│  Uncertainty: (f_add - f_VAE)² + Var[f_VAE]             │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│           ACTIVE LEARNING                               │
│  Score candidates → Select top-N → Simulate experiments │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│         CONFORMAL PREDICTION                            │
│  Calibrated prediction intervals (90% coverage)         │
└─────────────────────────────────────────────────────────┘
```

### Performance

**Ensemble Results (Norman Dataset):**
- VAE alone: MSE=0.037, R²=0.80, r=0.90
- Additive alone: MSE=0.187, R²=0.01, r=0.90
- Weighted Ensemble: MSE=0.118 (weights: 0.37 additive, 0.63 VAE)
- Epistemic uncertainty correlates with error: r=0.64

---

## Installation & Setup

### Requirements

```bash
conda env create -f environment.yml
```

### Data Setup

Download the Norman 2019 dataset. 

The dataset should be in AnnData (H5AD) format with:
- Single perturbations (nperts==1)
- Combinatorial perturbations (nperts==2)
- Control samples (condition='ctrl')

---

## Quick Start

### 1. Train Ensemble

```bash
cd scripts/
python ensemble_add_vae.py
```

**Outputs:**
- `additive_vae_results.pkl` - Trained weights, test metrics, uncertainty estimates

### 2. Analyze Results

```bash
python ensemble_analyze.py
```

**Outputs (in `ensemble_results/`):**
- `model_comparison.png` - Performance comparison
- `uncertainty_vs_error.png` - Calibration plot
- `gene_uncertainty.png` - Top uncertain genes
- `uncertainty_distribution.png` - Uncertainty histogram
- `diversity_heatmap.png` - Model correlation

### 3. Run Active Learning

```bash
python active_learning.py
```

---

## Detailed Workflow

### Workflow 1: Ensemble Training (`ensemble_add_vae.py`)

```
Step 1: Data Loading
├─ Load Norman dataset (H5AD format)
├─ Extract single perturbations (nperts=1)
├─ Extract combo perturbations (nperts=2)
└─ Create binary perturbation matrices

Step 2: Data Splitting
├─ All singles in training set
├─ Combinations split: 60% train / 20% val / 20% test
└─ GEARS-style holdout evaluation

Step 3: Fit Additive Baseline
├─ For each gene g: Δ_g = mean(y | x_g=1)
└─ Prediction: f_add(x_AB) = Δ_A + Δ_B

Step 4: Train VAE
├─ Architecture: Encoder → Latent(64) → Decoder
├─ Loss: MSE + β*KL divergence (β=0.5)
├─ Training: 50 epochs, batch=256, Adam(lr=1e-3)
└─ Early stopping: patience=10

Step 5: Learn Ensemble Weights
├─ Normalize predictions on validation set
├─ Solve: min ||y - (w1*f_add + w2*f_VAE)||^2  s.t. w≥0
└─ Normalize: w <- w / ||w||^2

Step 6: Conformal Prediction
├─ Calibrate residuals on test set
├─ Compute gene-wise quantiles (90%)
└─ Construct prediction intervals
```

### Workflow 2: Active Learning (`active_learning.py`)

```
Step 1: Initialize
├─ Load pre-trained ensemble weights
├─ Start with 5% of combinations 
└─ Hold out 95% as candidate pool 

Step 2: Active Learning Loop (5 iterations)
├─ Refit Additive Baseline
│   └─ Update with current singles in training set
│
├─ Retrain VAE
│   ├─ 50 epochs initial training
│   └─ 30 epochs per iteration
│
├─ Evaluate on Test Set
│   ├─ Compute MSE and Pearson r
│   └─ Log metrics
│
├─ Score Candidates (select strategy)
│   ├─ Uncertainty: Σ(f_add - f_VAE)^2 + Var[f_VAE]
│   ├─ Synergy: ||f_ens(x_AB) - (Δ_A + Δ_B)||^2
│   ├─ Diversity: min ||f(x) - f(x')||^2 for x' in training
│   └─ Weighted: α*unc + β*syn + γ*div
│
├─ Select Top-15 Experiments
│   └─ Highest acquisition scores
│
└─ Simulate Experiments
    ├─ Retrieve ground truth from pool
    └─ Add to training set

Step 3: Compare Strategies
├─ Random, Uncertainty, Diversity, Synergy, Weighted
├─ Plot learning curves
└─ Generate summary table
```

---

## Data Format

### Input Data Structure

#### Perturbation Matrix (X)
```
Shape: (n_samples, n_genes)
Type: Binary matrix (0 or 1)
Meaning: X[i,j] = 1 if gene j is perturbed in sample i

Example:
         BRAF  KRAS  TP53  MYC
Sample 0    1     0     0    0   → BRAF single
Sample 1    0     1     0    0   → KRAS single
Sample 2    1     1     0    0   → BRAF+KRAS combo
Sample 3    0     0     1    1   → TP53+MYC combo
```

#### Expression Matrix (y)
```
Shape: (n_samples, n_genes)
Type: Float (log-normalized counts)
Meaning: Gene expression levels after perturbation

Example:
         BRAF    KRAS    TP53    MYC    ...
Sample 0  2.3     5.1     3.2    4.5    ...
Sample 1  4.2     1.8     3.1    4.3    ...
Sample 2  3.1     2.5     2.9    4.7    ...
```

### AnnData Format (H5AD)

The Norman dataset is stored in AnnData format with the following structure:

```
adata.obs columns:
  - condition: Perturbation string (e.g., "BRAF+KRAS", "ctrl")
  - nperts: Number of perturbed genes (0, 1, or 2)

adata.var:
  - Gene names as index

adata.X:
  - Expression matrix (cells x genes)
  - Log-normalized counts
```

### Data Splits

#### Combo Holdout Split (used in ensemble_add_vae.py)
```
Goal: Train on singles + some combos, test on held-out combos
Method: All singles in training, random split of combos

Split ratios:
  - Train: All singles + 60% of combos
  - Val: 20% of combos
  - Test: 20% of combos

Example counts (Norman dataset):
  - Singles: ~100 samples (all in training)
  - Combos: ~6,000 samples
    - Train: ~3,600
    - Val: ~1,200
    - Test: ~1,200
```

---

## Project Structure

```
combo-pert-pred/
├── README.md                    # This file
├── environment.yml              # Dependencies
├── scripts/                     # Main implementation
│   ├── README.md                # Detailed code documentation
│   ├── ensemble_add_vae.py      # Core ensemble (Additive + VAE)
│   ├── active_learning.py       # Active learning strategies
│   └── ensemble_analyze.py      # Visualization and analysis
|
└── ensemble_results/            # Ensemble results figures

```

---

