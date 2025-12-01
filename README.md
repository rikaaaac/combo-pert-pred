# Combinatorial Perturbation Prediction with Active Learning

A comprehensive framework for predicting gene expression changes from combinatorial genetic perturbations using ensemble machine learning and active learning strategies for optimal experimental design.

---

## Table of Contents

1. [Overview](#overview)
2. [Installation & Setup](#installation--setup)
3. [Detailed Workflow](#detailed-workflow)
4. [Data Format](#data-format)
5. [Evaluation Metrics](#evaluation-metrics)
6. [Output Files](#output-files)
7. [References](#references)

---

## Overview

This project implements an **ensemble-based prediction system** that combines multiple models to predict responses to combinatorial genetic perturbations. It includes:

* **Ensemble Prediction**: Combines GEARS, scLAMBDA, and baseline models
* **Uncertainty Quantification**: Estimates prediction confidence through model disagreement
* **Active Learning**: Recommends which experiments to perform next via:
  * **Synergy Detection**: Identifies non-additive gene interactions using genetic interaction (GI) scores
  * **Interpretability**: Uses SHAP analysis to explain model predictions

### Key Features

* **Multiple Models**: GEARS (graph-based network) + scLAMBDA (transformer) + baselines
* **Epistemic Uncertainty**: Quantifies model disagreement to identify knowledge gaps
* **Active Learning Strategies**: 
  * Random: baseline
  * Uncertainty: select where models disagree the most
  * Synergy: prioritizes non-addtive interactions
  * Diversity: sample from underrepresented regions
  * Oracle: select highest-error perturbations
  * Weighted combinations
* **Genetic Interaction Analysis**: GEARS-style genetic interaction scores for synergy quantification
* **SHAP Interpretability**: Identifies uncertainty drivers and hub genes
* **Experiment Optimization**: Reduces experimental costs compared to random sampling

### Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                   INPUT DATA                            │
│  Binary Perturbation Matrix (X) + Expression (y)        │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│                ENSEMBLE MODELS                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────────┐   │
│  │  GEARS   │  │ scLAMBDA │  │ Baselines (Mean&Add) │   │
│  └──────────┘  └──────────┘  └──────────────────────┘   │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│         ENSEMBLE AGGREGATION                            │
│  Mean Prediction + Epistemic Uncertainty (Variance)     │
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
│        INTERPRETABILITY (SHAP)                          │
│  Identify uncertainty drivers + hub genes               │
└─────────────────────────────────────────────────────────┘
```

---

## Detailed Workflow

### Workflow 1: Ensemble Prediction (`ensemble.py`)

```
Step 1: Data Loading and Preprocessing
├─ Load Norman dataset (H5AD format)
├─ Extract single perturbations (num_targets=1)
├─ Extract combo perturbations (num_targets=2)
└─ Create binary perturbation matrices

Step 2: Data Splitting
├─ Option A: split_by_perturbation()
│   └─ Split by unique perturbation patterns (unseen combinations)
└─ Option B: create_combo_splits()
    └─ GEARS-style: train on singles + some combos, test on held-out combos

Step 3: Model Loading
├─ Load GEARS model (graph neural network)
├─ Load scLAMBDA model (transformer with gene embeddings)
└─ Fit baseline models (mean and additive)

Step 4: Ensemble Prediction
├─ Convert gene pairs to binary perturbation vector
├─ Get predictions from all 4 models:
│   ├─ GEARS: Graph-based prediction
│   ├─ scLAMBDA: Transformer-based prediction
│   ├─ Mean baseline: Average single effects
│   └─ Additive baseline: Sum single effects
├─ Compute ensemble mean (average across models)
└─ Compute epistemic uncertainty (variance across models)

Step 5: Experiment Recommendation
├─ Score candidates by epistemic uncertainty
├─ Rank by total uncertainty (sum across genes)
└─ Return top-N most uncertain perturbations
```

### Workflow 2: Active Learning (`active_learning.py`)

```
Step 1: Initialize
└─ Load ensemble models

Step 2: Active Learning Loop (iterate N times)
├─ Update Models
│   ├─ Refit baselines with current training data
│   └─ Update GI scorer with new baseline effects
│
├─ Evaluate Current Model
│   ├─ Predict on test set
│   ├─ Compute MSE and Pearson correlation
│   └─ Log metrics
│
├─ Score Candidates
│   ├─ Filter to unseen perturbations
│   ├─ Compute scores based on strategy:
│   │   ├─ Uncertainty: Model disagreement
│   │   ├─ Synergy: GI score magnitude
│   │   ├─ Diversity: Gene coverage
│   │   └─ Weighted: α*uncertainty + β*synergy + γ*diversity
│   └─ Rank candidates by score
│
├─ Select Experiments
│   └─ Pick top-N highest-scoring perturbations
│
├─ Simulate Experiments
│   ├─ Retrieve ground truth from pool
│   └─ Add to training set
│
└─ Log Results
    └─ Record selected perturbations, metrics, scores

Step 3: SHAP Analysis
├─ Explain Uncertainty
│   ├─ Compute SHAP values for uncertainty function
│   └─ Identify genes driving model disagreement
│
└─ Explain Synergy
    ├─ Compute SHAP values for GI score function
    └─ Identify epistasis hub genes


Step 4: Strategy Comparison
├─ Run multiple strategies in parallel
├─ Plot learning curves (MSE vs. training size)
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
  - condition: Perturbation string (e.g., "CBL+CNN1", "ctrl")
  - num_targets: Number of perturbed genes (0, 1, or 2)
  - cell_type: Cell type information
  - replicate: Biological replicate

adata.var:
  - Gene names as index
  - Gene metadata

adata.X:
  - Expression matrix (cells x genes)
  - Typically log-normalized counts
```

### Data Splits

#### Perturbation-based Split
```
Goal: Test generalization to unseen gene combinations
Method: Split by unique perturbation patterns
Example:
  Train: {(A,B), (C,D), (E,F), ...}
  Test: {(G,H), (I,J), ...}
```

#### Combo Holdout Split (GEARS-style, used in this script) 
```
Goal: Train on singles + some combos, test on held-out combos
Method: All singles in training, random split of combos
Example:
  Train: All singles + 60% of combos
  Val: 20% of combos
  Test: 20% of combos
```

---

## Evaluation Metrics

### Prediction Quality

#### Mean Squared Error (MSE)
```python
mse = np.mean((y_pred - y_true) ** 2)
```
- **Lower is better**
- Measures average squared difference across all genes and samples
- Sensitive to outliers

#### Pearson Correlation Coefficient (r)
```python
# Per-gene correlation, then averaged
pearson_per_gene = []
for i in range(n_genes):
    r = np.corrcoef(y_pred[:, i], y_true[:, i])[0, 1]
    pearson_per_gene.append(r)
mean_pearson = np.mean(pearson_per_gene)
```
- **Higher is better** (range: -1 to 1)
- Measures linear correlation per gene
- More robust to scale differences

### Uncertainty Metrics

#### Epistemic Uncertainty
```python
# Variance across models
uncertainty = np.var([pred_gears, pred_sclambda, pred_mean, pred_additive], axis=0)
mean_uncertainty = np.mean(uncertainty)
```
- **Interpretation**: Model disagreement
- High uncertainty = models don't agree → knowledge gap
- Used for active learning acquisition

### Genetic Interaction Metrics

#### GI Score Magnitude
```python
gi = observed_AB - (ctrl + delta_A + delta_B)
gi_magnitude = np.linalg.norm(gi)  # L2 norm
```
- **Higher = more non-additive interaction**
- Positive: Synergistic
- Negative: Antagonistic
- Near-zero: Additive

### Active Learning Metrics

#### Sample Efficiency
- **Definition**: MSE achieved with N samples
- **Comparison**: MSE curve vs. training size
- **Goal**: Achieve target MSE with fewer experiments

#### Improvement Over Random
```python
improvement = (mse_random - mse_strategy) / mse_random * 100
```
- **Interpretation**: Percentage reduction in error
- Typical range: 20-60% for good strategies

---

## Output Files

### From `active_learning.py`

When running `compare_all_strategies()` or `run_simulation()`, the following files are generated:

#### 1. `results.json`
```json
{
  "uncertainty": {
    "iteration": [1, 2, 3, ...],
    "test_mse": [0.523, 0.489, 0.456, ...],
    "test_pearson": [0.45, 0.52, 0.58, ...],
    "n_training_samples": [100, 110, 120, ...],
    "mean_uncertainty": [0.089, 0.082, 0.076, ...],
    "mean_gi_score": [0.234, 0.241, 0.238, ...],
    "strategy": ["uncertainty", "uncertainty", ...]
  },
  "synergy": { ... },
  ...
}
```
- **Format**: JSON
- **Content**: Iteration metrics for all strategies
- **Use**: Easy parsing and plotting

#### 2. `results.pkl`
```python
# Same as JSON but includes selected_perturbations
{
  "uncertainty": {
    ...,
    "selected_perturbations": [
      [("BRAF", "KRAS", 0.89), ("TP53", "MYC", 0.85), ...],  # Iter 1
      [("EGFR", "PTEN", 0.82), ("AKT1", "PIK3CA", 0.79), ...]  # Iter 2
    ]
  }
}
```
- **Format**: Python pickle
- **Content**: Full results including selected gene pairs
- **Use**: Detailed analysis of selection decisions

#### 3. `strategy_comparison.png`
- **Format**: PNG image
- **Content**: 2×2 subplot figure
  - Top-left: MSE vs. training samples
  - Top-right: Pearson r vs. training samples
  - Bottom-left: Mean uncertainty vs. iteration
  - Bottom-right: Mean GI score vs. iteration
- **Use**: Visual comparison of strategy performance

#### 4. `summary.txt`
```
================================================================================
SUMMARY
================================================================================
Strategy             Final MSE       Final r  vs Random
--------------------------------------------------------------------------------
random                  0.4523        0.5234   baseline
uncertainty             0.3812        0.6123     +15.7%
synergy                 0.4102        0.5789     +9.3%
diversity               0.4234        0.5645     +6.4%
oracle                  0.2987        0.7012     +34.0%
weighted_unc_syn        0.3689        0.6341     +18.4%
weighted_all            0.3756        0.6198     +17.0%
================================================================================
```
- **Format**: Plain text
- **Content**: Final metrics and improvement over random
- **Use**: Quick summary for reporting

#### 5. `shap_uncertainty.png`
- **Format**: PNG image
- **Content**: SHAP summary plot for uncertainty drivers
- **Features shown**: Top 20 genes by SHAP importance
- **Color scale**: Feature value (0=not perturbed, 1=perturbed)
- **Interpretation**: Genes on top drive model uncertainty most

#### 6. `shap_synergy.png`
- **Format**: PNG image
- **Content**: SHAP summary plot for epistasis hub genes
- **Features shown**: Top 20 genes by SHAP importance
- **Interpretation**: Genes on top are frequently involved in synergistic interactions

#### 7. `shap_gene_comparison.png`
- **Format**: PNG image
- **Content**: Horizontal bar chart comparing uncertainty vs. synergy importance
- **Features shown**: Top 15 genes by combined importance
- **Bars**: Blue (uncertainty driver), Orange (epistasis hub)
- **Interpretation**: Genes with both bars are high-priority targets

---

## Best Practices

### 1. Data Preparation
- Ensure gene names match across all datasets
- Normalize expression data (log-transform + scaling)
- Filter low-quality cells/genes before training
- Balance single vs. combo perturbations if possible
- Train GEARS and scLAMBDA on other datasets if needed

### 2. Model Training
- Train on diverse perturbation set (not just related pathways)
- Include sufficient single perturbations for baseline models
- Validate on held-out combinations, not random splits
- Monitor epistemic uncertainty to detect out-of-distribution samples

### 3. Active Learning Strategy Selection

| Phase | Recommended Strategy | Rationale |
|-------|---------------------|-----------|
| **Early (0-20% labeled)** | Diversity (γ=1.0) | Ensure broad gene coverage |
| **Mid (20-60% labeled)** | Weighted (α=0.5, β=0.3, γ=0.2) | Balance exploration/exploitation |
| **Late (60%+ labeled)** | Uncertainty (α=1.0) | Focus on remaining knowledge gaps |
| **Synergy discovery** | Synergy (β=1.0) or Weighted (β=0.5) | Prioritize non-additive interactions |

### 4. Interpretation
- **High uncertainty + High GI**: Priority targets (uncertain AND synergistic)
- **High uncertainty + Low GI**: Uncertain but likely additive
- **Low uncertainty + High GI**: Confident synergistic predictions
- **Low uncertainty + Low GI**: Well-understood additive effects

## References

1. **GEARS**: Roohani et al. (2023)
   - Title: "Predicting transcriptional outcomes of novel multigene perturbations with GEARS"
   - Journal: Nature Biotechnology
   - Link: https://www.nature.com/articles/s41587-023-01905-6

2. **scLAMBDA**: Wang et al. (2024)
   - Title: "Modeling and predicting single-cell multi-gene perturbation responses with scLAMBDA"
   - Preprint: https://www.biorxiv.org/content/10.1101/2024.12.04.626878v1

3. **Norman Dataset**: Norman et al. (2019)
   - Title: "Exploring genetic interaction manifolds constructed from rich single-cell phenotypes"
   - Journal: Science
   - Link: https://science.sciencemag.org/content/365/6455/786

4. **Epistemic Uncertainty**: Kendall & Gal (2017)
   - Title: "What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?"
   - Journal: arXiv
   - Link: https://arxiv.org/abs/1703.04977