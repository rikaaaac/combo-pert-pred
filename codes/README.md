# Combinatorial Perturbation Prediction & Active Learning

This repository implements an ensemble-based framework for predicting combinatorial genetic perturbation effects, with active learning capabilities for experimental design optimization.

---

## Overview

The framework combines multiple state-of-the-art models (GEARS, scLAMBDA) with baseline methods to:
1. **Predict** gene expression changes from combinatorial perturbations
2. **Quantify uncertainty** through ensemble disagreement
3. **Recommend experiments** using active learning strategies
4. **Analyze synergy** via genetic interaction (GI) scores
5. **Interpret predictions** using SHAP analysis

---

## File Structure

### `ensemble.py`
Core ensemble prediction system combining multiple perturbation models.

### `active_learning.py`
Active learning loop with multiple acquisition strategies and interpretability tools.

---

## Components

### 1. **PerturbationDataProcessor** (`ensemble.py`)

Standardizes data formats across different perturbation prediction models.

#### Key Methods

- **`load_norman_data(data_path)`**
  - Loads Norman et al. CRISPR combinatorial perturbation dataset
  - Separates single vs. combo perturbations
  - Returns binary perturbation matrices (X) and expression profiles (y)

- **`split_by_perturbation(X, y, train_ratio, val_ratio, test_ratio)`**
  - Splits data by unique perturbation combinations (not samples)
  - Tests model's ability to predict unseen gene combinations
  - Returns stratified train/val/test splits

- **`create_combo_splits(X_single, y_single, X_combo, y_combo, combo_test_ratio)`**
  - GEARS-style split: train on all singles + some combos
  - Test on held-out combinations
  - Simulates realistic experimental design scenario

---

### 2. **Ensemble** (`ensemble.py`)

Ensemble model combining GEARS, scLAMBDA, and baseline predictors.

#### Architecture

**Component Models:**
1. **GEARS** - Graph neural network for perturbation prediction
2. **scLAMBDA** - Transformer-based model with gene embeddings
3. **Mean Baseline** - Average single perturbation effects
4. **Additive Baseline** - Linear combination of single effects

#### Key Methods

- **`load_models(...)`**
  - Loads all pre-trained models
  - Arguments:
    - `gears_model_dir`: Saved GEARS model directory
    - `gears_data_path`: GEARS data directory
    - `sclambda_model_path`: scLAMBDA checkpoint path
    - `sclambda_adata_path`: AnnData for scLAMBDA
    - `sclambda_embeddings_path`: Gene embeddings pickle
    - `norman_data_path`: Norman dataset for baselines

- **`predict_ensemble(X_test)`**
  - Generates predictions from all models
  - Returns:
    - `ensemble_mean`: Average prediction across models
    - `epistemic_uncertainty`: Variance across model predictions (disagreement)
    - `predictions`: Individual model outputs
  - Handles unknown genes gracefully (fallback to baselines)

- **`recommend_experiments(candidate_perturbations, n_recommend)`**
  - Ranks candidate gene pairs by epistemic uncertainty
  - Returns top-N most uncertain experiments
  - Prioritizes experiments where models disagree most

---

### 3. **GeneticInteractionScorer** (`active_learning.py`)

Computes genetic interaction (GI) scores to quantify synergy and epistasis.

#### GEARS GI Score Method

```
GI = observed_AB - expected_AB
where:
  expected_AB = ctrl + Δ_A + Δ_B
  Δ_A = y_A - ctrl
  Δ_B = y_B - ctrl
```

- **Positive GI** → Synergistic interaction (combo effect > sum of singles)
- **Negative GI** → Antagonistic interaction (buffering/suppression)
- **Near-zero GI** → Additive (no interaction)

#### Key Methods

- **`gears_gi_score(y_ctrl, y_A, y_B, y_AB)`**
  - Computes GI score for a single gene pair
  - Returns per-gene GI profile

- **`compute_gi_for_candidates(candidates)`**
  - Batch computes GI magnitude (L2 norm) for candidate pairs
  - Used in synergy-based active learning

- **`classify_interaction(y_ctrl, y_A, y_B, y_AB, threshold)`**
  - Classifies as 'synergistic', 'antagonistic', or 'additive'

---

### 4. **SHAPAnalyzer** (`active_learning.py`)

SHAP-based interpretability to identify:
- **Uncertainty drivers**: Genes that cause model disagreement
- **Epistasis hub genes**: Genes frequently involved in synergistic interactions

#### Key Methods

- **`explain_uncertainty(X_samples, n_background, n_explain)`**
  - Computes SHAP values for uncertainty function
  - Identifies which genes drive high prediction uncertainty
  - Returns top genes ranked by SHAP importance

- **`explain_synergy(X_samples, gi_scorer, n_background, n_explain)`**
  - Computes SHAP values for GI score function
  - Identifies hub genes involved in epistatic interactions
  - Returns top epistasis hub genes

- **`plot_uncertainty_summary(save_path)` / `plot_synergy_summary(save_path)`**
  - Generates SHAP summary plots
  - Visualizes feature importance for uncertainty/synergy

- **`plot_gene_comparison(top_n, save_path)`**
  - Compares genes that drive uncertainty vs. synergy
  - Identifies genes that are both uncertain AND synergistic

---

### 5. **ActiveLearningLoop** (`active_learning.py`)

Main active learning framework for experimental design.

#### Selection Strategies

| Strategy | α | β | γ | Description |
|----------|---|---|---|-------------|
| **Random** | - | - | - | Uniform sampling (baseline) |
| **Uncertainty** | 1.0 | 0 | 0 | Select highest model disagreement |
| **Synergy** | 0 | 1.0 | 0 | Select highest GI scores (synergistic pairs) |
| **Diversity** | 0 | 0 | 1.0 | Prefer underrepresented genes |
| **Oracle** | - | - | - | Cheat mode: select highest-error pairs (upper bound) |
| **Weighted** | α | β | γ | Combine: `α*uncertainty + β*synergy + γ*diversity` |

#### Scoring Functions

- **`compute_uncertainty_scores(candidates)`**
  - Epistemic uncertainty from ensemble disagreement
  - Sum of variance across genes

- **`compute_synergy_scores(candidates)`**
  - GI-based synergy scores (GEARS method)
  - L2 norm of genetic interaction profile

- **`compute_diversity_scores(candidates, already_selected, X_train)`**
  - Diversity score = 1/(1 + gene_frequency)
  - Prefers genes that haven't been tested much

- **`compute_oracle_scores(candidates, ground_truth)`**
  - Oracle using ground truth
  - Selects pairs with highest actual prediction error

#### Active Learning Workflow

**`run_simulation(X_train_init, y_train_init, X_test, y_test, X_pool, y_pool, ...)`**

```
For iteration in 1..N:
  1. Update baseline models with current training data
  2. Evaluate on test set (MSE, Pearson r, uncertainty)
  3. Get candidate pool (unseen perturbations)
  4. Score candidates using selection strategy
  5. Select top-N experiments
  6. Simulate experiments (retrieve from ground truth pool)
  7. Add to training set
  8. Log metrics
```

Returns iteration history: MSE, Pearson correlation, uncertainty, GI scores

---

## Usage Examples

### Basic Ensemble Prediction

```python
from ensemble import Ensemble

# Initialize ensemble
ensemble = Ensemble(sclambda_repo_path='/path/to/scLAMBDA')

# Load pre-trained models
ensemble.load_models(
    gears_model_dir='./gears_model',
    gears_data_path='./gears_data',
    gears_data_name='norman',
    sclambda_model_path='./sclambda_model',
    sclambda_adata_path='./norman_data.h5ad',
    sclambda_embeddings_path='./gene_embeddings.pkl',
    norman_data_path='./norman_data.h5ad'
)

# Create data splits
splits = ensemble.data_processor.create_combo_splits(
    X_single=ensemble.X_single,
    y_single=ensemble.y_single,
    X_combo=ensemble.X_combo,
    y_combo=ensemble.y_combo,
    combo_test_ratio=0.2
)

# Predict with uncertainty
pred_mean, uncertainties, individual_preds = ensemble.predict_ensemble(splits['X_test'])

# Recommend experiments
candidates = [('BRAF', 'KRAS'), ('TP53', 'MYC'), ...]
recommendations = ensemble.recommend_experiments(candidates, n_recommend=10)
```

### Active Learning with Strategy Comparison

```python
from active_learning import ActiveLearningLoop, compare_all_strategies

# Run active learning
al = ActiveLearningLoop(ensemble)

history = al.run_simulation(
    X_train_init=X_train_init,
    y_train_init=y_train_init,
    X_test=X_test,
    y_test=y_test,
    X_pool=X_pool,
    y_pool=y_pool,
    n_iterations=15,
    n_select_per_iter=10,
    strategy='weighted',
    alpha=0.5,  # uncertainty weight
    beta=0.3,   # synergy weight
    gamma=0.2   # diversity weight
)

# Compare all strategies
results = compare_all_strategies(
    ensemble, X_train_init, y_train_init,
    X_test, y_test, X_pool, y_pool,
    n_iterations=15,
    n_select=10,
    result_dir='./results'
)
```

### SHAP Interpretability Analysis

```python
from active_learning import ActiveLearningLoop

al = ActiveLearningLoop(ensemble)

# Run SHAP analysis
shap_results = al.run_shap_analysis(
    X_combo=ensemble.X_combo,
    n_background=100,
    n_explain=50
)

# Generate plots
al.plot_shap_results(save_dir='./results')

# Access top genes
uncertainty_drivers = shap_results['uncertainty']['top_genes']
epistasis_hubs = shap_results['synergy']['top_genes']
```

---

## Evaluation Metrics

### Prediction Quality
- **MSE**: Mean squared error between predicted and true expression
- **Pearson r**: Per-gene correlation, averaged across all genes

### Active Learning Metrics
- **Mean Uncertainty**: Average epistemic uncertainty (ensemble variance)
- **Mean GI Score**: Average genetic interaction magnitude
- **Training Efficiency**: MSE vs. number of training samples

### Interpretability
- **SHAP Importance**: Feature importance for uncertainty/synergy functions
- **Hub Gene Identification**: Genes frequently involved in epistasis

---

## Key Features

### Uncertainty Quantification
- **Epistemic uncertainty** via ensemble disagreement
- Captures model uncertainty (not aleatoric/noise)
- Identifies regions of input space where models disagree

### Genetic Interaction Analysis
- **GEARS-style GI scores**
- Quantifies deviation from additivity
- Classifies synergistic vs. antagonistic interactions

### Active Learning Strategies
- **Uncertainty sampling**: Exploit model disagreement
- **Synergy-aware**: Prioritize non-additive combinations
- **Diversity**: Ensure broad gene coverage
- **Weighted hybrid**: Balance multiple objectives

### Interpretability
- **SHAP analysis** for feature importance
- Identifies uncertainty drivers
- Discovers epistasis hub genes
- Guides biological hypothesis generation

---

## Data Format

### Input: Perturbation Matrix (X)
- Shape: `(n_samples, n_genes)`
- Binary matrix: `X[i,j] = 1` if gene j is perturbed in sample i
- Single perturbations: one 1 per row
- Combo perturbations: two or more 1s per row

### Output: Expression Profiles (y)
- Shape: `(n_samples, n_genes)`
- Gene expression levels (typically log-normalized counts)
- Represents cellular response to perturbation

---

## Workflow Summary

### Training Phase
1. Load Norman et al. dataset (h5ad format)
2. Split into singles/combos
3. Load pre-trained GEARS and scLAMBDA models
4. Fit baseline models (mean/additive) on singles

### Prediction Phase
1. Convert gene pairs to binary perturbation matrix
2. Get predictions from all 4 models
3. Compute ensemble mean and variance
4. Return predictions + uncertainty estimates

### Active Learning Phase
1. Initialize with small training set
2. For each iteration:
   - Evaluate current model on test set
   - Score candidate experiments (uncertainty/synergy/diversity)
   - Select top-N experiments
   - Simulate experiments (add to training)
   - Retrain baselines
3. Compare strategies via learning curves

### Interpretability Phase
1. Run SHAP analysis on uncertainty function
2. Run SHAP analysis on GI score function
3. Identify uncertainty drivers and epistasis hubs
4. Generate summary plots

---

## Requirements

- `numpy`, `pandas`: Data manipulation
- `scanpy`: Single-cell data processing (AnnData)
- `torch`: Deep learning (GEARS/scLAMBDA)
- `shap`: Interpretability analysis
- `matplotlib`: Visualization

---

## Output Files

When running `active_learning.py`, the following files are saved:

- `results.json`: Active learning metrics (JSON format)
- `results.pkl`: Full results including selected perturbations (pickle)
- `strategy_comparison.png`: Learning curves comparing all strategies
- `summary.txt`: Summary table of final performance
- `shap_uncertainty.png`: SHAP summary for uncertainty drivers
- `shap_synergy.png`: SHAP summary for epistasis hubs
- `shap_gene_comparison.png`: Bar plot comparing uncertainty vs. synergy importance

---

## Experimental Design Recommendations

### When to use each strategy:

- **Uncertainty**: When model disagreement indicates knowledge gaps
- **Synergy**: When interested in discovering non-additive interactions
- **Diversity**: When broad gene coverage is important (early exploration)
- **Weighted (0.4-0.3-0.3)**: Balanced approach for general use
- **Weighted (0.5-0.5-0)**: Focus on uncertain synergistic pairs

### Best Practices:

1. Start with diversity to ensure broad coverage
2. Transition to uncertainty/synergy as data accumulates
3. Use SHAP analysis to identify high-priority genes
4. Run oracle to establish upper bound on performance
5. Monitor mean GI scores to track synergy discovery

---

## References

- **GEARS**: Roohani et al. (2023) - "Predicting transcriptional outcomes of novel multigene perturbations with GEARS"
- **scLAMBDA**: Rosen et al. (2024) - "Predicting transcriptional responses to novel chemical perturbations using deep generative model for drug discovery"
- **Norman et al.**: Norman et al. (2019) - "Exploring genetic interaction manifolds constructed from rich single-cell phenotypes"

---

## Citation

If you use this code, please cite the original model papers (GEARS, scLAMBDA) and the Norman dataset.

---

## License

Check individual model repositories for licensing information.
