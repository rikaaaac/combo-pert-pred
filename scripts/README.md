# scripts/

Ensemble implementation for combinatorial perturbation prediction using Additive + VAE models.

## Files

### `ensemble_add_vae.py`

**Purpose:** Core ensemble implementation combining additive baseline and VAE for predicting combinatorial gene perturbation effects.

**Models:**
- **Additive Baseline:** Linear superposition of single-gene effects (gene_AB = gene_A + gene_B)
- **VAE:** Variational autoencoder capturing nonlinear epistatic interactions
- **Ensemble:** Weighted combination with learned weights

**Key Functions:**
```python
load_norman_data(data_path)           # Load and preprocess Norman 2019 dataset
create_splits(X_single, y_single, X_combo, y_combo)  # Create train/val/test splits
fit_additive(X_single, y_single)      # Fit additive baseline
VAEWrapper(dim, device)               # VAE training and prediction
AdditiveVAEEnsemble(effects, vae)     # Ensemble with uncertainty quantification
```

**Usage:**
```bash
python ensemble_add_vae.py
```

**Outputs:**
- `additive_vae_results.pkl` - Model weights, test MSE, epistemic uncertainty, conformal intervals

**Requirements:**
- PyTorch
- scanpy, numpy, scikit-learn
- Norman dataset: `adata_norman_preprocessed.h5ad`

**Performance:**
- VAE: MSE=0.037, R²=0.80, r=0.90
- Additive: MSE=0.187, R²=0.01, r=0.90
- Ensemble: MSE=0.118 (weights: 0.37 additive, 0.63 VAE)

---

### `active_learning.py`

**Purpose:** Active learning loop for selecting informative experiments using acquisition strategies.

**Acquisition Strategies:**
1. **Random:** Uniform sampling (baseline)
2. **Uncertainty:** Epistemic uncertainty from additive-VAE disagreement
3. **Synergy:** GEARS-style genetic interaction scores
4. **Diversity:** Expression-space diversity
5. **Weighted:** Linear combination (α=0.5 uncertainty, β=0.5 synergy)

**Key Classes:**
```python
ActiveLearning(ensemble, gene_names)  # Main AL loop
  .compute_uncertainty_scores()       # Model disagreement
  .compute_synergy_scores()           # GI scores
  .compute_diversity_scores()         # Expression distance
  .run_simulation()                   # Run AL iterations
```

**Usage:**
```bash
python active_learning.py
```

**Configuration (in main()):**
- Initial training: 5% of combinations
- Pool: 95% 
- Iterations: 5
- Batch size: 15 samples per iteration
- VAE epochs: 50 initial, 30 per iteration

**Outputs:**
- `results.pkl` - Full results for all strategies
- `strategy_comparison.png` - MSE and Pearson r curves
- `summary.txt` - Final comparison table

**Requirements:**
- All dependencies from `ensemble_add_vae.py`
- matplotlib

**Expected Behavior:**
With proper training (50+ VAE epochs), uncertainty/weighted strategies should match or outperform random baseline. Low correlations (r<0.1) indicate insufficient VAE training.

---

### `ensemble_analyze.py`

**Purpose:** Post-hoc analysis and visualization of ensemble predictions and uncertainty.

**Analysis Modules:**
1. **Model Comparison:** MSE, MAE, R², Pearson r across models
2. **Uncertainty Calibration:** Correlation between epistemic uncertainty and prediction error
3. **Gene-Level Analysis:** Top genes by epistemic uncertainty
4. **Diversity Analysis:** Model prediction correlation heatmap

**Key Functions:**
```python
plot_model_comparison()               # Bar plots of metrics
plot_uncertainty_vs_error()           # Scatter plot (r=0.64)
plot_gene_uncertainty()               # Top 30 genes
plot_uncertainty_distribution()       # Histogram
plot_diversity_heatmap()              # Correlation matrix
```

**Usage:**
```bash
python ensemble_analyze.py
```

**Inputs:**
- Trained ensemble from `ensemble_add_vae.py`
- Test set predictions

**Outputs (to `ensemble_results/`):**
- `model_comparison.png` - Performance metrics
- `uncertainty_vs_error.png` - Calibration plot
- `gene_uncertainty.png` - Top uncertain genes
- `uncertainty_distribution.png` - Distribution histogram
- `diversity_heatmap.png` - Model correlation

**Key Findings:**
- Mitochondrial genes (MT-CO3, MALAT1) have highest uncertainty
- Epistemic uncertainty correlates with error (r=0.64)
- Additive-VAE predictions highly correlated (r=0.99)

---

## Workflow

**1. Train Ensemble:**
```bash
python ensemble_add_vae.py
```
→ Produces `additive_vae_results.pkl`

**2. Analyze Results:**
```bash
python ensemble_analyze.py
```
→ Produces visualizations in `ensemble_results/`

**3. Run Active Learning:**
```bash
python active_learning.py
```
→ Compares acquisition strategies

