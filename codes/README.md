# Combinatorial Perturbation Prediction & Active Learning

This README.md details how to run the model.

## Installation & Setup

### Requirements

```bash
pip install numpy pandas scanpy torch matplotlib shap
```

### Dependencies

- **numpy** (>= 1.23.5)
- **scanpy** (>= 1.11.5)
- **torch** (>= 2.9.1)
- **torch-geometric** (>= 2.7.0)
- **shap** (>= 0.49.1)
- **cell-gears** (0.1.2)
- scLAMBDA repo

### Pre-trained Models

You need to have the following pre-trained models:

1. **GEARS model**: Trained on norman dataset
   - `model.pt` and `config.pkl` in model directory
   - GEARS data in `gears_data/` directory

2. **scLAMBDA model**: Trained on preprocessed data
   - `ckpt.pth` checkpoint file
   - Gene embeddings pickle file
   - Preprocessed AnnData file

3. **Norman dataset**: For baseline training
   - `norman_perturbseq_preprocessed_hvg_filtered.h5ad`

---

## Quick Start

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

# Predict with uncertainty
pred_mean, uncertainties, individual_preds = ensemble.predict_ensemble(X_test)

print(f"Ensemble MSE: {np.mean((pred_mean - y_test)**2):.4f}")
print(f"Mean uncertainty: {np.mean(uncertainties):.4f}")
```

### Run Active Learning

```python
from active_learning import ActiveLearningLoop

# Initialize active learning
al = ActiveLearningLoop(ensemble)

# Run simulation with weighted strategy
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
```
---

## Usage Examples

### Example 1: Basic Ensemble Prediction

```python
from ensemble import Ensemble
import numpy as np

# Initialize
ensemble = Ensemble(sclambda_repo_path='/path/to/scLAMBDA')

# Load models
ensemble.load_models(
    gears_model_dir='./gears_model',
    gears_data_path='./gears_data',
    gears_data_name='norman',
    sclambda_model_path='./sclambda_model',
    sclambda_adata_path='./data.h5ad',
    sclambda_embeddings_path='./embeddings.pkl',
    norman_data_path='./norman_data.h5ad'
)

# Create splits
splits = ensemble.data_processor.create_combo_splits(
    X_single=ensemble.X_single,
    y_single=ensemble.y_single,
    X_combo=ensemble.X_combo,
    y_combo=ensemble.y_combo,
    combo_test_ratio=0.2
)

# Predict
pred_mean, uncertainties, individual_preds = ensemble.predict_ensemble(splits['X_test'])

# Evaluate
mse = np.mean((pred_mean - splits['y_test']) ** 2)
print(f"Ensemble MSE: {mse:.4f}")

# Individual model performance
for model_name, preds in individual_preds.items():
    mse_model = np.mean((preds - splits['y_test']) ** 2)
    print(f"{model_name} MSE: {mse_model:.4f}")
```

### Example 2: Experiment Recommendation

```python
# Get candidate gene pairs (all pairwise combos)
candidates = []
for i, gene1 in enumerate(ensemble.gene_names[:50]):
    for gene2 in ensemble.gene_names[i+1:50]:
        candidates.append((gene1, gene2))

# Recommend experiments
recommendations = ensemble.recommend_experiments(candidates, n_recommend=10)

print("Top 10 Recommended Experiments:")
for i, (gene1, gene2, score) in enumerate(recommendations):
    print(f"{i+1}. {gene1} × {gene2} (uncertainty: {score:.4f})")
```

### Example 3: Active Learning with Uncertainty Strategy

```python
from active_learning import ActiveLearningLoop

# Initialize
al = ActiveLearningLoop(ensemble)

# Create initial/pool split
n_init = int(len(splits['X_train']) * 0.2)
X_train_init = splits['X_train'][:n_init]
y_train_init = splits['y_train'][:n_init]
X_pool = splits['X_train'][n_init:]
y_pool = splits['y_train'][n_init:]

# Run simulation
history = al.run_simulation(
    X_train_init=X_train_init,
    y_train_init=y_train_init,
    X_test=splits['X_test'],
    y_test=splits['y_test'],
    X_pool=X_pool,
    y_pool=y_pool,
    n_iterations=15,
    n_select_per_iter=10,
    strategy='uncertainty'
)

# Plot learning curve
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(history['n_training_samples'], history['test_mse'], marker='o')
plt.xlabel('Number of Training Samples')
plt.ylabel('Test MSE')
plt.title('Active Learning: Uncertainty Strategy')
plt.grid(True)
plt.savefig('learning_curve.png')
```

### Example 4: Weighted Strategy (Balanced)

```python
# Run with weighted strategy
history = al.run_simulation(
    X_train_init=X_train_init.copy(),
    y_train_init=y_train_init.copy(),
    X_test=splits['X_test'],
    y_test=splits['y_test'],
    X_pool=X_pool,
    y_pool=y_pool,
    n_iterations=15,
    n_select_per_iter=10,
    strategy='weighted',
    alpha=0.4,   # 40% uncertainty
    beta=0.3,    # 30% synergy
    gamma=0.3    # 30% diversity
)

print(f"Final MSE: {history['test_mse'][-1]:.4f}")
print(f"Final Pearson r: {history['test_pearson'][-1]:.4f}")
```

### Example 5: Strategy Comparison

```python
from active_learning import compare_all_strategies, plot_strategy_comparison, print_summary_table

# Compare all strategies
results = compare_all_strategies(
    ensemble=ensemble,
    X_train_init=X_train_init,
    y_train_init=y_train_init,
    X_test=splits['X_test'],
    y_test=splits['y_test'],
    X_pool=X_pool,
    y_pool=y_pool,
    n_iterations=15,
    n_select=10,
    result_dir='./results'
)

# Generate comparison plot
plot_strategy_comparison(results, save_path='./results/comparison.png')

# Print summary
print_summary_table(results, save_path='./results/summary.txt')
```

### Example 6: SHAP Interpretability Analysis

```python
from active_learning import ActiveLearningLoop

# Initialize
al = ActiveLearningLoop(ensemble)

# Run SHAP analysis
shap_results = al.run_shap_analysis(
    X_combo=ensemble.X_combo,
    n_background=100,
    n_explain=50
)

# Plot results
al.plot_shap_results(save_dir='./shap_results')

# Access top genes
print("\nTop 5 Uncertainty Drivers:")
for gene, importance in shap_results['uncertainty']['top_genes'][:5]:
    print(f"  {gene}: {importance:.4f}")

print("\nTop 5 Epistasis Hub Genes:")
for gene, importance in shap_results['synergy']['top_genes'][:5]:
    print(f"  {gene}: {importance:.4f}")
```

### Example 7: Genetic Interaction Analysis

```python
from active_learning import GeneticInteractionScorer

# Initialize
gi_scorer = GeneticInteractionScorer(ensemble)

# Analyze specific gene pairs
test_pairs = [
    ('BRAF', 'KRAS'),
    ('TP53', 'MYC'),
    ('EGFR', 'PTEN')
]

gi_scores = gi_scorer.compute_gi_for_candidates(test_pairs)

print("Genetic Interaction Analysis:")
for (g1, g2), score in zip(test_pairs, gi_scores):
    print(f"{g1} × {g2}: GI magnitude = {score:.4f}")
```

### Example 8: Custom Active Learning Strategy

```python
# Define custom selection function
def custom_strategy(al, candidates, n_select, X_train):
    """
    Custom strategy: 70% uncertainty, 30% synergy
    """
    unc_scores = al.compute_uncertainty_scores(candidates)
    syn_scores = al.compute_synergy_scores(candidates)

    # Weighted combination
    combined_scores = 0.7 * unc_scores + 0.3 * syn_scores

    # Select top-N
    top_indices = np.argsort(combined_scores)[-n_select:][::-1]
    return [(candidates[i][0], candidates[i][1], combined_scores[i])
            for i in top_indices]

# Run simulation with custom strategy
history = al.run_simulation(
    X_train_init=X_train_init,
    y_train_init=y_train_init,
    X_test=splits['X_test'],
    y_test=splits['y_test'],
    X_pool=X_pool,
    y_pool=y_pool,
    n_iterations=15,
    n_select_per_iter=10,
    strategy='weighted',
    alpha=0.7,
    beta=0.3,
    gamma=0.0
)
```