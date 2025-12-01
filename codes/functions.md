## Function Reference

### `ensemble.py`

#### Class: `PerturbationDataProcessor`

Standardizes data format across different perturbation prediction models.

##### `__init__()`
- **Purpose**: Initialize processor
- **Attributes**:
  - `gene_names`: List of gene names
  - `gene_list`: Same as gene_names (for compatibility)
  - `pert_to_idx`: Dictionary mapping perturbations to indices

##### `load_norman_data(data_path)`
- **Purpose**: Load and preprocess Norman et al. CRISPR dataset
- **Input**:
  - `data_path` (str): Path to H5AD file
- **Output**:
  - `X_single` (np.ndarray): Binary perturbation matrix for singles
  - `y_single` (np.ndarray): Expression profiles for singles
  - `X_combo` (np.ndarray): Binary perturbation matrix for combos
  - `y_combo` (np.ndarray): Expression profiles for combos
  - `gene_names` (list): Gene names
- **Process**:
  1. Reads AnnData object from H5AD
  2. Filters samples by `num_targets` (1 or 2)
  3. Parses `condition` column to create binary matrices
  4. Returns separate arrays for singles and combos

##### `split_by_perturbation(X, y, train_ratio, val_ratio, test_ratio, random_state)`
- **Purpose**: Split data by unique perturbation combinations
- **Input**:
  - `X` (np.ndarray): Perturbation matrix
  - `y` (np.ndarray): Expression matrix
  - `train_ratio` (float): Train split ratio (default: 0.7)
  - `val_ratio` (float): Validation split ratio (default: 0.15)
  - `test_ratio` (float): Test split ratio (default: 0.15)
  - `random_state` (int): Random seed
- **Output**: Dictionary containing:
  - `X_train, y_train`: Training data
  - `X_val, y_val`: Validation data
  - `X_test, y_test`: Test data
  - `train_idx, val_idx, test_idx`: Sample indices
  - `train_perts, val_perts, test_perts`: Perturbation sets
  - `split_type`: 'perturbation'
- **Key Feature**: Tests ability to predict unseen gene combinations

##### `create_combo_splits(X_single, y_single, X_combo, y_combo, combo_test_ratio, random_state)`
- **Purpose**: GEARS-style split (train on singles + some combos, test on held-out combos)
- **Input**:
  - `X_single, y_single`: Single perturbation data
  - `X_combo, y_combo`: Combo perturbation data
  - `combo_test_ratio` (float): Fraction of combos to hold out (default: 0.2)
  - `random_state` (int): Random seed
- **Output**: Dictionary containing:
  - `X_train, y_train`: All singles + training combos
  - `X_val, y_val`: Validation combos
  - `X_test, y_test`: Test combos
  - `n_singles_in_train`: Number of single perturbations
  - `combo_train_idx, combo_val_idx, combo_test_idx`: Indices
  - `split_type`: 'combo_holdout'
- **Use Case**: Simulates realistic scenario where you train on singles first

##### `_create_perturbation_matrix(adata_subset, n_genes)`
- **Purpose**: Internal method to create binary perturbation matrix
- **Input**:
  - `adata_subset`: AnnData subset
  - `n_genes` (int): Number of genes
- **Output**: Binary matrix (n_samples × n_genes)
- **Process**: Parses condition strings like "CBL+CNN1" and sets corresponding indices to 1

---

#### Class: `Ensemble`

Main ensemble model combining GEARS, scLAMBDA, and baselines.

##### `__init__(sclambda_repo_path)`
- **Purpose**: Initialize ensemble
- **Input**:
  - `sclambda_repo_path` (str): Path to cloned scLAMBDA repository
- **Attributes**:
  - `gears_model`: GEARSWrapper instance
  - `sclambda_model`: scLAMBDAWrapper instance
  - `baseline_models`: Dictionary of baseline models
  - `data_processor`: PerturbationDataProcessor instance
  - `gene_effects`: Single perturbation effects matrix
  - `fitted`: Boolean flag

##### `load_models(gears_model_dir, gears_data_path, sclambda_model_path, sclambda_adata_path, sclambda_embeddings_path, norman_data_path, gears_data_name)`
- **Purpose**: Load all pre-trained models and fit baselines
- **Input**:
  - `gears_model_dir` (str): Directory with GEARS model.pt and config.pkl
  - `gears_data_path` (str): GEARS data directory
  - `gears_data_name` (str): Dataset name (e.g., 'norman')
  - `sclambda_model_path` (str): scLAMBDA checkpoint directory
  - `sclambda_adata_path` (str): AnnData path for scLAMBDA
  - `sclambda_embeddings_path` (str): Gene embeddings pickle
  - `norman_data_path` (str): Norman H5AD for baseline fitting
- **Process**:
  1. Loads GEARS pre-trained model
  2. Loads scLAMBDA pre-trained model
  3. Loads Norman data
  4. Fits baseline models (mean and additive)
  5. Sets `fitted=True`

##### `_fit_baselines()`
- **Purpose**: Internal method to fit baseline models from single perturbations
- **Process**:
  1. For each gene, compute average expression change across all samples where that gene is perturbed
  2. Store in `gene_effects` matrix (n_genes_pert × n_genes_expr)
  3. Create mean and additive baseline models
- **Models**:
  - **Mean baseline**: Average effect per perturbed gene
  - **Additive baseline**: Sum of individual gene effects

##### `predict_baseline(X, model)`
- **Purpose**: Predict using baseline models
- **Input**:
  - `X` (np.ndarray): Binary perturbation matrix
  - `model` (str): 'additive' or 'mean'
- **Output**: Predicted expression matrix
- **Math**:
  - Additive: `y_pred = X @ gene_effects`
  - Mean: `y_pred = (X @ gene_effects) / n_perturbed_genes`

##### `predict_ensemble(X_test)`
- **Purpose**: Generate ensemble predictions with uncertainty quantification
- **Input**:
  - `X_test` (np.ndarray): Binary perturbation matrix (n_samples × n_genes)
- **Output**: Tuple of:
  - `ensemble_mean` (np.ndarray): Mean prediction across models
  - `epistemic_uncertainty` (np.ndarray): Variance across models
  - `predictions` (dict): Individual model predictions
    - `'gears'`: GEARS predictions
    - `'sclambda'`: scLAMBDA predictions
    - `'mean'`: Mean baseline predictions
    - `'additive'`: Additive baseline predictions
- **Process**:
  1. Convert binary vectors to gene lists
  2. Get GEARS predictions (with fallback for unknown genes)
  3. Get scLAMBDA predictions
  4. Get baseline predictions
  5. Stack predictions (4 × n_samples × n_genes)
  6. Compute mean and variance across model axis
- **Error Handling**: Falls back to baseline for genes unknown to GEARS

##### `recommend_experiments(candidate_perturbations, n_recommend)`
- **Purpose**: Recommend experiments based on epistemic uncertainty
- **Input**:
  - `candidate_perturbations` (list): List of (gene1, gene2) tuples
  - `n_recommend` (int): Number of experiments to recommend
- **Output**: List of (gene1, gene2, uncertainty_score) tuples, ranked by uncertainty
- **Process**:
  1. Convert gene pairs to binary matrix
  2. Get ensemble predictions and uncertainties
  3. Compute total uncertainty (sum across genes)
  4. Sort by uncertainty descending
  5. Return top-N

##### `_binary_to_perturbation_str(x)`
- **Purpose**: Convert binary vector to perturbation string
- **Input**: Binary vector (n_genes,)
- **Output**: String like 'CBL+CNN1' or 'BRAF+ctrl'
- **Note**: Single perturbations get '+ctrl' appended

##### `_binary_to_gene_list(x)`
- **Purpose**: Convert binary vector to gene list
- **Input**: Binary vector (n_genes,)
- **Output**: List of gene names like ['CBL', 'CNN1']

---

### `active_learning.py`

#### Class: `GeneticInteractionScorer`

Computes genetic interaction (GI) scores using GEARS method.

##### `__init__(ensemble)`
- **Purpose**: Initialize GI scorer
- **Input**:
  - `ensemble`: Ensemble instance
- **Attributes**:
  - `ensemble`: Reference to ensemble
  - `gene_names`: Gene names from ensemble
  - `gene_effects`: Single perturbation effects from ensemble

##### `gears_gi_score(y_ctrl, y_A, y_B, y_AB)`
- **Purpose**: Compute GEARS-style genetic interaction score
- **Input**:
  - `y_ctrl` (np.ndarray): Control expression
  - `y_A` (np.ndarray): Gene A perturbation expression
  - `y_B` (np.ndarray): Gene B perturbation expression
  - `y_AB` (np.ndarray): Gene A+B combo perturbation expression
- **Output**: GI score vector (n_genes,)
- **Formula**:
  ```
  Δ_A = y_A - y_ctrl
  Δ_B = y_B - y_ctrl
  expected_AB = y_ctrl + Δ_A + Δ_B
  GI = y_AB - expected_AB
  ```
- **Interpretation**:
  - **Positive GI**: Synergistic (combo effect > sum of singles)
  - **Negative GI**: Antagonistic (buffering/suppression)
  - **Zero GI**: Additive (no interaction)

##### `compute_gi_for_candidates(candidates)`
- **Purpose**: Batch compute GI scores for candidate gene pairs
- **Input**:
  - `candidates` (list): List of (gene1, gene2) tuples
- **Output**: GI magnitude scores (n_candidates,)
- **Process**:
  1. For each candidate pair:
     - Get single perturbation effects from `gene_effects`
     - Predict combo effect using ensemble
     - Compute GI score
     - Take L2 norm (magnitude)
  2. Return array of GI magnitudes

##### `classify_interaction(y_ctrl, y_A, y_B, y_AB, threshold)`
- **Purpose**: Classify interaction type
- **Input**:
  - Same as `gears_gi_score`
  - `threshold` (float): Classification threshold (default: 0.1)
- **Output**: String: 'synergistic', 'antagonistic', or 'additive'

---

#### Class: `SHAPAnalyzer`

SHAP-based interpretability for perturbation predictions.

##### `__init__(ensemble)`
- **Purpose**: Initialize SHAP analyzer
- **Input**:
  - `ensemble`: Ensemble instance
- **Attributes**:
  - `ensemble`: Reference to ensemble
  - `gene_names`: Gene names
  - `explainers`: Dictionary of SHAP explainers
  - `shap_values_cache`: Cache for computed SHAP values

##### `_create_uncertainty_function()`
- **Purpose**: Create function that returns uncertainty given perturbation
- **Output**: Function `predict_uncertainty(X)` that returns uncertainty scores
- **Used by**: SHAP KernelExplainer

##### `_create_gi_function(gi_scorer)`
- **Purpose**: Create function that returns GI score given perturbation
- **Input**:
  - `gi_scorer`: GeneticInteractionScorer instance
- **Output**: Function `predict_gi(X)` that returns GI scores
- **Used by**: SHAP KernelExplainer

##### `explain_uncertainty(X_samples, n_background, n_explain)`
- **Purpose**: Identify genes that drive model uncertainty
- **Input**:
  - `X_samples` (np.ndarray): Sample perturbations to explain
  - `n_background` (int): Number of background samples for SHAP (default: 100)
  - `n_explain` (int): Number of samples to explain (default: 50)
- **Output**: Dictionary containing:
  - `shap_values`: SHAP values (n_explain × n_genes)
  - `gene_importance`: Mean absolute SHAP values per gene
  - `top_genes`: List of (gene_name, importance) tuples
  - `X_explain`: Explained samples
- **Interpretation**: High SHAP value = perturbing this gene causes high uncertainty

##### `explain_synergy(X_samples, gi_scorer, n_background, n_explain)`
- **Purpose**: Identify epistasis hub genes
- **Input**:
  - Same as `explain_uncertainty` plus `gi_scorer`
- **Output**: Same structure as `explain_uncertainty`
- **Interpretation**: High SHAP value = perturbing this gene causes high GI scores

##### `plot_uncertainty_summary(save_path)`
- **Purpose**: Generate SHAP summary plot for uncertainty
- **Input**:
  - `save_path` (str): Path to save plot
- **Output**: PNG file with SHAP summary plot
- **Note**: Must run `explain_uncertainty()` first

##### `plot_synergy_summary(save_path)`
- **Purpose**: Generate SHAP summary plot for synergy
- **Input**:
  - `save_path` (str): Path to save plot
- **Output**: PNG file with SHAP summary plot
- **Note**: Must run `explain_synergy()` first

##### `plot_gene_comparison(top_n, save_path)`
- **Purpose**: Compare uncertainty vs. synergy importance for top genes
- **Input**:
  - `top_n` (int): Number of top genes to display (default: 15)
  - `save_path` (str): Path to save plot
- **Output**: Horizontal bar chart comparing both importances
- **Insight**: Identifies genes that are both uncertain AND synergistic

---

#### Class: `ActiveLearningLoop`

Main active learning framework with multiple acquisition strategies.

##### `__init__(ensemble)`
- **Purpose**: Initialize active learning loop
- **Input**:
  - `ensemble`: Ensemble instance
- **Attributes**:
  - `ensemble`: Reference to ensemble
  - `gi_scorer`: GeneticInteractionScorer instance
  - `shap_analyzer`: SHAPAnalyzer instance
  - `history`: Dictionary tracking metrics across iterations

##### `get_candidate_pool(exclude_seen)`
- **Purpose**: Generate all unseen perturbation pairs
- **Input**:
  - `exclude_seen` (set): Set of (gene1, gene2) tuples to exclude
- **Output**: List of (gene1, gene2) tuples
- **Process**: Generate all pairwise combinations excluding already-tested pairs

##### `get_seen_perturbations(X)`
- **Purpose**: Extract seen perturbation pairs from training data
- **Input**:
  - `X` (np.ndarray): Training perturbation matrix
- **Output**: Set of (gene1, gene2) tuples (sorted)

##### `_candidates_to_matrix(candidates)`
- **Purpose**: Convert candidate gene pairs to perturbation matrix
- **Input**:
  - `candidates` (list): List of (gene1, gene2) tuples
- **Output**: Binary matrix (n_candidates × n_genes)

##### `compute_uncertainty_scores(candidates)`
- **Purpose**: Compute epistemic uncertainty scores
- **Input**:
  - `candidates` (list): Candidate gene pairs
- **Output**: Normalized uncertainty scores (0-1)
- **Method**: Sum of variance across genes, min-max normalized

##### `compute_synergy_scores(candidates)`
- **Purpose**: Compute GI-based synergy scores
- **Input**:
  - `candidates` (list): Candidate gene pairs
- **Output**: Normalized synergy scores (0-1)
- **Method**: GI magnitude (L2 norm), min-max normalized

##### `compute_diversity_scores(candidates, already_selected, X_train)`
- **Purpose**: Compute diversity scores (prefer underrepresented genes)
- **Input**:
  - `candidates` (list): Candidate gene pairs
  - `already_selected` (list): Already selected in this iteration
  - `X_train` (np.ndarray): Current training data
- **Output**: Normalized diversity scores (0-1)
- **Formula**: `score = 1 / (1 + gene_frequency)`
- **Method**: Prefers genes that haven't been tested much

##### `compute_oracle_scores(candidates, ground_truth)`
- **Purpose**: Oracle scores using ground truth (upper bound)
- **Input**:
  - `candidates` (list): Candidate gene pairs
  - `ground_truth` (dict): Dictionary mapping (gene1, gene2) → expression
- **Output**: Normalized error scores (0-1)
- **Method**: Selects pairs with highest actual prediction error
- **Use Case**: Establishes upper bound on active learning performance

##### `select_next_experiments(candidates, n_select, strategy, alpha, beta, gamma, X_train, ground_truth)`
- **Purpose**: Select next experiments using specified strategy
- **Input**:
  - `candidates` (list): Candidate gene pairs
  - `n_select` (int): Number to select
  - `strategy` (str): Strategy name
  - `alpha, beta, gamma` (float): Weights for weighted strategy
  - `X_train` (np.ndarray): Current training data
  - `ground_truth` (dict): Ground truth (for oracle only)
- **Output**: List of (gene1, gene2, score) tuples
- **Strategies**:
  - **'random'**: Uniform random sampling
  - **'uncertainty'**: Select highest uncertainty
  - **'synergy'**: Select highest GI scores
  - **'diversity'**: Greedy diversity selection
  - **'oracle'**: Select highest error (requires ground truth)
  - **'weighted'**: `α*uncertainty + β*synergy + γ*diversity`

##### `_greedy_diversity_selection(candidates, n_select, X_train)`
- **Purpose**: Greedy algorithm for diversity selection
- **Process**:
  1. Start with empty selection
  2. For each slot:
     - Compute diversity scores for remaining candidates
     - Select candidate with highest diversity
     - Update gene counts
     - Repeat
- **Ensures**: Maximum gene coverage

##### `evaluate(X_test, y_test)`
- **Purpose**: Evaluate current model on test set
- **Input**:
  - `X_test, y_test`: Test data
- **Output**: Dictionary containing:
  - `mse`: Mean squared error
  - `pearson`: Mean Pearson correlation across genes
  - `mean_uncertainty`: Mean epistemic uncertainty

##### `run_simulation(X_train_init, y_train_init, X_test, y_test, X_pool, y_pool, n_iterations, n_select_per_iter, strategy, alpha, beta, gamma)`
- **Purpose**: Run full active learning simulation
- **Input**:
  - `X_train_init, y_train_init`: Initial training set
  - `X_test, y_test`: Test set (fixed)
  - `X_pool, y_pool`: Pool of unlabeled data (ground truth for simulation)
  - `n_iterations` (int): Number of AL iterations
  - `n_select_per_iter` (int): Experiments to select per iteration
  - `strategy` (str): Selection strategy
  - `alpha, beta, gamma` (float): Weights for weighted strategy
- **Output**: History dictionary with iteration metrics
- **Process**: See "Detailed Workflow" section above

##### `run_shap_analysis(X_combo, n_background, n_explain)`
- **Purpose**: Run SHAP analysis for uncertainty and synergy
- **Input**:
  - `X_combo` (np.ndarray): Combo perturbations to analyze
  - `n_background` (int): Background samples for SHAP
  - `n_explain` (int): Samples to explain
- **Output**: Dictionary with:
  - `'uncertainty'`: Uncertainty SHAP results
  - `'synergy'`: Synergy SHAP results

##### `plot_shap_results(save_dir)`
- **Purpose**: Generate all SHAP plots
- **Input**:
  - `save_dir` (str): Directory to save plots
- **Output**: Three PNG files:
  - `shap_uncertainty.png`
  - `shap_synergy.png`
  - `shap_gene_comparison.png`

---

#### Utility Functions

##### `compare_all_strategies(ensemble, X_train_init, y_train_init, X_test, y_test, X_pool, y_pool, n_iterations, n_select, result_dir)`
- **Purpose**: Compare all selection strategies in parallel
- **Input**:
  - Ensemble and data splits
  - `n_iterations` (int): AL iterations
  - `n_select` (int): Experiments per iteration
  - `result_dir` (str): Directory to save results
- **Output**: Dictionary with results for each strategy
- **Strategies Tested**:
  - random
  - uncertainty
  - synergy
  - diversity
  - oracle
  - weighted_unc_syn (α=0.5, β=0.5)
  - weighted_all (α=0.4, β=0.3, γ=0.3)

##### `save_results(results, result_dir)`
- **Purpose**: Save results to JSON and pickle
- **Input**:
  - `results` (dict): Results from `compare_all_strategies`
  - `result_dir` (str): Output directory
- **Output**:
  - `results.json`: Metrics only
  - `results.pkl`: Full results with selected perturbations

##### `plot_strategy_comparison(results, save_path)`
- **Purpose**: Generate 2×2 comparison plot
- **Input**:
  - `results` (dict): Results dictionary
  - `save_path` (str): Path to save figure
- **Output**: PNG with 4 subplots:
  1. MSE vs. training size
  2. Pearson r vs. training size
  3. Uncertainty vs. iteration
  4. GI score vs. iteration

##### `print_summary_table(results, save_path)`
- **Purpose**: Print and save summary table
- **Input**:
  - `results` (dict): Results dictionary
  - `save_path` (str): Path to save text file (optional)
- **Output**: Table showing final MSE, Pearson r, and improvement vs. random