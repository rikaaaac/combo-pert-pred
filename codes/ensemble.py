import numpy as np
import pandas as pd
import scanpy as sc
import torch
import sys
import os
import warnings
warnings.filterwarnings('ignore')
from scipy.optimize import nnls
from sklearn.linear_model import Ridge

from ensemble_full_pipeline import GEARSWrapper, scLAMBDAWrapper

class PerturbationDataProcessor:
    """standardize data format for all models"""
    
    def __init__(self):
        self.gene_names = None
        self.gene_list = None
        self.pert_to_idx = {}

    def split_by_perturbation(self, X: np.ndarray, y: np.ndarray, 
                               train_ratio: float = 0.7,
                               val_ratio: float = 0.15,
                               test_ratio: float = 0.15,
                               random_state: int = 42) -> dict:
        """
        Split by unique perturbation combinations.
        Tests whether model can predict effects of unseen gene combinations.
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
        
        np.random.seed(random_state)
        
        # get unique perturbation patterns (as tuples for hashing)
        pert_patterns = [tuple(np.where(x > 0)[0]) for x in X]
        unique_perts = list(set(pert_patterns))
        
        n_perts = len(unique_perts)
        perm = np.random.permutation(n_perts)
        
        train_end = int(n_perts * train_ratio)
        val_end = int(n_perts * (train_ratio + val_ratio))
        
        train_perts = set([unique_perts[i] for i in perm[:train_end]])
        val_perts = set([unique_perts[i] for i in perm[train_end:val_end]])
        test_perts = set([unique_perts[i] for i in perm[val_end:]])
        
        # assign samples to splits based on their perturbation pattern
        train_idx, val_idx, test_idx = [], [], []
        for i, pattern in enumerate(pert_patterns):
            if pattern in train_perts:
                train_idx.append(i)
            elif pattern in val_perts:
                val_idx.append(i)
            else:
                test_idx.append(i)
        
        train_idx = np.array(train_idx)
        val_idx = np.array(val_idx)
        test_idx = np.array(test_idx)
        
        return {
            'X_train': X[train_idx], 'y_train': y[train_idx],
            'X_val': X[val_idx], 'y_val': y[val_idx],
            'X_test': X[test_idx], 'y_test': y[test_idx],
            'train_idx': train_idx, 'val_idx': val_idx, 'test_idx': test_idx,
            'train_perts': train_perts, 'val_perts': val_perts, 'test_perts': test_perts,
            'split_type': 'perturbation'
        }
    
    def create_combo_splits(self, X_single: np.ndarray, y_single: np.ndarray,
                            X_combo: np.ndarray, y_combo: np.ndarray,
                            combo_test_ratio: float = 0.2,
                            random_state: int = 42) -> dict:
        """
        GEARS-style split: train on singles + some combos, test on held-out combos.
        """
        np.random.seed(random_state)
        
        n_combo = X_combo.shape[0]
        combo_indices = np.random.permutation(n_combo)
        
        test_size = int(n_combo * combo_test_ratio)
        val_size = int(n_combo * combo_test_ratio)
        
        combo_test_idx = combo_indices[:test_size]
        combo_val_idx = combo_indices[test_size:test_size + val_size]
        combo_train_idx = combo_indices[test_size + val_size:]
        
        X_train = np.vstack([X_single, X_combo[combo_train_idx]])
        y_train = np.vstack([y_single, y_combo[combo_train_idx]])
        
        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_combo[combo_val_idx],
            'y_val': y_combo[combo_val_idx],
            'X_test': X_combo[combo_test_idx],
            'y_test': y_combo[combo_test_idx],
            'n_singles_in_train': len(X_single),
            'combo_train_idx': combo_train_idx,
            'combo_val_idx': combo_val_idx,
            'combo_test_idx': combo_test_idx,
            'split_type': 'combo_holdout'
        }
        
    def load_norman_data(self, data_path: str) -> tuple:
        """
        Load and preprocess Norman et al. data.
        returns:
            (X_single, y_single, X_combo, y_combo, gene_names)
        """
        adata = sc.read_h5ad(data_path)
        
        # Make sure we have a 'condition' column
        if 'perturbation' in adata.obs.columns and 'condition' not in adata.obs.columns:
            adata.obs['condition'] = adata.obs['perturbation']

        # Normalize perturbation names (same as done for scLAMBDA and GEARS)
        if 'condition' in adata.obs.columns:
            # 1. Replace underscore with '+' for multi-gene perturbations first
            adata.obs['condition'] = adata.obs['condition'].astype(str).str.replace('_', '+')
            # 2. Handle control conditions: normalize to 'ctrl' (case-insensitive)
            def normalize_control(s):
                parts = s.split('+')
                parts = ['ctrl' if p.lower() == 'control' else p for p in parts]
                return '+'.join(parts)
            adata.obs['condition'] = adata.obs['condition'].apply(normalize_control)
        
        # extract perturbation information
        adata.obs['num_targets'] = adata.obs['nperts']

        single_perts = adata[adata.obs['num_targets'] == 1].copy()
        combo_perts = adata[adata.obs['num_targets'] == 2].copy()
        
        self.gene_names = adata.var_names.tolist()
        self.gene_list = self.gene_names
        n_genes = len(self.gene_names)
        
        X_single = self._create_perturbation_matrix(single_perts, n_genes)
        y_single = single_perts.X.toarray() if hasattr(single_perts.X, 'toarray') else single_perts.X
        
        X_combo = self._create_perturbation_matrix(combo_perts, n_genes)
        y_combo = combo_perts.X.toarray() if hasattr(combo_perts.X, 'toarray') else combo_perts.X
        
        print(f"Loaded Norman data: {len(X_single)} singles, {len(X_combo)} combos")
        
        return X_single, y_single, X_combo, y_combo, self.gene_names
    
    def _create_perturbation_matrix(self, adata_subset, n_genes):
        """Create binary perturbation matrix from AnnData."""
        X = np.zeros((len(adata_subset), n_genes))

        gene_to_idx = {gene: idx for idx, gene in enumerate(self.gene_list)}

        for i, (_, row) in enumerate(adata_subset.obs.iterrows()):
            pert = row['condition']

            if pert == 'ctrl' or pert.lower() == 'control':
                continue

            genes = pert.split('+')
            for gene in genes:
                gene = gene.strip()
                if gene in gene_to_idx:
                    X[i, gene_to_idx[gene]] = 1

        return X

# ================================================================================
# conformal prediction
# ================================================================================

class ConformalPredictor:
    """
    Conformal prediction for calibrated uncertainty quantification.
    """

    def __init__(self, alpha: float = 0.1):
        """
        alpha = 0.1 for 90% coverage
        """
        self.alpha = alpha
        self.calibration_scores = None
        self.is_calibrated = False

    def calibrate(self, y_cal: np.ndarray, y_pred_cal: np.ndarray):
        """
        Calibrate using validation/calibration set.
        Args:
            y_cal: True values from calibration set (n_samples, n_genes)
            y_pred_cal: Predicted values from calibration set (n_samples, n_genes)
        """
        # Compute non-conformity scores (absolute residuals per gene)
        self.calibration_scores = np.abs(y_cal - y_pred_cal)
        self.is_calibrated = True

        print(f"Conformal predictor calibrated with {len(y_cal)} samples")
        print(f"Target coverage: {100*(1-self.alpha):.1f}%")

    def predict_interval(self, y_pred: np.ndarray) -> tuple:
        """
        Compute prediction intervals for new predictions.

        Args:
            y_pred: Point predictions (n_samples, n_genes)

        Returns:
            (lower_bound, upper_bound, interval_width)
        """
        if not self.is_calibrated:
            raise ValueError("Must call calibrate() before predict_interval()")

        # compute quantile of calibration scores for each gene
        n_cal = len(self.calibration_scores)
        q_level = np.ceil((n_cal + 1) * (1 - self.alpha)) / n_cal
        q_level = min(q_level, 1.0)  # ensure it doesn't exceed 1.0

        # compute gene-wise quantiles
        quantiles = np.quantile(self.calibration_scores, q_level, axis=0)

        # prediction intervals
        lower = y_pred - quantiles
        upper = y_pred + quantiles
        width = upper - lower

        return lower, upper, width

    def compute_coverage(self, y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> dict:
        """
        Compute empirical coverage of prediction intervals.
        Args:
            y_true: True values
            lower: Lower bounds of prediction intervals
            upper: Upper bounds of prediction intervals
        Returns:
            Dictionary with coverage statistics.
        """
        # check if true values fall within intervals
        covered = (y_true >= lower) & (y_true <= upper)

        # overall coverage (across all genes and samples)
        overall_coverage = np.mean(covered)

        # per-gene coverage
        gene_coverage = np.mean(covered, axis=0)

        # per-sample coverage
        sample_coverage = np.mean(covered, axis=1)

        return {
            'overall_coverage': overall_coverage,
            'target_coverage': 1 - self.alpha,
            'gene_coverage_mean': np.mean(gene_coverage),
            'gene_coverage_std': np.std(gene_coverage),
            'sample_coverage_mean': np.mean(sample_coverage),
            'sample_coverage_std': np.std(sample_coverage),
            'gene_coverage': gene_coverage,
            'sample_coverage': sample_coverage
        }


# ================================================================================
# ensemble
# ================================================================================

class Ensemble:
    """Ensemble with GEARS + scLAMBDA + baselines"""
    
    def __init__(self, sclambda_repo_path: str = None, conformal_alpha: float = 0.1):
        """
        args:
            sclambda_repo_path: Path to cloned scLAMBDA repo
            conformal_alpha: same as alpha above (0.1 for 90% coverage)
        """
        self.gears_model = GEARSWrapper()
        self.sclambda_model = scLAMBDAWrapper(sclambda_path=sclambda_repo_path)
        self.baseline_models = None
        self.data_processor = PerturbationDataProcessor()
        self.gene_effects = None
        self.fitted = False

        # Weighted ensemble components
        self.model_weights = None  # Learned weights for each model
        self.use_learned_weights = False

        # Conformal prediction
        self.conformal_predictor = ConformalPredictor(alpha=conformal_alpha)

        # Store validation data for weight learning and calibration
        self.val_predictions = None
        self.val_targets = None
        
    def load_models(self, 
                    gears_model_dir: str,
                    gears_data_path: str,
                    sclambda_model_path: str,
                    sclambda_adata_path: str,
                    sclambda_embeddings_path: str,
                    norman_data_path: str,
                    gears_data_name: str = None,
                    custom_gears_adata_path: str = None):
        """
        Load all pre-trained models.
        
        args:
            gears_model_dir: Directory where GEARS model is saved (e.g., './gears_model')
            gears_data_path: GEARS data directory (e.g., './data') - working directory for GEARS
            sclambda_model_path: Directory where scLAMBDA model is saved
            sclambda_adata_path: Path to adata for scLAMBDA
            sclambda_embeddings_path: Path to gene embeddings .pkl file
            norman_data_path: Path to Norman h5ad for baseline fitting
            gears_data_name: Built-in dataset name ('norman', 'adamson', 'dixit') or None for custom
            custom_gears_adata_path: Path to custom h5ad file for GEARS (required if gears_data_name is None)

        """
        print("Loading models...")
        
        # load GEARS
        self.gears_model.load_pretrained_gears(
            model_dir=gears_model_dir,
            data_path=gears_data_path,
            data_name=gears_data_name,
            custom_adata_path=custom_gears_adata_path
        )
        
        # load scLAMBDA
        self.sclambda_model.load_pretrained_sclambda(
            model_path=sclambda_model_path,
            adata_path=sclambda_adata_path,
            gene_embeddings_path=sclambda_embeddings_path
        )
        
        # load data for baselines
        self.X_single, self.y_single, self.X_combo, self.y_combo, self.gene_names = \
            self.data_processor.load_norman_data(norman_data_path)
            
        # NOTE: Do NOT call _fit_baselines() here to avoid data leakage
        # Baselines should be fitted on training data only, after creating splits
        # Call _fit_baselines(X_single_train, y_single_train) separately with training data
        
        self.fitted = True
        print("All models loaded!")
        print("NOTE: Baselines not fitted yet. Call _fit_baselines(X_single_train, y_single_train) with training data to avoid data leakage.")

        
    def _fit_baselines(self, X_single_train=None, y_single_train=None):
        """
        Fit additive and mean baseline models.

        IMPORTANT: Should only be called with TRAINING data to avoid data leakage.
        If X_single_train and y_single_train are not provided, uses self.X_single and self.y_single.
        """
        self.baseline_models = {}

        # Use provided training data or fall back to all data (for backward compatibility)
        if X_single_train is not None and y_single_train is not None:
            X_train = X_single_train
            y_train = y_single_train
            print(f"Fitting baselines on {len(X_train)} TRAINING singles (no data leakage)")
        else:
            X_train = self.X_single
            y_train = self.y_single
            print(f"WARNING: Fitting baselines on ALL {len(X_train)} singles (potential data leakage)")

        n_genes_pert = X_train.shape[1]
        n_genes_expr = y_train.shape[1]

        self.gene_effects = np.zeros((n_genes_pert, n_genes_expr))

        for gene_idx in range(n_genes_pert):
            mask = X_train[:, gene_idx] == 1
            if mask.sum() > 0:
                self.gene_effects[gene_idx] = np.mean(y_train[mask], axis=0)

        # Store separately to make it clear these are the same effects
        # but used differently in predict_baseline()
        self.baseline_models['mean'] = self.gene_effects
        self.baseline_models['additive'] = self.gene_effects

        print(f"Baselines fitted: {(self.gene_effects != 0).any(axis=1).sum()} genes have learned effects")
    
    def predict_baseline(self, X, model='additive'):
        """Predict using baseline models."""
        if self.gene_effects is None:
            raise ValueError("Baselines must be fitted first")
        
        if model == 'additive':
            return X @ self.gene_effects
        elif model == 'mean':
            n_perts = X.sum(axis=1, keepdims=True)
            n_perts = np.where(n_perts == 0, 1, n_perts)
            return (X @ self.gene_effects) / n_perts
        else:
            raise ValueError(f"Unknown model: {model}")
    
    def _binary_to_perturbation_str(self, x: np.ndarray) -> str:
        """Convert binary vector to perturbation string like 'CBL+CNN1'."""
        gene_indices = np.where(x > 0)[0]
        genes = [self.gene_names[i] for i in gene_indices]
        if len(genes) == 0:
            return 'ctrl'
        elif len(genes) == 1:
            return genes[0] + '+ctrl'
        return '+'.join(genes)
    
    def _binary_to_gene_list(self, x: np.ndarray) -> list:
        """Convert binary vector to gene list like ['CBL', 'CNN1']."""
        gene_indices = np.where(x > 0)[0]
        return [self.gene_names[i] for i in gene_indices]

    def learn_ensemble_weights(self, X_val: np.ndarray, y_val: np.ndarray,
                               method: str = 'nnls', normalize: bool = True,
                               diversity_penalty: float = 0.0):
        """
        learn weights using validation data.

        Args:
            X_val: Validation input features (n_samples, n_genes_pert)
            y_val: Validation targets (n_samples, n_genes_expr)
            method: 'nnls' (non-negative least squares), 'ridge', 'uniform', or 'diverse_nnls'
            normalize: Whether to normalize weights to sum to 1
            diversity_penalty: FIX 5 - Penalty for model similarity (0.0 = no penalty, 0.1 = moderate, 0.5 = strong)

        Returns:
            Learned weights for [gears, sclambda, mean, additive]
        """
        print(f"\nLearning ensemble weights using {method}...")

        # get predictions from all models on validation set
        predictions = {}
        n_samples = X_val.shape[0]
        n_genes_expr = y_val.shape[1]

        # GEARS predictions with improved error handling
        gears_known_genes = set(self.gears_model.pert_data.pert_list) if hasattr(self.gears_model.pert_data, 'pert_list') else set()
        pert_lists = [self._binary_to_gene_list(x) for x in X_val]
        gears_preds = np.zeros((n_samples, n_genes_expr))

        n_gears_success = 0
        n_gears_fallback_sclambda = 0
        n_gears_fallback_baseline = 0
        first_result_logged = False

        for i, pert_list in enumerate(pert_lists):
            if all(gene in gears_known_genes for gene in pert_list) and len(pert_list) > 0:
                try:
                    result = self.gears_model.batch_predict([pert_list])

                    # Debug: log first result
                    if not first_result_logged:
                        print(f"DEBUG: GEARS result type: {type(result)}")
                        if isinstance(result, dict):
                            print(f"DEBUG: GEARS result keys: {result.keys()}")
                        first_result_logged = True

                    # Handle different possible return structures from GEARS
                    if isinstance(result, dict):
                        pred = result.get('pred', None)
                        if pred is not None:
                            if isinstance(pred, (list, np.ndarray)):
                                pred_array = np.array(pred)
                                if pred_array.ndim > 1:
                                    gears_preds[i] = pred_array[0]
                                else:
                                    gears_preds[i] = pred_array
                                n_gears_success += 1
                            else:
                                raise ValueError(f"Unexpected prediction format: {type(pred)}")
                        else:
                            if 'prediction' in result:
                                pred_array = np.array(result['prediction'])
                                gears_preds[i] = pred_array[0] if pred_array.ndim > 1 else pred_array
                                n_gears_success += 1
                            else:
                                raise ValueError(f"GEARS returned dict without 'pred' key. Keys: {list(result.keys())}")
                    elif isinstance(result, (list, np.ndarray)):
                        pred_array = np.array(result)
                        if pred_array.ndim > 1:
                            gears_preds[i] = pred_array[0]
                        else:
                            gears_preds[i] = pred_array
                        n_gears_success += 1
                    else:
                        raise ValueError(f"Unexpected GEARS return type: {type(result)}")
                except Exception as e:
                    # FIX #4: Use scLAMBDA as fallback to preserve diversity
                    if n_gears_fallback_sclambda + n_gears_fallback_baseline < 3:
                        print(f"Warning: GEARS prediction failed for {pert_list}: {e}")

                    # Try scLAMBDA first (preserves diversity)
                    pert_str = self._binary_to_perturbation_str(X_val[i])
                    try:
                        gears_preds[i] = self.sclambda_model.predict([pert_str], return_type='mean')[0]
                        n_gears_fallback_sclambda += 1
                    except:
                        # Last resort: additive baseline
                        gears_preds[i] = self.predict_baseline(X_val[i:i+1], model='additive')[0]
                        n_gears_fallback_baseline += 1
            else:
                # FIX #4: Use scLAMBDA for unknown genes instead of additive baseline
                pert_str = self._binary_to_perturbation_str(X_val[i])
                try:
                    gears_preds[i] = self.sclambda_model.predict([pert_str], return_type='mean')[0]
                    n_gears_fallback_sclambda += 1
                except:
                    gears_preds[i] = self.predict_baseline(X_val[i:i+1], model='additive')[0]
                    n_gears_fallback_baseline += 1

        print(f"GEARS predictions: {n_gears_success} successful, {n_gears_fallback_sclambda} using scLAMBDA fallback, "
              f"{n_gears_fallback_baseline} using baseline fallback")
        predictions['gears'] = gears_preds

        # scLAMBDA predictions
        pert_strings = [self._binary_to_perturbation_str(x) for x in X_val]
        sclambda_results = self.sclambda_model.predict(pert_strings, return_type='mean')
        predictions['sclambda'] = sclambda_results

        # baseline predictions
        predictions['mean'] = self.predict_baseline(X_val, model='mean')
        predictions['additive'] = self.predict_baseline(X_val, model='additive')

        # Validate all predictions have the same shape
        expected_shape = predictions['gears'].shape
        for model_name, preds in predictions.items():
            if preds.shape != expected_shape:
                raise ValueError(
                    f"Shape mismatch: {model_name} predictions have shape {preds.shape}, "
                    f"expected {expected_shape}"
                )

        # store validation predictions
        self.val_predictions = predictions
        self.val_targets = y_val

        # stack predictions: shape (4, n_samples, n_genes_expr)
        pred_stack = np.stack([
            predictions['gears'],
            predictions['sclambda'],
            predictions['mean'],
            predictions['additive']
        ], axis=0)

        n_models = pred_stack.shape[0]

        if method == 'uniform':
            # equal weights
            weights = np.ones(n_models) / n_models
            print(f"Using uniform weights: {weights}")

        elif method == 'nnls':
            # non-negative least squares per gene
            weights_per_gene = []

            for gene_idx in range(n_genes_expr):
                # stack: (n_samples, n_models)
                A = pred_stack[:, :, gene_idx].T
                b = y_val[:, gene_idx]

                # solve non-negative least squares
                w, _ = nnls(A, b)
                weights_per_gene.append(w)

            # average weights across genes
            weights = np.mean(weights_per_gene, axis=0)

            # normalize if requested
            if normalize:
                weight_sum = weights.sum()
                if weight_sum > 0:
                    weights = weights / weight_sum

            print(f"Learned NNLS weights: GEARS={weights[0]:.3f}, scLAMBDA={weights[1]:.3f}, "
                  f"mean={weights[2]:.3f}, additive={weights[3]:.3f}")

        elif method == 'nnls_global':
            # FIX #1: Global NNLS - solve ONE optimization problem for all genes jointly
            # This is more stable and faster than per-gene NNLS
            print("Using global NNLS (optimizes MSE across all genes jointly)...")

            # Flatten everything: (n_samples * n_genes, n_models)
            A = pred_stack.transpose(1, 2, 0).reshape(-1, n_models)  # (n_samples*n_genes, 4)
            b = y_val.reshape(-1)  # (n_samples*n_genes,)

            # Solve global NNLS
            weights, residual = nnls(A, b)

            # Normalize
            if normalize and weights.sum() > 0:
                weights = weights / weights.sum()

            print(f"Learned global NNLS weights: GEARS={weights[0]:.3f}, scLAMBDA={weights[1]:.3f}, "
                  f"mean={weights[2]:.3f}, additive={weights[3]:.3f}")
            print(f"NNLS residual norm: {residual:.6f}")

        elif method == 'ridge':
            # ridge regression to learn weights
            # reshape: (n_samples * n_genes_expr, n_models)
            A = pred_stack.transpose(1, 2, 0).reshape(-1, n_models)
            b = y_val.reshape(-1)

            ridge = Ridge(alpha=1.0, fit_intercept=False, positive=True)
            ridge.fit(A, b)
            weights = ridge.coef_

            if normalize:
                weight_sum = weights.sum()
                if weight_sum > 0:
                    weights = weights / weight_sum

            print(f"Learned Ridge weights: GEARS={weights[0]:.3f}, scLAMBDA={weights[1]:.3f}, "
                  f"mean={weights[2]:.3f}, additive={weights[3]:.3f}")

        elif method == 'diverse_nnls':
            # FIX 5: NNLS with diversity penalty
            from scipy.optimize import minimize
            from scipy.stats import pearsonr

            # Compute pairwise correlations between models
            print("Computing model correlations for diversity penalty...")
            correlations = np.zeros((n_models, n_models))
            for i in range(n_models):
                for j in range(n_models):
                    if i != j:
                        pred_i = pred_stack[i].flatten()
                        pred_j = pred_stack[j].flatten()
                        corr, _ = pearsonr(pred_i, pred_j)
                        correlations[i, j] = corr
                    else:
                        correlations[i, j] = 1.0

            print(f"Model correlations range: [{correlations[np.triu_indices_from(correlations, k=1)].min():.3f}, "
                  f"{correlations[np.triu_indices_from(correlations, k=1)].max():.3f}]")

            def objective(w):
                # Accuracy term: MSE
                weighted_pred = np.tensordot(w, pred_stack, axes=([0], [0]))
                mse = np.mean((weighted_pred - y_val) ** 2)

                # Diversity penalty: penalize high correlation
                # Sum of w[i] * w[j] * corr[i,j] for all i < j
                diversity_term = 0.0
                for i in range(n_models):
                    for j in range(i+1, n_models):
                        diversity_term += w[i] * w[j] * correlations[i, j]

                total_loss = mse + diversity_penalty * diversity_term
                return total_loss

            # Constraints: w >= 0, sum(w) = 1
            constraints = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
            ]
            bounds = [(0.0, 1.0) for _ in range(n_models)]

            # Initial guess: uniform weights
            w0 = np.ones(n_models) / n_models

            # Optimize
            result = minimize(objective, w0, method='SLSQP', bounds=bounds, constraints=constraints)

            if result.success:
                weights = result.x
                print(f"Learned diverse NNLS weights: GEARS={weights[0]:.3f}, scLAMBDA={weights[1]:.3f}, "
                      f"mean={weights[2]:.3f}, additive={weights[3]:.3f}")
                print(f"Final objective: MSE={objective(weights):.6f} (with diversity penalty={diversity_penalty})")
            else:
                print(f"Warning: Optimization failed, falling back to uniform weights")
                weights = np.ones(n_models) / n_models

        else:
            raise ValueError(f"Unknown weight learning method: {method}")

        self.model_weights = weights
        self.use_learned_weights = True

        # compute validation MSE with learned weights
        weighted_pred = np.tensordot(weights, pred_stack, axes=([0], [0]))
        val_mse = np.mean((weighted_pred - y_val) ** 2)
        print(f"Validation MSE with learned weights: {val_mse:.6f}")

        return weights

    def save_ensemble_weights(self, save_path: str):
        """
        Save learned ensemble weights to disk.

        Args:
            save_path: Path to save weights (e.g., 'ensemble_weights.pkl')
        """
        if self.model_weights is None:
            raise ValueError("No weights to save. Call learn_ensemble_weights() first.")

        import pickle
        from datetime import datetime

        weights_dict = {
            'model_weights': self.model_weights,
            'use_learned_weights': self.use_learned_weights,
            'timestamp': datetime.now().isoformat(),
            'model_names': ['gears', 'sclambda', 'mean', 'additive']
        }

        with open(save_path, 'wb') as f:
            pickle.dump(weights_dict, f)

        print(f"\nSaved ensemble weights to {save_path}")
        print(f"   GEARS:     {self.model_weights[0]:.4f}")
        print(f"   scLAMBDA:  {self.model_weights[1]:.4f}")
        print(f"   Mean:      {self.model_weights[2]:.4f}")
        print(f"   Additive:  {self.model_weights[3]:.4f}")

    def load_ensemble_weights(self, load_path: str):
        """
        Load learned ensemble weights from disk.

        Args:
            load_path: Path to load weights from

        Returns:
            True if weights loaded successfully, False if file doesn't exist
        """
        import pickle
        import os

        if not os.path.exists(load_path):
            return False

        with open(load_path, 'rb') as f:
            weights_dict = pickle.load(f)

        self.model_weights = weights_dict['model_weights']
        self.use_learned_weights = weights_dict['use_learned_weights']

        print(f"\nLoaded ensemble weights from {load_path}")
        print(f"   Saved: {weights_dict.get('timestamp', 'unknown date')}")
        print(f"   GEARS:     {self.model_weights[0]:.4f}")
        print(f"   scLAMBDA:  {self.model_weights[1]:.4f}")
        print(f"   Mean:      {self.model_weights[2]:.4f}")
        print(f"   Additive:  {self.model_weights[3]:.4f}")

        return True

    def calibrate_conformal_predictor(self, X_val: np.ndarray = None, y_val: np.ndarray = None):
        """
        Calibrate conformal predictor using validation data.

        IMPORTANT: Use SEPARATE calibration data (not the same as weight-learning data)
        to avoid overly optimistic coverage estimates.

        Args:
            X_val: Validation inputs (if None, uses stored val_predictions)
            y_val: Validation targets (if None, uses stored val_targets)
        """
        if X_val is not None and y_val is not None:
            # FIX #3: Fresh predictions on NEW data (best practice)
            print("\nGenerating predictions for conformal calibration on fresh data...")
            pred_mean, _, _ = self.predict_ensemble(X_val, return_intervals=False)
            y_pred_cal = pred_mean
            y_cal = y_val
        elif self.val_predictions is not None and self.val_targets is not None:
            # FIX #3: WARN if reusing weight-learning data
            print("\n" + "="*70)
            print("WARNING: Calibrating conformal predictor on SAME data used for weight learning!")
            print("="*70)
            print("This may lead to:")
            print("  - Overly optimistic coverage estimates (residuals too small)")
            print("  - Poor calibration on test set (data leakage)")
            print("\nRECOMMENDED: Provide separate X_val, y_val for calibration.")
            print("  Example: Split validation into 60% weight-learning, 40% calibration")
            print("="*70 + "\n")

            # use stored validation predictions
            print("Using stored validation predictions for conformal calibration...")
            pred_stack = np.stack([
                self.val_predictions['gears'],
                self.val_predictions['sclambda'],
                self.val_predictions['mean'],
                self.val_predictions['additive']
            ], axis=0)

            if self.use_learned_weights and self.model_weights is not None:
                y_pred_cal = np.tensordot(self.model_weights, pred_stack, axes=([0], [0]))
            else:
                y_pred_cal = np.mean(pred_stack, axis=0)

            y_cal = self.val_targets
        else:
            raise ValueError("Must provide validation data or call learn_ensemble_weights first")

        # calibrate conformal predictor
        self.conformal_predictor.calibrate(y_cal, y_pred_cal)

    def predict_ensemble(self, X_test: np.ndarray, return_intervals: bool = True) -> tuple:
        """
        Ensemble prediction with uncertainty quantification.

        Args:
            X_test: Test input features
            return_intervals: Whether to compute conformal prediction intervals

        Returns:
            If return_intervals=True:
                (ensemble_mean, epistemic_uncertainty, model_predictions,
                 conformal_lower, conformal_upper, conformal_width)
            If return_intervals=False:
                (ensemble_mean, epistemic_uncertainty, model_predictions)
        """
        if not self.fitted:
            raise ValueError("Models must be loaded before prediction")

        predictions = {}
        n_samples = X_test.shape[0]
        n_genes_expr = self.y_single.shape[1]

        # get GEARS known genes
        gears_known_genes = set(self.gears_model.pert_data.pert_list) if hasattr(self.gears_model.pert_data, 'pert_list') else set()

        # GEARS predictions with improved error handling and debugging
        pert_lists = [self._binary_to_gene_list(x) for x in X_test]
        gears_preds = np.zeros((n_samples, n_genes_expr))

        n_gears_success = 0
        n_gears_fallback_sclambda = 0
        n_gears_fallback_baseline = 0
        first_result_logged = False

        for i, pert_list in enumerate(pert_lists):
            # check if all genes in this perturbation are known to GEARS
            if all(gene in gears_known_genes for gene in pert_list) and len(pert_list) > 0:
                try:
                    result = self.gears_model.batch_predict([pert_list])

                    # Debug: log first result to understand structure
                    if not first_result_logged:
                        print(f"DEBUG: GEARS result type: {type(result)}")
                        if isinstance(result, dict):
                            print(f"DEBUG: GEARS result keys: {result.keys()}")
                        first_result_logged = True

                    # Handle different possible return structures from GEARS
                    if isinstance(result, dict):
                        pred = result.get('pred', None)
                        if pred is not None:
                            # Handle both array and list formats
                            if isinstance(pred, (list, np.ndarray)):
                                pred_array = np.array(pred)
                                if pred_array.ndim > 1:
                                    gears_preds[i] = pred_array[0]
                                else:
                                    gears_preds[i] = pred_array
                                n_gears_success += 1
                            else:
                                raise ValueError(f"Unexpected prediction format: {type(pred)}")
                        else:
                            # Check for other possible keys
                            if 'prediction' in result:
                                pred_array = np.array(result['prediction'])
                                gears_preds[i] = pred_array[0] if pred_array.ndim > 1 else pred_array
                                n_gears_success += 1
                            else:
                                raise ValueError(f"GEARS returned dict without 'pred' key. Keys: {list(result.keys())}")
                    elif isinstance(result, (list, np.ndarray)):
                        # Direct array/list return
                        pred_array = np.array(result)
                        if pred_array.ndim > 1:
                            gears_preds[i] = pred_array[0]
                        else:
                            gears_preds[i] = pred_array
                        n_gears_success += 1
                    else:
                        raise ValueError(f"Unexpected GEARS return type: {type(result)}")
                except Exception as e:
                    # FIX #4: Use scLAMBDA as fallback to preserve diversity
                    if n_gears_fallback_sclambda + n_gears_fallback_baseline < 3:
                        print(f"Warning: GEARS prediction failed for {pert_list}: {e}")

                    # Try scLAMBDA first (preserves diversity)
                    pert_str = self._binary_to_perturbation_str(X_test[i])
                    try:
                        gears_preds[i] = self.sclambda_model.predict([pert_str], return_type='mean')[0]
                        n_gears_fallback_sclambda += 1
                    except:
                        # Last resort: additive baseline
                        gears_preds[i] = self.predict_baseline(X_test[i:i+1], model='additive')[0]
                        n_gears_fallback_baseline += 1
            else:
                # FIX #4: Use scLAMBDA for unknown genes instead of additive baseline
                pert_str = self._binary_to_perturbation_str(X_test[i])
                try:
                    gears_preds[i] = self.sclambda_model.predict([pert_str], return_type='mean')[0]
                    n_gears_fallback_sclambda += 1
                except:
                    unknown_genes = [g for g in pert_list if g not in gears_known_genes]
                    if n_gears_fallback_sclambda + n_gears_fallback_baseline < 3:
                        print(f"Warning: Genes unknown to GEARS: {unknown_genes[:3]}. Using baseline.")
                    gears_preds[i] = self.predict_baseline(X_test[i:i+1], model='additive')[0]
                    n_gears_fallback_baseline += 1

        print(f"GEARS predictions: {n_gears_success} successful, {n_gears_fallback_sclambda} using scLAMBDA fallback, "
              f"{n_gears_fallback_baseline} using baseline fallback")
        predictions['gears'] = gears_preds

        # scLAMBDA predictions
        pert_strings = [self._binary_to_perturbation_str(x) for x in X_test]
        sclambda_results = self.sclambda_model.predict(pert_strings, return_type='mean')
        predictions['sclambda'] = sclambda_results

        # baseline predictions
        predictions['mean'] = self.predict_baseline(X_test, model='mean')
        predictions['additive'] = self.predict_baseline(X_test, model='additive')

        # Debug: print shapes
        print(f"GEARS predictions shape: {predictions['gears'].shape}")
        print(f"scLAMBDA predictions shape: {predictions['sclambda'].shape}")
        print(f"Mean baseline predictions shape: {predictions['mean'].shape}")
        print(f"Additive baseline predictions shape: {predictions['additive'].shape}")

        # Validate all predictions have the same shape
        expected_shape = predictions['gears'].shape
        for model_name, preds in predictions.items():
            if preds.shape != expected_shape:
                raise ValueError(
                    f"Shape mismatch: {model_name} predictions have shape {preds.shape}, "
                    f"expected {expected_shape}"
                )

        # ensemble statistics
        pred_stack = np.stack([
            predictions['gears'],
            predictions['sclambda'],
            predictions['mean'],
            predictions['additive']
        ], axis=0)

        # use learned weights or uniform weights
        if self.use_learned_weights and self.model_weights is not None:
            print(f"Using learned weights: {self.model_weights}")
            ensemble_mean = np.tensordot(self.model_weights, pred_stack, axes=([0], [0]))
        else:
            print("Using uniform weights (equal averaging)")
            ensemble_mean = np.mean(pred_stack, axis=0)

        # epistemic uncertainty (model disagreement)
        epistemic_uncertainty = np.var(pred_stack, axis=0)

        # Check model diversity
        self._check_model_diversity(predictions)

        # conformal prediction intervals (aleatoric uncertainty)
        if return_intervals and self.conformal_predictor.is_calibrated:
            lower, upper, width = self.conformal_predictor.predict_interval(ensemble_mean)
            return ensemble_mean, epistemic_uncertainty, predictions, lower, upper, width
        elif return_intervals and not self.conformal_predictor.is_calibrated:
            print("Warning: Conformal predictor not calibrated. Call calibrate_conformal_predictor() first.")
            return ensemble_mean, epistemic_uncertainty, predictions, None, None, None
        else:
            return ensemble_mean, epistemic_uncertainty, predictions

    def _check_model_diversity(self, predictions: dict):
        """
        Check if models are producing diverse predictions.
        Prints correlation matrix between model predictions.
        """
        from scipy.stats import pearsonr

        model_names = list(predictions.keys())
        n_models = len(model_names)

        print("\n" + "="*60)
        print("MODEL DIVERSITY CHECK")
        print("="*60)

        # Flatten predictions for correlation
        flat_preds = {name: pred.flatten() for name, pred in predictions.items()}

        # Compute pairwise correlations
        print("\nPairwise correlations between model predictions:")
        print("(Values close to 1.0 indicate models are too similar)")
        print("")
        for i in range(n_models):
            for j in range(i+1, n_models):
                name1, name2 = model_names[i], model_names[j]
                corr, _ = pearsonr(flat_preds[name1], flat_preds[name2])

                # Flag if correlation is too high
                flag = "TOO SIMILAR" if corr > 0.95 else ""
                print(f"  {name1:12s} vs {name2:12s}: {corr:.4f} {flag}")

        # Compute variance of predictions
        pred_stack = np.stack(list(predictions.values()), axis=0)
        mean_variance = np.mean(np.var(pred_stack, axis=0))
        print(f"\nMean variance across models: {mean_variance:.6f}")
        if mean_variance < 1e-6:
            print("  WARNING: Very low variance - models may be identical!")

        print("="*60 + "\n")
    
    def recommend_experiments(self, candidate_perturbations: list, n_recommend: int = 10) -> list:
        """
        Recommend top experiments based on epistemic uncertainty.

        args:
            candidate_perturbations: List of (gene1, gene2) tuples or lists of gene names
            n_recommend: Number of experiments to recommend

        returns:
            List of (gene1, gene2, uncertainty_score) tuples
        """
        if self.gene_names is None:
            raise ValueError("Gene names not initialized. Must load data first.")
        
        X_candidates = np.zeros((len(candidate_perturbations), len(self.gene_names)))
        valid_indices = []

        for i, pert in enumerate(candidate_perturbations):
            # Handle both tuple and list formats
            if isinstance(pert, (tuple, list)) and len(pert) >= 2:
                gene1, gene2 = pert[0], pert[1]
            elif isinstance(pert, (tuple, list)) and len(pert) == 1:
                # Single gene - duplicate it for combo
                gene1 = gene2 = pert[0]
            else:
                print(f"Warning: Skipping invalid perturbation format at index {i}: {pert}")
                continue
                
            if gene1 in self.gene_names and gene2 in self.gene_names:
                idx1 = self.gene_names.index(gene1)
                idx2 = self.gene_names.index(gene2)
                X_candidates[i, [idx1, idx2]] = 1.0
                valid_indices.append(i)

        if len(valid_indices) == 0:
            print("Warning: No valid gene pairs found")
            return []

        X_valid = X_candidates[valid_indices]
        _, uncertainties, _ = self.predict_ensemble(X_valid, return_intervals=False)
        uncertainty_scores = np.sum(uncertainties, axis=1)

        n_recommend = min(n_recommend, len(valid_indices))
        top_indices = np.argsort(uncertainty_scores)[-n_recommend:]

        recommended = []
        for idx in reversed(top_indices):
            original_idx = valid_indices[idx]
            gene1, gene2 = candidate_perturbations[original_idx]
            score = uncertainty_scores[idx]
            recommended.append((gene1, gene2, score))

        return recommended

    def print_ensemble_summary(self):
        """Print a summary of the ensemble configuration."""
        print("\n" + "="*60)
        print("ENSEMBLE CONFIGURATION SUMMARY")
        print("="*60)

        print("\nModels:")
        print("  - GEARS")
        print("  - scLAMBDA")
        print("  - Mean baseline")
        print("  - Additive baseline")

        print("\nWeighting:")
        if self.use_learned_weights and self.model_weights is not None:
            print("  Using LEARNED weights:")
            print(f"    GEARS:     {self.model_weights[0]:.4f}")
            print(f"    scLAMBDA:  {self.model_weights[1]:.4f}")
            print(f"    Mean:      {self.model_weights[2]:.4f}")
            print(f"    Additive:  {self.model_weights[3]:.4f}")
        else:
            print("  Using UNIFORM weights (0.25 each)")

        print("\nUncertainty Quantification:")
        print("  - Epistemic: Model disagreement (variance)")
        if self.conformal_predictor.is_calibrated:
            print(f"  - Aleatoric: Conformal prediction intervals")
            print(f"    Target coverage: {(1-self.conformal_predictor.alpha)*100:.1f}%")
            print(f"    Calibration samples: {len(self.conformal_predictor.calibration_scores)}")
        else:
            print("  - Aleatoric: Not calibrated (call calibrate_conformal_predictor)")

        print("\n" + "="*60)

    def diagnose_diversity(self, X_test: np.ndarray = None, n_samples: int = 100):
        """
        FIX DIAGNOSTIC: Check if ensemble has sufficient diversity for active learning.

        Args:
            X_test: Test data to use for diagnostic (optional, will use X_combo if not provided)
            n_samples: Number of samples to use for diagnostic

        Returns:
            dict with diagnostic results and pass/fail status
        """
        from scipy.stats import pearsonr

        print("\n" + "="*60)
        print("DIVERSITY DIAGNOSTIC FOR ACTIVE LEARNING")
        print("="*60)

        results = {
            'max_correlation': None,
            'mean_variance': None,
            'uncertainty_range': None,
            'all_checks_passed': False
        }

        # Use provided test data or default to combo data
        if X_test is None:
            if hasattr(self, 'X_combo'):
                X_test = self.X_combo[:min(n_samples, len(self.X_combo))]
            else:
                print("ERROR: No test data available. Provide X_test or load data first.")
                return results
        else:
            X_test = X_test[:min(n_samples, len(X_test))]

        # 1. Check model correlations
        print("\n1. Model Correlation Check")
        print("-" * 60)

        if self.val_predictions is not None:
            correlations = np.zeros((4, 4))
            model_names = ['gears', 'sclambda', 'mean', 'additive']

            for i, name1 in enumerate(model_names):
                for j, name2 in enumerate(model_names):
                    if i != j and name1 in self.val_predictions and name2 in self.val_predictions:
                        pred_i = self.val_predictions[name1].flatten()
                        pred_j = self.val_predictions[name2].flatten()
                        corr, _ = pearsonr(pred_i, pred_j)
                        correlations[i, j] = corr
                    elif i == j:
                        correlations[i, j] = 1.0

            max_corr = np.max(correlations[np.triu_indices_from(correlations, k=1)])
            results['max_correlation'] = max_corr

            if max_corr > 0.95:
                print(f"FAIL: Maximum correlation = {max_corr:.3f} (target: < 0.95)")
                print("   Models are too similar - active learning will not work!")
            else:
                print(f"PASS: Maximum correlation = {max_corr:.3f}")
        else:
            print("SKIP: No validation predictions available")

        # 2. Check variance across models
        print("\n2. Model Variance Check")
        print("-" * 60)

        _, _, predictions = self.predict_ensemble(X_test, return_intervals=False)
        pred_stack = np.stack(list(predictions.values()), axis=0)
        mean_var = np.mean(np.var(pred_stack, axis=0))
        results['mean_variance'] = mean_var

        if mean_var < 0.01:
            print(f"FAIL: Mean variance = {mean_var:.6f} (target: > 0.01)")
            print("   Models predict too similarly - no diversity for active learning!")
        else:
            print(f"PASS: Mean variance = {mean_var:.6f}")

        # 3. Check uncertainty distribution
        print("\n3. Uncertainty Range Check")
        print("-" * 60)

        _, uncertainties, _ = self.predict_ensemble(X_test, return_intervals=False)
        unc_sum = np.sum(uncertainties, axis=1)
        unc_range = unc_sum.max() - unc_sum.min()
        unc_std = np.std(unc_sum)
        results['uncertainty_range'] = unc_range
        results['uncertainty_std'] = unc_std

        if unc_range < 0.1:
            print(f"FAIL: Uncertainty range = {unc_range:.6f} (target: > 0.1)")
            print(f"   Uncertainty std = {unc_std:.6f}")
            print("   Uncertainty is too uniform - cannot distinguish informative samples!")
        else:
            print(f"PASS: Uncertainty range = {unc_range:.6f}")
            print(f"   Uncertainty std = {unc_std:.6f}")

        # Overall assessment
        print("\n" + "="*60)
        print("OVERALL ASSESSMENT")
        print("="*60)

        checks_passed = []
        if results['max_correlation'] is not None:
            checks_passed.append(results['max_correlation'] < 0.95)
        checks_passed.append(results['mean_variance'] > 0.01)
        checks_passed.append(results['uncertainty_range'] > 0.1)

        results['all_checks_passed'] = all(checks_passed)

        if results['all_checks_passed']:
            print("ALL CHECKS PASSED")
            print("   Ensemble has sufficient diversity for active learning!")
        else:
            print("SOME CHECKS FAILED")
            print("   Recommendations:")
            if results['max_correlation'] is not None and results['max_correlation'] > 0.95:
                print("   1. Retrain GEARS on custom data (Fix #1)")
                print("   2. Add more diverse models (Random Forest, Ridge with interactions)")
            if results['mean_variance'] < 0.01:
                print("   3. Use MC Dropout for scLAMBDA (Fix #4)")
                print("   4. Bootstrap training data for different models")
            if results['uncertainty_range'] < 0.1:
                print("   5. Use diversity-aware weight learning (Fix #5)")

        print("="*60 + "\n")

        return results

# ================================================================================
# usage
# ================================================================================

if __name__ == "__main__":
    print("Combinatorial Perturbation Prediction Ensemble")
    print("=" * 50)
    print("\nStrategy:")
    print("  - GEARS: Train on built-in 'norman' dataset")
    print("  - scLAMBDA")
    print("  - Evaluation")
    print("=" * 50)

    SCLAMBDA_REPO = '/insomnia001/depts/edu/BIOLBC3141_Fall2025/rc3517/ml-stats/scLAMBDA'
    DATA_DIR = '/insomnia001/depts/edu/BIOLBC3141_Fall2025/rc3517/ml-stats/data'
    
    GEARS_MODEL_DIR = f'{DATA_DIR}/gears_model'  # has config.pkl and model.pt
    GEARS_DATA_PATH = f'{DATA_DIR}/gears_data'  # contains downloaded GEARS data

    SCLAMBDA_MODEL_PATH = f'{DATA_DIR}/sclambda_model'  # has ckpt.pth
    SCLAMBDA_ADATA_PATH = f'{DATA_DIR}/adata_norman_preprocessed.h5ad'
    SCLAMBDA_EMBEDDINGS_PATH = f'{DATA_DIR}/GPT_3_5_gene_embeddings_3-large.pickle'
    NORMAN_DATA_PATH = f'{DATA_DIR}/adata_norman_preprocessed.h5ad'

    # ==========================================
    # step 3: load ensemble WITHOUT fitting baselines yet (to avoid data leakage)
    # ==========================================
    print("\n[3/6] Loading ensemble (GEARS=norman, scLAMBDA=custom, eval=custom)...")
    ensemble = Ensemble(sclambda_repo_path=SCLAMBDA_REPO)

    # Load GEARS and scLAMBDA models
    # Use GEARS built-in 'norman' dataset to avoid data loading issues
    ensemble.gears_model.load_pretrained_gears(
        model_dir=GEARS_MODEL_DIR,
        data_path=GEARS_DATA_PATH,
        data_name='norman',  # Use built-in norman dataset
        custom_adata_path=None  # No custom data needed
    )
    ensemble.sclambda_model.load_pretrained_sclambda(
        model_path=SCLAMBDA_MODEL_PATH,
        adata_path=SCLAMBDA_ADATA_PATH,
        gene_embeddings_path=SCLAMBDA_EMBEDDINGS_PATH
    )

    # Load Norman data for splitting
    ensemble.X_single, ensemble.y_single, ensemble.X_combo, ensemble.y_combo, ensemble.gene_names = \
        ensemble.data_processor.load_norman_data(NORMAN_DATA_PATH)

    # Create splits BEFORE fitting baselines to avoid data leakage
    print("\n[4/6] Creating data splits (BEFORE fitting baselines to avoid leakage)...")
    splits = ensemble.data_processor.create_combo_splits(
        X_single=ensemble.X_single,
        y_single=ensemble.y_single,
        X_combo=ensemble.X_combo,
        y_combo=ensemble.y_combo,
        combo_test_ratio=0.2,
        random_state=42
    )

    # NOW fit baselines on TRAINING data only
    print("\nFitting baselines on TRAINING singles only...")
    # Extract training singles from the combined train set
    n_train_singles = splits['n_singles_in_train']
    X_single_train = splits['X_train'][:n_train_singles]
    y_single_train = splits['y_train'][:n_train_singles]

    ensemble._fit_baselines(X_single_train=X_single_train, y_single_train=y_single_train)
    ensemble.fitted = True

    print(f"\nData splits:")
    print(f"  Training samples: {len(splits['X_train'])} ({n_train_singles} singles + {len(splits['X_train']) - n_train_singles} combos)")
    print(f"  Validation samples: {len(splits['X_val'])} (all combos)")
    print(f"  Test samples: {len(splits['X_test'])} (all combos)")
    
    # ==========================================
    # step 5: Split validation into val + calibration sets
    # ==========================================
    print("\n[5/6] Splitting validation set into weight-learning and calibration sets...")
    n_val = len(splits['X_val'])
    val_indices = np.random.RandomState(42).permutation(n_val)

    # Use 60% for weight learning, 40% for calibration
    n_weight_learn = int(0.6 * n_val)

    X_weight = splits['X_val'][val_indices[:n_weight_learn]]
    y_weight = splits['y_val'][val_indices[:n_weight_learn]]

    X_cal = splits['X_val'][val_indices[n_weight_learn:]]
    y_cal = splits['y_val'][val_indices[n_weight_learn:]]

    print(f"  Weight learning: {len(X_weight)} samples")
    print(f"  Calibration: {len(X_cal)} samples")

    # ==========================================
    # step 6: learn ensemble weights
    # ==========================================
    print("\n[6/6] Learning ensemble weights on weight-learning set...")
    ensemble.learn_ensemble_weights(
        X_val=X_weight,
        y_val=y_weight,
        method='nnls_global',  # FIX #1: Use global NNLS for better stability
        normalize=True
    )

    # Save learned weights to disk for reuse
    WEIGHTS_PATH = f'{DATA_DIR}/ensemble_weights.pkl'
    ensemble.save_ensemble_weights(WEIGHTS_PATH)

    # ==========================================
    # step 7: calibrate conformal predictor on separate calibration set
    # ==========================================
    print("\n[7/7] Calibrating conformal predictor on calibration set...")
    ensemble.calibrate_conformal_predictor(X_val=X_cal, y_val=y_cal)

    # Print ensemble configuration
    ensemble.print_ensemble_summary()

    # ==========================================
    # step 8: evaluate on test set with conformal intervals
    # ==========================================
    print("\n[8/8] Evaluating on test set with conformal prediction intervals...")
    results = ensemble.predict_ensemble(splits['X_test'], return_intervals=True)
    pred_mean = results[0]
    uncertainties = results[1]
    individual_preds = results[2]
    conformal_lower = results[3]
    conformal_upper = results[4]
    conformal_width = results[5]

    # Compute metrics
    mse = np.mean((pred_mean - splits['y_test']) ** 2)

    print(f"\nTest metrics:")
    print(f"  MSE (weighted ensemble): {mse:.6f}")
    print(f"  Mean epistemic uncertainty: {np.mean(uncertainties):.6f}")

    if conformal_lower is not None:
        # Conformal prediction statistics
        coverage_stats = ensemble.conformal_predictor.compute_coverage(
            splits['y_test'], conformal_lower, conformal_upper
        )
        print(f"\nConformal prediction statistics:")
        print(f"  Target coverage: {coverage_stats['target_coverage']*100:.1f}%")
        print(f"  Empirical coverage: {coverage_stats['overall_coverage']*100:.1f}%")
        print(f"  Mean interval width: {np.mean(conformal_width):.6f}")
        print(f"  Per-gene coverage (mean±std): {coverage_stats['gene_coverage_mean']*100:.1f}% ± {coverage_stats['gene_coverage_std']*100:.1f}%")
        print(f"  Per-sample coverage (mean±std): {coverage_stats['sample_coverage_mean']*100:.1f}% ± {coverage_stats['sample_coverage_std']*100:.1f}%")

    # Compare individual model performance
    print(f"\nIndividual model MSE:")
    for model_name, preds in individual_preds.items():
        model_mse = np.mean((preds - splits['y_test']) ** 2)
        print(f"  {model_name}: {model_mse:.6f}")

    # experiment recommendations
    candidate_experiments = [(ensemble.gene_names[i], ensemble.gene_names[i+1])
                             for i in range(min(20, len(ensemble.gene_names)-1))]
    recommendations = ensemble.recommend_experiments(candidate_experiments, n_recommend=5)

    print(f"\nTop 5 Recommended Experiments:")
    for i, (gene1, gene2, score) in enumerate(recommendations):
        print(f"  {i+1}. {gene1} x {gene2} (uncertainty: {score:.4f})")