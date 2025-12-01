import numpy as np
import pandas as pd
import scanpy as sc
import torch
import sys
import os
import warnings
warnings.filterwarnings('ignore')

from ensemble_full_pipeline import GEARSWrapper, scLAMBDAWrapper

class PerturbationDataProcessor:
    """Standardize data format for all models"""
    
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
        
        # extract perturbation information
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

            if pert == 'ctrl' or pert == 'control':
                continue

            genes = pert.split('+')
            for gene in genes:
                gene = gene.strip()
                if gene in gene_to_idx:
                    X[i, gene_to_idx[gene]] = 1

        return X


# ================================================================================
# ensemble
# ================================================================================

class Ensemble:
    """Ensemble with GEARS + scLAMBDA + baselines"""
    
    def __init__(self, sclambda_repo_path: str = None):
        """
        args:
            sclambda_repo_path: Path to cloned scLAMBDA repo
        """
        self.gears_model = GEARSWrapper()
        self.sclambda_model = scLAMBDAWrapper(sclambda_path=sclambda_repo_path)
        self.baseline_models = None
        self.data_processor = PerturbationDataProcessor()
        self.gene_effects = None
        self.fitted = False
        
    def load_models(self, 
                    gears_model_dir: str,
                    gears_data_path: str,
                    sclambda_model_path: str,
                    sclambda_adata_path: str,
                    sclambda_embeddings_path: str,
                    norman_data_path: str,
                    gears_data_name: str = None):
        """
        Load all pre-trained models.
        
        args:
            gears_model_dir: Directory where GEARS model is saved (e.g., './gears_model')
            gears_data_path: GEARS data directory (e.g., './data')
            sclambda_model_path: Directory where scLAMBDA model is saved
            sclambda_adata_path: Path to adata for scLAMBDA
            sclambda_embeddings_path: Path to gene embeddings .pkl file
            norman_data_path: Path to Norman h5ad for baseline fitting
            gears_data_name: None bc using our custom dataset

        """
        print("Loading models...")
        
        # load GEARS
        self.gears_model.load_pretrained_gears(
            model_dir=gears_model_dir,
            data_path=gears_data_path,
            data_name=gears_data_name
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
            
        self._fit_baselines()
        
        self.fitted = True
        print("All models loaded!")
        
    def _fit_baselines(self):
        """Fit additive and mean baseline models."""
        self.baseline_models = {}

        n_genes_pert = self.X_single.shape[1]
        n_genes_expr = self.y_single.shape[1]
        
        self.gene_effects = np.zeros((n_genes_pert, n_genes_expr))
        
        for gene_idx in range(n_genes_pert):
            mask = self.X_single[:, gene_idx] == 1
            if mask.sum() > 0:
                self.gene_effects[gene_idx] = np.mean(self.y_single[mask], axis=0)
        
        self.baseline_models['mean'] = self.gene_effects
        self.baseline_models['additive'] = self.gene_effects
    
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
        
    def predict_ensemble(self, X_test: np.ndarray) -> tuple:
        """
        Ensemble prediction with uncertainty quantification.

        Returns:
            (ensemble_mean, epistemic_uncertainty, model_predictions)
        """
        if not self.fitted:
            raise ValueError("Models must be loaded before prediction")

        predictions = {}
        n_samples = X_test.shape[0]
        n_genes_expr = self.y_single.shape[1]

        # Get GEARS known genes
        gears_known_genes = set(self.gears_model.pert_data.pert_list) if hasattr(self.gears_model.pert_data, 'pert_list') else set()

        # GEARS predictions with error handling
        pert_lists = [self._binary_to_gene_list(x) for x in X_test]
        gears_preds = np.zeros((n_samples, n_genes_expr))

        for i, pert_list in enumerate(pert_lists):
            # Check if all genes in this perturbation are known to GEARS
            if all(gene in gears_known_genes for gene in pert_list):
                try:
                    result = self.gears_model.batch_predict([pert_list])
                    gears_preds[i] = result.get('pred', np.zeros(n_genes_expr))[0]
                except Exception as e:
                    print(f"Warning: GEARS prediction failed for {pert_list}: {e}")
                    # Use baseline prediction as fallback
                    gears_preds[i] = self.predict_baseline(X_test[i:i+1], model='additive')[0]
            else:
                # Use baseline prediction for unknown genes
                unknown_genes = [g for g in pert_list if g not in gears_known_genes]
                if i == 0:  # Only print once
                    print(f"Warning: Some genes unknown to GEARS (e.g., {unknown_genes[:3]}). Using baseline predictions.")
                gears_preds[i] = self.predict_baseline(X_test[i:i+1], model='additive')[0]

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

        # ensemble statistics
        pred_stack = np.stack([
            predictions['gears'],
            predictions['sclambda'],
            predictions['mean'],
            predictions['additive']
        ], axis=0)

        ensemble_mean = np.mean(pred_stack, axis=0)
        epistemic_uncertainty = np.var(pred_stack, axis=0)

        return ensemble_mean, epistemic_uncertainty, predictions
    
    def recommend_experiments(self, candidate_perturbations: list, n_recommend: int = 10) -> list:
        """
        Recommend top experiments based on epistemic uncertainty.
        
        args:
            candidate_perturbations: List of (gene1, gene2) tuples
            n_recommend: Number of experiments to recommend
            
        returns:
            List of (gene1, gene2, uncertainty_score) tuples
        """
        X_candidates = np.zeros((len(candidate_perturbations), len(self.gene_names)))
        valid_indices = []
        
        for i, (gene1, gene2) in enumerate(candidate_perturbations):
            if gene1 in self.gene_names and gene2 in self.gene_names:
                idx1 = self.gene_names.index(gene1)
                idx2 = self.gene_names.index(gene2)
                X_candidates[i, [idx1, idx2]] = 1.0
                valid_indices.append(i)
        
        if len(valid_indices) == 0:
            print("Warning: No valid gene pairs found")
            return []
        
        X_valid = X_candidates[valid_indices]
        _, uncertainties, _ = self.predict_ensemble(X_valid)
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

# ================================================================================
# usage
# ================================================================================

if __name__ == "__main__":
    print("Combinatorial Perturbation Prediction Ensemble")
    print("=" * 50)
    print("\nStrategy:")
    print("  - GEARS: Train on built-in 'norman' dataset (pre-processed by GEARS)")
    print("  - scLAMBDA: Train on custom preprocessed data")
    print("  - Evaluation: Use custom preprocessed data")
    print("=" * 50)

    SCLAMBDA_REPO = '/insomnia001/depts/edu/BIOLBC3141_Fall2025/rc3517/ml-stats/scLAMBDA'
    DATA_DIR = '/insomnia001/depts/edu/BIOLBC3141_Fall2025/rc3517/ml-stats/data'
    
    GEARS_MODEL_DIR = f'{DATA_DIR}/gears_model'  # has config.pkl and model.pt
    GEARS_DATA_PATH = f'{DATA_DIR}/gears_data'  # contains downloaded GEARS data

    SCLAMBDA_MODEL_PATH = f'{DATA_DIR}/sclambda_model'  # has ckpt.pth
    SCLAMBDA_ADATA_PATH = f'{DATA_DIR}/norman_perturbseq_preprocessed_hvg_filtered.h5ad'
    SCLAMBDA_EMBEDDINGS_PATH = f'{DATA_DIR}/GPT_3_5_gene_embeddings_3-large.pickle'
    NORMAN_DATA_PATH = f'{DATA_DIR}/norman_perturbseq_preprocessed_hvg_filtered.h5ad'

    # ==========================================
    # step 3: load ensemble and evaluate
    # ==========================================
    print("\n[3/3] Loading ensemble (GEARS=norman, scLAMBDA=custom, eval=custom)...")
    ensemble = Ensemble(sclambda_repo_path=SCLAMBDA_REPO)
    ensemble.load_models(
        gears_model_dir=GEARS_MODEL_DIR,
        gears_data_path=GEARS_DATA_PATH,
        gears_data_name='norman',
        sclambda_model_path=SCLAMBDA_MODEL_PATH,
        sclambda_adata_path=SCLAMBDA_ADATA_PATH,
        sclambda_embeddings_path=SCLAMBDA_EMBEDDINGS_PATH,
        norman_data_path=NORMAN_DATA_PATH
    )
    
    # create splits
    splits = ensemble.data_processor.create_combo_splits(
        X_single=ensemble.X_single,
        y_single=ensemble.y_single,
        X_combo=ensemble.X_combo,
        y_combo=ensemble.y_combo,
        combo_test_ratio=0.2,
        random_state=42
    )
    
    # evaluate
    print("\nEvaluating on test set...")
    pred_mean, uncertainties, individual_preds = ensemble.predict_ensemble(splits['X_test'])
    mse = np.mean((pred_mean - splits['y_test']) ** 2)
    
    print(f"\nTest metrics:")
    print(f"  MSE: {mse:.4f}")
    print(f"  Mean epistemic uncertainty: {np.mean(uncertainties):.4f}")
    
    # experiment recommendations
    candidate_experiments = [(ensemble.gene_names[i], ensemble.gene_names[i+1]) 
                             for i in range(min(20, len(ensemble.gene_names)-1))]
    recommendations = ensemble.recommend_experiments(candidate_experiments, n_recommend=5)
    
    print(f"\nTop 5 Recommended Experiments:")
    for i, (gene1, gene2, score) in enumerate(recommendations):
        print(f"  {i+1}. {gene1} x {gene2} (uncertainty: {score:.4f})")