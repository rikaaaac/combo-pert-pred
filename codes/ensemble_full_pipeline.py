"""
GEARS + scLAMBDA + baseline ensemble for combinatorial perturbation prediction

- GEARS: pip install cell-gears (https://github.com/snap-stanford/GEARS)
- scLAMBDA: git clone https://github.com/gefeiwang/scLAMBDA

"""

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# ================================================================================
# GEARS Integration
# ================================================================================

class GEARSWrapper:
    """GEARS model wrapper"""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = None
        self.pert_data = None
        self.fitted = False
        
    def load_pretrained_gears(self, model_dir: str, data_path: str, data_name: str = None,
                              split: str = 'simulation', seed: int = 1,
                              batch_size: int = 32, test_batch_size: int = 128):
        """
        Load pre-trained GEARS model.

        args:
            model_dir: Directory where model was saved (e.g., 'gears_model')
            data_path: Path to data directory (e.g., './data')
            data_name: Name of dataset if using built-in ('norman', 'adamson', 'dixit')
                       or None if using custom processed data
            split: Split type (needed for dataloader creation)
            seed: Random seed (needed for dataloader creation)
            batch_size: Batch size for dataloaders
            test_batch_size: Test batch size for dataloaders
        """
        from gears import PertData, GEARS

        # load perturbation data
        self.pert_data = PertData(data_path)

        if data_name is not None:
            # use built-in dataset
            self.pert_data.load(data_name=data_name)
        else:
            # load custom processed data
            self.pert_data.load(data_path=data_path)


        if hasattr(self.pert_data.adata.obs['condition'], 'values'):
            # force conversion to avoid pandas/scipy incompatibility
            ctrl_mask = (self.pert_data.adata.obs['condition'] == 'ctrl').values
            self.pert_data.adata.obs['_ctrl_mask'] = ctrl_mask

        # prepare split and dataloaders (required before GEARS initialization)
        self.pert_data.prepare_split(split=split, seed=seed)
        self.pert_data.get_dataloader(batch_size=batch_size, test_batch_size=test_batch_size)

        # initialize and load pre-trained model
        self.model = GEARS(self.pert_data, device=self.device)
        self.model.load_pretrained(model_dir)
        self.fitted = True

        print(f"GEARS model loaded from {model_dir}")
    
    def train_gears(self, model_dir: str, data_path: str, data_name: str = None,
                split: str = 'simulation', seed: int = 1,
                hidden_size: int = 64, epochs: int = 20, 
                batch_size: int = 32, test_batch_size: int = 128):
        """
        Train a new GEARS model.
        
        args:
            model_dir: Directory to save model
            data_path: Path to data directory OR path to specific h5ad file
            data_name: Dataset name ('norman', 'adamson', 'dixit') or None for custom data
            split: Split type ('simulation', 'combo_seen0', 'combo_seen1', 'combo_seen2')
            seed: Random seed
            hidden_size: Hidden layer size
            epochs: Number of training epochs
            batch_size: Training batch size
            test_batch_size: Test batch size
        """
        from gears import PertData, GEARS
        
        # Use built-in GEARS dataset
        if data_name not in ['norman', 'adamson', 'dixit']:
            raise ValueError(f"data_name must be 'norman', 'adamson', or 'dixit', got: {data_name}")

        self.pert_data = PertData(data_path)
        self.pert_data.load(data_name=data_name)
        print(f"Loaded GEARS built-in dataset: {data_name}")

        self.pert_data.prepare_split(split=split, seed=seed)
        self.pert_data.get_dataloader(batch_size=batch_size, test_batch_size=test_batch_size)
        
        self.model = GEARS(self.pert_data, device=self.device)
        self.model.model_initialize(hidden_size=hidden_size)
        self.model.train(epochs=epochs)
        self.model.save_model(model_dir)
        self.fitted = True
        
        print(f"GEARS model trained and saved to {model_dir}")
        
    def predict_perturbation(self, genes: list) -> np.ndarray:
        """
        Predict perturbation effect using GEARS.
        
        args:
            genes: List of gene symbols, e.g., ['CBL', 'CNN1'] for combo or ['FEV'] for single
            
        returns:
            predicted_expression: Predicted gene expression (mean and variance)
        """
        if not self.fitted:
            raise ValueError("Model must be loaded before prediction")
        
        # GEARS expects list of lists for batch prediction
        result = self.model.predict([genes])
        return result
    
    def predict_genetic_interaction(self, gene1: str, gene2: str, GI_genes_file: str = None):
        """
        Predict genetic interaction score.
        
        args:
            gene1, gene2: Gene symbols
            GI_genes_file: Optional file with genes to compute GI for
        """
        if not self.fitted:
            raise ValueError("Model must be loaded before prediction")
            
        return self.model.GI_predict([gene1, gene2], GI_genes_file=GI_genes_file)

    def batch_predict(self, perturbation_list: list) -> dict:
        """
        Batch prediction for multiple perturbations.
        
        args:
            perturbation_list: List of gene lists, e.g., [['CBL', 'CNN1'], ['FEV'], ['SAMD1', 'ZBTB1']]
            
        returns:
            Dictionary with predictions
        """
        if not self.fitted:
            raise ValueError("Model must be loaded before prediction")
        
        return self.model.predict(perturbation_list)


# ================================================================================
# scLAMBDA integration
# ================================================================================

class scLAMBDAWrapper:
    """scLAMBDA model wrapper following actual API"""
    
    def __init__(self, sclambda_path: str = None):
        """
        args:
            sclambda_path: Path to cloned scLAMBDA repo (adds to sys.path if needed)
        """
        if sclambda_path is not None and sclambda_path not in sys.path:
            sys.path.insert(0, sclambda_path)
        
        self.model = None
        self.adata = None
        self.gene_embeddings = None
        self.fitted = False
        
    def load_data_and_embeddings(self, adata_path: str, gene_embeddings_path: str):
        """
        Load data and gene embeddings required for scLAMBDA.
        
        args:
            adata_path: Path to h5ad file
            gene_embeddings_path: Path to gene embeddings dict (pickle or npy)
        """
        import pickle
        
        self.adata = sc.read_h5ad(adata_path)
        
        # load gene embeddings - expects dict {"gene_name": np.array}
        if gene_embeddings_path.endswith('.pkl') or gene_embeddings_path.endswith('.pickle'):
            with open(gene_embeddings_path, 'rb') as f:
                self.gene_embeddings = pickle.load(f)
        elif gene_embeddings_path.endswith('.npy'):
            self.gene_embeddings = np.load(gene_embeddings_path, allow_pickle=True).item()
        else:
            raise ValueError("Gene embeddings must be .pkl, .pickle, or .npy file")
    
        # make sure there's a column called condition
        if 'perturbation' in self.adata.obs.columns:
            self.adata.obs['condition'] = self.adata.obs['perturbation']
        
        print(f"Loaded data with {self.adata.n_obs} cells and {len(self.gene_embeddings)} gene embeddings")
        
    def train_sclambda(self, model_path: str, multi_gene: bool = True, 
                       split_type: str = None, seed: int = 0):
        """
        Train scLAMBDA model.
        
        args:
            model_path: Directory to save model
            multi_gene: True for two-gene perturbations, False for single-gene
            split_type: 'single' for single-gene, 'all_train' for no holdout, or None for default
            seed: Random seed for split
        """
        import sclambda
        
        if self.adata is None or self.gene_embeddings is None:
            raise ValueError("Must call load_data_and_embeddings first")
        
        # data split
        if split_type == 'single':
            self.adata, split = sclambda.utils.data_split(self.adata, split_type='single', seed=seed)
        elif split_type == 'all_train':
            self.adata, split = sclambda.utils.data_split(self.adata, split_type='all_train', seed=seed)
        else:
            self.adata, split = sclambda.utils.data_split(self.adata, seed=seed)
        
        # initialize and train
        self.model = sclambda.model.Model(
            self.adata,
            self.gene_embeddings,
            model_path=model_path,
            multi_gene=multi_gene
        )
        self.model.train()
        self.fitted = True
        
        print(f"scLAMBDA model trained and saved to {model_path}")
        
    def load_pretrained_sclambda(self, model_path: str, adata_path: str = None,
                                  gene_embeddings_path: str = None, multi_gene: bool = True):
        """
        Load pre-trained scLAMBDA model.

        args:
            model_path: Directory where model was saved
            adata_path: Path to h5ad file (needed if not already loaded)
            gene_embeddings_path: Path to gene embeddings (needed if not already loaded)
            multi_gene: Whether model was trained for multi-gene perturbations
        """
        import sclambda
        import pickle

        # Load gene embeddings separately (don't load full dataset)
        if self.gene_embeddings is None and gene_embeddings_path is not None:
            if gene_embeddings_path.endswith('.pkl') or gene_embeddings_path.endswith('.pickle'):
                with open(gene_embeddings_path, 'rb') as f:
                    self.gene_embeddings = pickle.load(f)
            elif gene_embeddings_path.endswith('.npy'):
                self.gene_embeddings = np.load(gene_embeddings_path, allow_pickle=True).item()
            print(f"Loaded {len(self.gene_embeddings)} gene embeddings")

        # Load only a minimal subset of cells for model initialization
        # The pretrained weights don't depend on these cells - they're only needed for structure
        if self.adata is None and adata_path is not None:
            print("Loading minimal dataset for model initialization...")
            adata_full = sc.read_h5ad(adata_path)
            print(f"Full dataset: {adata_full.n_obs} cells")

            # Take only 500 cells for initialization (much faster than 100K+ cells)
            n_init_cells = min(500, adata_full.n_obs)
            self.adata = adata_full[:n_init_cells, :].copy()

            # Make sure there's a column called condition
            if 'perturbation' in self.adata.obs.columns:
                self.adata.obs['condition'] = self.adata.obs['perturbation']

            del adata_full
            print(f"Using {self.adata.n_obs} cells for initialization (speeds up loading by ~100x)")
        elif self.adata is None:
            raise ValueError("Must provide adata_path or call load_data_and_embeddings first")

        # data split is required for scLAMBDA model initialization
        # use 'all_train' to avoid creating actual train/test splits since using pretrained model
        self.adata, split = sclambda.utils.data_split(self.adata, split_type='all_train', seed=0)

        # initialize model - this computes perturbation embeddings for only 500 cells (not 100K!)
        print("Initializing scLAMBDA model architecture...")
        self.model = sclambda.model.Model(
            self.adata,
            self.gene_embeddings,
            model_path=model_path,
            multi_gene=multi_gene
        )

        # Load pretrained weights using scLAMBDA's built-in method
        print("Loading pretrained weights from ckpt.pth...")
        self.model.load_pretrain()
        self.fitted = True
        print(f"scLAMBDA model loaded from {model_path}")
        
    def predict(self, perturbations: list, return_type: str = 'mean') -> np.ndarray:
        """
        Predict perturbation effects.

        args:
            perturbations: List of perturbation strings, e.g., ['CBL+CNN1', 'FEV+ctrl']
            return_type: 'mean' for mean expression, 'cells' for single-cell predictions

        returns:
            Predicted expression as numpy array
        """
        if not self.fitted:
            raise ValueError("Model must be loaded before prediction")

        # Filter perturbations - scLAMBDA with multi_gene=True requires exactly 2 genes
        # Also check if genes have embeddings
        valid_perts = []
        invalid_indices = []
        for i, pert in enumerate(perturbations):
            genes = pert.split('+')
            # Remove 'ctrl' from gene list for counting
            actual_genes = [g for g in genes if g != 'ctrl']

            # Check if all genes have embeddings
            has_embeddings = all(g in self.gene_embeddings for g in actual_genes)

            if len(actual_genes) >= 1 and has_embeddings:  # At least 1 real gene with embeddings
                valid_perts.append(pert)
            else:
                invalid_indices.append(i)

        # Print warning if some genes were skipped
        if len(invalid_indices) > 0:
            missing_genes = set()
            for idx in invalid_indices[:5]:  # Show up to 5 examples
                genes = perturbations[idx].split('+')
                actual_genes = [g for g in genes if g != 'ctrl']
                for g in actual_genes:
                    if g not in self.gene_embeddings:
                        missing_genes.add(g)
            if missing_genes:
                print(f"Warning: {len(invalid_indices)} perturbations skipped due to missing gene embeddings (e.g., {list(missing_genes)[:3]})")

        # Get predictions for valid perturbations
        if len(valid_perts) > 0:
            results_dict = self.model.predict(valid_perts, return_type=return_type)
        else:
            results_dict = {}

        # Build output array, using zeros for invalid perturbations
        predictions = []
        valid_idx = 0
        for i, pert in enumerate(perturbations):
            if i in invalid_indices:
                # Use zero vector for control/invalid samples
                if len(results_dict) > 0:
                    example_pred = list(results_dict.values())[0]
                    predictions.append(np.zeros_like(example_pred))
                else:
                    predictions.append(np.zeros(self.model.x_dim))
            else:
                predictions.append(results_dict[valid_perts[valid_idx]])
                valid_idx += 1

        # Convert to array and ensure correct shape (n_samples, n_genes)
        predictions_array = np.array(predictions)

        # Squeeze out extra dimensions if present
        while predictions_array.ndim > 2:
            predictions_array = np.squeeze(predictions_array, axis=1)

        return predictions_array
    
    def generate(self, perturbations: list, return_type: str = 'cells') -> np.ndarray:
        """
        Generate new perturbed cells.

        args:
            perturbations: List of perturbation strings
            return_type: 'mean' or 'cells'

        returns:
            Generated expression profiles as numpy array
        """
        if not self.fitted:
            raise ValueError("Model must be loaded before prediction")

        # scLAMBDA returns a dict {perturbation: generated_cells}
        results_dict = self.model.generate(perturbations, return_type=return_type)

        # Convert to numpy array in the same order as input
        generated = np.array([results_dict[pert] for pert in perturbations])

        return generated


# ================================================================================
# unified data preprocessing
# ================================================================================

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
    
    GEARS_MODEL_DIR = f'{DATA_DIR}/gears_model'  # will be created
    GEARS_DATA_PATH = f'{DATA_DIR}/gears_data'  # contains downloaded GEARS data

    SCLAMBDA_MODEL_PATH = f'{DATA_DIR}/sclambda_model'  # will be created
    SCLAMBDA_ADATA_PATH = f'{DATA_DIR}/norman_perturbseq_preprocessed_hvg_filtered.h5ad'
    SCLAMBDA_EMBEDDINGS_PATH = f'{DATA_DIR}/GPT_3_5_gene_embeddings_3-large.pickle'
    NORMAN_DATA_PATH = f'{DATA_DIR}/norman_perturbseq_preprocessed_hvg_filtered.h5ad'

    # ==========================================
    # step 1: train GEARS (using built-in norman dataset)
    # ==========================================
    print("\n[1/3] Training GEARS with built-in norman dataset...")
    gears = GEARSWrapper()
    gears.train_gears(
        model_dir=GEARS_MODEL_DIR,
        data_path=GEARS_DATA_PATH,
        data_name='norman',
        split='simulation',
        seed=1,
        hidden_size=64,
        epochs=10
    )
    
    # ==========================================
    # step 2: train scLAMBDA (using custom preprocessed data)
    # ==========================================
    print("\n[2/3] Training scLAMBDA with custom preprocessed data...")
    sclambda = scLAMBDAWrapper(sclambda_path=SCLAMBDA_REPO)
    sclambda.load_data_and_embeddings(SCLAMBDA_ADATA_PATH, SCLAMBDA_EMBEDDINGS_PATH)
    sclambda.train_sclambda(
        model_path=SCLAMBDA_MODEL_PATH,
        multi_gene=True,
        seed=0
    )
    
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