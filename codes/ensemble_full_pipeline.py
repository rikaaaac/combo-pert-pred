"""
ensemble_full_pipeline.py used for training GEARS + scLAMBDA Ensemble
use ensemble.py for learning weights, calibrated uncertainty, and evaluation
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
# GEARS
# ================================================================================

class GEARSWrapper:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = None
        self.pert_data = None
        self.fitted = False
        
    def load_pretrained_gears(self, model_dir: str, data_path: str, data_name: str = None,
                              split: str = 'simulation', seed: int = 1,
                              batch_size: int = 32, test_batch_size: int = 128,
                              custom_adata_path: str = None):
        """
        Load the gears model

        args:
            model_dir: Directory where model was saved (e.g., 'gears_model')
            data_path: Path to data directory (e.g., './data') - working directory for GEARS
            data_name: Name of dataset if using built-in ('norman', 'adamson', 'dixit')
                      If None and custom_adata_path is provided, will load custom data
            split: Split type (needed for dataloader creation)
            seed: Random seed (needed for dataloader creation)
            batch_size: Batch size for dataloaders
            test_batch_size: Test batch size for dataloaders
            custom_adata_path: Path to custom h5ad file (required if data_name is None)
        """
        from gears import PertData, GEARS

        # Ensure data_path directory exists (required for PertData initialization)
        if not os.path.exists(data_path):
            print(f"Warning: data_path directory does not exist: {data_path}")
            print(f"Creating directory: {data_path}")
            os.makedirs(data_path, exist_ok=True)
        
        # load perturbation data
        self.pert_data = PertData(data_path)

        if data_name is not None:
            # use built-in dataset
            self.pert_data.load(data_name=data_name)
            print(f"Loaded GEARS built-in dataset: {data_name}")
        else:
            # load custom processed data from h5ad file
            if custom_adata_path is None:
                raise ValueError("Either data_name or custom_adata_path must be provided")
            
            if not os.path.exists(custom_adata_path):
                raise FileNotFoundError(f"Custom data file not found: {custom_adata_path}")
            
            print(f"Loading custom GEARS dataset from: {custom_adata_path}")
            # Read h5ad file and process using new_data_process
            try:
                print(f"Reading h5ad file: {custom_adata_path}")
                adata = sc.read_h5ad(custom_adata_path)
                
                # Verify file was read successfully
                if adata is None:
                    raise ValueError(f"Failed to read h5ad file: {custom_adata_path} (returned None)")
                
                # Verify it's a valid AnnData object
                if not hasattr(adata, 'obs') or not hasattr(adata, 'var'):
                    raise ValueError(f"Invalid AnnData object read from: {custom_adata_path}")
                
                print(f"Successfully read h5ad file: {adata.n_obs} cells, {adata.n_vars} genes")
                
            except Exception as e:
                raise RuntimeError(
                    f"Failed to read h5ad file '{custom_adata_path}': {str(e)}\n"
                    f"Please verify the file exists and is a valid h5ad file."
                ) from e
            
            # Verify required columns exist
            if 'condition' not in adata.obs.columns:
                # Check if 'perturbation' column exists as alternative
                if 'perturbation' in adata.obs.columns:
                    print("Warning: 'condition' column not found, using 'perturbation' column")
                    adata.obs['condition'] = adata.obs['perturbation']
                else:
                    raise ValueError(
                        f"AnnData must have 'condition' or 'perturbation' column in obs for GEARS. "
                        f"Found columns: {list(adata.obs.columns)}"
                    )
            
            # Verify condition column has data
            if adata.obs['condition'].isna().all():
                raise ValueError("'condition' column exists but all values are NaN")

            # GEARS expects specific format - prepare the data
            print(f"Preparing data for GEARS new_data_process()...")

            # Ensure gene_symbols column exists (GEARS requirement)
            if 'gene_symbols' not in adata.var.columns:
                print("Adding 'gene_symbols' column to var (using index)")
                adata.var['gene_symbols'] = adata.var_names.tolist()

            # Ensure gene_name column exists in var (some GEARS versions need this)
            if 'gene_name' not in adata.var.columns:
                print("Adding 'gene_name' column to var (using gene_symbols)")
                adata.var['gene_name'] = adata.var.get('gene_symbols', adata.var_names).tolist()

            # Normalize perturbation names to match GEARS format
            print("Normalizing perturbation names...")
            # Replace underscore with '+' for multi-gene perturbations
            adata.obs['condition'] = adata.obs['condition'].astype(str).str.replace('_', '+')

            # Ensure ctrl is used (not control)
            def normalize_control(s):
                parts = s.split('+')
                parts = ['ctrl' if p.lower() == 'control' else p for p in parts]
                return '+'.join(parts)
            adata.obs['condition'] = adata.obs['condition'].apply(normalize_control)

            print(f"Sample conditions: {adata.obs['condition'].unique()[:5].tolist()}")

            # BYPASS new_data_process() - manually set up GEARS data structures
            print(f"Manually setting up GEARS data structures (bypassing new_data_process)...")

            # CRITICAL: Prepare adata BEFORE setting it to self.pert_data.adata
            # Add required columns for prepare_split
            if 'condition_name' not in adata.obs.columns:
                adata.obs['condition_name'] = adata.obs['condition']

            # Ensure obs columns are pandas Series (not numpy arrays) for prepare_split
            import pandas as pd
            for col in adata.obs.columns:
                if not isinstance(adata.obs[col], pd.Series):
                    adata.obs[col] = pd.Series(adata.obs[col].values, index=adata.obs.index)

            # Now set the adata (after preprocessing)
            self.pert_data.adata = adata

            # Set dataset_path (required by prepare_split)
            self.pert_data.dataset_path = data_path

            # Extract unique perturbations and create perturbation list
            self.pert_data.gene_names = adata.var_names.tolist()

            # Get unique perturbations from condition column
            unique_conditions = adata.obs['condition'].unique()
            self.pert_data.pert_list = [p for p in unique_conditions if p != 'ctrl']

            # Set dataset name
            self.pert_data.dataset_name = 'norman_custom'

            # pert_names should be a numpy array (not a set) for GEARS
            self.pert_data.pert_names = np.array(self.pert_data.pert_list)

            # Ensure ctrl_adata exists (required by GEARS)
            ctrl_mask = adata.obs['condition'] == 'ctrl'
            if ctrl_mask.sum() > 0:
                self.pert_data.ctrl_adata = adata[ctrl_mask].copy()
            else:
                print("Warning: No control samples found!")

            print(f"Successfully set up custom GEARS dataset:")
            print(f"  - {self.pert_data.adata.n_obs} cells")
            print(f"  - {len(self.pert_data.gene_names)} genes")
            print(f"  - {len(self.pert_data.pert_list)} unique perturbations")
            print(f"  - {ctrl_mask.sum()} control cells")
            print(f"  - dataset_path: {self.pert_data.dataset_path}")

            # Create dataset_processed (required by get_dataloader)
            # This is normally done by GEARS data processing, but we bypassed it
            print(f"Creating dataset_processed for dataloader...")
            from collections import defaultdict
            self.pert_data.dataset_processed = defaultdict(list)

            # Group cells by perturbation condition
            for idx, condition in enumerate(adata.obs['condition']):
                # For each cell, store its index grouped by condition
                # get_dataloader will use this to create cell graphs
                self.pert_data.dataset_processed[condition].append(idx)

            print(f"  - Processed {len(self.pert_data.dataset_processed)} unique conditions")

            # Create node_map_pert (required by GEARS model initialization)
            # Maps gene names to node indices
            self.pert_data.node_map_pert = {gene: idx for idx, gene in enumerate(self.pert_data.gene_names)}

            # Set ctrl_str (control string identifier)
            self.pert_data.ctrl_str = 'ctrl'

            # Ensure adata.X is sparse matrix (GEARS expects this)
            if not hasattr(adata.X, 'toarray'):
                print(f"Converting dense matrix to sparse format for GEARS...")
                from scipy.sparse import csr_matrix
                adata.X = csr_matrix(adata.X)
                print(f"  - Converted to sparse matrix: {adata.X.shape}")

            # Create non_zeros_gene_idx (required by GEARS)
            # This identifies which genes have non-zero expression
            if 'non_zeros_gene_idx' not in adata.uns:
                print(f"Computing non-zero gene indices...")
                # Find genes with non-zero expression across all cells
                X_dense = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X

                non_zero_genes = {}
                for idx, gene in enumerate(self.pert_data.gene_names):
                    if np.any(X_dense[:, idx] != 0):
                        non_zero_genes[gene] = idx

                adata.uns['non_zeros_gene_idx'] = non_zero_genes
                print(f"  - Found {len(non_zero_genes)} genes with non-zero expression")

            print(f"  - Created node_map_pert with {len(self.pert_data.node_map_pert)} genes")

        # Ensure condition column exists and create control mask
        if 'condition' not in self.pert_data.adata.obs.columns:
            raise ValueError("GEARS data must have 'condition' column in obs")
        
        if hasattr(self.pert_data.adata.obs['condition'], 'values'):
            # force conversion to avoid pandas/scipy incompatibility
            ctrl_mask = (self.pert_data.adata.obs['condition'] == 'ctrl').values
            self.pert_data.adata.obs['_ctrl_mask'] = ctrl_mask

        # prepare split and dataloaders (required before GEARS initialization)
        self.pert_data.prepare_split(split=split, seed=seed)
        self.pert_data.get_dataloader(batch_size=batch_size, test_batch_size=test_batch_size)

        # initialize and load model
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
            data_path: Path to data directory (for built-in datasets)
            data_name: Dataset name ('norman', 'adamson', 'dixit')
            split: Split type ('simulation', 'combo_seen0', 'combo_seen1', 'combo_seen2')
            seed: Random seed
            hidden_size: Hidden layer size
            epochs: Number of training epochs
            batch_size: Training batch size
            test_batch_size: Test batch size
        """
        from gears import PertData, GEARS

        if data_name is None or data_name not in ['norman', 'adamson', 'dixit']:
            raise ValueError(f"data_name must be 'norman', 'adamson', or 'dixit', got: {data_name}")

        self.pert_data = PertData(data_path)
        self.pert_data.load(data_name=data_name)
        print(f"Loaded GEARS built-in dataset: {data_name}")

        # prepare split and dataloaders (required before GEARS initialization)
        self.pert_data.prepare_split(split=split, seed=seed)
        self.pert_data.get_dataloader(batch_size=batch_size, test_batch_size=test_batch_size)

        import gears.utils as gears_utils

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
# sclambda
# ================================================================================

class scLAMBDAWrapper:
    def __init__(self, sclambda_path: str = None):
        """
        args:
            sclambda_path: Path to cloned scLAMBDA repo
        """
        if sclambda_path is not None and sclambda_path not in sys.path:
            sys.path.insert(0, sclambda_path)
        
        self.model = None
        self.adata = None
        self.gene_embeddings = None
        self.fitted = False
        
    def load_data_and_embeddings(self, adata_path: str, gene_embeddings_path: str):
        """
        Load data and gene embeddings required for scLAMBDA
        
        args:
            adata_path: Path to h5ad file
            gene_embeddings_path: Path to gene embeddings dict
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

        # Check for NaN/Inf values in gene embeddings
        n_bad_embeddings = 0
        for gene, emb in self.gene_embeddings.items():
            if np.any(np.isnan(emb)) or np.any(np.isinf(emb)):
                n_bad_embeddings += 1
                self.gene_embeddings[gene] = np.nan_to_num(emb, nan=0.0, posinf=0.0, neginf=0.0)
        if n_bad_embeddings > 0:
            print(f"Warning: Found and fixed {n_bad_embeddings} gene embeddings with NaN/Inf values")
    
        # make sure there's a column called condition
        if 'perturbation' in self.adata.obs.columns:
            self.adata.obs['condition'] = self.adata.obs['perturbation']

        # normalize perturbation names
        if 'condition' in self.adata.obs.columns:
            # 1. Replace underscore with '+' for multi-gene perturbations first
            # scLAMBDA expects gene names separated by '+' (e.g., 'AHR+FEV' not 'AHR_FEV')
            self.adata.obs['condition'] = self.adata.obs['condition'].astype(str).str.replace('_', '+')
            # 2. Handle control conditions: normalize both to 'ctrl'
            # Use str.replace with whole word matching by splitting and rejoining
            def normalize_control(s):
                parts = s.split('+')
                parts = ['ctrl' if p.lower() == 'control' else p for p in parts]
                return '+'.join(parts)
            self.adata.obs['condition'] = self.adata.obs['condition'].apply(normalize_control)

        # Check for NaN/Inf values in expression data
        from scipy import sparse
        if sparse.issparse(self.adata.X):
            # For sparse matrices, convert to dense for checking
            X_dense = self.adata.X.toarray()
            has_nan = np.any(np.isnan(X_dense))
            has_inf = np.any(np.isinf(X_dense))
            if has_nan or has_inf:
                n_nan = np.sum(np.isnan(X_dense)) if has_nan else 0
                n_inf = np.sum(np.isinf(X_dense)) if has_inf else 0
                if has_nan:
                    print(f"Warning: Found {n_nan} NaN values in expression data. Replacing with 0.")
                if has_inf:
                    print(f"Warning: Found {n_inf} Inf values in expression data. Replacing with 0.")
                X_clean = np.nan_to_num(X_dense, nan=0.0, posinf=0.0, neginf=0.0)
                self.adata.X = sparse.csr_matrix(X_clean)
        else:
            # For dense matrices
            if np.any(np.isnan(self.adata.X)):
                n_nan = np.sum(np.isnan(self.adata.X))
                print(f"Warning: Found {n_nan} NaN values in expression data. Replacing with 0.")
                self.adata.X = np.nan_to_num(self.adata.X, nan=0.0)
            if np.any(np.isinf(self.adata.X)):
                n_inf = np.sum(np.isinf(self.adata.X))
                print(f"Warning: Found {n_inf} Inf values in expression data. Replacing with 0.")
                self.adata.X = np.nan_to_num(self.adata.X, nan=0.0, posinf=0.0, neginf=0.0)

        print(f"Loaded data with {self.adata.n_obs} cells and {len(self.gene_embeddings)} gene embeddings")

        # Print data statistics to help debug potential issues
        from scipy import sparse
        if sparse.issparse(self.adata.X):
            X_sample = self.adata.X[:1000].toarray()  # Sample for efficiency
        else:
            X_sample = self.adata.X[:1000] if self.adata.X.shape[0] > 1000 else self.adata.X

        print(f"Expression data stats: min={X_sample.min():.3f}, max={X_sample.max():.3f}, mean={X_sample.mean():.3f}, std={X_sample.std():.3f}")
        
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

        # filter out perturbations with genes that don't have embeddings
        print(f"Initial dataset: {self.adata.n_obs} cells with {len(self.adata.obs['condition'].unique())} unique perturbations")

        valid_mask = []
        missing_genes = set()
        for condition in self.adata.obs['condition']:
            genes = condition.split('+')
            actual_genes = [g for g in genes if g != 'ctrl']
            has_all_embeddings = all(g in self.gene_embeddings for g in actual_genes)
            valid_mask.append(has_all_embeddings)

            if not has_all_embeddings:
                for g in actual_genes:
                    if g not in self.gene_embeddings:
                        missing_genes.add(g)

        if missing_genes:
            print(f"Warning: Filtering out perturbations with {len(missing_genes)} genes lacking embeddings")
            print(f"  Examples of missing genes: {list(missing_genes)[:10]}")

        # filter to valid perturbations
        self.adata = self.adata[valid_mask, :].copy()
        print(f"Filtered dataset: {self.adata.n_obs} cells with {len(self.adata.obs['condition'].unique())} unique perturbations")

        if self.adata.n_obs == 0:
            raise ValueError("No perturbations remain after filtering for gene embeddings")

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
                                  gene_embeddings_path: str = None, multi_gene: bool = True,
                                  seed: int = 0):
        """
        Load pre-trained scLAMBDA model.

        args:
            model_path: Directory where model was saved
            adata_path: Path to h5ad file (needed if not already loaded)
            gene_embeddings_path: Path to gene embeddings (needed if not already loaded)
            multi_gene: Whether model was trained for multi-gene perturbations
            seed: Random seed for cell sampling during initialization
        """
        import sclambda
        import pickle

        # load gene embeddings separately (don't load full dataset)
        if self.gene_embeddings is None and gene_embeddings_path is not None:
            if gene_embeddings_path.endswith('.pkl') or gene_embeddings_path.endswith('.pickle'):
                with open(gene_embeddings_path, 'rb') as f:
                    self.gene_embeddings = pickle.load(f)
            elif gene_embeddings_path.endswith('.npy'):
                self.gene_embeddings = np.load(gene_embeddings_path, allow_pickle=True).item()

            # Check for NaN/Inf values in gene embeddings
            n_bad_embeddings = 0
            for gene, emb in self.gene_embeddings.items():
                if np.any(np.isnan(emb)) or np.any(np.isinf(emb)):
                    n_bad_embeddings += 1
                    self.gene_embeddings[gene] = np.nan_to_num(emb, nan=0.0, posinf=0.0, neginf=0.0)
            if n_bad_embeddings > 0:
                print(f"Warning: Found and fixed {n_bad_embeddings} gene embeddings with NaN/Inf values")

            print(f"Loaded {len(self.gene_embeddings)} gene embeddings")

        # load only a minimal subset of cells for model initialization
        # the pretrained weights don't depend on these cells, only needed for structure
        # but need to cover all unique perturbations for proper initialization
        if self.adata is None and adata_path is not None:
            print("Loading minimal dataset for model initialization...")
            adata_full = sc.read_h5ad(adata_path)
            print(f"Full dataset: {adata_full.n_obs} cells")

            # make sure there's a column called condition
            if 'perturbation' in adata_full.obs.columns:
                adata_full.obs['condition'] = adata_full.obs['perturbation']

            # normalize perturbation names
            if 'condition' in adata_full.obs.columns:
                # 1. Replace underscore with '+' for multi-gene perturbations first
                # scLAMBDA expects gene names separated by '+' (e.g., 'AHR+FEV' not 'AHR_FEV')
                adata_full.obs['condition'] = adata_full.obs['condition'].astype(str).str.replace('_', '+')
                # 2. Handle control conditions: normalize both to 'ctrl'
                # Use str.replace with whole word matching by splitting and rejoining
                def normalize_control(s):
                    parts = s.split('+')
                    parts = ['ctrl' if p.lower() == 'control' else p for p in parts]
                    return '+'.join(parts)
                adata_full.obs['condition'] = adata_full.obs['condition'].apply(normalize_control)

            # filter out perturbations with genes that don't have embeddings
            valid_mask = []
            missing_genes = set()
            for condition in adata_full.obs['condition']:
                genes = condition.split('+')
                actual_genes = [g for g in genes if g != 'ctrl']
                has_all_embeddings = all(g in self.gene_embeddings for g in actual_genes)
                valid_mask.append(has_all_embeddings)

                if not has_all_embeddings:
                    for g in actual_genes:
                        if g not in self.gene_embeddings:
                            missing_genes.add(g)

            if missing_genes:
                print(f"Warning: Filtering out perturbations with {len(missing_genes)} genes lacking embeddings")
                print(f"  Examples of missing genes: {list(missing_genes)[:10]}")

            # filter to valid perturbations
            adata_full = adata_full[valid_mask, :].copy()
            print(f"Filtered to {adata_full.n_obs} cells after removing perturbations with missing gene embeddings")

            # Check for NaN/Inf values in expression data
            from scipy import sparse
            if sparse.issparse(adata_full.X):
                # For sparse matrices, convert to dense for checking
                X_dense = adata_full.X.toarray()
                has_nan = np.any(np.isnan(X_dense))
                has_inf = np.any(np.isinf(X_dense))
                if has_nan or has_inf:
                    n_nan = np.sum(np.isnan(X_dense)) if has_nan else 0
                    n_inf = np.sum(np.isinf(X_dense)) if has_inf else 0
                    if has_nan:
                        print(f"Warning: Found {n_nan} NaN values in expression data. Replacing with 0.")
                    if has_inf:
                        print(f"Warning: Found {n_inf} Inf values in expression data. Replacing with 0.")
                    X_clean = np.nan_to_num(X_dense, nan=0.0, posinf=0.0, neginf=0.0)
                    adata_full.X = sparse.csr_matrix(X_clean)
            else:
                # For dense matrices
                if np.any(np.isnan(adata_full.X)):
                    n_nan = np.sum(np.isnan(adata_full.X))
                    print(f"Warning: Found {n_nan} NaN values in expression data. Replacing with 0.")
                    adata_full.X = np.nan_to_num(adata_full.X, nan=0.0)
                if np.any(np.isinf(adata_full.X)):
                    n_inf = np.sum(np.isinf(adata_full.X))
                    print(f"Warning: Found {n_inf} Inf values in expression data. Replacing with 0.")
                    adata_full.X = np.nan_to_num(adata_full.X, nan=0.0, posinf=0.0, neginf=0.0)

            # get unique perturbations to ensure coverage
            unique_perts = adata_full.obs['condition'].unique()
            print(f"Found {len(unique_perts)} unique perturbations with complete embeddings")

            # sample cells to represent all unique perturbations
            # strategy: take at least 1 cell per perturbation, up to max_cells total
            max_init_cells = min(500, adata_full.n_obs)
            n_perts = len(unique_perts)
            
            # Set random seed for reproducibility
            rng = np.random.RandomState(seed=seed)
            
            if n_perts <= max_init_cells:
                # sample multiple cells per perturbation but evenly
                cells_per_pert = max(1, max_init_cells // n_perts)
                selected_indices = []
                
                for pert in unique_perts:
                    pert_mask = adata_full.obs['condition'] == pert
                    pert_indices = np.where(pert_mask)[0]
                    n_sample = min(cells_per_pert, len(pert_indices))
                    selected_indices.extend(
                        rng.choice(pert_indices, size=n_sample, replace=False).tolist()
                    )
                
                # if cells are still less than 500, add more random cells
                remaining = max_init_cells - len(selected_indices)
                if remaining > 0:
                    all_indices = set(range(adata_full.n_obs))
                    remaining_indices = list(all_indices - set(selected_indices))
                    if len(remaining_indices) > 0:
                        n_add = min(remaining, len(remaining_indices))
                        selected_indices.extend(
                            rng.choice(remaining_indices, size=n_add, replace=False).tolist()
                        )
                
                self.adata = adata_full[selected_indices, :].copy()
            else:
                # too many and just take one cell per perturbation
                selected_indices = []
                for pert in unique_perts:
                    pert_mask = adata_full.obs['condition'] == pert
                    pert_indices = np.where(pert_mask)[0]
                    if len(pert_indices) > 0:
                        selected_indices.append(rng.choice(pert_indices))
                
                self.adata = adata_full[selected_indices, :].copy()
                print(f"Warning: {n_perts} perturbations found, using 1 cell per perturbation ({len(selected_indices)} cells)")

            del adata_full
            print(f"Using {self.adata.n_obs} cells for initialization (covering {len(self.adata.obs['condition'].unique())} unique perturbations)")

        elif self.adata is None:
            raise ValueError("Must provide adata_path or call load_data_and_embeddings first")

        # data split is required for scLAMBDA model initialization
        # use 'all_train' to avoid creating actual train/test splits since using pretrained model
        self.adata, split = sclambda.utils.data_split(self.adata, split_type='all_train', seed=0)

        # initialize model, this computes perturbation embeddings for only the 500 cells defined above
        print("Initializing scLAMBDA model architecture...")
        self.model = sclambda.model.Model(
            self.adata,
            self.gene_embeddings,
            model_path=model_path,
            multi_gene=multi_gene
        )

        # load sclambda model
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

        # change perturbation names
        # 1. Replace underscore with '+' for multi-gene perturbations
        # scLAMBDA expects gene names separated by '+' ('AHR+FEV')
        perturbations = [p.replace('_', '+') for p in perturbations]
        # 2. Normalize 'control' to 'ctrl'
        def normalize_control(s):
            parts = s.split('+')
            parts = ['ctrl' if p.lower() == 'control' else p for p in parts]
            return '+'.join(parts)
        perturbations = [normalize_control(p) for p in perturbations]

        # filter perturbations, scLAMBDA with multi_gene=True requires exactly 2 genes
        # also check if genes have embeddings
        valid_perts = []
        invalid_indices = []
        for i, pert in enumerate(perturbations):
            genes = pert.split('+')
            # remove 'ctrl' from gene list for counting
            actual_genes = [g for g in genes if g != 'ctrl']

            # check if all genes have embeddings
            has_embeddings = all(g in self.gene_embeddings for g in actual_genes)

            if len(actual_genes) >= 1 and has_embeddings:  # at least 1 real gene with embeddings
                valid_perts.append(pert)
            else:
                invalid_indices.append(i)

        # print warning if some genes were skipped
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

        # get predictions for valid perturbations
        if len(valid_perts) > 0:
            results_dict = self.model.predict(valid_perts, return_type=return_type)
        else:
            results_dict = {}

        # build output array, using zeros for invalid perturbations
        predictions = []
        valid_idx = 0
        for i, pert in enumerate(perturbations):
            if i in invalid_indices:
                # use zero vector for control/invalid samples
                if len(results_dict) > 0:
                    example_pred = list(results_dict.values())[0]
                    predictions.append(np.zeros_like(example_pred))
                else:
                    # Fallback: use adata shape if model.x_dim not available
                    if hasattr(self.model, 'x_dim'):
                        predictions.append(np.zeros(self.model.x_dim))
                    elif self.adata is not None:
                        predictions.append(np.zeros(self.adata.n_vars))
                    else:
                        raise ValueError(
                            "Cannot determine output dimension: no valid predictions, "
                            "no model.x_dim, and no adata available"
                        )
            else:
                predictions.append(results_dict[valid_perts[valid_idx]])
                valid_idx += 1

        # convert to array and ensure correct shape (n_samples, n_genes)
        predictions_array = np.array(predictions)

        # squeeze out extra dimensions if present
        while predictions_array.ndim > 2:
            predictions_array = np.squeeze(predictions_array, axis=1)

        return predictions_array

    def predict_with_mc_dropout(self, perturbations: list, n_samples: int = 10,
                                 return_type: str = 'mean') -> tuple:
        """
        Predict with MC Dropout for increased diversity.

        Uses dropout at test time to generate stochastic predictions,
        increasing model diversity and better uncertainty estimates.

        args:
            perturbations: List of perturbation strings
            n_samples: Number of MC dropout samples
            return_type: 'mean' or 'cells'

        returns:
            (pred_mean, pred_std): mean and standard deviation across MC samples
        """
        if not self.fitted:
            raise ValueError("Model must be loaded before prediction")

        import torch

        # Normalize perturbation names (same as regular predict)
        perturbations = [p.replace('_', '+') for p in perturbations]
        def normalize_control(s):
            parts = s.split('+')
            parts = ['ctrl' if p.lower() == 'control' else p for p in parts]
            return '+'.join(parts)
        perturbations = [normalize_control(p) for p in perturbations]

        # Filter valid perturbations
        valid_perts = []
        invalid_indices = []
        for i, pert in enumerate(perturbations):
            genes = pert.split('+')
            actual_genes = [g for g in genes if g != 'ctrl']
            has_embeddings = all(g in self.gene_embeddings for g in actual_genes)

            if len(actual_genes) >= 1 and has_embeddings:
                valid_perts.append(pert)
            else:
                invalid_indices.append(i)

        if len(invalid_indices) > 0 and len(invalid_indices) < 10:
            missing_genes = set()
            for idx in invalid_indices[:5]:
                genes = perturbations[idx].split('+')
                actual_genes = [g for g in genes if g != 'ctrl']
                for g in actual_genes:
                    if g not in self.gene_embeddings:
                        missing_genes.add(g)
            if missing_genes:
                print(f"Warning: {len(invalid_indices)} perturbations skipped (MC dropout)")

        # run MC dropout
        if len(valid_perts) > 0:
            # set model to training mode to enable dropout
            if hasattr(self.model, 'Net'):
                self.model.Net.train()

            mc_predictions = []
            for sample_idx in range(n_samples):
                with torch.no_grad():  # No gradient needed, just forward pass
                    results_dict = self.model.predict(valid_perts, return_type=return_type)
                    mc_predictions.append(results_dict)

            # set back to eval mode
            if hasattr(self.model, 'Net'):
                self.model.Net.eval()

            # aggregate MC samples
            # convert list of dicts to dict of arrays
            all_keys = list(mc_predictions[0].keys())
            aggregated_preds = {}
            for key in all_keys:
                samples = np.array([mc_predictions[i][key] for i in range(n_samples)])
                aggregated_preds[key] = {
                    'mean': np.mean(samples, axis=0),
                    'std': np.std(samples, axis=0)
                }
        else:
            aggregated_preds = {}

        # output arrays
        predictions_mean = []
        predictions_std = []
        valid_idx = 0

        for i, pert in enumerate(perturbations):
            if i in invalid_indices:
                # 0 for invalid
                if len(aggregated_preds) > 0:
                    first_key = list(aggregated_preds.keys())[0]
                    example_mean = aggregated_preds[first_key]['mean']
                    predictions_mean.append(np.zeros_like(example_mean))
                    predictions_std.append(np.zeros_like(example_mean))
                else:
                    n_genes = self.adata.n_vars if self.adata is not None else 5001
                    predictions_mean.append(np.zeros(n_genes))
                    predictions_std.append(np.zeros(n_genes))
            else:
                pert_key = valid_perts[valid_idx]
                predictions_mean.append(aggregated_preds[pert_key]['mean'])
                predictions_std.append(aggregated_preds[pert_key]['std'])
                valid_idx += 1

        pred_mean = np.array(predictions_mean)
        pred_std = np.array(predictions_std)

        # squeeze extra dimensions
        while pred_mean.ndim > 2:
            pred_mean = np.squeeze(pred_mean, axis=1)
        while pred_std.ndim > 2:
            pred_std = np.squeeze(pred_std, axis=1)

        return pred_mean, pred_std

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

        # convert to np.array in the same order as input
        generated = np.array([results_dict[pert] for pert in perturbations])

        return generated

# ================================================================================
# unified data preprocessing
# ================================================================================

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

        # Normalize perturbation names (same as done for scLAMBDA)
        if 'condition' in adata.obs.columns:
            # 1. Replace underscore with '+' for multi-gene perturbations first
            adata.obs['condition'] = adata.obs['condition'].astype(str).str.replace('_', '+')
            # 2. Handle control conditions: normalize to 'ctrl'
            def normalize_control(s):
                parts = s.split('+')
                parts = ['ctrl' if p.lower() == 'control' else p for p in parts]
                return '+'.join(parts)
            adata.obs['condition'] = adata.obs['condition'].apply(normalize_control)

        adata.obs['num_targets'] = adata.obs['nperts']
        
        single_perts = adata[adata.obs['num_targets'] == 1]
        combo_perts = adata[adata.obs['num_targets'] == 2]

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

    def train_models(self,
                     gears_model_dir: str,
                     gears_data_path: str,
                     gears_data_name: str,
                     sclambda_model_path: str,
                     sclambda_adata_path: str,
                     sclambda_embeddings_path: str,
                     norman_data_path: str,
                     gears_epochs: int = 20,
                     gears_hidden_size: int = 64,
                     gears_split: str = 'simulation',
                     gears_seed: int = 1,
                     sclambda_multi_gene: bool = True,
                     sclambda_split_type: str = None,
                     sclambda_seed: int = 0):
        """
        Train GEARS and scLAMBDA models

        Args:
            gears_model_dir: Directory to save GEARS model
            gears_data_path: Path to GEARS data directory
            gears_data_name: Dataset name ('norman', 'adamson', 'dixit')
            sclambda_model_path: Directory to save scLAMBDA model
            sclambda_adata_path: Path to h5ad file for scLAMBDA
            sclambda_embeddings_path: Path to gene embeddings file
            norman_data_path: Path to Norman h5ad for baseline fitting
            gears_epochs: Number of GEARS training epochs
            gears_hidden_size: GEARS hidden layer size
            gears_split: GEARS split type
            gears_seed: GEARS random seed
            sclambda_multi_gene: Whether to train scLAMBDA for multi-gene perturbations
            sclambda_split_type: scLAMBDA split type
            sclambda_seed: scLAMBDA random seed
        """
        print("="*60)
        print("TRAINING PIPELINE: GEARS + scLAMBDA + Baselines")
        print("="*60)

        # ==========================================
        # Step 1: Train GEARS
        # ==========================================
        print("\n[1/3] Training GEARS...")
        print(f"  Data: {gears_data_name}")
        print(f"  Epochs: {gears_epochs}")
        print(f"  Hidden size: {gears_hidden_size}")
        print(f"  Split: {gears_split}")

        self.gears_model.train_gears(
            model_dir=gears_model_dir,
            data_path=gears_data_path,
            data_name=gears_data_name,
            split=gears_split,
            seed=gears_seed,
            hidden_size=gears_hidden_size,
            epochs=gears_epochs
        )

        # ==========================================
        # Step 2: Train scLAMBDA
        # ==========================================
        print("\n[2/3] Training scLAMBDA...")
        print(f"  Multi-gene: {sclambda_multi_gene}")
        print(f"  Split type: {sclambda_split_type}")

        self.sclambda_model.load_data_and_embeddings(
            adata_path=sclambda_adata_path,
            gene_embeddings_path=sclambda_embeddings_path
        )

        self.sclambda_model.train_sclambda(
            model_path=sclambda_model_path,
            multi_gene=sclambda_multi_gene,
            split_type=sclambda_split_type,
            seed=sclambda_seed
        )

        # ==========================================
        # Step 3: baseline models
        # ==========================================
        print("\n[3/3] Fitting baseline models...")

        self.X_single, self.y_single, self.X_combo, self.y_combo, self.gene_names = \
            self.data_processor.load_norman_data(norman_data_path)

        self._fit_baselines()

        self.fitted = True

        print("\n" + "="*60)
        print("TRAINING COMPLETE!")
        print("="*60)
        print(f"\nModels saved to:")
        print(f"  GEARS: {gears_model_dir}")
        print(f"  scLAMBDA: {sclambda_model_path}")
        print(f"\nBaseline models fitted on {len(self.X_single)} single perturbations")

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
            
        self._fit_baselines()
        
        self.fitted = True
        print("All models loaded!")
        
    def _fit_baselines(self, X_single_train=None, y_single_train=None):
        """
        Fit additive and mean baseline models.

        IMPORTANT: Should only be called with TRAINING data to avoid data leakage.
        If X_single_train and y_single_train are not provided, uses self.X_single and self.y_single.
        """
        self.baseline_models = {}

        # use provided training data or fall back to all data (for backward compatibility)
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
        
    def predict_ensemble(self, X_test: np.ndarray) -> tuple:
        """
        Ensemble prediction with uncertainty quantification, not calibrated uncertainty yet

        Returns:
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
        n_gears_fallback = 0
        first_result_logged = False

        for i, pert_list in enumerate(pert_lists):
            # Check if all genes in this perturbation are known to GEARS
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
                    if n_gears_fallback < 3:  # Only print first few errors
                        print(f"Warning: GEARS prediction failed for {pert_list}: {e}")
                    n_gears_fallback += 1
                    # Use baseline prediction as fallback
                    gears_preds[i] = self.predict_baseline(X_test[i:i+1], model='additive')[0]
            else:
                # Use baseline prediction for unknown genes
                unknown_genes = [g for g in pert_list if g not in gears_known_genes]
                if n_gears_fallback < 3:  # Only print first few warnings
                    print(f"Warning: Genes unknown to GEARS: {unknown_genes[:3]}. Using baseline.")
                n_gears_fallback += 1
                gears_preds[i] = self.predict_baseline(X_test[i:i+1], model='additive')[0]

        print(f"GEARS predictions: {n_gears_success} successful, {n_gears_fallback} using baseline fallback")
        predictions['gears'] = gears_preds

        # scLAMBDA predictions
        pert_strings = [self._binary_to_perturbation_str(x) for x in X_test]
        sclambda_results = self.sclambda_model.predict(pert_strings, return_type='mean')
        predictions['sclambda'] = sclambda_results

        # baseline predictions
        predictions['mean'] = self.predict_baseline(X_test, model='mean')
        predictions['additive'] = self.predict_baseline(X_test, model='additive')

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

        ensemble_mean = np.mean(pred_stack, axis=0)
        epistemic_uncertainty = np.var(pred_stack, axis=0)

        # Check model diversity
        self._check_model_diversity(predictions)

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
    print("="*70)
    print("ENSEMBLE TRAINING PIPELINE")
    print("="*70)
    print("\nThis script is for training GEARS and scLAMBDA models.")
    print("="*70)

    # paths
    SCLAMBDA_REPO = '/insomnia001/depts/edu/BIOLBC3141_Fall2025/rc3517/ml-stats/scLAMBDA'
    DATA_DIR = '/insomnia001/depts/edu/BIOLBC3141_Fall2025/rc3517/ml-stats/data'

    GEARS_MODEL_DIR = f'{DATA_DIR}/gears_model'  # will be created
    GEARS_DATA_PATH = f'{DATA_DIR}/gears_data'  # contains downloaded GEARS data

    SCLAMBDA_MODEL_PATH = f'{DATA_DIR}/sclambda_model'  # will be created
    SCLAMBDA_EMBEDDINGS_PATH = f'{DATA_DIR}/GPT_3_5_gene_embeddings_3-large.pickle'

    NORMAN_DATA_PATH = f'{DATA_DIR}/adata_norman_preprocessed.h5ad'

    # hyperparameters
    GEARS_EPOCHS = 20
    GEARS_HIDDEN_SIZE = 64

    # ==========================================
    # train all models using unified pipeline
    # ==========================================
    ensemble = Ensemble(sclambda_repo_path=SCLAMBDA_REPO)

    ensemble.train_models(
        gears_model_dir=GEARS_MODEL_DIR,
        gears_data_path=GEARS_DATA_PATH,
        gears_data_name='norman',  # use GEARS built-in dataset
        sclambda_model_path=SCLAMBDA_MODEL_PATH,
        sclambda_adata_path=NORMAN_DATA_PATH,
        sclambda_embeddings_path=SCLAMBDA_EMBEDDINGS_PATH,
        norman_data_path=NORMAN_DATA_PATH,
        gears_epochs=GEARS_EPOCHS,
        gears_hidden_size=GEARS_HIDDEN_SIZE,
        gears_split='simulation',
        gears_seed=1,
        sclambda_multi_gene=True,
        sclambda_split_type=None,
        sclambda_seed=0
    )

    # ==========================================
    # quick evaluation on test set
    # ==========================================
    print("\n" + "="*70)
    print("QUICK EVALUATION")
    print("="*70)

    # Create splits
    splits = ensemble.data_processor.create_combo_splits(
        X_single=ensemble.X_single,
        y_single=ensemble.y_single,
        X_combo=ensemble.X_combo,
        y_combo=ensemble.y_combo,
        combo_test_ratio=0.2,
        random_state=42
    )

    print(f"\nData splits:")
    print(f"  Training samples: {len(splits['X_train'])}")
    print(f"  Validation samples: {len(splits['X_val'])}")
    print(f"  Test samples: {len(splits['X_test'])}")

    # eval
    print("\nEvaluating ensemble on test set...")
    pred_mean, uncertainties, individual_preds = ensemble.predict_ensemble(splits['X_test'])
    mse = np.mean((pred_mean - splits['y_test']) ** 2)

    print(f"\nTest metrics:")
    print(f"  Ensemble MSE (uniform weights): {mse:.6f}")
    print(f"  Mean epistemic uncertainty: {np.mean(uncertainties):.6f}")

    # individual model performance
    print(f"\nIndividual model MSE:")
    for model_name, preds in individual_preds.items():
        model_mse = np.mean((preds - splits['y_test']) ** 2)
        print(f"  {model_name}: {model_mse:.6f}")

    # experiment recommendations
    print("\n" + "="*70)
    print("EXPERIMENT RECOMMENDATIONS")
    print("="*70)

    candidate_experiments = [(ensemble.gene_names[i], ensemble.gene_names[i+1])
                             for i in range(min(20, len(ensemble.gene_names)-1))]
    recommendations = ensemble.recommend_experiments(candidate_experiments, n_recommend=5)

    print(f"\nTop 5 experiments with highest epistemic uncertainty:")
    for i, (gene1, gene2, score) in enumerate(recommendations):
        print(f"  {i+1}. {gene1} x {gene2} (uncertainty: {score:.4f})")

    print("\n" + "="*70)
    print("TRAINING AND EVALUATION COMPLETE!")
    print("="*70)