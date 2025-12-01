import numpy as np
import pandas as pd
import scanpy as sc
import torch
import sys
import os
import warnings
warnings.filterwarnings('ignore')

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
        
        # load data if not already loaded
        if self.adata is None and adata_path is not None:
            self.load_data_and_embeddings(adata_path, gene_embeddings_path)
        elif self.adata is None:
            raise ValueError("Must provide adata_path or call load_data_and_embeddings first")
        
        # initialize model with saved weights
        self.model = sclambda.model.Model(
            self.adata,
            self.gene_embeddings,
            model_path=model_path,
            multi_gene=multi_gene
        )
        # scLAMBDA loads weights automatically from model_path during init
        self.fitted = True
        
        print(f"scLAMBDA model loaded from {model_path}")
        
    def predict(self, perturbations: list, return_type: str = 'mean') -> np.ndarray:
        """
        Predict perturbation effects.
        
        args:
            perturbations: List of perturbation strings, e.g., ['CBL+CNN1', 'FEV+ctrl']
            return_type: 'mean' for mean expression, 'cells' for single-cell predictions
            
        returns:
            Predicted expression
        """
        if not self.fitted:
            raise ValueError("Model must be loaded before prediction")
        
        return self.model.predict(perturbations, return_type=return_type)
    
    def generate(self, perturbations: list, return_type: str = 'cells') -> np.ndarray:
        """
        Generate new perturbed cells.
        
        args:
            perturbations: List of perturbation strings
            return_type: 'mean' or 'cells'
            
        returns:
            Generated expression profiles
        """
        if not self.fitted:
            raise ValueError("Model must be loaded before prediction")
        
        return self.model.generate(perturbations, return_type=return_type)

# ================================================================================
# usage
# ================================================================================

if __name__ == "__main__":
    print("Combinatorial Perturbation Prediction Ensemble")
    print("=" * 50)
    print("  test if sclambda works")
    print("=" * 50)

    SCLAMBDA_REPO = '/insomnia001/depts/edu/BIOLBC3141_Fall2025/rc3517/ml-stats/scLAMBDA'
    DATA_DIR = '/insomnia001/depts/edu/BIOLBC3141_Fall2025/rc3517/ml-stats/data'

    SCLAMBDA_MODEL_PATH = f'{DATA_DIR}/sclambda_model'  # will be created
    SCLAMBDA_ADATA_PATH = f'{DATA_DIR}/norman_perturbseq_preprocessed_hvg_filtered.h5ad'
    SCLAMBDA_EMBEDDINGS_PATH = f'{DATA_DIR}/GPT_3_5_gene_embeddings_3-large.pickle'
    NORMAN_DATA_PATH = f'{DATA_DIR}/norman_perturbseq_preprocessed_hvg_filtered.h5ad'
    
    # ==========================================
    # train scLAMBDA only (using custom preprocessed data)
    # ==========================================
    print("\n[2/3] Training scLAMBDA with custom preprocessed data...")
    sclambda = scLAMBDAWrapper(sclambda_path=SCLAMBDA_REPO)
    sclambda.load_data_and_embeddings(SCLAMBDA_ADATA_PATH, SCLAMBDA_EMBEDDINGS_PATH)
    sclambda.train_sclambda(
        model_path=SCLAMBDA_MODEL_PATH,
        multi_gene=True,
        seed=0
    )