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