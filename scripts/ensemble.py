"""
Ensemble for Combinatorial Perturbation Prediction

3-Model Ensemble with Diverse Objectives:
- Additive baseline: Linear sum of single-gene effects
- VAE: with moderate KL weight (beta=0.5) to encourage latent diversity
- GNN: Graph Neural Network with heavy dropout (0.4) for regularization

Diversity is achieved through different inductive biases and training objectives

# VAE predictions are sampled from the prior conditioned on perturbation.
# This provides exploratory diversity rather than point-optimal prediction.

"""

import numpy as np
import pandas as pd
import scanpy as sc
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import os
from scipy.optimize import nnls
from sklearn.metrics import mean_squared_error 

# ============================================================================
# load data
# ============================================================================

def load_norman_data(data_path):
    print("Loading Norman dataset...")
    adata = sc.read_h5ad(data_path)

    if 'condition' not in adata.obs.columns and 'perturbation' in adata.obs.columns:
        adata.obs['condition'] = adata.obs['perturbation']

    adata.obs['condition'] = adata.obs['condition'].astype(str).str.replace('_', '+')

    def normalize_control(s):
        parts = s.split('+')
        parts = ['ctrl' if p.lower() == 'control' else p for p in parts]
        return '+'.join(parts)

    adata.obs['condition'] = adata.obs['condition'].apply(normalize_control)

    single_perts = adata[adata.obs['nperts'] == 1].copy()
    combo_perts = adata[adata.obs['nperts'] == 2].copy()

    gene_names = adata.var_names.tolist()

    X_single = create_perturbation_matrix(single_perts, gene_names)
    y_single = single_perts.X.toarray() if hasattr(single_perts.X, 'toarray') else single_perts.X

    X_combo = create_perturbation_matrix(combo_perts, gene_names)
    y_combo = combo_perts.X.toarray() if hasattr(combo_perts.X, 'toarray') else combo_perts.X

    print(f"Loaded: {len(X_single)} singles, {len(X_combo)} combos, {len(gene_names)} genes")

    return X_single, y_single, X_combo, y_combo, gene_names


def create_perturbation_matrix(adata_subset, gene_names):
    X = np.zeros((len(adata_subset), len(gene_names)))
    gene_to_idx = {gene: idx for idx, gene in enumerate(gene_names)}

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


def create_splits(X_single, y_single, X_combo, y_combo, test_ratio=0.2, random_state=42):
    np.random.seed(random_state)

    n_combo = len(X_combo)
    indices = np.random.permutation(n_combo)

    test_size = int(n_combo * test_ratio)
    val_size = int(n_combo * test_ratio)

    test_idx = indices[:test_size]
    val_idx = indices[test_size:test_size + val_size]
    train_idx = indices[test_size + val_size:]

    X_train = np.vstack([X_single, X_combo[train_idx]])
    y_train = np.vstack([y_single, y_combo[train_idx]])

    X_val = X_combo[val_idx]
    y_val = y_combo[val_idx]

    X_test = X_combo[test_idx]
    y_test = y_combo[test_idx]

    print(f"\nSplits: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test,
        'n_singles': len(X_single)
    }


# ============================================================================
# VAE Model
# ============================================================================

class VAE(nn.Module):
    """Variational Autoencoder for gene expression prediction."""

    def __init__(self, input_dim, latent_dim=64, hidden_dims=[512, 256]):
        super(VAE, self).__init__()

        self.latent_dim = latent_dim

        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = h_dim

        self.encoder = nn.Sequential(*encoder_layers)
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)

        # Decoder
        decoder_layers = []
        prev_dim = latent_dim + input_dim  # Concatenate latent + perturbation
        for h_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = h_dim

        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, pert):
        # Concatenate latent vector with perturbation info
        combined = torch.cat([z, pert], dim=1)
        return self.decoder(combined)

    def forward(self, x, pert):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, pert)
        return recon, mu, logvar


class VAEWrapper:
    """Wrapper for VAE model training and prediction."""

    def __init__(self, input_dim, device='cpu'):
        self.input_dim = input_dim
        self.device = device
        self.model = None

    def train_model(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=256, lr=0.001):
        print("\n1. Training VAE...")
        print(f"   latent_dim=64, hidden=[512, 256], epochs={epochs}")

        self.model = VAE(self.input_dim, latent_dim=64, hidden_dims=[512, 256]).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        # Create data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train)
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        self.model.train()
        best_loss = float('inf')
        patience = 10
        patience_counter = 0

        for epoch in range(epochs):
            total_loss = 0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

                optimizer.zero_grad()

                recon, mu, logvar = self.model(batch_y, batch_x)

                # VAE loss = Reconstruction + KL divergence
                recon_loss = F.mse_loss(recon, batch_y, reduction='sum')
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recon_loss + 0.5 * kl_loss  # Beta=0.5 for more diversity (higher KL weight)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader.dataset)

            # Validation
            if (epoch + 1) % 10 == 0:
                val_loss = self._validate(X_val, y_val)
                print(f"   Epoch {epoch+1}/{epochs}: Train={avg_loss:.4f}, Val={val_loss:.4f}")

                if val_loss < best_loss:
                    best_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    print(f"   Early stopping at epoch {epoch+1}")
                    break

        print("   DONE - VAE trained")

    def _validate(self, X_val, y_val):
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_val).to(self.device)
            y_tensor = torch.FloatTensor(y_val).to(self.device)

            recon, mu, logvar = self.model(y_tensor, X_tensor)
            recon_loss = F.mse_loss(recon, y_tensor, reduction='mean')
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + 0.5 * kl_loss  # Match training beta

        self.model.train()
        return loss.item()

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)

            # For prediction, sample from prior
            batch_size = X_tensor.shape[0]
            z = torch.randn(batch_size, self.model.latent_dim).to(self.device)

            predictions = self.model.decode(z, X_tensor)
            return predictions.cpu().numpy()

# ============================================================================
# baseline models
# ============================================================================

def fit_baselines(X_single, y_single):
    print("\nFitting baseline models...")

    n_genes_pert = X_single.shape[1]
    n_genes_expr = y_single.shape[1]

    gene_effects = np.zeros((n_genes_pert, n_genes_expr))

    for gene_idx in range(n_genes_pert):
        mask = X_single[:, gene_idx] == 1
        if mask.sum() > 0:
            gene_effects[gene_idx] = np.mean(y_single[mask], axis=0)

    print(f"Baselines fitted: {(gene_effects != 0).any(axis=1).sum()} genes with effects")
    return gene_effects


def predict_baseline(X, gene_effects, model='additive'):
    if model == 'additive':
        return X @ gene_effects
    elif model == 'mean':
        n_perts = X.sum(axis=1, keepdims=True)
        n_perts = np.where(n_perts == 0, 1, n_perts)
        return (X @ gene_effects) / n_perts
    else:
        raise ValueError(f"Unknown baseline model: {model}")


# ============================================================================
# ensemble
# ============================================================================

class Ensemble:

    def __init__(self, gene_names, gene_effects):
        self.gene_names = gene_names
        self.gene_effects = gene_effects
        self.n_genes = len(gene_names)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")

        self.vae_model = VAEWrapper(self.n_genes, device=self.device)
        self.model_weights = None

    def train_ml_models(self, X_train, y_train, X_val=None, y_val=None, epochs=50):
        print("\n" + "="*60)
        print("Training models...")
        print("="*60)

        # Split validation from train if not provided
        if X_val is None or y_val is None:
            val_size = int(0.1 * len(X_train))
            indices = np.random.permutation(len(X_train))
            val_idx = indices[:val_size]
            train_idx = indices[val_size:]

            X_val = X_train[val_idx]
            y_val = y_train[val_idx]
            X_train = X_train[train_idx]
            y_train = y_train[train_idx]

        print("\nDiversity Strategy:")
        print("  - VAE (beta=0.5)")

        # 1. VAE: High beta for more exploration
        print("\n[1/3] Training VAE (beta=0.5, latent=64)...")
        self.vae_model.train_model(X_train, y_train, X_val, y_val, epochs=epochs)

        print("\n" + "="*60)
        print("TRAINING COMPLETE")
        print("  - VAE: Probabilistic latent space")
        print("="*60)

    def get_predictions(self, X):
        """Get predictions from all models."""
        predictions = {}

        # 1. Additive baseline
        predictions['additive'] = predict_baseline(X, self.gene_effects, 'additive')

        # 2. VAE (probabilistic latent space)
        predictions['vae'] = self.vae_model.predict(X)

        return predictions

    def _normalize_preds(self, preds):
        """
        Normalize predictions per model (zero mean, unit variance)
        """
        mu = preds.mean(axis=0, keepdims=True)
        std = preds.std(axis=0, keepdims=True) + 1e-8
        return (preds - mu) / std


    def learn_weights(self, X_val, y_val):
        """Learn optimal ensemble weights using NNLS."""
        print("\n" + "="*60)
        print("Learning Ensemble Weights (Global NNLS)")
        print("="*60)

        predictions = self.get_predictions(X_val)

        pred_stack = np.stack([
            self._normalize_preds(predictions['additive']),
            self._normalize_preds(predictions['vae']),
            self._normalize_preds(predictions['gnn'])
        ], axis=0)


        print(f"\nPredictions shape: {pred_stack.shape}")

        # Check individual model performance
        print("\nIndividual Model Validation MSE:")
        for i, name in enumerate(['additive', 'vae']):
            mse = np.mean((predictions[name] - y_val) ** 2)
            print(f"  {name:15s}: {mse:.6f}")

        self._check_diversity(predictions)

        n_models = pred_stack.shape[0]
        A = pred_stack.transpose(1, 2, 0).reshape(-1, n_models)
        b = y_val.reshape(-1)

        print("\nSolving ridge...")
        lam = 1e-2  # try 1e-3, 1e-2, 1e-1 if needed

        # ridge solution
        w = np.linalg.solve(
            A.T @ A + lam * np.eye(n_models),
            A.T @ b
        )

        # Enforce non-negativity
        w = np.clip(w, 0, None)

        # Normalize to simplex
        if w.sum() > 0:
            weights = w / w.sum()
        else:
            print("WARNING: All weights zero! Using uniform.")
            weights = np.ones(n_models) / n_models


        self.model_weights = weights

        print(f"\nLearned Weights:")
        print(f"  Additive:     {weights[0]:.4f}")
        print(f"  VAE:          {weights[1]:.4f}")
        print(f"  GNN:          {weights[2]:.4f}")

        weighted_pred = np.tensordot(weights, pred_stack, axes=([0], [0]))
        val_mse = np.mean((weighted_pred - y_val) ** 2)
        print(f"\nEnsemble Validation MSE: {val_mse:.6f}")
        print("="*60)

        return weights

    def _check_diversity(self, predictions):
        """Check pairwise correlation between models."""
        from scipy.stats import pearsonr

        print("\nModel Diversity Check:")
        print("-" * 60)

        model_names = list(predictions.keys())

        for i, name1 in enumerate(model_names):
            for j, name2 in enumerate(model_names):
                if i < j:
                    pred1 = predictions[name1].flatten()
                    pred2 = predictions[name2].flatten()
                    
                    # Check for constant arrays
                    if np.std(pred1) < 1e-10 or np.std(pred2) < 1e-10:
                        print(f"  {name1:12s} vs {name2:12s}: CONSTANT ⚠️")
                        continue
                    
                    corr, _ = pearsonr(pred1, pred2)
                    
                    if corr > 0.95:
                        flag = "⚠️ TOO SIMILAR"
                    elif corr < 0.7:
                        flag = "✓ GOOD DIVERSITY"
                    else:
                        flag = "✓"
                    print(f"  {name1:12s} vs {name2:12s}: {corr:.4f} {flag}")

        print("-" * 60)

    def predict_ensemble(self, X):
        """Make ensemble predictions with epistemic uncertainty."""
        predictions = self.get_predictions(X)

        pred_stack = np.stack([
            self._normalize_preds(predictions['additive']),
            self._normalize_preds(predictions['vae']),
            self._normalize_preds(predictions['gnn'])
        ], axis=0)


        ensemble_mean = np.tensordot(self.model_weights, pred_stack, axes=([0], [0]))

        # Epistemic uncertainty = variance across models (weighted)
        # Only include models with non-zero weight for meaningful uncertainty
        active_models = self.model_weights > 0.01
        if active_models.sum() > 1:
            active_preds = pred_stack[active_models]
            epistemic_unc = np.var(pred_stack[active_models], axis=0)
        else:
            epistemic_unc = np.var(pred_stack[active_models], axis=0)

        return ensemble_mean, epistemic_unc, predictions


# ============================================================================
# conformal prediction
# ============================================================================

class ConformalPredictor:
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.calibration_scores = None

    def calibrate(self, y_cal, y_pred_cal):
        print("\n" + "="*60)
        print("Conformal Prediction Calibration")
        print("="*60)

        self.calibration_scores = np.abs(y_cal - y_pred_cal)

        print(f"Calibrated on {len(y_cal)} samples")
        print(f"Target coverage: {100*(1-self.alpha):.1f}%")
        print(f"Mean calibration score: {self.calibration_scores.mean():.6f}")
        print("="*60)

    def predict_interval(self, y_pred):
        n_cal = len(self.calibration_scores)
        q_level = np.ceil((n_cal + 1) * (1 - self.alpha)) / n_cal
        q_level = min(q_level, 1.0)

        quantiles = np.quantile(self.calibration_scores, q_level, axis=0)

        lower = y_pred - quantiles
        upper = y_pred + quantiles
        width = upper - lower

        return lower, upper, width

    def compute_coverage(self, y_true, lower, upper):
        covered = (y_true >= lower) & (y_true <= upper)
        return {
            'overall_coverage': np.mean(covered),
            'target_coverage': 1 - self.alpha,
            'per_gene_coverage': np.mean(covered, axis=0),
            'per_sample_coverage': np.mean(covered, axis=1)
        }


# ============================================================================
# main
# ============================================================================

def main():
    print("="*60)
    print("Ensemble for Combinatorial Perturbation Prediction")
    print("="*60)
    print("\nModels: Additive, VAE, GNN")
    print("="*60)

    DATA_DIR = '/insomnia001/depts/edu/BIOLBC3141_Fall2025/rc3517/ml-stats/data'
    ADATA_PATH = f'{DATA_DIR}/adata_norman_preprocessed.h5ad'

    # Load data
    X_single, y_single, X_combo, y_combo, gene_names = load_norman_data(ADATA_PATH)

    # Create splits
    splits = create_splits(X_single, y_single, X_combo, y_combo)

    # Fit baselines
    gene_effects = fit_baselines(
        splits['X_train'][:splits['n_singles']],
        splits['y_train'][:splits['n_singles']]
    )

    # Create ensemble
    ensemble = Ensemble(gene_names, gene_effects)

    # Train ML models
    ensemble.train_ml_models(splits['X_train'], splits['y_train'], epochs=50)

    # Split validation
    n_val = len(splits['X_val'])
    val_indices = np.random.RandomState(42).permutation(n_val)
    n_weight = int(0.6 * n_val)

    X_weight = splits['X_val'][val_indices[:n_weight]]
    y_weight = splits['y_val'][val_indices[:n_weight]]
    X_cal = splits['X_val'][val_indices[n_weight:]]
    y_cal = splits['y_val'][val_indices[n_weight:]]

    print(f"\nValidation split: {n_weight} weight-learning, {len(X_cal)} calibration")

    # Learn weights
    weights = ensemble.learn_weights(X_weight, y_weight)

    # Calibrate conformal predictor
    pred_cal, _, _ = ensemble.predict_ensemble(X_cal)
    conformal = ConformalPredictor(alpha=0.1)
    conformal.calibrate(y_cal, pred_cal)

    # Evaluate on test
    print("\n" + "="*60)
    print("Test Set Evaluation")
    print("="*60)

    pred_test, unc_test, individual_preds = ensemble.predict_ensemble(splits['X_test'])

    test_mse = mean_squared_error(splits['y_test'], pred_test)
    print(f"\nEnsemble MSE: {test_mse:.6f}")
    print(f"Mean epistemic uncertainty: {np.mean(unc_test):.6f}")

    print("\nIndividual Model MSE:")
    for name, preds in individual_preds.items():
        mse = mean_squared_error(splits['y_test'], preds)
        print(f"  {name:15s}: {mse:.6f}")

    lower, upper, width = conformal.predict_interval(pred_test)
    coverage_stats = conformal.compute_coverage(splits['y_test'], lower, upper)

    print(f"\nConformal Prediction:")
    print(f"  Target coverage:   {coverage_stats['target_coverage']*100:.1f}%")
    print(f"  Empirical coverage: {coverage_stats['overall_coverage']*100:.1f}%")
    print(f"  Mean interval width: {np.mean(width):.6f}")

    print("\n" + "="*60)
    print("Done!")
    print("="*60)

    # Save results
    results = {
        'weights': weights,
        'test_mse': test_mse,
        'individual_mse': {name: mean_squared_error(splits['y_test'], preds)
                          for name, preds in individual_preds.items()},
        'coverage': coverage_stats,
        'mean_uncertainty': np.mean(unc_test)
    }

    with open('ensemble_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    print("\nResults saved to ensemble_results.pkl")


if __name__ == "__main__":
    main()