"""
Additive + VAE Ensemble for Combinatorial Perturbation Prediction

Models:
- Additive baseline: linear superposition of single-gene effects
- VAE: probabilistic latent-variable model capturing nonlinear interactions
"""

import numpy as np
import scanpy as sc
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error

# ============================================================================
# data loading
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


def create_perturbation_matrix(adata, gene_names):
    X = np.zeros((len(adata), len(gene_names)))
    g2i = {g: i for i, g in enumerate(gene_names)}

    for i, pert in enumerate(adata.obs['condition']):
        if pert == 'ctrl':
            continue
        for g in pert.split('+'):
            if g in g2i:
                X[i, g2i[g]] = 1
    return X


def create_splits(X_single, y_single, X_combo, y_combo, test_ratio=0.2, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X_combo))

    n_test = int(test_ratio * len(idx))
    n_val = int(test_ratio * len(idx))
    test, val, train = idx[:n_test], idx[n_test:n_test+n_val], idx[n_test+n_val:]

    X_train = np.vstack([X_single, X_combo[train]])
    y_train = np.vstack([y_single, y_combo[train]])

    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_combo[val],
        'y_val': y_combo[val],
        'X_test': X_combo[test],
        'y_test': y_combo[test],
        'n_singles': len(X_single)
    }


# ============================================================================
# additive baseline
# ============================================================================

def fit_additive(X_single, y_single):
    n_genes = X_single.shape[1]
    effects = np.zeros((n_genes, y_single.shape[1]))

    for g in range(n_genes):
        mask = X_single[:, g] == 1
        if mask.any():
            effects[g] = y_single[mask].mean(axis=0)

    print(f"Additive effects learned for {(effects != 0).any(axis=1).sum()} genes")
    return effects


def predict_additive(X, effects):
    return X @ effects


# ============================================================================
# VAE
# ============================================================================

class VAE(nn.Module):
    def __init__(self, dim, latent=64, hidden=(512, 256)):
        super().__init__()
        enc = []
        d = dim
        for h in hidden:
            enc += [nn.Linear(d, h), nn.ReLU()]
            d = h
        self.encoder = nn.Sequential(*enc)

        self.mu = nn.Linear(d, latent)
        self.logvar = nn.Linear(d, latent)

        dec = []
        d = latent + dim
        for h in reversed(hidden):
            dec += [nn.Linear(d, h), nn.ReLU()]
            d = h
        dec.append(nn.Linear(d, dim))
        self.decoder = nn.Sequential(*dec)

        self.latent = latent

    def forward(self, y, x):
        h = self.encoder(y)
        mu, logvar = self.mu(h), self.logvar(h)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        recon = self.decoder(torch.cat([z, x], dim=1))
        return recon, mu, logvar


class VAEWrapper:
    def __init__(self, dim, device):
        self.model = VAE(dim).to(device)
        self.device = device

    def train(self, X, y, Xv, yv, epochs=50, beta=0.5):
        opt = torch.optim.Adam(self.model.parameters(), 1e-3)
        loader = DataLoader(TensorDataset(
            torch.FloatTensor(X), torch.FloatTensor(y)),
            batch_size=256, shuffle=True)

        best = np.inf
        patience = 10
        wait = 0

        for e in range(epochs):
            self.model.train()
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                opt.zero_grad()
                recon, mu, logvar = self.model(yb, xb)
                rec = F.mse_loss(recon, yb, reduction='sum')
                kl = -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp())
                loss = rec + beta * kl
                loss.backward()
                opt.step()

            val_loss = self.validate(Xv, yv, beta)
            if val_loss < best:
                best, wait = val_loss, 0
            else:
                wait += 1
            if wait >= patience:
                break

    def validate(self, X, y, beta):
        self.model.eval()
        with torch.no_grad():
            xb = torch.FloatTensor(X).to(self.device)
            yb = torch.FloatTensor(y).to(self.device)
            recon, mu, logvar = self.model(yb, xb)
            return (F.mse_loss(recon, yb) + beta * torch.mean(mu**2)).item()

    def predict(self, X, n_samples=1):
        self.model.eval()
        preds = []
        with torch.no_grad():
            xb = torch.FloatTensor(X).to(self.device)
            for _ in range(n_samples):
                z = torch.randn(len(X), self.model.latent).to(self.device)
                preds.append(self.model.decoder(torch.cat([z, xb], 1)).cpu().numpy())
        return np.stack(preds)


# ============================================================================
# ensemble
# ============================================================================

class AdditiveVAEEnsemble:
    def __init__(self, effects, vae):
        self.effects = effects
        self.vae = vae
        self.weights = None

    def learn_weights(self, X, y):
        A = predict_additive(X, self.effects)
        V = self.vae.predict(X, n_samples=1)[0]

        A = (A - A.mean()) / (A.std() + 1e-8)
        V = (V - V.mean()) / (V.std() + 1e-8)

        w = np.linalg.lstsq(
            np.stack([A.flatten(), V.flatten()], 1),
            y.flatten(),
            rcond=None
        )[0]
        w = np.clip(w, 0, None)
        self.weights = w / (w.sum() + 1e-8)
        print(f"Weights → Additive: {self.weights[0]:.3f}, VAE: {self.weights[1]:.3f}")
        return self.weights

    def predict(self, X):
        A = predict_additive(X, self.effects)
        V = self.vae.predict(X, n_samples=10)
        V_mean = V.mean(axis=0)

        mean = self.weights[0] * A + self.weights[1] * V_mean
        epistemic = (A - V_mean) ** 2 + V.var(axis=0)
        return mean, epistemic


# ============================================================================
# conformal prediction
# ============================================================================

class ConformalPredictor:
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.scores = None

    def calibrate(self, y, yhat):
        self.scores = np.abs(y - yhat)

    def interval(self, yhat):
        q = np.quantile(self.scores, 1 - self.alpha, axis=0)
        return yhat - q, yhat + q


# ============================================================================
# main
# ============================================================================

def main():
    DATA = "/insomnia001/depts/edu/BIOLBC3141_Fall2025/rc3517/ml-stats/data/adata_norman_preprocessed.h5ad"

    Xs, ys, Xc, yc, genes = load_norman_data(DATA)
    splits = create_splits(Xs, ys, Xc, yc)

    # Additive baseline
    effects = fit_additive(
        splits['X_train'][:splits['n_singles']],
        splits['y_train'][:splits['n_singles']]
    )

    # VAE
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vae = VAEWrapper(len(genes), device)
    vae.train(splits['X_train'], splits['y_train'],
              splits['X_val'], splits['y_val'])

    # Ensemble
    ensemble = AdditiveVAEEnsemble(effects, vae)
    learned_weights = ensemble.learn_weights(splits['X_val'], splits['y_val'])

    pred_test, unc_test = ensemble.predict(splits['X_test'])
    test_mse = mean_squared_error(splits['y_test'], pred_test)

    print(f"Test MSE: {test_mse:.6f}")
    print(f"Mean epistemic uncertainty: {unc_test.mean():.6f}")
    print(f"Learned weights: {learned_weights}")

    # Conformal prediction
    cp = ConformalPredictor(alpha=0.1)
    cp.calibrate(splits['y_test'], pred_test)
    lower, upper = cp.interval(pred_test)
    width = upper - lower

    results = {
        'weights': learned_weights,
        'test_mse': test_mse,
        'mean_uncertainty': np.mean(unc_test),
        'coverage_lower': lower,
        'coverage_upper': upper,
        'coverage_width': width
    }

    with open("additive_vae_results.pkl", "wb") as f:
        pickle.dump(results, f)

    print("Done. Saved as additive_vae_results.pkl")


if __name__ == "__main__":
    main()
