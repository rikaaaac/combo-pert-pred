"""
Active Learning for Additive + VAE Ensemble

Acquisition strategies:
- Random: Uniform sampling (baseline)
- Uncertainty: Ensemble epistemic uncertainty
- Synergy: GEARS-style genetic interaction scores
- Diversity: Expression-space diversity
- Weighted: α*uncertainty + β*synergy + γ*diversity

Architecture:
- Reuses AdditiveVAEEnsemble from ensemble_add_vae.py
- Iteratively refits models on expanding training set
- Evaluates multiple acquisition strategies
"""

import os
import torch
import pickle
import numpy as np
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered in divide')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.linear_model import Ridge

from ensemble_add_vae import (
    load_norman_data,
    create_splits,
    fit_additive,
    predict_additive,
    AdditiveVAEEnsemble,
    VAEWrapper
)


# ============================================================================
# active learning loop
# ============================================================================

class ActiveLearning:
    """
    Active learning loop for AdditiveVAEEnsemble.

    Iteratively selects informative experiments based on:
    - Uncertainty: Epistemic uncertainty from ensemble disagreement
    - Synergy: GEARS-style genetic interaction scores
    - Diversity: Expression-space coverage
    """

    def __init__(self, ensemble, gene_names):
        self.ensemble = ensemble
        self.gene_names = gene_names
        self.gene_effects = ensemble.effects
        self.ridge_model = None
        self.history = {
            'iteration': [],
            'test_mse': [],
            'test_r': [],
            'n_training_samples': [],
            'mean_uncertainty': [],
            'strategy': []
        }

    def get_candidate_pool(self, X_train, X_pool):
        """
        Return unseen perturbation pairs from pool.

        Args:
            X_train: Current training perturbations
            X_pool: Available pool of perturbations

        Returns:
            List of (gene1, gene2, pool_index) tuples
        """
        seen = set()
        for x in X_train:
            idxs = tuple(sorted(np.where(x > 0)[0]))
            if len(idxs) == 2:
                seen.add(idxs)
        candidates = []
        for i, x in enumerate(X_pool):
            idxs = tuple(sorted(np.where(x > 0)[0]))
            if len(idxs) == 2 and idxs not in seen:
                g1, g2 = self.gene_names[idxs[0]], self.gene_names[idxs[1]]
                candidates.append((g1, g2, i))
        return candidates

    def _candidates_to_matrix(self, candidates):
        """
        Convert candidate gene pairs to perturbation matrix.

        Args:
            candidates: List of (gene1, gene2, pool_index) tuples

        Returns:
            Binary perturbation matrix (n_candidates, n_genes)
        """
        X = np.zeros((len(candidates), len(self.gene_names)))
        gene_to_idx = {g: i for i, g in enumerate(self.gene_names)}
        for i, (g1, g2, _) in enumerate(candidates):
            X[i, gene_to_idx[g1]] = 1
            X[i, gene_to_idx[g2]] = 1
        return X

    def compute_uncertainty_scores(self, candidates):
        """
        Compute epistemic uncertainty from ensemble disagreement.

        Higher scores indicate regions of high model uncertainty.
        """
        X_cand = self._candidates_to_matrix(candidates)
        pred_mean, epistemic_unc = self.ensemble.predict(X_cand)
        scores = np.sum(epistemic_unc, axis=1)
        return self._normalize(scores)

    def compute_diversity_scores(self, candidates, X_train):
        """
        Compute expression-space diversity scores.

        Higher scores indicate perturbations far from training set in expression space.
        """
        X_cand = self._candidates_to_matrix(candidates)
        pred_cand, _ = self.ensemble.predict(X_cand)
        pred_train, _ = self.ensemble.predict(X_train)
        distances = euclidean_distances(pred_cand, pred_train)
        scores = distances.min(axis=1)
        return self._normalize(scores)

    def compute_synergy_scores(self, candidates):
        """
        Compute GEARS-style genetic interaction scores.

        GI = observed_AB - (ctrl + Δ_A + Δ_B)
        Higher scores indicate stronger synergistic/antagonistic effects.
        """
        scores = np.zeros(len(candidates))
        y_ctrl = np.zeros(self.gene_effects.shape[1])
        for i, (g1, g2, _) in enumerate(candidates):
            if g1 not in self.gene_names or g2 not in self.gene_names:
                continue
            idx1, idx2 = self.gene_names.index(g1), self.gene_names.index(g2)
            y_A, y_B = y_ctrl + self.gene_effects[idx1], y_ctrl + self.gene_effects[idx2]
            x_combo = np.zeros((1, len(self.gene_names)))
            x_combo[0, idx1] = 1
            x_combo[0, idx2] = 1
            y_AB, _ = self.ensemble.predict(x_combo)
            expected = y_A + y_B - y_ctrl
            scores[i] = np.linalg.norm(y_AB[0] - expected)
        return self._normalize(scores)

    def _normalize(self, scores):
        """Normalize scores to [0, 1] range."""
        if scores.max() > scores.min():
            return (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
        return np.zeros_like(scores)

    def select_next_experiments(self, candidates, n_select, strategy='uncertainty',
                                a=1.0, b=0.0, c=0.0, X_train=None):
        """
        Select next experiments using specified acquisition strategy.

        Args:
            candidates: List of (gene1, gene2, pool_index) tuples
            n_select: Number of experiments to select
            strategy: 'random', 'uncertainty', 'synergy', 'diversity', or 'weighted'
            a, b, c: Weights for weighted strategy (uncertainty, synergy, diversity)
            X_train: Current training set (required for diversity)

        Returns:
            List of selected (gene1, gene2, pool_index) tuples
        """
        n_select = min(n_select, len(candidates))
        if strategy == 'random':
            idxs = np.random.choice(len(candidates), size=n_select, replace=False)
            return [candidates[i] for i in idxs]
        elif strategy == 'uncertainty':
            scores = self.compute_uncertainty_scores(candidates)
        elif strategy == 'synergy':
            scores = self.compute_synergy_scores(candidates)
        elif strategy == 'diversity':
            scores = self.compute_diversity_scores(candidates, X_train)
        elif strategy == 'weighted':
            scores = a*self.compute_uncertainty_scores(candidates) + \
                     b*self.compute_synergy_scores(candidates) + \
                     c*self.compute_diversity_scores(candidates, X_train)
            scores /= (a+b+c)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        top_idxs = np.argsort(scores)[-n_select:][::-1]
        return [candidates[i] for i in top_idxs]

    def evaluate(self, X_test, y_test):
        """
        Evaluate ensemble on test set.

        Returns:
            Dictionary with mse, pearson, and mean_uncertainty
        """
        pred_mean, epistemic_unc = self.ensemble.predict(X_test)
        mse = np.mean((pred_mean - y_test)**2)
        pearson_per_gene = [np.corrcoef(pred_mean[:, i], y_test[:, i])[0,1]
                            for i in range(pred_mean.shape[1])]
        pearson_per_gene = [r for r in pearson_per_gene if not np.isnan(r)]
        mean_r = np.mean(pearson_per_gene) if pearson_per_gene else 0.0
        mean_unc = np.mean(epistemic_unc)
        return {'mse': mse, 'pearson': mean_r, 'mean_uncertainty': mean_unc}

    def run_simulation(self, X_train_init, y_train_init, X_test, y_test,
                       X_pool, y_pool, X_val, y_val, n_iterations=10, n_select=15,
                       strategy='uncertainty', alpha=1.0, beta=0.0, gamma=0.0,
                       vae_epochs=30):
        """
        Run active learning simulation.

        Iteratively:
        1. Fit models on current training set
        2. Select informative experiments
        3. Add selected experiments to training set
        4. Evaluate on test set

        Args:
            X_train_init, y_train_init: Initial training data
            X_test, y_test: Test data
            X_pool, y_pool: Pool of available experiments
            X_val, y_val: Validation data for VAE training
            n_iterations: Number of AL iterations
            n_select: Number of experiments to select per iteration
            strategy: Acquisition strategy
            alpha, beta, gamma: Weights for weighted strategy
            vae_epochs: Number of epochs for VAE retraining

        Returns:
            History dictionary with metrics over iterations
        """
        X_train, y_train = X_train_init.copy(), y_train_init.copy()
        ground_truth = {tuple(sorted(np.where(x>0)[0])): y_pool[i]
                        for i, x in enumerate(X_pool)}
        print(f"\nActive Learning: {strategy}, Init={len(X_train)}, Pool={len(X_pool)}, Test={len(X_test)}")
        for iteration in range(n_iterations):
            print(f"\nIteration {iteration+1}/{n_iterations}")

            # Update additive baseline
            single_mask = X_train.sum(axis=1) == 1
            if single_mask.sum() > 0:
                self.gene_effects = fit_additive(X_train[single_mask], y_train[single_mask])
                self.ensemble.effects = self.gene_effects

            # Retrain VAE
            self.ensemble.vae.train(X_train, y_train, X_val, y_val, epochs=vae_epochs)

            # Evaluate
            metrics = self.evaluate(X_test, y_test)

            # Get candidates and select
            candidates = self.get_candidate_pool(X_train, X_pool)
            if not candidates:
                print("No more candidates.")
                break
            selected = self.select_next_experiments(
                candidates, n_select, strategy=strategy, a=alpha, b=beta, c=gamma, X_train=X_train)

            # Add selected to training set
            new_X, new_y = [], []
            for g1, g2, idx in selected:
                indices = tuple(sorted([self.gene_names.index(g1), self.gene_names.index(g2)]))
                if indices in ground_truth:
                    x_new = np.zeros(len(self.gene_names))
                    x_new[indices[0]] = 1
                    x_new[indices[1]] = 1
                    new_X.append(x_new)
                    new_y.append(ground_truth[indices])
            if new_X:
                X_train = np.vstack([X_train, np.array(new_X)])
                y_train = np.vstack([y_train, np.array(new_y)])

            # Log
            self.history['iteration'].append(iteration+1)
            self.history['test_mse'].append(metrics['mse'])
            self.history['test_r'].append(metrics['pearson'])
            self.history['n_training_samples'].append(len(X_train))
            self.history['mean_uncertainty'].append(metrics['mean_uncertainty'])
            self.history['strategy'].append(strategy)

            print(f"  MSE={metrics['mse']:.6f}, r={metrics['pearson']:.4f}, n={len(X_train)}")

        return self.history

# ============================================================================
# strategy comparison
# ============================================================================

def compare_strategies(ensemble, gene_names, X_train_init, y_train_init,
                       X_test, y_test, X_pool, y_pool, X_val, y_val,
                       n_iterations=10, n_select=15,
                       result_dir='active_learning_results'):
    """
    Compare all acquisition strategies.

    Args:
        ensemble: AdditiveVAEEnsemble instance
        gene_names: List of gene names
        X_train_init, y_train_init: Initial training data
        X_test, y_test: Test data
        X_pool, y_pool: Pool of available experiments
        X_val, y_val: Validation data
        n_iterations: Number of AL iterations
        n_select: Experiments to select per iteration
        result_dir: Directory to save results

    Returns:
        Dictionary of results for each strategy
    """
    os.makedirs(result_dir, exist_ok=True)
    strategies = {
        'random': {'strategy': 'random'},
        'uncertainty': {'strategy': 'uncertainty'},
        'weighted': {'strategy': 'weighted', 'alpha': 0.5, 'beta': 0.5, 'gamma': 0.0},
        # Removed diversity and synergy to fit in 2 hours
    }
    results = {}
    for name, params in strategies.items():
        print(f"\n{'='*60}\nStrategy: {name.upper()}\n{'='*60}")
        al = ActiveLearning(ensemble, gene_names)
        history = al.run_simulation(
            X_train_init.copy(), y_train_init.copy(),
            X_test, y_test, X_pool, y_pool, X_val, y_val,
            n_iterations=n_iterations, n_select=n_select, **params
        )
        results[name] = history

    # Save results
    with open(os.path.join(result_dir, 'results.pkl'), 'wb') as f:
        pickle.dump(results, f)
    print(f"\nResults saved to {os.path.join(result_dir, 'results.pkl')}")
    return results


# ============================================================================
# plotting and summary
# ============================================================================

def plot_comparison(results, save_path='strategy_comparison.png'):
    """Plot strategy comparison (MSE and Pearson r vs training size)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = {
        'random': 'gray',
        'uncertainty': 'blue',
        'diversity': 'orange',
        'synergy': 'green',
        'weighted': 'brown'
    }

    for name, history in results.items():
        c = colors.get(name, 'black')
        axes[0].plot(history['n_training_samples'], history['test_mse'],
                     color=c, marker='o', markersize=4, label=name)
        axes[1].plot(history['n_training_samples'], history['test_r'],
                     color=c, marker='o', markersize=4, label=name)

    axes[0].set_xlabel('Training Samples')
    axes[0].set_ylabel('Test MSE')
    axes[0].set_title('MSE vs Training Size')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Training Samples')
    axes[1].set_ylabel('Test Pearson r')
    axes[1].set_title('Correlation vs Training Size')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {save_path}")


def print_summary(results, save_path=None):
    """Print and optionally save summary table."""
    lines = [
        "=" * 80,
        "ACTIVE LEARNING SUMMARY",
        "=" * 80,
        f"{'Strategy':<25} {'Final MSE':>12} {'Final r':>12} {'vs Random':>12}",
        "-" * 80
    ]

    random_mse = results.get('random', {}).get('test_mse', [None])[-1]

    for name, history in results.items():
        mse = history['test_mse'][-1]
        r = history['test_r'][-1]
        if random_mse and name != 'random':
            imp_str = f"{(100*(random_mse-mse)/random_mse):+.1f}%"
        else:
            imp_str = "baseline"
        lines.append(f"{name:<25} {mse:>12.6f} {r:>12.4f} {imp_str:>12}")

    lines.append("=" * 80)

    print('\n'.join(lines))
    if save_path:
        with open(save_path, 'w') as f:
            f.write('\n'.join(lines))
        print(f"\nSummary saved to {save_path}")

# ============================================================================
# main
# ============================================================================

def main():
    DATA_PATH = "/insomnia001/depts/edu/BIOLBC3141_Fall2025/rc3517/ml-stats/data/adata_norman_preprocessed.h5ad"
    RESULT_DIR = "/insomnia001/depts/edu/users/rc3517/active_learning_results"
    os.makedirs(RESULT_DIR, exist_ok=True)

    # Load data
    print("Loading Norman dataset...")
    Xs, ys, Xc, yc, genes = load_norman_data(DATA_PATH)
    splits = create_splits(Xs, ys, Xc, yc)

    # Create initial training set (5% of available data)
    n_init = max(50, int(len(splits['X_train']) * 0.05))
    X_train_init = splits['X_train'][:n_init]
    y_train_init = splits['y_train'][:n_init]
    X_pool = splits['X_train'][n_init:]
    y_pool = splits['y_train'][n_init:]

    print(f"Initial training: {len(X_train_init)}")
    print(f"Pool: {len(X_pool)}")
    print(f"Test: {len(splits['X_test'])}")

    # Fit additive baseline on singles
    print("\nFitting additive baseline...")
    effects = fit_additive(
        splits['X_train'][:splits['n_singles']],
        splits['y_train'][:splits['n_singles']]
    )

    # Initialize VAE
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    vae = VAEWrapper(len(genes), device)

    # Create ensemble
    ensemble = AdditiveVAEEnsemble(effects, vae)

    # Set weights manually (from ensemble.py)
    print("\nSetting ensemble weights manually...")
    ensemble.weights = np.array([0.37, 0.63])  # [additive, VAE]
    print(f"Weights → Additive: {ensemble.weights[0]:.3f}, VAE: {ensemble.weights[1]:.3f}")

    # Train initial VAE
    print("\nTraining initial VAE...")
    vae.train(X_train_init, y_train_init, splits['X_val'], splits['y_val'], epochs=50)

    # Run strategy comparison
    print("\nComparing active learning strategies...")
    results = compare_strategies(
        ensemble, genes, X_train_init, y_train_init,
        splits['X_test'], splits['y_test'],
        X_pool, y_pool, splits['X_val'], splits['y_val'],
        n_iterations=5, 
        n_select=15,
        result_dir=RESULT_DIR
    )

    # Plot and summarize
    plot_comparison(results, save_path=os.path.join(RESULT_DIR, 'strategy_comparison.png'))
    print_summary(results, save_path=os.path.join(RESULT_DIR, 'summary.txt'))

    print(f"\nDone. Results saved to {RESULT_DIR}")


if __name__ == "__main__":
    main()
