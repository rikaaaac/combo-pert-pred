"""
Quick Active Learning Demo (Reduced parameters for fast execution)
This is a subset of active_learning.py
Key reductions:
- Fewer iterations: 6 instead of 15
- Fewer selections per iteration: 3 instead of 10
- 5 strategies: random, uncertainty, synergy, diversity, weighted_all
- weighted_all uses: α=0.5 (uncertainty) + β=0.3 (synergy) + γ=0.2 (diversity)
- Reduced SHAP samples: 10 background, 1 explain
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import shap
import warnings
import os
import json
import pickle
warnings.filterwarnings('ignore')

from ensemble_full_pipeline import GEARSWrapper, scLAMBDAWrapper

class GeneticInteractionScorer:
    """
    compute genetic interaction scores using GEARS method.
    """

    def __init__(self, ensemble):
        self.ensemble = ensemble
        self.gene_names = ensemble.gene_names
        self.gene_effects = ensemble.gene_effects

    def gears_gi_score(self, y_ctrl: np.ndarray, y_A: np.ndarray,
                       y_B: np.ndarray, y_AB: np.ndarray) -> np.ndarray:
        """
        compute genetic interaction scores
        positive GI = synergistic
        negative GI = antagonistic
        """
        delta_A = y_A - y_ctrl
        delta_B = y_B - y_ctrl

        expected_AB = y_ctrl + delta_A + delta_B
        observed_AB = y_AB

        gi = observed_AB - expected_AB
        return gi

    def compute_gi_for_candidates(self, candidates: list) -> np.ndarray:
        """
        compute GI scores for candidate perturbations
        args:
            candidates: List of (gene1, gene2) tuples
        returns:
            GI magnitude scores (n_candidates,)
        """
        scores = np.zeros(len(candidates))

        # control mean
        if hasattr(self.ensemble, 'y_control') and self.ensemble.y_control is not None:
            y_ctrl = np.mean(self.ensemble.y_control, axis=0)
        else:
            y_ctrl = np.zeros(self.gene_effects.shape[1])

        for i, (g1, g2) in enumerate(candidates):
            if g1 not in self.gene_names or g2 not in self.gene_names:
                continue

            idx1 = self.gene_names.index(g1)
            idx2 = self.gene_names.index(g2)

            # single perturbation effects
            y_A = y_ctrl + self.gene_effects[idx1]
            y_B = y_ctrl + self.gene_effects[idx2]

            # predicted combo
            x_combo = np.zeros((1, len(self.gene_names)))
            x_combo[0, idx1] = 1
            x_combo[0, idx2] = 1

            pred_mean, _, _ = self.ensemble.predict_ensemble(x_combo)
            y_AB = pred_mean[0]

            # GI scores
            gi = self.gears_gi_score(y_ctrl, y_A, y_B, y_AB)
            scores[i] = np.linalg.norm(gi)

        return scores


class SHAPAnalyzer:
    """
    SHAP-based interpretability for perturbation predictions
    identifies:
    - genes driving model uncertainty (where to experiment next)
    - hub genes (frequent synergy participants)
    """

    def __init__(self, ensemble):

        self.ensemble = ensemble
        self.gene_names = ensemble.gene_names
        self.explainers = {}
        self.shap_values_cache = {}

    def _create_uncertainty_function(self):
        """function returning uncertainty given perturbation matrix"""
        def predict_uncertainty(X):
            if X.ndim == 1:
                X = X.reshape(1, -1)
            _, uncertainties, _ = self.ensemble.predict_ensemble(X)
            return uncertainties.mean(axis=1)
        return predict_uncertainty

    def _create_gi_function(self, gi_scorer: GeneticInteractionScorer):
        """function returning GI score given perturbation matrix"""
        def predict_gi(X):
            if X.ndim == 1:
                X = X.reshape(1, -1)

            scores = np.zeros(X.shape[0])
            for i in range(X.shape[0]):
                indices = np.where(X[i] > 0)[0]
                if len(indices) == 2:
                    g1 = self.gene_names[indices[0]]
                    g2 = self.gene_names[indices[1]]
                    scores[i] = gi_scorer.compute_gi_for_candidates([(g1, g2)])[0]
            return scores
        return predict_gi

    def explain_uncertainty(self, X_samples: np.ndarray,
                           n_background: int = 100,
                           n_explain: int = 3) -> dict:
        """
        identify genes that drive model uncertainty
        high SHAP value = perturbing this gene leads to high uncertainty
        """
        print("Computing SHAP values for uncertainty...")

        background = shap.sample(X_samples, min(n_background, len(X_samples)))
        uncertainty_fn = self._create_uncertainty_function()

        explainer = shap.KernelExplainer(uncertainty_fn, background)

        X_explain = X_samples[:min(n_explain, len(X_samples))]
        shap_values = explainer.shap_values(X_explain)

        gene_importance = np.abs(shap_values).mean(axis=0)

        top_indices = np.argsort(gene_importance)[::-1]
        top_genes = [(self.gene_names[i], gene_importance[i]) for i in top_indices[:20]]

        results = {
            'shap_values': shap_values,
            'gene_importance': gene_importance,
            'top_genes': top_genes,
            'X_explain': X_explain
        }

        self.shap_values_cache['uncertainty'] = results

        print(f"\nTop 10 genes driving uncertainty:")
        for gene, importance in top_genes[:10]:
            print(f"  {gene}: {importance:.4f}")

        return results

    def plot_uncertainty_summary(self, save_path: str = None):
        """SHAP summary plot for uncertainty."""
        if 'uncertainty' not in self.shap_values_cache:
            raise ValueError("Run explain_uncertainty() first")

        results = self.shap_values_cache['uncertainty']

        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            results['shap_values'],
            results['X_explain'],
            feature_names=self.gene_names,
            show=False,
            max_display=20
        )
        plt.title('SHAP: Genes Driving Model Uncertainty')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved to {save_path}")


class ActiveLearningLoop:
    def __init__(self, ensemble):
        self.ensemble = ensemble
        self.gi_scorer = GeneticInteractionScorer(ensemble)
        self.shap_analyzer = SHAPAnalyzer(ensemble)

        self.history = {
            'iteration': [],
            'selected_perturbations': [],
            'test_mse': [],
            'test_pearson': [],
            'n_training_samples': [],
            'mean_uncertainty': [],
            'mean_gi_score': [],
            'strategy': []
        }

    def get_candidate_pool(self, exclude_seen: set = None) -> list:
        """generate unseen perturbation pairs."""
        if exclude_seen is None:
            exclude_seen = set()

        candidates = []
        gene_names = self.ensemble.gene_names

        for i, g1 in enumerate(gene_names):
            for g2 in gene_names[i+1:]:
                pair = tuple(sorted([g1, g2]))
                if pair not in exclude_seen:
                    candidates.append((g1, g2))

        return candidates

    def get_seen_perturbations(self, X: np.ndarray) -> set:
        """extract seen perturbation pairs."""
        seen = set()
        gene_names = self.ensemble.gene_names

        for x in X:
            indices = np.where(x > 0)[0]
            if len(indices) == 2:
                pair = tuple(sorted([gene_names[indices[0]], gene_names[indices[1]]]))
                seen.add(pair)

        return seen

    def _candidates_to_matrix(self, candidates: list) -> np.ndarray:
        """convert candidates to perturbation matrix."""
        gene_names = self.ensemble.gene_names
        X = np.zeros((len(candidates), len(gene_names)))

        for i, (g1, g2) in enumerate(candidates):
            if g1 in gene_names and g2 in gene_names:
                X[i, gene_names.index(g1)] = 1
                X[i, gene_names.index(g2)] = 1

        return X

    # ==========================================
    # Scoring functions
    # ==========================================

    def compute_uncertainty_scores(self, candidates: list) -> np.ndarray:
        """Epistemic uncertainty from model disagreement."""
        X_candidates = self._candidates_to_matrix(candidates)
        _, uncertainties, _ = self.ensemble.predict_ensemble(X_candidates)

        scores = np.sum(uncertainties, axis=1)

        if scores.max() > scores.min():
            scores = (scores - scores.min()) / (scores.max() - scores.min())

        return scores

    def compute_synergy_scores(self, candidates: list) -> np.ndarray:
        """GI-based synergy scores using GEARS method."""
        scores = self.gi_scorer.compute_gi_for_candidates(candidates)

        if scores.max() > scores.min():
            scores = (scores - scores.min()) / (scores.max() - scores.min())

        return scores

    def compute_diversity_scores(self, candidates: list,
                                  already_selected: list = None,
                                  X_train: np.ndarray = None) -> np.ndarray:
        """Diversity scores - prefer underrepresented genes."""
        gene_names = self.ensemble.gene_names
        n_candidates = len(candidates)

        gene_counts = defaultdict(int)

        if X_train is not None:
            for x in X_train:
                for idx in np.where(x > 0)[0]:
                    gene_counts[gene_names[idx]] += 1

        if already_selected:
            for g1, g2 in already_selected:
                gene_counts[g1] += 1
                gene_counts[g2] += 1

        scores = np.zeros(n_candidates)
        for i, (g1, g2) in enumerate(candidates):
            coverage = gene_counts.get(g1, 0) + gene_counts.get(g2, 0)
            scores[i] = 1.0 / (1.0 + coverage)

        if scores.max() > scores.min():
            scores = (scores - scores.min()) / (scores.max() - scores.min())

        return scores

    # ==========================================
    # Selection
    # ==========================================

    def select_next_experiments(self, candidates: list, n_select: int,
                                 strategy: str = 'uncertainty',
                                 alpha: float = 1.0,
                                 beta: float = 0.0,
                                 gamma: float = 0.0,
                                 X_train: np.ndarray = None) -> list:
        """select next experiments."""
        n_select = min(n_select, len(candidates))

        if strategy == 'random':
            selected_idx = np.random.choice(len(candidates), size=n_select, replace=False)
            return [(candidates[i][0], candidates[i][1], 0.0) for i in selected_idx]

        elif strategy == 'uncertainty':
            scores = self.compute_uncertainty_scores(candidates)

        elif strategy == 'synergy':
            scores = self.compute_synergy_scores(candidates)

        elif strategy == 'diversity':
            return self._greedy_diversity_selection(candidates, n_select, X_train)

        elif strategy == 'weighted':
            scores = np.zeros(len(candidates))

            if alpha > 0:
                scores += alpha * self.compute_uncertainty_scores(candidates)
            if beta > 0:
                scores += beta * self.compute_synergy_scores(candidates)
            if gamma > 0:
                scores += gamma * self.compute_diversity_scores(candidates, X_train=X_train)

            total_weight = alpha + beta + gamma
            if total_weight > 0:
                scores /= total_weight

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        top_indices = np.argsort(scores)[-n_select:][::-1]

        return [(candidates[i][0], candidates[i][1], scores[i]) for i in top_indices]

    def _greedy_diversity_selection(self, candidates: list, n_select: int,
                                     X_train: np.ndarray = None) -> list:
        """greedy diversity selection."""
        selected = []
        remaining = list(range(len(candidates)))

        for _ in range(n_select):
            if not remaining:
                break

            remaining_candidates = [candidates[i] for i in remaining]
            scores = self.compute_diversity_scores(
                remaining_candidates,
                already_selected=[(s[0], s[1]) for s in selected],
                X_train=X_train
            )

            best_local_idx = np.argmax(scores)
            best_global_idx = remaining[best_local_idx]

            g1, g2 = candidates[best_global_idx]
            selected.append((g1, g2, scores[best_local_idx]))
            remaining.remove(best_global_idx)

        return selected

    # ==========================================
    # Simulation
    # ==========================================

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """evaluate on test set."""
        pred_mean, uncertainties, _ = self.ensemble.predict_ensemble(X_test)

        mse = np.mean((pred_mean - y_test) ** 2)

        pearson_per_gene = []
        for i in range(pred_mean.shape[1]):
            r = np.corrcoef(pred_mean[:, i], y_test[:, i])[0, 1]
            if not np.isnan(r):
                pearson_per_gene.append(r)
        mean_pearson = np.mean(pearson_per_gene) if pearson_per_gene else 0.0

        return {
            'mse': mse,
            'pearson': mean_pearson,
            'mean_uncertainty': np.mean(uncertainties)
        }

    def run_simulation(self, X_train_init: np.ndarray, y_train_init: np.ndarray,
                       X_test: np.ndarray, y_test: np.ndarray,
                       X_pool: np.ndarray, y_pool: np.ndarray,
                       n_iterations: int = 10,
                       n_select_per_iter: int = 5,
                       strategy: str = 'uncertainty',
                       alpha: float = 1.0,
                       beta: float = 0.0,
                       gamma: float = 0.0):
        """run active learning simulation."""
        X_train = X_train_init.copy()
        y_train = y_train_init.copy()

        # ground truth lookup
        ground_truth = {}
        gene_names = self.ensemble.gene_names
        for i in range(X_pool.shape[0]):
            indices = np.where(X_pool[i] > 0)[0]
            if len(indices) == 2:
                pair = tuple(sorted([gene_names[indices[0]], gene_names[indices[1]]]))
                ground_truth[pair] = y_pool[i]

        self.history = {k: [] for k in self.history}

        strategy_str = strategy
        if strategy == 'weighted':
            strategy_str = f"weighted(α={alpha},β={beta},γ={gamma})"

        print(f"\nActive Learning Simulation")
        print(f"  Strategy: {strategy_str}")
        print(f"  GI method: GEARS")
        print(f"  Initial: {len(X_train)}, Pool: {len(X_pool)}, Test: {len(X_test)}")
        print("=" * 60)

        for iteration in range(n_iterations):
            # update baselines
            single_mask = X_train.sum(axis=1) == 1
            if single_mask.sum() > 0:
                self.ensemble.X_single = X_train[single_mask]
                self.ensemble.y_single = y_train[single_mask]
                self.ensemble._fit_baselines()
                self.gi_scorer = GeneticInteractionScorer(self.ensemble)

            metrics = self.evaluate(X_test, y_test)

            # candidates
            seen = self.get_seen_perturbations(X_train)
            candidates = self.get_candidate_pool(exclude_seen=seen)

            # filter to pool
            pool_pairs = set()
            for i in range(X_pool.shape[0]):
                indices = np.where(X_pool[i] > 0)[0]
                if len(indices) == 2:
                    pair = tuple(sorted([gene_names[indices[0]], gene_names[indices[1]]]))
                    pool_pairs.add(pair)
            candidates = [(g1, g2) for g1, g2 in candidates
                         if tuple(sorted([g1, g2])) in pool_pairs]

            if len(candidates) == 0:
                print(f"  Iter {iteration + 1}: No more candidates")
                break

            # GI score for logging
            gi_scores = self.compute_synergy_scores(candidates[:100])
            mean_gi = np.mean(gi_scores) if len(gi_scores) > 0 else 0

            # select
            selected = self.select_next_experiments(
                candidates, n_select_per_iter,
                strategy=strategy,
                alpha=alpha, beta=beta, gamma=gamma,
                X_train=X_train
            )

            # add to training
            new_X, new_y = [], []
            for g1, g2, _ in selected:
                pair = tuple(sorted([g1, g2]))
                if pair in ground_truth:
                    x_new = np.zeros(len(gene_names))
                    x_new[gene_names.index(g1)] = 1
                    x_new[gene_names.index(g2)] = 1
                    new_X.append(x_new)
                    new_y.append(ground_truth[pair])

            if new_X:
                X_train = np.vstack([X_train, np.array(new_X)])
                y_train = np.vstack([y_train, np.array(new_y)])

                # CRITICAL: Retrain ensemble with new data
                single_mask = X_train.sum(axis=1) == 1
                combo_mask = X_train.sum(axis=1) == 2

                if combo_mask.sum() > 0:
                    self.ensemble.X_combo = X_train[combo_mask]
                    self.ensemble.y_combo = y_train[combo_mask]

                if single_mask.sum() > 0:
                    self.ensemble.X_single = X_train[single_mask]
                    self.ensemble.y_single = y_train[single_mask]
                    self.ensemble._fit_baselines()
                    # Update GI scorer with new baselines
                    self.gi_scorer = GeneticInteractionScorer(self.ensemble)

                print(f"    -> Retrained with {len(X_train)} samples ({combo_mask.sum()} combos, {single_mask.sum()} singles)")

            # log
            self.history['iteration'].append(iteration + 1)
            self.history['selected_perturbations'].append(selected)
            self.history['test_mse'].append(metrics['mse'])
            self.history['test_pearson'].append(metrics['pearson'])
            self.history['n_training_samples'].append(len(X_train))
            self.history['mean_uncertainty'].append(metrics['mean_uncertainty'])
            self.history['mean_gi_score'].append(mean_gi)
            self.history['strategy'].append(strategy)

            print(f"  Iter {iteration + 1}: MSE={metrics['mse']:.4f}, "
                  f"r={metrics['pearson']:.4f}, GI={mean_gi:.4f}, n={len(X_train)}")

        return self.history

    # ==========================================
    # SHAP analysis
    # ==========================================

    def run_shap_analysis(self, X_combo: np.ndarray,
                          n_background: int = 100,
                          n_explain: int = 3):
        """Run SHAP analysis for uncertainty."""

        print("\n" + "="*60)
        print("SHAP ANALYSIS")
        print("="*60)

        unc_results = self.shap_analyzer.explain_uncertainty(
            X_combo, n_background=n_background, n_explain=n_explain
        )

        return {'uncertainty': unc_results}

    def plot_shap_results(self, save_dir: str = '.'):
        """Plot SHAP results."""
        self.shap_analyzer.plot_uncertainty_summary(f'{save_dir}/shap_uncertainty.png')


def compare_strategies(ensemble, X_train_init, y_train_init,
                       X_test, y_test, X_pool, y_pool,
                       n_iterations=6, n_select=3, result_dir='results_quick'):
    """Compare 5 selection strategies - QUICK VERSION."""

    # Create result directory
    os.makedirs(result_dir, exist_ok=True)

    # Run strategies with stronger weight differences
    strategies = {
        'random': {'strategy': 'random'},
        'uncertainty': {'strategy': 'weighted', 'alpha': 1.0, 'beta': 0.0, 'gamma': 0.0},
        'synergy': {'strategy': 'weighted', 'alpha': 0.0, 'beta': 1.0, 'gamma': 0.0},
        'diversity': {'strategy': 'weighted', 'alpha': 0.0, 'beta': 0.0, 'gamma': 1.0},
        'weighted_all': {'strategy': 'weighted', 'alpha': 0.5, 'beta': 0.3, 'gamma': 0.2},
    }

    results = {}

    for name, params in strategies.items():
        print(f"\n{'='*60}")
        print(f"Strategy: {name.upper()}")
        print('='*60)

        al = ActiveLearningLoop(ensemble)

        single_mask = X_train_init.sum(axis=1) == 1
        if single_mask.sum() > 0:
            ensemble.X_single = X_train_init[single_mask]
            ensemble.y_single = y_train_init[single_mask]
            ensemble._fit_baselines()

        history = al.run_simulation(
            X_train_init.copy(), y_train_init.copy(),
            X_test, y_test, X_pool, y_pool,
            n_iterations=n_iterations,
            n_select_per_iter=n_select,
            **params
        )

        results[name] = history

    # Save results to file
    save_results(results, result_dir)

    return results


def save_results(results: dict, result_dir: str):
    """Save results to JSON and pickle files."""
    # Convert results to JSON-serializable format
    results_json = {}
    for name, history in results.items():
        results_json[name] = {
            'iteration': history['iteration'],
            'test_mse': history['test_mse'],
            'test_pearson': history['test_pearson'],
            'n_training_samples': history['n_training_samples'],
            'mean_uncertainty': history['mean_uncertainty'],
            'mean_gi_score': history['mean_gi_score'],
            'strategy': history['strategy']
        }

    # Save as JSON
    json_path = os.path.join(result_dir, 'results.json')
    with open(json_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"\nSaved JSON results to {json_path}")

    # Save full results as pickle (includes selected_perturbations)
    pickle_path = os.path.join(result_dir, 'results.pkl')
    with open(pickle_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"Saved pickle results to {pickle_path}")


def plot_strategy_comparison(results: dict, save_path: str = 'strategy_comparison.png'):
    """Plot strategy comparison."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    colors = {
        'random': 'gray',
        'uncertainty': 'blue',
        'synergy': 'green',
        'diversity': 'orange',
        'weighted_all': 'purple'
    }

    linestyles = {
        'random': '--',
        'uncertainty': '-',
        'synergy': '-',
        'diversity': '-',
        'weighted_all': '-.'
    }

    for name, history in results.items():
        c = colors.get(name, 'black')
        ls = linestyles.get(name, '-')

        axes[0, 0].plot(history['n_training_samples'], history['test_mse'],
                       color=c, linestyle=ls, marker='o', markersize=4, label=name)
        axes[0, 1].plot(history['n_training_samples'], history['test_pearson'],
                       color=c, linestyle=ls, marker='o', markersize=4, label=name)
        axes[1, 0].plot(history['iteration'], history['mean_uncertainty'],
                       color=c, linestyle=ls, marker='o', markersize=4, label=name)
        axes[1, 1].plot(history['iteration'], history['mean_gi_score'],
                       color=c, linestyle=ls, marker='o', markersize=4, label=name)

    axes[0, 0].set_xlabel('Training Samples'); axes[0, 0].set_ylabel('Test MSE')
    axes[0, 0].set_title('MSE vs Training Size'); axes[0, 0].legend(fontsize=8); axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set_xlabel('Training Samples'); axes[0, 1].set_ylabel('Test Pearson r')
    axes[0, 1].set_title('Correlation vs Training Size'); axes[0, 1].legend(fontsize=8); axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].set_xlabel('Iteration'); axes[1, 0].set_ylabel('Mean Uncertainty')
    axes[1, 0].set_title('Epistemic Uncertainty'); axes[1, 0].legend(fontsize=8); axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].set_xlabel('Iteration'); axes[1, 1].set_ylabel('Mean GI Score')
    axes[1, 1].set_title('Genetic Interaction Score'); axes[1, 1].legend(fontsize=8); axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\nSaved to {save_path}")


def print_summary_table(results: dict, save_path: str = None):
    """Print and optionally save summary."""
    summary_lines = []
    summary_lines.append("=" * 80)
    summary_lines.append("QUICK DEMO SUMMARY")
    summary_lines.append("=" * 80)
    summary_lines.append(f"{'Strategy':<20} {'Final MSE':>12} {'Final r':>12} {'vs Random':>12}")
    summary_lines.append("-" * 80)

    random_mse = results.get('random', {}).get('test_mse', [None])[-1]

    for name, history in results.items():
        mse = history['test_mse'][-1]
        r = history['test_pearson'][-1]

        if random_mse and name != 'random':
            imp = ((random_mse - mse) / random_mse) * 100
            imp_str = f"{imp:+.1f}%"
        else:
            imp_str = "baseline"

        summary_lines.append(f"{name:<20} {mse:>12.4f} {r:>12.4f} {imp_str:>12}")

    summary_lines.append("=" * 80)

    # Print to console
    for line in summary_lines:
        print(line)

    # Save to file if path provided
    if save_path:
        with open(save_path, 'w') as f:
            f.write('\n'.join(summary_lines))
        print(f"\nSummary saved to {save_path}")


# ================================================================================
# Main - QUICK VERSION
# ================================================================================

if __name__ == "__main__":
    from ensemble import Ensemble

    SCLAMBDA_REPO = '/insomnia001/depts/edu/BIOLBC3141_Fall2025/rc3517/ml-stats/scLAMBDA'
    DATA_DIR = '/insomnia001/depts/edu/BIOLBC3141_Fall2025/rc3517/ml-stats/data'
    RESULT_DIR = '/insomnia001/depts/edu/BIOLBC3141_Fall2025/rc3517/ml-stats/active_learning_results_quick'

    # Create result directory
    os.makedirs(RESULT_DIR, exist_ok=True)
    print(f"Results will be saved to: {RESULT_DIR}\n")

    print("Loading ensemble...")
    ensemble = Ensemble(sclambda_repo_path=SCLAMBDA_REPO)
    ensemble.load_models(
        gears_model_dir=f'{DATA_DIR}/gears_model',
        gears_data_path=f'{DATA_DIR}/gears_data',
        gears_data_name='norman',
        sclambda_model_path=f'{DATA_DIR}/sclambda_model',
        sclambda_adata_path=f'{DATA_DIR}/norman_perturbseq_preprocessed_hvg_filtered.h5ad',
        sclambda_embeddings_path=f'{DATA_DIR}/GPT_3_5_gene_embeddings_3-large.pickle',
        norman_data_path=f'{DATA_DIR}/norman_perturbseq_preprocessed_hvg_filtered.h5ad'
    )

    splits = ensemble.data_processor.create_combo_splits(
        X_single=ensemble.X_single,
        y_single=ensemble.y_single,
        X_combo=ensemble.X_combo,
        y_combo=ensemble.y_combo,
        combo_test_ratio=0.3,
        random_state=42
    )

    # Reduced from 0.2 to 0.1 to leave more room for active learning improvements
    n_init = int(len(splits['X_train']) * 0.1)
    X_train_init = splits['X_train'][:n_init]
    y_train_init = splits['y_train'][:n_init]
    X_pool = splits['X_train'][n_init:]
    y_pool = splits['y_train'][n_init:]

    print(f"\nInitial: {len(X_train_init)}, Pool: {len(X_pool)}, Test: {len(splits['X_test'])}")

    # QUICK VERSION: 5 strategies, 6 iterations, 3 selections per iteration
    print("\n*** QUICK DEMO MODE ***")
    print("Running with reduced parameters:")
    print("  - 5 strategies (random, uncertainty, synergy, diversity, weighted_all)")
    print("  - 6 iterations instead of 15")
    print("  - 3 selections per iteration instead of 10")
    print("  - 10% initial training (reduced from 20%) to show AL improvements")
    print("  - Model retraining after each iteration (CRITICAL FIX)")
    print("  - 10 SHAP background samples instead of 100")
    print("  - 1 SHAP explanation sample instead of 3")
    print()

    results = compare_strategies(
        ensemble, X_train_init, y_train_init,
        splits['X_test'], splits['y_test'], X_pool, y_pool,
        n_iterations=6, n_select=3,
        result_dir=RESULT_DIR
    )

    plot_strategy_comparison(results, save_path=os.path.join(RESULT_DIR, 'strategy_comparison.png'))
    print_summary_table(results, save_path=os.path.join(RESULT_DIR, 'summary.txt'))

    # SHAP - reduced samples
    print("\nRunning SHAP analysis (quick mode)...")
    al = ActiveLearningLoop(ensemble)
    al.run_shap_analysis(ensemble.X_combo, n_background=10, n_explain=1)
    al.plot_shap_results(RESULT_DIR)

    print(f"\n✓ Done! All results saved to {RESULT_DIR}")
    print("\nTo run the full version with all strategies and iterations,")
    print("use the original active_learning.py script.")
