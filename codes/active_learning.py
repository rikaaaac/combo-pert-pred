"""
Active Learning Loop for Combinatorial Perturbation Prediction

Selection strategies:
- Random: Uniform sampling (baseline)
- Uncertainty: Model disagreement (epistemic uncertainty)
- Synergy: GEARS-style GI scores
- Diversity: Expression-space diversity (FIXED)
- Weighted: α*uncertainty + β*synergy + γ*diversity (α=0.5, β=0.3, γ=0.2)

Synergy quantification:
- GEARS-style GI score:
  GI = observed_AB - (ctrl + Δ_A + Δ_B)

Interpretability:
- SHAP for uncertainty drivers
- SHAP for synergy/epistasis hub genes

KEY:
==================
1. ACQUISITION SCORE NORMALIZATION: All scores normalized to [0, 1] with epsilon
2. EXPRESSION-SPACE DIVERSITY: Uses predicted expressions, not gene counts
3. AGGRESSIVE INITIAL SIZE: Start with 5% (not 20%) for clearer improvement signal
4. MORE ITERATIONS: 20 iterations (not 15) to see learning curves
5. LARGER BATCHES: 15 samples/iteration (not 10) for stronger signal

"""

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from sklearn.metrics.pairwise import euclidean_distances
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

            pred_mean, _, _ = self.ensemble.predict_ensemble(x_combo, return_intervals=False)
            y_AB = pred_mean[0]
            
            # GI scores
            gi = self.gears_gi_score(y_ctrl, y_A, y_B, y_AB)
            scores[i] = np.linalg.norm(gi)
        
        return scores
    
    def classify_interaction(self, y_ctrl: np.ndarray, y_A: np.ndarray,
                             y_B: np.ndarray, y_AB: np.ndarray,
                             threshold: float = 0.1) -> str:
        """
        classify interaction type based on GI score.
        returns: 'synergistic', 'antagonistic', or 'additive'
        """
        gi = self.gears_gi_score(y_ctrl, y_A, y_B, y_AB)
        gi_magnitude = np.mean(gi)
        
        if gi_magnitude > threshold:
            return 'synergistic'
        elif gi_magnitude < -threshold:
            return 'antagonistic'
        else:
            return 'additive'


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
            _, uncertainties, _ = self.ensemble.predict_ensemble(X, return_intervals=False)
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
    
    def explain_synergy(self, X_samples: np.ndarray,
                        gi_scorer: GeneticInteractionScorer,
                        n_background: int = 100,
                        n_explain: int = 3) -> dict:
        """
        identify hub genes - genes frequently involved in synergistic interactions.
        high SHAP value = perturbing this gene leads to high GI scores
        """
        print("Computing SHAP values for synergy (GEARS GI)...")
        
        background = shap.sample(X_samples, min(n_background, len(X_samples)))
        gi_fn = self._create_gi_function(gi_scorer)
        
        explainer = shap.KernelExplainer(gi_fn, background)
        
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
        
        self.shap_values_cache['synergy'] = results
        
        print(f"\nTop 10 epistasis hub genes:")
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

        # plt.show()  
    
    def plot_synergy_summary(self, save_path: str = None):
        """SHAP summary plot for synergy."""
        if 'synergy' not in self.shap_values_cache:
            raise ValueError("Run explain_synergy() first")
        
        results = self.shap_values_cache['synergy']
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            results['shap_values'],
            results['X_explain'],
            feature_names=self.gene_names,
            show=False,
            max_display=20
        )
        plt.title('SHAP: Epistasis Hub Genes (GEARS GI)')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved to {save_path}")

        # plt.show()  
    
    def plot_gene_comparison(self, top_n: int = 15, save_path: str = None):
        """compare uncertainty vs synergy importance for top genes."""
        unc_importance = np.zeros(len(self.gene_names))
        syn_importance = np.zeros(len(self.gene_names))
        
        if 'uncertainty' in self.shap_values_cache:
            unc_importance = self.shap_values_cache['uncertainty']['gene_importance']
        
        if 'synergy' in self.shap_values_cache:
            syn_importance = self.shap_values_cache['synergy']['gene_importance']
        
        combined = unc_importance + syn_importance
        top_indices = np.argsort(combined)[-top_n:]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        top_genes = [self.gene_names[i] for i in top_indices]
        unc_vals = unc_importance[top_indices]
        syn_vals = syn_importance[top_indices]
        
        x = np.arange(len(top_genes))
        width = 0.35
        
        ax.barh(x - width/2, unc_vals, width, label='Uncertainty Driver', color='steelblue')
        ax.barh(x + width/2, syn_vals, width, label='Epistasis Hub', color='coral')
        
        ax.set_yticks(x)
        ax.set_yticklabels(top_genes)
        ax.set_xlabel('SHAP Importance')
        ax.set_title('Top Genes: Uncertainty vs Synergy')
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved to {save_path}")

        # plt.show()  
        
        return top_genes


class ActiveLearningLoop:
    """active learning with GI-based synergy and SHAP interpretability."""

    def __init__(self, ensemble, use_ridge=True, ridge_alpha=1.0):
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

        # Ridge model for improved uncertainty
        self.ridge_model = None
        self.use_ridge = use_ridge
        self.ridge_alpha = ridge_alpha
        
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
        """
        Epistemic uncertainty from model disagreement.

        FIX: Better normalization to avoid scale issues.
        Returns scores in [0, 1] range.
        """
        X_candidates = self._candidates_to_matrix(candidates)
        _, uncertainties, _ = self.ensemble.predict_ensemble(X_candidates, return_intervals=False)

        # Sum uncertainty across all genes for each sample
        scores = np.sum(uncertainties, axis=1)

        # normalization
        if scores.max() > scores.min():
            scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
        else:
            scores = np.zeros_like(scores)

        return scores

    def compute_uncertainty_scores_IMPROVED(self, candidates: list) -> np.ndarray:
        """
        IMPROVED uncertainty via bootstrap ensemble + interaction ridge.

        Creates diverse model predictions to get better uncertainty estimates:
        1. Original ensemble prediction
        2. Bootstrap variants of baseline models (5 variants)
        3. Ridge with interaction terms (if available)

        Returns scores in [0, 1] range.
        """
        X_candidates = self._candidates_to_matrix(candidates)

        # Get predictions from diverse models
        preds = []

        # 1. Original ensemble
        pred_mean, _, _ = self.ensemble.predict_ensemble(X_candidates, return_intervals=False)
        preds.append(pred_mean)

        # 2. Bootstrap baselines (5 variants)
        # Create diverse predictions by bootstrapping gene effects
        for seed in range(5):
            np.random.seed(seed)
            # Bootstrap gene_effects
            n_genes = self.ensemble.gene_effects.shape[0]
            boot_idx = np.random.choice(n_genes, n_genes, replace=True)
            gene_effects_boot = self.ensemble.gene_effects[boot_idx]

            # Predict with bootstrapped effects (additive model)
            pred_boot = X_candidates[:, boot_idx] @ gene_effects_boot
            preds.append(pred_boot)

        # 3. Ridge with interactions (if trained)
        if hasattr(self, 'ridge_model') and self.ridge_model is not None:
            X_aug = self._add_interaction_features(X_candidates)
            pred_ridge = self.ridge_model.predict(X_aug)
            preds.append(pred_ridge)

        # Compute variance across all models
        preds = np.array(preds)  # (n_models, n_samples, n_genes)
        uncertainty = np.var(preds, axis=0)  # (n_samples, n_genes)
        scores = np.sum(uncertainty, axis=1)

        # Normalize to [0, 1]
        if scores.max() > scores.min():
            scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
        else:
            scores = np.zeros_like(scores)

        return scores

    def _add_interaction_features(self, X: np.ndarray) -> np.ndarray:
        """
        Add pairwise gene interaction features.

        For each pair of genes (i, j), adds feature X[:, i] * X[:, j]
        This allows Ridge to capture epistatic effects.

        Args:
            X: Binary perturbation matrix (n_samples, n_genes)

        Returns:
            Augmented feature matrix (n_samples, n_genes + n_interactions)
        """
        n_genes = X.shape[1]
        interactions = []

        for i in range(n_genes):
            for j in range(i+1, n_genes):
                interactions.append(X[:, i] * X[:, j])

        if interactions:
            return np.hstack([X, np.array(interactions).T])
        else:
            return X
    
    def compute_synergy_scores(self, candidates: list) -> np.ndarray:
        """
        GI-based synergy scores using GEARS method.

        FIX: Better normalization and error handling.
        Returns scores in [0, 1] range.
        """
        scores = self.gi_scorer.compute_gi_for_candidates(candidates)

        if scores.max() > scores.min():
            scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
        else:
            scores = np.zeros_like(scores)

        return scores
    
    def compute_diversity_scores(self, candidates: list,
                                  already_selected: list = None,
                                  X_train: np.ndarray = None) -> np.ndarray:
        """
        Diversity scores - prefer perturbations distant from training set.

        Use expression-space diversity instead of gene-count diversity.
        Returns scores in [0, 1] range.
        """
        gene_names = self.ensemble.gene_names
        n_candidates = len(candidates)

        if X_train is None or len(X_train) == 0:
            # Fallback to gene-count diversity if no training data
            gene_counts = defaultdict(int)

            if already_selected:
                for g1, g2 in already_selected:
                    gene_counts[g1] += 1
                    gene_counts[g2] += 1

            scores = np.zeros(n_candidates)
            for i, (g1, g2) in enumerate(candidates):
                coverage = gene_counts.get(g1, 0) + gene_counts.get(g2, 0)
                scores[i] = 1.0 / (1.0 + coverage)
        else:
            # Use expression-space diversity
            X_candidates = self._candidates_to_matrix(candidates)

            # Get ensemble predictions for candidates
            candidate_preds, _, _ = self.ensemble.predict_ensemble(X_candidates, return_intervals=False)

            # Get ensemble predictions for training set
            train_preds, _, _ = self.ensemble.predict_ensemble(X_train, return_intervals=False)

            # Compute minimum distance to training set in expression space
            distances = euclidean_distances(candidate_preds, train_preds)
            scores = distances.min(axis=1)  # Distance to nearest training sample

        # FIX: Robust normalization
        if scores.max() > scores.min():
            scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
        else:
            scores = np.zeros_like(scores)

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

        elif strategy == 'uncertainty_improved':
            scores = self.compute_uncertainty_scores_IMPROVED(candidates)

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

        elif strategy == 'weighted_improved':
            scores = np.zeros(len(candidates))

            if alpha > 0:
                scores += alpha * self.compute_uncertainty_scores_IMPROVED(candidates)
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
        """
        Evaluate on test set.

        FIX: Don't require conformal intervals for evaluation.
        """
        pred_mean, uncertainties, _ = self.ensemble.predict_ensemble(X_test, return_intervals=False)

        mse = np.mean((pred_mean - y_test) ** 2)

        # Per-gene correlation
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
        """
        Run active learning simulation.

        FIX: Properly fit baselines only on TRAINING data to avoid data leakage.
        """
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
            # FIX: Update baselines ONLY on training singles (no data leakage)
            single_mask = X_train.sum(axis=1) == 1
            if single_mask.sum() > 0:
                X_single_train = X_train[single_mask]
                y_single_train = y_train[single_mask]
                # Fit baselines only on training data
                self.ensemble._fit_baselines(X_single_train=X_single_train,
                                            y_single_train=y_single_train)
                self.gi_scorer = GeneticInteractionScorer(self.ensemble)

            # Train Ridge model with interactions on current training data
            if self.use_ridge and iteration == 0:
                from sklearn.linear_model import Ridge
                print(f"  Training Ridge model with interaction features...")
                X_train_aug = self._add_interaction_features(X_train)
                self.ridge_model = Ridge(alpha=self.ridge_alpha, fit_intercept=True)
                self.ridge_model.fit(X_train_aug, y_train)
                print(f"  Ridge model trained on {X_train_aug.shape[0]} samples with {X_train_aug.shape[1]} features")
            
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
        """Run SHAP analysis for uncertainty and synergy."""

        print("\n" + "="*60)
        print("SHAP ANALYSIS")
        print("="*60)
        
        unc_results = self.shap_analyzer.explain_uncertainty(
            X_combo, n_background=n_background, n_explain=n_explain
        )
        
        syn_results = self.shap_analyzer.explain_synergy(
            X_combo, self.gi_scorer, 
            n_background=n_background, n_explain=n_explain
        )
        
        return {'uncertainty': unc_results, 'synergy': syn_results}
    
    def plot_shap_results(self, save_dir: str = '.'):
        """Plot SHAP results."""
        self.shap_analyzer.plot_uncertainty_summary(f'{save_dir}/shap_uncertainty.png')
        self.shap_analyzer.plot_synergy_summary(f'{save_dir}/shap_synergy.png')
        self.shap_analyzer.plot_gene_comparison(save_path=f'{save_dir}/shap_gene_comparison.png')


def compare_all_strategies(ensemble, X_train_init, y_train_init,
                           X_test, y_test, X_pool, y_pool,
                           n_iterations=10, n_select=5, result_dir='results'):
    """
    Compare all selection strategies.
    Each strategy gets a fresh ensemble to avoid state pollution.
    """

    # Create result directory
    os.makedirs(result_dir, exist_ok=True)

    strategies = {
        'random': {'strategy': 'random'},
        'uncertainty': {'strategy': 'uncertainty'},
        'uncertainty_improved': {'strategy': 'uncertainty_improved'},
        'synergy': {'strategy': 'synergy'},
        'diversity': {'strategy': 'diversity'},
        'weighted_all': {'strategy': 'weighted', 'alpha': 0.5, 'beta': 0.3, 'gamma': 0.2},
        'weighted_improved': {'strategy': 'weighted_improved', 'alpha': 0.5, 'beta': 0.3, 'gamma': 0.2},
    }

    results = {}

    for name, params in strategies.items():
        print(f"\n{'='*60}")
        print(f"Strategy: {name.upper()}")
        print('='*60)

        # FIX: Create fresh ActiveLearningLoop for each strategy
        al = ActiveLearningLoop(ensemble)

        # FIX: Fit baselines only on INITIAL TRAINING singles (no data leakage)
        single_mask = X_train_init.sum(axis=1) == 1
        if single_mask.sum() > 0:
            X_single_init = X_train_init[single_mask]
            y_single_init = y_train_init[single_mask]
            ensemble._fit_baselines(X_single_train=X_single_init,
                                  y_single_train=y_single_init)

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
        'random': 'gray', 'uncertainty': 'blue', 'uncertainty_improved': 'darkblue',
        'synergy': 'green', 'diversity': 'orange', 'weighted_all': 'brown',
        'weighted_improved': 'darkred'
    }

    linestyles = {
        'random': '--', 'uncertainty': '-', 'uncertainty_improved': '-',
        'synergy': '-', 'diversity': '-', 'weighted_all': '-.',
        'weighted_improved': '-.'
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
    # plt.show()  # Commented out for HPC (no display)


def print_summary_table(results: dict, save_path: str = None):
    """Print and optionally save summary."""
    summary_lines = []
    summary_lines.append("=" * 80)
    summary_lines.append("SUMMARY")
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
# Main
# ================================================================================

if __name__ == "__main__":
    from ensemble import Ensemble

    SCLAMBDA_REPO = '/insomnia001/depts/edu/BIOLBC3141_Fall2025/rc3517/ml-stats/scLAMBDA'
    DATA_DIR = '/insomnia001/depts/edu/BIOLBC3141_Fall2025/rc3517/ml-stats/data'
    RESULT_DIR = '/insomnia001/depts/edu/users/rc3517/active_learning_results'

    os.makedirs(RESULT_DIR, exist_ok=True)
    print(f"Results will be saved to: {RESULT_DIR}\n")

    print("Loading ensemble...")
    ensemble = Ensemble(sclambda_repo_path=SCLAMBDA_REPO)

    # Load models WITHOUT fitting baselines (to avoid data leakage)
    ensemble.load_models(
        gears_model_dir=f'{DATA_DIR}/gears_model',
        gears_data_path=f'{DATA_DIR}/gears_data',
        gears_data_name='norman',
        sclambda_model_path=f'{DATA_DIR}/sclambda_model',
        sclambda_adata_path=f'{DATA_DIR}/adata_norman_preprocessed.h5ad',
        sclambda_embeddings_path=f'{DATA_DIR}/GPT_3_5_gene_embeddings_3-large.pickle',
        norman_data_path=f'{DATA_DIR}/adata_norman_preprocessed.h5ad'
    )

    # load pre-computed ensemble weights (from ensemble.py)
    WEIGHTS_PATH = f'{DATA_DIR}/ensemble_weights.pkl'
    weights_loaded = ensemble.load_ensemble_weights(WEIGHTS_PATH)
    if not weights_loaded:
        print(f"\n No saved weights found at {WEIGHTS_PATH}")
        print("   Ensemble will use uniform weights (0.25 each)")
        print("   Run ensemble.py first to learn optimal weights!")

    # Create splits BEFORE fitting baselines to avoid data leakage
    print("\nCreating data splits (BEFORE fitting baselines to avoid data leakage)...")
    splits = ensemble.data_processor.create_combo_splits(
        X_single=ensemble.X_single,
        y_single=ensemble.y_single,
        X_combo=ensemble.X_combo,
        y_combo=ensemble.y_combo,
        combo_test_ratio=0.3,
        random_state=42
    )

    # Start with 5% of training data for active learning
    n_init = max(50, int(len(splits['X_train']) * 0.05))  # 5% of training data
    X_train_init = splits['X_train'][:n_init]
    y_train_init = splits['y_train'][:n_init]
    X_pool = splits['X_train'][n_init:]
    y_pool = splits['y_train'][n_init:]

    print(f"\nACTIVE LEARNING SETUP:")
    print(f"  Initial training: {len(X_train_init):,} samples (5% of available data)")
    print(f"  Pool: {len(X_pool):,} samples")
    print(f"  Test: {len(splits['X_test']):,} samples")

    # FIT BASELINES ONLY ON INITIAL TRAINING SINGLES (NO DATA LEAKAGE)
    print("\nFitting baselines on INITIAL TRAINING singles only (no data leakage)...")
    single_mask = X_train_init.sum(axis=1) == 1
    if single_mask.sum() > 0:
        X_single_init = X_train_init[single_mask]
        y_single_init = y_train_init[single_mask]
        ensemble._fit_baselines(X_single_train=X_single_init,
                               y_single_train=y_single_init)
        print(f"  Fitted on {len(X_single_init)} training singles (no data leakage)")
    else:
        # If no singles in initial training, use all singles from the full training split
        # (still avoiding test set leakage)
        print("  WARNING: No single perturbations in initial training set")
        print("  Fitting baselines on all training singles from the split...")
        n_train_singles = splits['n_singles_in_train']
        X_single_train_all = splits['X_train'][:n_train_singles]
        y_single_train_all = splits['y_train'][:n_train_singles]
        ensemble._fit_baselines(X_single_train=X_single_train_all,
                               y_single_train=y_single_train_all)
        print(f"  Fitted on {len(X_single_train_all)} training singles")

    print("\nComparing strategies...")
    results = compare_all_strategies(
        ensemble, X_train_init, y_train_init,
        splits['X_test'], splits['y_test'], X_pool, y_pool,
        n_iterations=10,  
        n_select=15,      
        result_dir=RESULT_DIR
    )

    plot_strategy_comparison(results, save_path=os.path.join(RESULT_DIR, 'strategy_comparison.png'))
    print_summary_table(results, save_path=os.path.join(RESULT_DIR, 'summary.txt'))

    # SHAP
    print("\nRunning SHAP analysis...")
    al = ActiveLearningLoop(ensemble)
    al.run_shap_analysis(ensemble.X_combo, n_background=100, n_explain=3)
    al.plot_shap_results(RESULT_DIR)

    print(f"\nDone! All results saved to {RESULT_DIR}")