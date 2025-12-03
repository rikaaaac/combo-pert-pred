import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import os
import json
import warnings
warnings.filterwarnings('ignore')

from ensemble import Ensemble

class EnsembleAnalyzer:
    """Comprehensive analysis of ensemble model performance"""

    def __init__(self, ensemble, splits, output_dir='ensemble_results'):
        self.ensemble = ensemble
        self.splits = splits
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.metrics = {}
        self.predictions = None
        self.uncertainties = None

    def evaluate_individual_models(self):
        """Evaluate each model individually on test set"""
        print("\n" + "="*70)
        print("INDIVIDUAL MODEL EVALUATION")
        print("="*70)

        X_test = self.splits['X_test']
        y_test = self.splits['y_test']

        # Get all predictions
        _, self.uncertainties, self.predictions = self.ensemble.predict_ensemble(X_test)

        # Evaluate each model
        for model_name in ['gears', 'sclambda', 'mean', 'additive']:
            preds = self.predictions[model_name]

            # Compute metrics
            mse = np.mean((preds - y_test) ** 2)
            mae = np.mean(np.abs(preds - y_test))

            # Pearson correlation (per-sample)
            correlations = []
            for i in range(len(preds)):
                if np.std(preds[i]) > 0 and np.std(y_test[i]) > 0:
                    corr, _ = pearsonr(preds[i], y_test[i])
                    correlations.append(corr)
            mean_corr = np.mean(correlations) if correlations else 0.0

            # R-squared
            ss_res = np.sum((y_test - preds) ** 2)
            ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
            r2 = 1 - (ss_res / ss_tot)

            self.metrics[model_name] = {
                'mse': mse,
                'mae': mae,
                'pearson_r': mean_corr,
                'r2': r2
            }

            print(f"\n{model_name.upper():15s}: MSE={mse:.4f}, MAE={mae:.4f}, "
                  f"Pearson r={mean_corr:.4f}, R²={r2:.4f}")

        # Evaluate ensemble
        ensemble_pred = np.mean(np.stack([
            self.predictions['gears'],
            self.predictions['sclambda'],
            self.predictions['mean'],
            self.predictions['additive']
        ], axis=0), axis=0)

        mse = np.mean((ensemble_pred - y_test) ** 2)
        mae = np.mean(np.abs(ensemble_pred - y_test))

        correlations = []
        for i in range(len(ensemble_pred)):
            if np.std(ensemble_pred[i]) > 0 and np.std(y_test[i]) > 0:
                corr, _ = pearsonr(ensemble_pred[i], y_test[i])
                correlations.append(corr)
        mean_corr = np.mean(correlations) if correlations else 0.0

        ss_res = np.sum((y_test - ensemble_pred) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        r2 = 1 - (ss_res / ss_tot)

        self.metrics['ensemble'] = {
            'mse': mse,
            'mae': mae,
            'pearson_r': mean_corr,
            'r2': r2
        }

        print(f"\n{'ENSEMBLE':15s}: MSE={mse:.4f}, MAE={mae:.4f}, "
              f"Pearson r={mean_corr:.4f}, R²={r2:.4f}")
        print("="*70)

        return self.metrics

    def plot_model_comparison(self):
        """Create comprehensive model comparison plots"""
        print("\nGenerating model comparison plots...")

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Ensemble Model Performance Comparison', fontsize=16, fontweight='bold')

        models = ['gears', 'sclambda', 'mean', 'additive', 'ensemble']
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']

        # 1. MSE Comparison
        ax = axes[0, 0]
        mse_values = [self.metrics[m]['mse'] for m in models]
        bars = ax.bar(models, mse_values, color=colors, alpha=0.8, edgecolor='black')
        ax.set_ylabel('Mean Squared Error', fontsize=12, fontweight='bold')
        ax.set_title('MSE (Lower is Better)', fontsize=13, fontweight='bold')
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, mse_values)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.4f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')

        # Highlight ensemble
        bars[-1].set_edgecolor('black')
        bars[-1].set_linewidth(3)

        # 2. Pearson Correlation
        ax = axes[0, 1]
        corr_values = [self.metrics[m]['pearson_r'] for m in models]
        bars = ax.bar(models, corr_values, color=colors, alpha=0.8, edgecolor='black')
        ax.set_ylabel('Pearson Correlation', fontsize=12, fontweight='bold')
        ax.set_title('Pearson r (Higher is Better)', fontsize=13, fontweight='bold')
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_ylim([0, max(corr_values) * 1.2])

        for i, (bar, val) in enumerate(zip(bars, corr_values)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.4f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')

        bars[-1].set_edgecolor('black')
        bars[-1].set_linewidth(3)

        # 3. R² Comparison
        ax = axes[1, 0]
        r2_values = [self.metrics[m]['r2'] for m in models]
        bars = ax.bar(models, r2_values, color=colors, alpha=0.8, edgecolor='black')
        ax.set_ylabel('R² Score', fontsize=12, fontweight='bold')
        ax.set_title('R² (Higher is Better)', fontsize=13, fontweight='bold')
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        for i, (bar, val) in enumerate(zip(bars, r2_values)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.4f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')

        bars[-1].set_edgecolor('black')
        bars[-1].set_linewidth(3)

        # 4. MAE Comparison
        ax = axes[1, 1]
        mae_values = [self.metrics[m]['mae'] for m in models]
        bars = ax.bar(models, mae_values, color=colors, alpha=0.8, edgecolor='black')
        ax.set_ylabel('Mean Absolute Error', fontsize=12, fontweight='bold')
        ax.set_title('MAE (Lower is Better)', fontsize=13, fontweight='bold')
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        for i, (bar, val) in enumerate(zip(bars, mae_values)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.4f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')

        bars[-1].set_edgecolor('black')
        bars[-1].set_linewidth(3)

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/model_comparison.png', dpi=300, bbox_inches='tight')
        print(f"  → Saved: {self.output_dir}/model_comparison.png")
        plt.close()

    def plot_uncertainty_distribution(self):
        """Plot uncertainty distribution across test samples"""
        print("\nGenerating uncertainty distribution plots...")

        # Compute per-sample total uncertainty
        per_sample_uncertainty = np.sum(self.uncertainties, axis=1)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Epistemic Uncertainty Analysis', fontsize=16, fontweight='bold')

        # 1. Histogram of uncertainty
        ax = axes[0, 0]
        ax.hist(per_sample_uncertainty, bins=50, color='#3498db', alpha=0.7, edgecolor='black')
        ax.axvline(np.mean(per_sample_uncertainty), color='red', linestyle='--',
                   linewidth=2, label=f'Mean: {np.mean(per_sample_uncertainty):.2f}')
        ax.axvline(np.median(per_sample_uncertainty), color='green', linestyle='--',
                   linewidth=2, label=f'Median: {np.median(per_sample_uncertainty):.2f}')
        ax.set_xlabel('Total Uncertainty per Sample', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax.set_title('Distribution of Uncertainty Scores', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3, linestyle='--')

        # 2. Box plot
        ax = axes[0, 1]
        bp = ax.boxplot([per_sample_uncertainty],
                        labels=['Test Set'],
                        patch_artist=True,
                        widths=0.5)
        bp['boxes'][0].set_facecolor('#3498db')
        bp['boxes'][0].set_alpha(0.7)
        ax.set_ylabel('Total Uncertainty', fontsize=12, fontweight='bold')
        ax.set_title('Uncertainty Distribution Summary', fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        # Add statistics text
        stats_text = (f"Min: {np.min(per_sample_uncertainty):.2f}\n"
                     f"Q1: {np.percentile(per_sample_uncertainty, 25):.2f}\n"
                     f"Median: {np.median(per_sample_uncertainty):.2f}\n"
                     f"Q3: {np.percentile(per_sample_uncertainty, 75):.2f}\n"
                     f"Max: {np.max(per_sample_uncertainty):.2f}")
        ax.text(1.15, 0.5, stats_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='center',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # 3. Cumulative distribution
        ax = axes[1, 0]
        sorted_unc = np.sort(per_sample_uncertainty)
        cumulative = np.arange(1, len(sorted_unc) + 1) / len(sorted_unc)
        ax.plot(sorted_unc, cumulative, linewidth=2, color='#3498db')
        ax.set_xlabel('Total Uncertainty', fontsize=12, fontweight='bold')
        ax.set_ylabel('Cumulative Probability', fontsize=12, fontweight='bold')
        ax.set_title('Cumulative Distribution Function', fontsize=13, fontweight='bold')
        ax.grid(alpha=0.3, linestyle='--')

        # Mark top 5% uncertainty threshold
        threshold_95 = np.percentile(per_sample_uncertainty, 95)
        ax.axvline(threshold_95, color='red', linestyle='--', linewidth=2,
                  label=f'95th percentile: {threshold_95:.2f}')
        ax.legend(fontsize=10)

        # 4. Top uncertainty samples
        ax = axes[1, 1]
        top_n = 20
        top_indices = np.argsort(per_sample_uncertainty)[-top_n:]
        top_uncertainties = per_sample_uncertainty[top_indices]

        y_pos = np.arange(top_n)
        bars = ax.barh(y_pos, top_uncertainties, color='#e74c3c', alpha=0.8, edgecolor='black')
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f'Sample {i}' for i in top_indices], fontsize=8)
        ax.set_xlabel('Total Uncertainty', fontsize=12, fontweight='bold')
        ax.set_title(f'Top {top_n} Most Uncertain Samples', fontsize=13, fontweight='bold')
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.invert_yaxis()

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/uncertainty_distribution.png', dpi=300, bbox_inches='tight')
        print(f"  → Saved: {self.output_dir}/uncertainty_distribution.png")
        plt.close()

        return per_sample_uncertainty

    def plot_uncertainty_vs_error(self):
        """Plot relationship between uncertainty and prediction error"""
        print("\nGenerating uncertainty vs error analysis...")

        y_test = self.splits['y_test']
        ensemble_pred = np.mean(np.stack([
            self.predictions['gears'],
            self.predictions['sclambda'],
            self.predictions['mean'],
            self.predictions['additive']
        ], axis=0), axis=0)

        # Compute per-sample metrics
        per_sample_uncertainty = np.sum(self.uncertainties, axis=1)
        per_sample_error = np.mean((ensemble_pred - y_test) ** 2, axis=1)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Uncertainty vs Prediction Error', fontsize=16, fontweight='bold')

        # 1. Scatter plot
        ax = axes[0]
        scatter = ax.scatter(per_sample_uncertainty, per_sample_error,
                           alpha=0.5, s=20, c=per_sample_error,
                           cmap='viridis', edgecolors='black', linewidth=0.5)
        ax.set_xlabel('Total Uncertainty', fontsize=12, fontweight='bold')
        ax.set_ylabel('Mean Squared Error', fontsize=12, fontweight='bold')
        ax.set_title('Uncertainty vs Error (per sample)', fontsize=13, fontweight='bold')
        ax.grid(alpha=0.3, linestyle='--')

        # Add correlation
        corr, p_val = pearsonr(per_sample_uncertainty, per_sample_error)
        ax.text(0.05, 0.95, f'Pearson r = {corr:.3f}\np-value = {p_val:.2e}',
               transform=ax.transAxes, fontsize=11,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.colorbar(scatter, ax=ax, label='MSE')

        # 2. Binned analysis
        ax = axes[1]
        n_bins = 10
        bins = np.percentile(per_sample_uncertainty, np.linspace(0, 100, n_bins+1))
        bin_centers = []
        bin_errors = []
        bin_stds = []

        for i in range(n_bins):
            mask = (per_sample_uncertainty >= bins[i]) & (per_sample_uncertainty < bins[i+1])
            if mask.sum() > 0:
                bin_centers.append((bins[i] + bins[i+1]) / 2)
                bin_errors.append(np.mean(per_sample_error[mask]))
                bin_stds.append(np.std(per_sample_error[mask]))

        ax.errorbar(bin_centers, bin_errors, yerr=bin_stds,
                   fmt='o-', linewidth=2, markersize=8, capsize=5,
                   color='#3498db', ecolor='#e74c3c', capthick=2)
        ax.set_xlabel('Uncertainty (binned)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Mean Prediction Error', fontsize=12, fontweight='bold')
        ax.set_title('Error vs Uncertainty (binned analysis)', fontsize=13, fontweight='bold')
        ax.grid(alpha=0.3, linestyle='--')

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/uncertainty_vs_error.png', dpi=300, bbox_inches='tight')
        print(f"  → Saved: {self.output_dir}/uncertainty_vs_error.png")
        plt.close()

    def save_summary(self):
        """Save comprehensive summary to text file"""
        print("\nGenerating summary report...")

        summary_path = f'{self.output_dir}/summary.txt'

        with open(summary_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("ENSEMBLE MODEL ANALYSIS SUMMARY\n")
            f.write("="*80 + "\n\n")

            # Dataset info
            f.write("DATASET INFORMATION\n")
            f.write("-" * 80 + "\n")
            f.write(f"Test samples: {self.splits['X_test'].shape[0]}\n")
            f.write(f"Number of genes (features): {self.splits['X_test'].shape[1]}\n")
            f.write(f"Expression dimensions: {self.splits['y_test'].shape[1]}\n")
            f.write(f"Training samples: {self.splits['X_train'].shape[0]}\n")
            f.write(f"Validation samples: {self.splits['X_val'].shape[0]}\n\n")

            # Model performance
            f.write("MODEL PERFORMANCE\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Model':<15s} {'MSE':>10s} {'MAE':>10s} {'Pearson r':>12s} {'R²':>10s}\n")
            f.write("-" * 80 + "\n")

            for model in ['gears', 'sclambda', 'mean', 'additive', 'ensemble']:
                metrics = self.metrics[model]
                f.write(f"{model:<15s} "
                       f"{metrics['mse']:>10.6f} "
                       f"{metrics['mae']:>10.6f} "
                       f"{metrics['pearson_r']:>12.6f} "
                       f"{metrics['r2']:>10.6f}\n")

            f.write("\n")

            # Best model per metric
            f.write("BEST MODELS BY METRIC\n")
            f.write("-" * 80 + "\n")

            best_mse = min(self.metrics.items(), key=lambda x: x[1]['mse'])
            best_pearson = max(self.metrics.items(), key=lambda x: x[1]['pearson_r'])
            best_r2 = max(self.metrics.items(), key=lambda x: x[1]['r2'])

            f.write(f"Lowest MSE:         {best_mse[0]} ({best_mse[1]['mse']:.6f})\n")
            f.write(f"Highest Pearson r:  {best_pearson[0]} ({best_pearson[1]['pearson_r']:.6f})\n")
            f.write(f"Highest R²:         {best_r2[0]} ({best_r2[1]['r2']:.6f})\n\n")

            # Uncertainty statistics
            per_sample_uncertainty = np.sum(self.uncertainties, axis=1)
            mean_uncertainty_per_gene = np.mean(self.uncertainties, axis=0)

            f.write("UNCERTAINTY ANALYSIS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Mean uncertainty (per sample):     {np.mean(per_sample_uncertainty):.6f}\n")
            f.write(f"Median uncertainty (per sample):   {np.median(per_sample_uncertainty):.6f}\n")
            f.write(f"Std uncertainty (per sample):      {np.std(per_sample_uncertainty):.6f}\n")
            f.write(f"Min uncertainty:                   {np.min(per_sample_uncertainty):.6f}\n")
            f.write(f"Max uncertainty:                   {np.max(per_sample_uncertainty):.6f}\n")
            f.write(f"95th percentile:                   {np.percentile(per_sample_uncertainty, 95):.6f}\n")
            f.write(f"99th percentile:                   {np.percentile(per_sample_uncertainty, 99):.6f}\n\n")

            # Uncertainty vs error correlation
            y_test = self.splits['y_test']
            ensemble_pred = np.mean(np.stack([
                self.predictions['gears'],
                self.predictions['sclambda'],
                self.predictions['mean'],
                self.predictions['additive']
            ], axis=0), axis=0)
            per_sample_error = np.mean((ensemble_pred - y_test) ** 2, axis=1)
            corr, p_val = pearsonr(per_sample_uncertainty, per_sample_error)

            f.write("UNCERTAINTY-ERROR RELATIONSHIP\n")
            f.write("-" * 80 + "\n")
            f.write(f"Correlation (uncertainty vs error): {corr:.6f}\n")
            f.write(f"P-value:                            {p_val:.2e}\n")
            f.write(f"Interpretation: {'Significant' if p_val < 0.05 else 'Not significant'} ")
            f.write(f"{'positive' if corr > 0 else 'negative'} correlation\n\n")

            # Model disagreement insights
            f.write("MODEL AGREEMENT ANALYSIS\n")
            f.write("-" * 80 + "\n")

            # Compute pairwise model disagreements
            model_names = ['gears', 'sclambda', 'mean', 'additive']
            f.write("Average pairwise MSE between models:\n")
            for i, m1 in enumerate(model_names):
                for m2 in model_names[i+1:]:
                    mse_diff = np.mean((self.predictions[m1] - self.predictions[m2]) ** 2)
                    f.write(f"  {m1} vs {m2}: {mse_diff:.6f}\n")

            f.write("\n")
            f.write("="*80 + "\n")
            f.write("Generated by ensemble_analyze.py\n")
            f.write("="*80 + "\n")

        print(f"  → Saved: {summary_path}")

        # Also save as JSON
        json_path = f'{self.output_dir}/metrics.json'
        with open(json_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        print(f"  → Saved: {json_path}")


# ================================================================================
# MAIN EXECUTION
# ================================================================================

if __name__ == "__main__":
    print("="*70)
    print("ENSEMBLE MODEL COMPREHENSIVE ANALYSIS")
    print("="*70)
    print("\nThis script loads pre-trained models and generates:")
    print("  1. Individual model performance metrics")
    print("  2. Model comparison visualizations")
    print("  3. Uncertainty distribution analysis")
    print("  4. Uncertainty vs error correlation")
    print("  5. Comprehensive summary report")
    print("="*70)

    # Configure paths - MODIFY THESE FOR YOUR SYSTEM
    SCLAMBDA_REPO = '/insomnia001/depts/edu/BIOLBC3141_Fall2025/rc3517/ml-stats/scLAMBDA'
    DATA_DIR = '/insomnia001/depts/edu/BIOLBC3141_Fall2025/rc3517/ml-stats/data'

    GEARS_MODEL_DIR = f'{DATA_DIR}/gears_model'
    GEARS_DATA_PATH = f'{DATA_DIR}/gears_data'
    SCLAMBDA_MODEL_PATH = f'{DATA_DIR}/sclambda_model'
    SCLAMBDA_ADATA_PATH = f'{DATA_DIR}/norman_perturbseq_preprocessed_hvg_filtered.h5ad'
    SCLAMBDA_EMBEDDINGS_PATH = f'{DATA_DIR}/GPT_3_5_gene_embeddings_3-large.pickle'
    NORMAN_DATA_PATH = f'{DATA_DIR}/norman_perturbseq_preprocessed_hvg_filtered.h5ad'

    # Load ensemble
    print("\n[1/5] Loading ensemble models...")
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

    # Create splits
    print("\n[2/5] Creating data splits...")
    splits = ensemble.data_processor.create_combo_splits(
        X_single=ensemble.X_single,
        y_single=ensemble.y_single,
        X_combo=ensemble.X_combo,
        y_combo=ensemble.y_combo,
        combo_test_ratio=0.2,
        random_state=42
    )

    # Initialize analyzer
    print("\n[3/5] Initializing analyzer...")
    analyzer = EnsembleAnalyzer(ensemble, splits, output_dir='/insomnia001/depts/edu/BIOLBC3141_Fall2025/rc3517/ml-stats/ensemble_results')

    # Run analysis
    print("\n[4/5] Evaluating models...")
    analyzer.evaluate_individual_models()

    print("\n[5/5] Generating visualizations and reports...")
    analyzer.plot_model_comparison()
    analyzer.plot_uncertainty_distribution()
    analyzer.plot_uncertainty_vs_error()
    analyzer.save_summary()

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\nResults saved to: {analyzer.output_dir}/")
    print("\nGenerated files:")
    print("  - summary.txt: Comprehensive text report")
    print("  - metrics.json: Machine-readable metrics")
    print("  - model_comparison.png: Performance comparison across models")
    print("  - uncertainty_distribution.png: Uncertainty analysis")
    print("  - uncertainty_vs_error.png: Calibration analysis")
    print("="*70)
