"""
Analyze and visualize Additive + VAE ensemble predictions.
Generates publication-quality plots.
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import torch
from sklearn.metrics import mean_squared_error
from sklearn.calibration import calibration_curve

from ensemble_add_vae import (
    load_norman_data,
    create_splits,
    fit_additive,
    AdditiveVAEEnsemble,
    predict_additive,
    VAEWrapper,
    ConformalPredictor
)

# ============================================================================
# Utility metrics
# ============================================================================

def compute_metrics(y_true, y_pred):
    mse = np.mean((y_pred - y_true) ** 2)
    mae = np.mean(np.abs(y_pred - y_true))

    per_sample_corr = []
    for i in range(len(y_pred)):
        if np.std(y_pred[i]) > 0 and np.std(y_true[i]) > 0:
            r, _ = pearsonr(y_pred[i], y_true[i])
            per_sample_corr.append(r)
    pearson_r = np.mean(per_sample_corr) if per_sample_corr else 0.0

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot

    return {"mse": mse, "mae": mae, "pearson_r": pearson_r, "r2": r2}

# ============================================================================
# Plotting functions
# ============================================================================

def plot_model_comparison(metrics, output_dir):
    models = ["additive", "vae", "ensemble"]
    colors = ["#9b59b6", "#2ecc71", "#1abc9c"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle("Model Performance Comparison", fontsize=16, fontweight="bold")

    for ax, metric, title, ylabel in zip(
        axes.flatten(),
        ["mse", "pearson_r", "r2", "mae"],
        ["MSE (↓)", "Pearson r (↑)", "R² (↑)", "MAE (↓)"],
        ["MSE", "Pearson r", "R²", "MAE"]
    ):
        values = [metrics[m][metric] for m in models]
        bars = ax.bar(models, values, color=colors, edgecolor="black")
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.set_xticklabels(models, rotation=45, ha="right")

        for b, v in zip(bars, values):
            ax.text(b.get_x() + b.get_width()/2, b.get_height(),
                    f"{v:.4f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    path = os.path.join(output_dir, "model_comparison.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved {path}")

def plot_uncertainty_distribution(epistemic_unc, output_dir):
    per_sample_unc = np.sum(epistemic_unc, axis=1)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(per_sample_unc, bins=50, edgecolor="black", alpha=0.75)
    ax.axvline(np.mean(per_sample_unc), color="red", linestyle="--",
               label=f"Mean = {np.mean(per_sample_unc):.4f}")
    ax.set_title("Epistemic Uncertainty Distribution", fontweight="bold")
    ax.set_xlabel("Total Uncertainty per Sample")
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(alpha=0.3, linestyle="--")
    path = os.path.join(output_dir, "uncertainty_distribution.png")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"✓ Saved {path}")
    return per_sample_unc

def plot_uncertainty_vs_error(ensemble, splits, epistemic_unc, output_dir):
    y_test = splits["y_test"]
    pred_mean, _ = ensemble.predict(splits["X_test"])
    per_sample_unc = np.sum(epistemic_unc, axis=1)
    per_sample_err = np.mean((pred_mean - y_test) ** 2, axis=1)

    fig, ax = plt.subplots(figsize=(8, 6))
    sc_plot = ax.scatter(per_sample_unc, per_sample_err,
                         c=per_sample_err, cmap="viridis",
                         alpha=0.6, edgecolor="black", linewidth=0.3)
    r, _ = pearsonr(per_sample_unc, per_sample_err)
    ax.set_title(f"Uncertainty vs Error (r = {r:.3f})", fontweight="bold")
    ax.set_xlabel("Total Uncertainty")
    ax.set_ylabel("MSE per Sample")
    ax.grid(alpha=0.3, linestyle="--")
    plt.colorbar(sc_plot, ax=ax, label="MSE")

    path = os.path.join(output_dir, "uncertainty_vs_error.png")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"✓ Saved {path}")

def plot_diversity_heatmap(predictions, output_dir):
    model_names = ["additive", "vae"]
    n = len(model_names)
    corr = np.zeros((n, n))
    for i, m1 in enumerate(model_names):
        for j, m2 in enumerate(model_names):
            x, y = predictions[m1].flatten(), predictions[m2].flatten()
            corr[i, j] = pearsonr(x, y)[0] if np.std(x) > 0 else 1.0

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr, cmap="RdYlGn_r", vmin=0, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([m.upper() for m in model_names])
    ax.set_yticklabels([m.upper() for m in model_names])
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{corr[i, j]:.3f}",
                    ha="center", va="center", fontweight="bold")
    ax.set_title("Model Prediction Correlation\n(Lower = More Diverse)", fontweight="bold")
    plt.colorbar(im, ax=ax)
    path = os.path.join(output_dir, "diversity_heatmap.png")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"✓ Saved {path}")
    return corr

def plot_gene_uncertainty(epistemic_unc, gene_names, output_dir, top_k=30):
    mean_unc = epistemic_unc.mean(axis=0)
    top_idx = np.argsort(mean_unc)[-top_k:]
    genes = np.array(gene_names)[top_idx]
    values = mean_unc[top_idx]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(genes, values, edgecolor="black")
    ax.set_title(f"Top {top_k} Genes by Epistemic Uncertainty", fontweight="bold")
    ax.set_xlabel("Mean Uncertainty")
    ax.grid(alpha=0.3, linestyle="--")
    path = os.path.join(output_dir, "gene_uncertainty.png")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"✓ Saved {path}")

# ============================================================================
# Main
# ============================================================================

def main():
    DATA_DIR = "/insomnia001/depts/edu/BIOLBC3141_Fall2025/rc3517/ml-stats/data"
    OUTPUT_DIR = "/insomnia001/depts/edu/users/rc3517/ensemble_results"
    ADATA_PATH = os.path.join(DATA_DIR, "adata_norman_preprocessed.h5ad")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load data
    Xs, ys, Xc, yc, gene_names = load_norman_data(ADATA_PATH)
    splits = create_splits(Xs, ys, Xc, yc)

    # Fit additive baseline
    effects = fit_additive(splits['X_train'][:splits['n_singles']],
                           splits['y_train'][:splits['n_singles']])

    # Train VAE
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vae = VAEWrapper(len(gene_names), device)
    vae.train(splits['X_train'], splits['y_train'],
              splits['X_val'], splits['y_val'])

    # Ensemble
    ensemble = AdditiveVAEEnsemble(effects, vae)
    ensemble.learn_weights(splits['X_val'], splits['y_val'])

    # Predictions
    pred_mean, epistemic_unc = ensemble.predict(splits['X_test'])
    test_mse = mean_squared_error(splits['y_test'], pred_mean)
    print(f"Test MSE: {test_mse:.6f}")
    print(f"Mean epistemic uncertainty: {epistemic_unc.mean():.6f}")

    # Conformal prediction
    cp = ConformalPredictor(alpha=0.1)
    cp.calibrate(splits['y_test'], pred_mean)
    lower, upper = cp.interval(pred_mean)
    width = upper - lower

    # Compute metrics
    predictions = {
        "additive": predict_additive(splits["X_test"], effects),
        "vae": vae.predict(splits["X_test"], n_samples=10).mean(axis=0)
    }
    metrics = {name: compute_metrics(splits["y_test"], pred) for name, pred in predictions.items()}
    metrics["ensemble"] = compute_metrics(splits["y_test"], pred_mean)

    # Generate plots
    plot_model_comparison(metrics, OUTPUT_DIR)
    plot_uncertainty_distribution(epistemic_unc, OUTPUT_DIR)
    plot_uncertainty_vs_error(ensemble, splits, epistemic_unc, OUTPUT_DIR)
    plot_diversity_heatmap(predictions, OUTPUT_DIR)
    plot_gene_uncertainty(epistemic_unc, gene_names, OUTPUT_DIR)

    # Save metrics
    with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n✓ ANALYSIS COMPLETE")
    print(f"Results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
