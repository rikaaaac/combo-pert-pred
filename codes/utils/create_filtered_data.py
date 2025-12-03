#!/usr/bin/env python
"""
Create filtered dataset removing perturbations with genes missing from embeddings
"""

import scanpy as sc
import pickle
import numpy as np

DATA_DIR = '/Users/rikac/Documents/ml_stats/project/data'
ADATA_PATH = f'{DATA_DIR}/norman_perturbseq_preprocessed_hvg.h5ad'
EMBEDDINGS_PATH = f'{DATA_DIR}/GPT_3_5_gene_embeddings_3-large.pickle'
OUTPUT_PATH = f'{DATA_DIR}/norman_perturbseq_preprocessed_hvg_filtered.h5ad'

print("="*70)
print("Creating Filtered Dataset for scLAMBDA")
print("="*70)

# Load data
print("\n[1/4] Loading data...")
adata = sc.read_h5ad(ADATA_PATH)
print(f"  Original: {adata.n_obs} cells, {adata.n_vars} genes")
print(f"  Conditions: {adata.obs['condition'].nunique()}")

# Load embeddings
print("\n[2/4] Loading gene embeddings...")
with open(EMBEDDINGS_PATH, 'rb') as f:
    gene_embeddings = pickle.load(f)
print(f"  Embeddings for {len(gene_embeddings)} genes")

# Identify cells to keep
print("\n[3/4] Filtering cells...")
missing_genes = set()
keep_mask = np.ones(adata.n_obs, dtype=bool)

for idx, condition in enumerate(adata.obs['condition']):
    if condition in ['control', 'ctrl']:
        continue  # Keep all control cells

    # Extract genes from condition
    genes = [g.strip() for g in condition.split('+')]

    # Check if all genes have embeddings
    genes_without_embeddings = [g for g in genes if g not in gene_embeddings and g not in ['ctrl', 'control']]

    if genes_without_embeddings:
        keep_mask[idx] = False
        missing_genes.update(genes_without_embeddings)

# Create filtered dataset
adata_filtered = adata[keep_mask].copy()

print(f"\n  Missing genes found: {sorted(missing_genes)}")
print(f"  Removed cells: {(~keep_mask).sum()}")
print(f"  Kept cells: {keep_mask.sum()}")
print(f"  Filtered: {adata_filtered.n_obs} cells, {adata_filtered.n_vars} genes")
print(f"  Conditions: {adata_filtered.obs['condition'].nunique()}")

# Verify all remaining conditions have embeddings
print("\n[4/4] Verifying filtered dataset...")
remaining_genes = set()
for condition in adata_filtered.obs['condition'].unique():
    if condition in ['control', 'ctrl']:
        continue
    genes = [g.strip() for g in condition.split('+')]
    for gene in genes:
        if gene not in ['ctrl', 'control']:
            remaining_genes.add(gene)

missing_in_filtered = [g for g in remaining_genes if g not in gene_embeddings]

if missing_in_filtered:
    print(f"  ✗ ERROR: Still have missing genes: {missing_in_filtered}")
else:
    print(f"  ✓ All {len(remaining_genes)} remaining perturbed genes have embeddings!")

# Save filtered dataset
print(f"\n[5/5] Saving filtered dataset...")
adata_filtered.write_h5ad(OUTPUT_PATH)
print(f"  ✓ Saved to: {OUTPUT_PATH}")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"  Original cells:   {adata.n_obs:,}")
print(f"  Filtered cells:   {adata_filtered.n_obs:,}")
print(f"  Removed cells:    {adata.n_obs - adata_filtered.n_obs:,} ({100*(adata.n_obs - adata_filtered.n_obs)/adata.n_obs:.2f}%)")
print(f"  Original conditions: {adata.obs['condition'].nunique()}")
print(f"  Filtered conditions: {adata_filtered.obs['condition'].nunique()}")
print(f"  Removed conditions:  {adata.obs['condition'].nunique() - adata_filtered.obs['condition'].nunique()}")
print(f"\n  Genes without embeddings: {sorted(missing_genes)}")

print("\n" + "="*70)
print("NEXT STEPS")
print("="*70)
print("\n1. Update your ensemble.py to use the filtered dataset:")
print(f"   SCLAMBDA_ADATA_PATH = '{OUTPUT_PATH}'")
print(f"   NORMAN_DATA_PATH = '{OUTPUT_PATH}'")
print("\n2. Run test_sclambda.py again with filtered data to verify")
print("\n3. Submit your ensemble training job!")

print("\n" + "="*70)
