#!/usr/bin/env python
"""
Diagnose gene name mismatches between data and embeddings
"""

import scanpy as sc
import pickle
import sys

DATA_DIR = '/Users/rikac/Documents/ml_stats/project/data'
ADATA_PATH = f'{DATA_DIR}/norman_perturbseq_preprocessed_hvg.h5ad'
EMBEDDINGS_PATH = f'{DATA_DIR}/GPT_3_5_gene_embeddings_3-large.pickle'

print("="*70)
print("Gene Mismatch Diagnostic")
print("="*70)

# Load data
print("\n[1/3] Loading data...")
adata = sc.read_h5ad(ADATA_PATH)
print(f"  ✓ Loaded: {adata.n_obs} cells, {adata.n_vars} genes")

# Load embeddings
print("\n[2/3] Loading gene embeddings...")
with open(EMBEDDINGS_PATH, 'rb') as f:
    gene_embeddings = pickle.load(f)
print(f"  ✓ Loaded embeddings for {len(gene_embeddings)} genes")

# Extract perturbed genes from conditions
print("\n[3/3] Extracting perturbed genes from conditions...")
perturbed_genes = set()
for condition in adata.obs['condition'].unique():
    if condition in ['control', 'ctrl']:
        continue
    # Split by '+' for combo perturbations
    genes = condition.split('+')
    for gene in genes:
        gene = gene.strip()
        if gene not in ['control', 'ctrl']:
            perturbed_genes.add(gene)

print(f"  Found {len(perturbed_genes)} unique perturbed genes")

# Check which genes are missing from embeddings
missing_genes = []
for gene in perturbed_genes:
    if gene not in gene_embeddings:
        missing_genes.append(gene)

print("\n" + "="*70)
print("RESULTS")
print("="*70)

if missing_genes:
    print(f"\n⚠️  Found {len(missing_genes)} genes in perturbations but NOT in embeddings:")
    print("-"*70)
    for i, gene in enumerate(sorted(missing_genes)[:20], 1):
        print(f"  {i:2d}. {gene}")
    if len(missing_genes) > 20:
        print(f"  ... and {len(missing_genes) - 20} more")

    # Count how many cells have these missing genes
    print("\n" + "="*70)
    print("IMPACT ANALYSIS")
    print("="*70)
    affected_cells = 0
    affected_conditions = 0
    for condition in adata.obs['condition'].unique():
        genes = [g.strip() for g in condition.split('+')]
        if any(g in missing_genes for g in genes):
            affected_conditions += 1
            affected_cells += (adata.obs['condition'] == condition).sum()

    print(f"  Affected conditions: {affected_conditions}/{adata.obs['condition'].nunique()}")
    print(f"  Affected cells: {affected_cells}/{adata.n_obs} ({100*affected_cells/adata.n_obs:.2f}%)")

    print("\n" + "="*70)
    print("SOLUTIONS")
    print("="*70)
    print("\nOption 1 (RECOMMENDED): Filter out perturbations with missing genes")
    print("  - Remove cells with conditions containing missing genes")
    print("  - Keep only perturbations where all genes have embeddings")
    print("\nOption 2: Get embeddings for missing genes")
    print("  - Use GPT API to generate embeddings for missing genes")
    print("  - Or use a different embedding source")
    print("\nOption 3: Use placeholder embeddings")
    print("  - Replace missing genes with mean embedding (not recommended)")

    print("\n" + "="*70)
    print("ACTION REQUIRED")
    print("="*70)
    print("\nCreate filtered dataset with Option 1? (y/n)")

else:
    print("\n✓ SUCCESS! All perturbed genes have embeddings!")
    print("  You can proceed with training.")

# Sample embedding keys to help debug
print("\n" + "="*70)
print("EMBEDDING SAMPLE (first 20 genes)")
print("="*70)
sample_genes = list(gene_embeddings.keys())[:20]
for gene in sample_genes:
    print(f"  - {gene}")

print("\n" + "="*70)
print("PERTURBED GENES SAMPLE (first 20)")
print("="*70)
for gene in sorted(list(perturbed_genes))[:20]:
    status = "✓" if gene in gene_embeddings else "✗"
    print(f"  {status} {gene}")
