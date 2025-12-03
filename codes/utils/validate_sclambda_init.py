#!/usr/bin/env python
"""
Validate scLAMBDA initialization before running full ensemble
Tests the exact code path that's failing
"""

import sys
import scanpy as sc
import pickle

# HPC paths
SCLAMBDA_REPO = '/insomnia001/depts/edu/BIOLBC3141_Fall2025/rc3517/ml-stats/scLAMBDA'
DATA_DIR = '/insomnia001/depts/edu/BIOLBC3141_Fall2025/rc3517/ml-stats/data'

SCLAMBDA_ADATA_PATH = f'{DATA_DIR}/norman_perturbseq_preprocessed_hvg_filtered.h5ad'
SCLAMBDA_EMBEDDINGS_PATH = f'{DATA_DIR}/GPT_3_5_gene_embeddings_3-large.pickle'
MODEL_PATH = f'{DATA_DIR}/sclambda_test_model'

print("="*70)
print("scLAMBDA Initialization Validation")
print("="*70)

# Step 1: Add scLAMBDA to path
print("\n[1/6] Adding scLAMBDA to path...")
if SCLAMBDA_REPO not in sys.path:
    sys.path.insert(0, SCLAMBDA_REPO)
print(f"  ✓ Added: {SCLAMBDA_REPO}")

# Step 2: Load data
print("\n[2/6] Loading AnnData...")
adata = sc.read_h5ad(SCLAMBDA_ADATA_PATH)
print(f"  ✓ Loaded: {adata.n_obs} cells, {adata.n_vars} genes")

# Step 3: CHECK AND FIX CONTROL FORMAT
print("\n[3/6] Checking control format...")
control_variants = [c for c in adata.obs['condition'].unique()
                   if 'control' in c.lower() or 'ctrl' in c.lower()]
print(f"  Control-related conditions: {control_variants}")

# Check if any genes will be parsed as 'control'
problem_found = False
for cond in adata.obs['condition'].unique():
    genes = [g.strip() for g in cond.split('+')]
    if 'control' in genes:
        problem_found = True
        print(f"  ✗ PROBLEM: '{cond}' contains 'control' as gene!")
        break

if problem_found:
    print("\n  Applying fix: replacing 'control' with 'ctrl'...")
    adata.obs['condition'] = adata.obs['condition'].astype(str).str.replace('control', 'ctrl')

    # Verify fix
    for cond in adata.obs['condition'].unique():
        genes = [g.strip() for g in cond.split('+')]
        if 'control' in genes:
            print(f"  ✗ ERROR: Still have 'control' in '{cond}'")
            sys.exit(1)

    print("  ✓ Fixed! Saving corrected dataset...")
    adata.write_h5ad(SCLAMBDA_ADATA_PATH)
    print(f"  ✓ Saved to: {SCLAMBDA_ADATA_PATH}")
else:
    print("  ✓ No 'control' issues found")

# Step 4: Load embeddings
print("\n[4/6] Loading gene embeddings...")
with open(SCLAMBDA_EMBEDDINGS_PATH, 'rb') as f:
    gene_embeddings = pickle.load(f)
print(f"  ✓ Loaded embeddings for {len(gene_embeddings)} genes")

# Step 5: Import and test scLAMBDA
print("\n[5/6] Testing scLAMBDA import and data_split...")
try:
    import sclambda
    print("  ✓ scLAMBDA imported")

    adata_split, split = sclambda.utils.data_split(adata, seed=0)
    print(f"  ✓ data_split successful ({adata_split.n_obs} cells)")

except Exception as e:
    print(f"  ✗ ERROR in data_split: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1) 

# Step 6: THE CRITICAL TEST - Model initialization
print("\n[6/6] Testing Model initialization (this is where it failed)...")
try:
    model = sclambda.model.Model(
        adata_split,
        gene_embeddings,
        model_path=MODEL_PATH,
        multi_gene=True
    )
    print("  ✓ Model initialized successfully!")

except KeyError as e:
    print(f"  ✗ ERROR: KeyError '{e}'")
    print("\n  This means scLAMBDA tried to look up this as a gene in embeddings.")
    print("  Check your condition column for this string.")
    import traceback
    traceback.print_exc()
    sys.exit(1)

except Exception as e:
    print(f"  ✗ ERROR: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*70)
print("SUCCESS! scLAMBDA initialization works!")
print("="*70)
print("\nYou can now safely run the full ensemble script.")
print("The scLAMBDA portion should work correctly.")
print("\n" + "="*70)
