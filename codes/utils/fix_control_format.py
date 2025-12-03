#!/usr/bin/env python
"""
Check and fix control condition format for scLAMBDA
"""

import scanpy as sc
import numpy as np

DATA_DIR = '/Users/rikac/Documents/ml_stats/project/data'
INPUT_PATH = f'{DATA_DIR}/norman_perturbseq_preprocessed_hvg_filtered.h5ad'
OUTPUT_PATH = f'{DATA_DIR}/norman_perturbseq_preprocessed_hvg_filtered_v2.h5ad'

print("="*70)
print("Fix Control Condition Format")
print("="*70)

# Load data
print("\n[1/3] Loading data...")
adata = sc.read_h5ad(INPUT_PATH)
print(f"  Loaded: {adata.n_obs} cells")

# Check control conditions
print("\n[2/3] Analyzing control conditions...")
conditions = adata.obs['condition'].unique()

control_variants = []
for cond in conditions:
    if 'control' in cond.lower() or 'ctrl' in cond.lower():
        control_variants.append(cond)

print(f"  Found control condition variants: {control_variants}")
print(f"  Count per variant:")
for variant in control_variants:
    count = (adata.obs['condition'] == variant).sum()
    print(f"    '{variant}': {count} cells")

# Check what scLAMBDA expects
print("\n" + "="*70)
print("scLAMBDA Control Format Requirements")
print("="*70)
print("\nAccording to scLAMBDA documentation:")
print("  - Single gene perturbations should be: 'GENE+ctrl' (e.g., 'FEV+ctrl')")
print("  - Control cells should be: 'ctrl' (lowercase)")
print("  - Combo perturbations should be: 'GENE1+GENE2' (no ctrl)")

# Fix control format
print("\n[3/3] Fixing control format...")
print("  Converting all control variants to 'ctrl'")

adata.obs['condition'] = adata.obs['condition'].astype(str)
adata.obs['condition'] = adata.obs['condition'].replace({
    'control': 'ctrl',
    'Control': 'ctrl',
    'CONTROL': 'ctrl'
})

print(f"  ✓ Updated control format")

# Verify
ctrl_count = (adata.obs['condition'] == 'ctrl').sum()
print(f"  Control cells: {ctrl_count}")

# Check for any conditions that still have 'control' in them
remaining_control = [c for c in adata.obs['condition'].unique() if 'control' in c.lower() and c != 'ctrl']
if remaining_control:
    print(f"  ⚠️  Still have conditions with 'control': {remaining_control[:10]}")
else:
    print(f"  ✓ No remaining 'control' variants")

# Show sample conditions
print("\n" + "="*70)
print("Sample Conditions (first 20)")
print("="*70)
sample_conditions = sorted(adata.obs['condition'].unique())[:20]
for cond in sample_conditions:
    count = (adata.obs['condition'] == cond).sum()
    print(f"  {cond:<30} ({count:>6} cells)")

# Save
print("\n" + "="*70)
print("Saving...")
print("="*70)
adata.write_h5ad(OUTPUT_PATH)
print(f"  ✓ Saved to: {OUTPUT_PATH}")

print("\n" + "="*70)
print("NEXT STEPS")
print("="*70)
print("\n1. Update test_sclambda.py to use:")
print(f"   ADATA_PATH = '{OUTPUT_PATH}'")
print("\n2. Run test_sclambda.py again to verify")
print("\n" + "="*70)
