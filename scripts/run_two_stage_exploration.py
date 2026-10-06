"""
Exploration Script for Two-Stage Feature Selection:
1. Locks 20% holdout test set (unseen).
2. Performs Stage 1: Hierarchical Collinearity Clustering & Pruning on the 80% train set.
3. Performs Stage 2: RFECV (Recursive Feature Elimination with 5-Fold Stratified CV).
4. Prints detailed findings: Collinear clusters, dropped features, and selected elite features.
"""

import os
import sys
sys.path.append(os.path.abspath("."))

from src.data_loader import load_dataset, get_stratified_split
from src.two_stage_selection import stage1_correlation_pruning, stage2_rfecv_selection
import pandas as pd
import numpy as np


def main():
    print("=" * 70)
    print("STEP 1: LOADING DATA & SPLITTING 80% TRAIN / 20% HOLDOUT TEST SET")
    print("=" * 70)
    X, y, target_names = load_dataset()
    X_train, X_test, y_train, y_test = get_stratified_split(X, y, test_size=0.20, random_state=42)
    
    print(f"Full Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Training Set (80%): {X_train.shape[0]} samples (Malignant={sum(y_train==1)}, Benign={sum(y_train==0)})")
    print(f"Test Set (20%):     {X_test.shape[0]} samples (Malignant={sum(y_test==1)}, Benign={sum(y_test==0)})")
    print(">> TEST SET SAFELY LOCKED AWAY. All selection occurs strictly on Training Set <<\n")

    print("=" * 70)
    print("STEP 2: STAGE 1 - COLLINEARITY & MULTICOLLINEARITY PRUNING (|r| >= 0.85)")
    print("=" * 70)
    
    # Calculate baseline collinear pairs
    corr_matrix = X_train.corr(method="spearman").abs()
    high_corr_pairs = []
    cols = X_train.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr_matrix.iloc[i, j]
            if r >= 0.85:
                high_corr_pairs.append((cols[i], cols[j], r))
                
    print(f"Total feature pairs with |r| >= 0.85: {len(high_corr_pairs)}")
    print("Sample extreme collinear pairs:")
    for feat1, feat2, r in sorted(high_corr_pairs, key=lambda x: x[2], reverse=True)[:6]:
        print(f"  - {feat1:<25} <--> {feat2:<25} : r = {r:.4f}")
        
    kept_features, dropped_features, _, _ = stage1_correlation_pruning(
        X_train, y_train, corr_threshold=0.85, method="spearman"
    )
    
    print(f"\nStage 1 Result: Reduced from 30 features down to {len(kept_features)} non-redundant features.")
    print("\nPruning Breakdown (Clusters formed by collinearity):")
    for kept, dropped in dropped_features.items():
        if dropped:
            print(f"  [Cluster] Retained: '{kept}' (Higher target alignment)")
            print(f"            Dropped redundant: {dropped}")
            
    print(f"\nRemaining {len(kept_features)} Features after Stage 1:")
    for idx, f in enumerate(kept_features, 1):
        print(f"  {idx:2d}. {f}")

    print("\n" + "=" * 70)
    print("STEP 3: STAGE 2 - RFECV OPTIMAL SUBSET SELECTION (5-Fold Stratified CV)")
    print("=" * 70)
    
    X_train_pruned = X_train[kept_features]
    
    # Stage 2 with regularized logistic regression estimator
    selected_features, rfecv = stage2_rfecv_selection(
        X_train_pruned, y_train, estimator_type="logistic", cv_folds=5, scoring="roc_auc", random_state=42
    )
    
    print(f"Optimal Number of Features determined by RFECV: {rfecv.n_features_}")
    print(f"Best Mean CV ROC-AUC: {rfecv.cv_results_['mean_test_score'].max():.4f}")
    print("\nFinal Selected Elite Feature Subset:")
    for idx, f in enumerate(selected_features, 1):
        print(f"  {idx:2d}. {f}")
        
    # Also test with Random Forest estimator for robustness comparison
    selected_features_rf, rfecv_rf = stage2_rfecv_selection(
        X_train_pruned, y_train, estimator_type="rf", cv_folds=5, scoring="roc_auc", random_state=42
    )
    print(f"\nRFECV with Random Forest Estimator:")
    print(f"Optimal Features: {rfecv_rf.n_features_}, Best Mean CV ROC-AUC: {rfecv_rf.cv_results_['mean_test_score'].max():.4f}")
    print(f"Features: {selected_features_rf}")


if __name__ == "__main__":
    main()
