"""
Two-Stage Feature Selection Module:
Stage 1: Collinearity & Multicollinearity Pruning via Hierarchical Correlation Clustering.
Stage 2: Optimal Minimal Subset Selection via Recursive Feature Elimination with Cross-Validation (RFECV).
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold


def stage1_correlation_pruning(
    X_train: pd.DataFrame, 
    y_train: pd.Series, 
    corr_threshold: float = 0.85,
    method: str = "spearman"
):
    """
    Stage 1: Hierarchical Collinearity Pruning.
    Groups features that have absolute correlation >= corr_threshold.
    Within each cluster, retains the single feature with highest correlation to the target y_train.
    
    Returns:
        kept_features (list): Non-redundant feature subset.
        dropped_features (dict): Mapping of kept_feature -> list of dropped redundant features.
        corr_matrix (DataFrame): The calculated feature correlation matrix.
        linkage_matrix (ndarray): Hierarchical clustering linkage matrix for dendrogram plotting.
    """
    corr_matrix = X_train.corr(method=method).abs()
    
    # Distance matrix: d = 1 - |r|
    dist_matrix = 1.0 - corr_matrix.values
    np.fill_diagonal(dist_matrix, 0)
    
    # Clip small negative values due to floating point inaccuracies
    dist_matrix = np.clip(dist_matrix, 0, 1)
    
    condensed_dist = squareform(dist_matrix, checks=False)
    linkage_matrix = linkage(condensed_dist, method="average")
    
    # Cluster distance threshold is 1 - corr_threshold
    dist_threshold = 1.0 - corr_threshold
    cluster_labels = fcluster(linkage_matrix, t=dist_threshold, criterion="distance")
    
    # Calculate association of each feature with target y_train
    target_corrs = X_train.apply(lambda col: abs(col.corr(y_train, method=method)))
    
    kept_features = []
    dropped_features = {}
    
    unique_clusters = np.unique(cluster_labels)
    for c in unique_clusters:
        cluster_cols = X_train.columns[cluster_labels == c].tolist()
        if len(cluster_cols) == 1:
            kept_features.append(cluster_cols[0])
            dropped_features[cluster_cols[0]] = []
        else:
            # Pick feature with highest target correlation
            best_feature = target_corrs[cluster_cols].idxmax()
            kept_features.append(best_feature)
            dropped = [col for col in cluster_cols if col != best_feature]
            dropped_features[best_feature] = dropped
            
    return kept_features, dropped_features, corr_matrix, linkage_matrix


def stage2_rfecv_selection(
    X_train_pruned: pd.DataFrame,
    y_train: pd.Series,
    estimator_type: str = "logistic",
    cv_folds: int = 5,
    scoring: str = "roc_auc",
    random_state: int = 42
):
    """
    Stage 2: RFECV (Recursive Feature Elimination with Cross-Validation).
    Operates on the non-redundant Stage 1 features to determine the minimal, optimal feature count.
    
    Returns:
        selected_features (list): Final optimal features.
        rfecv_selector (RFECV): Fitted RFECV object with cv_results_.
    """
    if estimator_type == "logistic":
        base_estimator = LogisticRegression(
            penalty="l2", C=1.0, solver="liblinear", random_state=random_state, max_iter=1000
        )
    elif estimator_type == "rf":
        base_estimator = RandomForestClassifier(
            n_estimators=100, max_depth=4, random_state=random_state
        )
    else:
        raise ValueError(f"Unsupported estimator_type: {estimator_type}")
        
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    
    # We must scale features before RFE if using logistic regression
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train_pruned)
    
    rfecv = RFECV(
        estimator=base_estimator,
        step=1,
        cv=cv,
        scoring=scoring,
        min_features_to_select=3
    )
    rfecv.fit(X_scaled, y_train)
    
    selected_mask = rfecv.support_
    selected_features = X_train_pruned.columns[selected_mask].tolist()
    
    return selected_features, rfecv
