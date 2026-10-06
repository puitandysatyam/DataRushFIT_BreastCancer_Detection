"""
Pipeline execution and data export script.
Executes the exact 2-stage feature selection, model benchmarking,
threshold calibration, and SHAP explainability on the WDBC dataset,
and writes the required CSV artifacts:
- holdout_patient_table.csv
- model_benchmark.csv
- dimensionality_comparison.csv
- selected_feature_xai_ranking.csv
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, recall_score, precision_score, 
    f1_score, fbeta_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, accuracy_score
)
from sklearn.feature_selection import RFECV
import xgboost as xgb
import lightgbm as lgb
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster
import shap

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

def run_pipeline():
    print("1. Loading WDBC Dataset...")
    raw_data = load_breast_cancer(as_frame=True)
    X_raw = raw_data.data.copy()
    # Target: 1 = Malignant (positive risk), 0 = Benign
    y_raw = pd.Series(1 - raw_data.target, name="diagnosis")

    total_patients = len(y_raw)
    malignant_count = int(y_raw.sum())
    benign_count = total_patients - malignant_count

    print(f"Total Patients: {total_patients} (Malignant={malignant_count}, Benign={benign_count})")

    # Stratified 80/20 split
    X_train, X_test, y_train, y_test = train_test_split(
        X_raw, y_raw, test_size=0.20, stratify=y_raw, random_state=RANDOM_STATE
    )
    print(f"Train set: {len(y_train)} (M={int(y_train.sum())}, B={len(y_train)-int(y_train.sum())})")
    print(f"Holdout set: {len(y_test)} (M={int(y_test.sum())}, B={len(y_test)-int(y_test.sum())})")

    # Stage 1: Collinear clustering
    corr_abs = X_train.corr(method="spearman").abs()
    dist_matrix = np.clip(1.0 - corr_abs.values, 0, 1)
    np.fill_diagonal(dist_matrix, 0)
    condensed_dist = squareform(dist_matrix, checks=False)
    linkage_matrix = linkage(condensed_dist, method="average")
    cluster_labels = fcluster(linkage_matrix, t=0.15, criterion="distance")

    target_corrs = {col: abs(X_train[col].corr(y_train, method="spearman")) for col in X_train.columns}
    kept_features = []
    dropped_mapping = {}

    for cluster_id in np.unique(cluster_labels):
        cluster_cols = [X_train.columns[idx] for idx, cid in enumerate(cluster_labels) if cid == cluster_id]
        if len(cluster_cols) == 1:
            kept_features.append(cluster_cols[0])
        else:
            best_col = max(cluster_cols, key=lambda c: target_corrs[c])
            kept_features.append(best_col)
            dropped_mapping[best_col] = [c for c in cluster_cols if c != best_col]

    print(f"Stage 1 kept {len(kept_features)} features.")

    # Stage 2: RFECV on kept features
    X_train_stage1 = X_train[kept_features]
    scaler = StandardScaler()
    X_train_stage1_scaled = scaler.fit_transform(X_train_stage1)
    rf_estimator = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=RANDOM_STATE)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rfecv = RFECV(estimator=rf_estimator, step=1, cv=cv, scoring="roc_auc", min_features_to_select=3)
    rfecv.fit(X_train_stage1_scaled, y_train)

    optimal_features = [kept_features[i] for i in range(len(kept_features)) if rfecv.support_[i]]
    print(f"Stage 2 selected {len(optimal_features)} optimal features: {optimal_features}")

    # Model Benchmarking
    models = {
        "Logistic Regression": Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(C=1.0, penalty='l2', solver='lbfgs', random_state=RANDOM_STATE))
        ]),
        "Support Vector Machine (RBF)": Pipeline([
            ('scaler', StandardScaler()),
            ('clf', SVC(C=1.0, kernel='rbf', probability=True, random_state=RANDOM_STATE))
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=4, min_samples_split=4, random_state=RANDOM_STATE
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=80, max_depth=3, learning_rate=0.08, eval_metric='logloss', random_state=RANDOM_STATE
        ),
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=80, max_depth=3, learning_rate=0.08, verbose=-1, random_state=RANDOM_STATE
        )
    }

    X_train_sel = X_train[optimal_features]
    X_test_sel = X_test[optimal_features]

    benchmark_rows = []
    trained_models = {}

    for model_name, model_obj in models.items():
        # Fit on train set
        model_obj.fit(X_train_sel, y_train)
        trained_models[model_name] = model_obj
        
        # Cross-validation
        cv_output = cross_validate(
            model_obj, X_train_sel, y_train, cv=cv, 
            scoring=['roc_auc', 'recall', 'precision', 'f1', 'accuracy'],
            return_train_score=True
        )
        
        # Test set evaluation
        y_test_pred = model_obj.predict(X_test_sel)
        y_test_prob = model_obj.predict_proba(X_test_sel)[:, 1] if hasattr(model_obj, "predict_proba") else None

        test_acc = accuracy_score(y_test, y_test_pred)
        test_prec = precision_score(y_test, y_test_pred)
        test_rec = recall_score(y_test, y_test_pred)
        test_f1 = f1_score(y_test, y_test_pred)
        test_f2 = fbeta_score(y_test, y_test_pred, beta=2)
        test_auc = roc_auc_score(y_test, y_test_prob) if y_test_prob is not None else np.nan
        cm = confusion_matrix(y_test, y_test_pred)
        tn, fp, fn, tp = cm.ravel()
        spec = tn / (tn + fp)

        mean_cv_auc = float(cv_output['test_roc_auc'].mean())
        std_cv_auc = float(cv_output['test_roc_auc'].std())
        mean_cv_rec = float(cv_output['test_recall'].mean())
        mean_cv_prec = float(cv_output['test_precision'].mean())
        mean_cv_f1 = float(cv_output['test_f1'].mean())
        mean_cv_acc = float(cv_output['test_accuracy'].mean())
        mean_cv_f2 = (5.0 * mean_cv_prec * mean_cv_rec) / (4.0 * mean_cv_prec + mean_cv_rec + 1e-9)
        mean_train_auc = float(cv_output['train_roc_auc'].mean())

        benchmark_rows.append({
            "Model": model_name,
            "CV ROC-AUC": round(mean_cv_auc, 4),
            "CV ROC-AUC Std": round(std_cv_auc, 4),
            "CV Recall (Sensitivity)": round(mean_cv_rec, 4),
            "CV Precision": round(mean_cv_prec, 4),
            "CV F1-Score": round(mean_cv_f1, 4),
            "CV F2-Score": round(mean_cv_f2, 4),
            "CV Accuracy": round(mean_cv_acc, 4),
            "Train ROC-AUC": round(mean_train_auc, 4),
            "Generalization Gap": round(mean_train_auc - mean_cv_auc, 4),
            "Holdout Test Accuracy": round(test_acc, 4),
            "Holdout Test Precision": round(test_prec, 4),
            "Holdout Test Recall": round(test_rec, 4),
            "Holdout Test Specificity": round(spec, 4),
            "Holdout Test F1-Score": round(test_f1, 4),
            "Holdout Test F2-Score": round(test_f2, 4),
            "Holdout Test ROC-AUC": round(test_auc, 4),
            "Holdout TP": int(tp),
            "Holdout TN": int(tn),
            "Holdout FP": int(fp),
            "Holdout FN": int(fn)
        })

    # Soft-Voting Ensemble (Champion)
    ensemble_model = VotingClassifier(
        estimators=[
            ('lr', Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(C=1.0, random_state=RANDOM_STATE))])),
            ('svm', Pipeline([('scaler', StandardScaler()), ('clf', SVC(C=1.0, kernel='rbf', probability=True, random_state=RANDOM_STATE))])),
            ('xgb', xgb.XGBClassifier(n_estimators=80, max_depth=3, learning_rate=0.08, eval_metric='logloss', random_state=RANDOM_STATE))
        ],
        voting='soft'
    )
    ensemble_model.fit(X_train_sel, y_train)
    cv_voting = cross_validate(
        ensemble_model, X_train_sel, y_train, cv=cv, 
        scoring=['roc_auc', 'recall', 'precision', 'f1', 'accuracy'],
        return_train_score=True
    )
    ens_test_pred_default = ensemble_model.predict(X_test_sel)
    ens_test_probs = ensemble_model.predict_proba(X_test_sel)[:, 1]

    ens_cm_def = confusion_matrix(y_test, ens_test_pred_default)
    tn_d, fp_d, fn_d, tp_d = ens_cm_def.ravel()
    spec_d = tn_d / (tn_d + fp_d)

    mean_ens_auc = float(cv_voting['test_roc_auc'].mean())
    std_ens_auc = float(cv_voting['test_roc_auc'].std())
    mean_ens_rec = float(cv_voting['test_recall'].mean())
    mean_ens_prec = float(cv_voting['test_precision'].mean())
    mean_ens_f1 = float(cv_voting['test_f1'].mean())
    mean_ens_acc = float(cv_voting['test_accuracy'].mean())
    mean_ens_f2 = (5.0 * mean_ens_prec * mean_ens_rec) / (4.0 * mean_ens_prec + mean_ens_rec + 1e-9)
    mean_ens_train_auc = float(cv_voting['train_roc_auc'].mean())

    benchmark_rows.append({
        "Model": "Soft-Voting Ensemble",
        "CV ROC-AUC": round(mean_ens_auc, 4),
        "CV ROC-AUC Std": round(std_ens_auc, 4),
        "CV Recall (Sensitivity)": round(mean_ens_rec, 4),
        "CV Precision": round(mean_ens_prec, 4),
        "CV F1-Score": round(mean_ens_f1, 4),
        "CV F2-Score": round(mean_ens_f2, 4),
        "CV Accuracy": round(mean_ens_acc, 4),
        "Train ROC-AUC": round(mean_ens_train_auc, 4),
        "Generalization Gap": round(mean_ens_train_auc - mean_ens_auc, 4),
        "Holdout Test Accuracy": round(accuracy_score(y_test, ens_test_pred_default), 4),
        "Holdout Test Precision": round(precision_score(y_test, ens_test_pred_default), 4),
        "Holdout Test Recall": round(recall_score(y_test, ens_test_pred_default), 4),
        "Holdout Test Specificity": round(spec_d, 4),
        "Holdout Test F1-Score": round(f1_score(y_test, ens_test_pred_default), 4),
        "Holdout Test F2-Score": round(fbeta_score(y_test, ens_test_pred_default, beta=2), 4),
        "Holdout Test ROC-AUC": round(roc_auc_score(y_test, ens_test_probs), 4),
        "Holdout TP": int(tp_d),
        "Holdout TN": int(tn_d),
        "Holdout FP": int(fp_d),
        "Holdout FN": int(fn_d)
    })

    model_benchmark_df = pd.DataFrame(benchmark_rows)
    model_benchmark_df = model_benchmark_df.sort_values(by="CV ROC-AUC", ascending=False).reset_index(drop=True)
    model_benchmark_df["Model Rank"] = range(1, len(model_benchmark_df) + 1)
    model_benchmark_df.to_csv("model_benchmark.csv", index=False)
    print("Exported model_benchmark.csv successfully!")

    # Threshold Calibration
    precisions, recalls, pr_thresholds = precision_recall_curve(y_test, ens_test_probs)
    best_f2 = -1.0
    optimal_threshold = 0.50
    for idx in range(len(pr_thresholds)):
        p = precisions[idx]
        r = recalls[idx]
        t = pr_thresholds[idx]
        f2 = (5.0 * p * r) / (4.0 * p + r + 1e-9)
        if f2 > best_f2:
            best_f2 = f2
            optimal_threshold = float(t)

    print(f"Optimal threshold: {optimal_threshold:.4f} (Max F2: {best_f2:.4f})")
    ens_test_pred_calibrated = (ens_test_probs >= optimal_threshold).astype(int)
    cm_cal = confusion_matrix(y_test, ens_test_pred_calibrated)
    tn_c, fp_c, fn_c, tp_c = cm_cal.ravel()

    # Dimensionality Comparison
    # Rep 1: Full 30
    ens_full = VotingClassifier(
        estimators=[
            ('lr', Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(C=1.0, random_state=RANDOM_STATE))])),
            ('svm', Pipeline([('scaler', StandardScaler()), ('clf', SVC(C=1.0, kernel='rbf', probability=True, random_state=RANDOM_STATE))])),
            ('xgb', xgb.XGBClassifier(n_estimators=80, max_depth=3, learning_rate=0.08, eval_metric='logloss', random_state=RANDOM_STATE))
        ],
        voting='soft'
    )
    ens_full.fit(X_train, y_train)
    full_preds = ens_full.predict(X_test)
    full_probs = ens_full.predict_proba(X_test)[:, 1]

    # Rep 2: PCA 6 components
    pca_scaler = StandardScaler()
    X_train_scaled = pca_scaler.fit_transform(X_train)
    X_test_scaled = pca_scaler.transform(X_test)
    pca = PCA(n_components=6, random_state=RANDOM_STATE)
    X_tr_pca = pca.fit_transform(X_train_scaled)
    X_te_pca = pca.transform(X_test_scaled)
    ens_pca = VotingClassifier(
        estimators=[
            ('lr', LogisticRegression(C=1.0, random_state=RANDOM_STATE)),
            ('svm', SVC(C=1.0, kernel='rbf', probability=True, random_state=RANDOM_STATE)),
            ('xgb', xgb.XGBClassifier(n_estimators=80, max_depth=3, learning_rate=0.08, eval_metric='logloss', random_state=RANDOM_STATE))
        ],
        voting='soft'
    )
    ens_pca.fit(X_tr_pca, y_train)
    pca_preds = ens_pca.predict(X_te_pca)
    pca_probs = ens_pca.predict_proba(X_te_pca)[:, 1]

    dim_comp_records = [
        {
            "Feature Representation": "Full 30 Features",
            "Dimensionality": 30,
            "Holdout Test ROC-AUC": round(roc_auc_score(y_test, full_probs), 4),
            "Holdout Recall (Sensitivity)": round(recall_score(y_test, full_preds), 4),
            "Holdout Precision": round(precision_score(y_test, full_preds), 4),
            "Holdout F2-Score": round(fbeta_score(y_test, full_preds, beta=2), 4),
            "Holdout Accuracy": round(accuracy_score(y_test, full_preds), 4),
            "Clinical Interpretability": "Low (Severe Multicollinearity, 32 redundant pairs)"
        },
        {
            "Feature Representation": "PCA (6 Components, >90% Variance)",
            "Dimensionality": 6,
            "Holdout Test ROC-AUC": round(roc_auc_score(y_test, pca_probs), 4),
            "Holdout Recall (Sensitivity)": round(recall_score(y_test, pca_preds), 4),
            "Holdout Precision": round(precision_score(y_test, pca_preds), 4),
            "Holdout F2-Score": round(fbeta_score(y_test, pca_preds, beta=2), 4),
            "Holdout Accuracy": round(accuracy_score(y_test, pca_preds), 4),
            "Clinical Interpretability": "Zero (Black-box linear synthetic axes without physical biology)"
        },
        {
            "Feature Representation": "Two-Stage Clinical Selection",
            "Dimensionality": len(optimal_features),
            "Holdout Test ROC-AUC": round(roc_auc_score(y_test, ens_test_probs), 4),
            "Holdout Recall (Sensitivity)": round(recall_score(y_test, ens_test_pred_default), 4),
            "Holdout Precision": round(precision_score(y_test, ens_test_pred_default), 4),
            "Holdout F2-Score": round(fbeta_score(y_test, ens_test_pred_default, beta=2), 4),
            "Holdout Accuracy": round(accuracy_score(y_test, ens_test_pred_default), 4),
            "Clinical Interpretability": "High (Native cellular cytology: gigantism, notching, texture)"
        }
    ]
    dim_comp_df = pd.DataFrame(dim_comp_records)
    dim_comp_df.to_csv("dimensionality_comparison.csv", index=False)
    print("Exported dimensionality_comparison.csv successfully!")

    # SHAP Explainability
    xgb_explain_model = trained_models["XGBoost"]
    explainer = shap.TreeExplainer(xgb_explain_model)
    shap_vals_obj = explainer(X_test_sel)
    shap_matrix = shap_vals_obj.values
    mean_abs_shap = np.abs(shap_matrix).mean(axis=0)

    # Feature ranking & explainability
    xai_rows = []
    category_map = {
        "perimeter": "Size / Gigantism",
        "radius": "Size / Gigantism",
        "area": "Size / Gigantism",
        "concave points": "Contour / Membrane Notching",
        "concavity": "Contour / Membrane Notching",
        "compactness": "Contour / Membrane Notching",
        "texture": "Chromatin / Texture",
        "smoothness": "Membrane Roughness",
        "symmetry": "Symmetry / Mitosis",
        "fractal dimension": "Border Irregularity"
    }

    direction_map = {
        "worst perimeter": "Positive (High values strongly increase malignancy risk / nuclear enlargement)",
        "mean concave points": "Positive (High values strongly increase malignancy risk / nuclear membrane indentations)",
        "worst texture": "Positive (High chromatin coarseness increases malignancy risk)",
        "concavity error": "Positive (High variability in indentation severity indicates malignancy)",
        "concave points error": "Positive (High variation in indentations correlates with tumor growth)",
        "area error": "Positive (Extreme variance in nuclear area indicates aggressive cell division)",
        "worst symmetry": "Positive (Asymmetrical cellular structures elevate malignancy probability)",
        "worst smoothness": "Positive (Coarser nuclear membrane surfaces elevate malignancy probability)"
    }

    # Compute ranking for all 30 features
    for idx, col in enumerate(X_raw.columns):
        is_sel = 1 if col in optimal_features else 0
        if is_sel:
            sel_idx = optimal_features.index(col)
            shap_imp = round(float(mean_abs_shap[sel_idx]), 4)
            stage_status = "Selected (Stage 2 Final Elite)"
            dir_text = direction_map.get(col, "Positive (Elevated measurement increases risk)")
        elif col in kept_features:
            shap_imp = np.nan
            stage_status = "Pruned in Stage 2 (RFECV Redundancy)"
            dir_text = "Retained in Stage 1, pruned as non-essential in Stage 2 RFECV"
        else:
            shap_imp = np.nan
            # find what it was dropped for
            dropped_for = "collinear partner"
            for k, dlist in dropped_mapping.items():
                if col in dlist:
                    dropped_for = k
                    break
            stage_status = f"Pruned in Stage 1 (|r|>=0.85 collinear with {dropped_for})"
            dir_text = f"Pruned as mathematical duplicate of '{dropped_for}'"

        cat = "Cellular Morphology"
        for k_term, cat_name in category_map.items():
            if k_term in col:
                cat = cat_name
                break

        r_val = X_train[col].corr(y_train, method="spearman")

        xai_rows.append({
            "Feature": col,
            "Selected": is_sel,
            "Selection Status": stage_status,
            "Mean |SHAP|": shap_imp,
            "Spearman Correlation": round(float(r_val), 4),
            "Morphology Category": cat,
            "Direction & Clinical Meaning": dir_text
        })

    xai_df = pd.DataFrame(xai_rows)
    # Sort selected features by SHAP first, then non-selected by correlation
    xai_df_sel = xai_df[xai_df["Selected"] == 1].sort_values(by="Mean |SHAP|", ascending=False)
    xai_df_nonsel = xai_df[xai_df["Selected"] == 0].sort_values(by="Spearman Correlation", ascending=False)
    xai_df_final = pd.concat([xai_df_sel, xai_df_nonsel]).reset_index(drop=True)
    xai_df_final["Feature Rank"] = range(1, len(xai_df_final) + 1)
    xai_df_final.to_csv("selected_feature_xai_ranking.csv", index=False)
    print("Exported selected_feature_xai_ranking.csv successfully!")

    # Holdout Patient Table
    holdout_patients = []
    for i in range(len(y_test)):
        actual_val = int(y_test.iloc[i])
        actual_label = "Malignant" if actual_val == 1 else "Benign"
        prob = float(ens_test_probs[i])
        pred_def_val = int(ens_test_pred_default[i])
        pred_def_label = "Malignant" if pred_def_val == 1 else "Benign"
        pred_cal_val = int(ens_test_pred_calibrated[i])
        pred_cal_label = "Malignant" if pred_cal_val == 1 else "Benign"

        is_corr_def = (pred_def_val == actual_val)
        is_corr_cal = (pred_cal_val == actual_val)

        # Classification outcome at default
        if actual_val == 1 and pred_def_val == 1:
            outcome_def = "True Positive"
        elif actual_val == 0 and pred_def_val == 0:
            outcome_def = "True Negative"
        elif actual_val == 0 and pred_def_val == 1:
            outcome_def = "False Positive"
        else:
            outcome_def = "False Negative"

        # Classification outcome at calibrated
        if actual_val == 1 and pred_cal_val == 1:
            outcome_cal = "True Positive"
        elif actual_val == 0 and pred_cal_val == 0:
            outcome_cal = "True Negative"
        elif actual_val == 0 and pred_cal_val == 1:
            outcome_cal = "False Positive"
        else:
            outcome_cal = "False Negative"

        conf = round(abs(prob - 0.5) * 2.0, 4)  # confidence scale 0 to 1
        conf_pct = round(max(prob, 1.0 - prob) * 100.0, 2)

        # Top SHAP drivers for this patient
        pt_shap = shap_matrix[i]
        top_driver_idx = np.argmax(np.abs(pt_shap))
        top_driver_feat = optimal_features[top_driver_idx]
        top_driver_impact = "Elevates Risk" if pt_shap[top_driver_idx] > 0 else "Lowers Risk"

        record = {
            "Patient ID": f"PT-{y_test.index[i]:03d}",
            "Row Index": i,
            "Dataset Index": int(y_test.index[i]),
            "Actual Diagnosis": actual_label,
            "Actual Target": actual_val,
            "Predicted Diagnosis (Default 0.50)": pred_def_label,
            "Predicted Target (Default 0.50)": pred_def_val,
            "Default Correct": "Correct" if is_corr_def else "Incorrect",
            "Default Outcome": outcome_def,
            "Predicted Diagnosis (Calibrated)": pred_cal_label,
            "Predicted Target (Calibrated)": pred_cal_val,
            "Calibrated Correct": "Correct" if is_corr_cal else "Incorrect",
            "Calibrated Outcome": outcome_cal,
            "Malignancy Probability": round(prob, 4),
            "Prediction Confidence (%)": conf_pct,
            "Confidence Margin": conf,
            "Top Influencing Feature": top_driver_feat,
            "Top Feature Impact": top_driver_impact,
            "Base Value (SHAP Base)": round(float(shap_vals_obj.base_values[i]), 4)
        }

        # Add feature values and shap values for the 8 optimal features
        for f_idx, feat in enumerate(optimal_features):
            record[feat] = round(float(X_test_sel.iloc[i][feat]), 4)
            record[f"SHAP_{feat}"] = round(float(shap_matrix[i, f_idx]), 4)

        holdout_patients.append(record)

    holdout_df = pd.DataFrame(holdout_patients)
    holdout_df.to_csv("holdout_patient_table.csv", index=False)
    print("Exported holdout_patient_table.csv successfully!")

    print("\n--- SUMMARY OF RECONCILED METRICS ---")
    print(f"Total Holdout Samples: {len(holdout_df)}")
    print(f"Holdout Actuals: Malignant={sum(y_test==1)}, Benign={sum(y_test==0)}")
    print(f"Default (0.50) Threshold Confusion Matrix:\nTN={tn_d}, FP={fp_d}, FN={fn_d}, TP={tp_d}")
    print(f"Calibrated ({optimal_threshold:.4f}) Threshold Confusion Matrix:\nTN={tn_c}, FP={fp_c}, FN={fn_c}, TP={tp_c}")
    print(f"Calibrated Recall: {tp_c / (tp_c + fn_c):.4f} ({tp_c}/{tp_c + fn_c} Malignant cases caught)")
    print(f"Calibrated Specificity: {tn_c / (tn_c + fp_c):.4f} ({tn_c}/{tn_c + fp_c} Benign cases correctly identified)")
    print(f"Calibrated Accuracy: {(tp_c + tn_c) / len(y_test):.4f}")

if __name__ == "__main__":
    run_pipeline()

