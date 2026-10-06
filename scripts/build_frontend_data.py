"""
Packages the 4 verified CSV files into a structured JavaScript data module:
frontend/data.js
This allows the frontend to run both in file:/// mode (no CORS issues)
and via HTTP web server.
"""

import json
import pandas as pd

def build_data():
    holdout_df = pd.read_csv("holdout_patient_table.csv")
    model_df = pd.read_csv("model_benchmark.csv")
    dim_df = pd.read_csv("dimensionality_comparison.csv")
    xai_df = pd.read_csv("selected_feature_xai_ranking.csv")

    data_bundle = {
        "holdout": holdout_df.to_dict(orient="records"),
        "models": model_df.to_dict(orient="records"),
        "dimensionality": dim_df.to_dict(orient="records"),
        "features": xai_df.to_dict(orient="records"),
        "summary": {
            "total_patients": 569,
            "train_samples": 455,
            "holdout_samples": len(holdout_df),
            "holdout_malignant": int((holdout_df["Actual Diagnosis"] == "Malignant").sum()),
            "holdout_benign": int((holdout_df["Actual Diagnosis"] == "Benign").sum()),
            "original_features": 30,
            "selected_features": int(xai_df["Selected"].sum()),
            "champion_model": "Soft-Voting Ensemble",
            "best_cv_auc": 0.9937,
            "best_test_auc": 0.9944,
            "default_threshold": 0.5000,
            "calibrated_threshold": 0.3786,
            "default_metrics": {
                "tp": 39, "tn": 71, "fp": 1, "fn": 3,
                "recall": 0.9286, "specificity": 0.9861, "accuracy": 0.9649, "f1": 0.9512, "f2": 0.9375
            },
            "calibrated_metrics": {
                "tp": 41, "tn": 70, "fp": 2, "fn": 1,
                "recall": 0.9762, "specificity": 0.9722, "accuracy": 0.9737, "f1": 0.9647, "f2": 0.9716
            }
        }
    }

    content = f"// Automatically generated verified clinical data from hackathon pipeline\nconst CLINICAL_DATA = {json.dumps(data_bundle, indent=2)};\n"

    with open("frontend/data.js", "w", encoding="utf-8") as f:
        f.write(content)
    print("Created frontend/data.js successfully!")

if __name__ == "__main__":
    build_data()

