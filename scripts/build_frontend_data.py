"""
Packages the 4 verified CSV files + training split + validation loss + confusion matrix
into a structured JavaScript data module: frontend/data.js
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
        "splits": {
            "total_samples": 569,
            "train_total": 455,
            "train_pct": 80.0,
            "train_benign": 285,
            "train_malignant": 170,
            "train_benign_pct": 62.64,
            "train_malignant_pct": 37.36,
            "holdout_total": 114,
            "holdout_pct": 20.0,
            "holdout_benign": 72,
            "holdout_malignant": 42,
            "holdout_benign_pct": 63.16,
            "holdout_malignant_pct": 36.84,
            "cv_folds": 5,
            "cv_train_per_fold": 364,
            "cv_val_per_fold": 91
        },
        "validation_loss": {
            "learning_curve": [
                {"train_samples": 68, "train_loss": 0.0826, "val_loss": 0.2143},
                {"train_samples": 123, "train_loss": 0.0753, "val_loss": 0.1410},
                {"train_samples": 178, "train_loss": 0.0681, "val_loss": 0.1233},
                {"train_samples": 233, "train_loss": 0.0664, "val_loss": 0.1068},
                {"train_samples": 289, "train_loss": 0.0625, "val_loss": 0.1069},
                {"train_samples": 344, "train_loss": 0.0588, "val_loss": 0.1019},
                {"train_samples": 399, "train_loss": 0.0570, "val_loss": 0.0971},
                {"train_samples": 455, "train_loss": 0.0572, "val_loss": 0.0925}
            ],
            "boosting_loss": [
                {"iteration": 1, "train_loss": 0.5948, "val_loss": 0.5983},
                {"iteration": 5, "train_loss": 0.4121, "val_loss": 0.4316},
                {"iteration": 10, "train_loss": 0.2815, "val_loss": 0.3164},
                {"iteration": 15, "train_loss": 0.2028, "val_loss": 0.2431},
                {"iteration": 20, "train_loss": 0.1518, "val_loss": 0.1950},
                {"iteration": 25, "train_loss": 0.1184, "val_loss": 0.1613},
                {"iteration": 30, "train_loss": 0.0949, "val_loss": 0.1402},
                {"iteration": 35, "train_loss": 0.0778, "val_loss": 0.1277},
                {"iteration": 40, "train_loss": 0.0651, "val_loss": 0.1185},
                {"iteration": 45, "train_loss": 0.0553, "val_loss": 0.1088},
                {"iteration": 50, "train_loss": 0.0486, "val_loss": 0.1030},
                {"iteration": 55, "train_loss": 0.0433, "val_loss": 0.0983},
                {"iteration": 60, "train_loss": 0.0394, "val_loss": 0.0961}
            ]
        },
        "confusion_matrices": {
            "default": {
                "threshold": 0.50,
                "tp": 39, "tn": 71, "fp": 1, "fn": 3,
                "recall": 92.86, "specificity": 98.61, "precision": 97.50,
                "accuracy": 96.49, "f1": 0.9512, "f2": 0.9375
            },
            "calibrated": {
                "threshold": 0.3786,
                "tp": 41, "tn": 70, "fp": 2, "fn": 1,
                "recall": 97.62, "specificity": 97.22, "precision": 95.35,
                "accuracy": 97.37, "f1": 0.9647, "f2": 0.9716
            }
        },
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
