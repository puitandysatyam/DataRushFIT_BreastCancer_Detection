"""
Generates the Power BI model.bim (TMSL/Tabular Model Definition) and report.json
for the Explainable Breast Cancer Classification project.
"""

import json
import os
import pandas as pd

def build_pbi_model():
    base_dir = os.path.abspath(".")
    csv_dir = base_dir.replace("\\", "/")
    
    # Read sample CSVs to extract column names and types
    holdout_df = pd.read_csv("holdout_patient_table.csv")
    benchmark_df = pd.read_csv("model_benchmark.csv")
    dim_df = pd.read_csv("dimensionality_comparison.csv")
    xai_df = pd.read_csv("selected_feature_xai_ranking.csv")

    def infer_type(dtype):
        if "int" in str(dtype):
            return "int64"
        elif "float" in str(dtype):
            return "double"
        elif "bool" in str(dtype):
            return "boolean"
        else:
            return "string"

    def make_columns(df):
        cols = []
        for col_name, dtype in df.dtypes.items():
            cols.append({
                "name": col_name,
                "dataType": infer_type(dtype),
                "sourceColumn": col_name,
                "summarizeBy": "none" if infer_type(dtype) == "string" else "sum"
            })
        return cols

    # Define DAX Measures
    measures = [
        {
            "name": "Total Patients",
            "expression": "COUNTROWS('Holdout_Patients')",
            "formatString": "#,0"
        },
        {
            "name": "Benign Count",
            "expression": "CALCULATE(COUNTROWS('Holdout_Patients'), 'Holdout_Patients'[Actual Diagnosis] = \"Benign\")",
            "formatString": "#,0"
        },
        {
            "name": "Malignant Count",
            "expression": "CALCULATE(COUNTROWS('Holdout_Patients'), 'Holdout_Patients'[Actual Diagnosis] = \"Malignant\")",
            "formatString": "#,0"
        },
        {
            "name": "Predicted Benign Count",
            "expression": "CALCULATE(COUNTROWS('Holdout_Patients'), 'Holdout_Patients'[Predicted Diagnosis (Calibrated)] = \"Benign\")",
            "formatString": "#,0"
        },
        {
            "name": "Predicted Malignant Count",
            "expression": "CALCULATE(COUNTROWS('Holdout_Patients'), 'Holdout_Patients'[Predicted Diagnosis (Calibrated)] = \"Malignant\")",
            "formatString": "#,0"
        },
        {
            "name": "Correct Predictions",
            "expression": "CALCULATE(COUNTROWS('Holdout_Patients'), 'Holdout_Patients'[Calibrated Correct] = \"Correct\")",
            "formatString": "#,0"
        },
        {
            "name": "Incorrect Predictions",
            "expression": "CALCULATE(COUNTROWS('Holdout_Patients'), 'Holdout_Patients'[Calibrated Correct] = \"Incorrect\")",
            "formatString": "#,0"
        },
        {
            "name": "True Positives (TP)",
            "expression": "CALCULATE(COUNTROWS('Holdout_Patients'), 'Holdout_Patients'[Calibrated Outcome] = \"True Positive\")",
            "formatString": "#,0"
        },
        {
            "name": "True Negatives (TN)",
            "expression": "CALCULATE(COUNTROWS('Holdout_Patients'), 'Holdout_Patients'[Calibrated Outcome] = \"True Negative\")",
            "formatString": "#,0"
        },
        {
            "name": "False Positives (FP)",
            "expression": "CALCULATE(COUNTROWS('Holdout_Patients'), 'Holdout_Patients'[Calibrated Outcome] = \"False Positive\")",
            "formatString": "#,0"
        },
        {
            "name": "False Negatives (FN)",
            "expression": "CALCULATE(COUNTROWS('Holdout_Patients'), 'Holdout_Patients'[Calibrated Outcome] = \"False Negative\")",
            "formatString": "#,0"
        },
        {
            "name": "Accuracy",
            "expression": "DIVIDE([True Positives (TP)] + [True Negatives (TN)], [Total Patients], 0)",
            "formatString": "0.0%"
        },
        {
            "name": "Precision",
            "expression": "DIVIDE([True Positives (TP)], [True Positives (TP)] + [False Positives (FP)], 0)",
            "formatString": "0.0%"
        },
        {
            "name": "Recall (Sensitivity)",
            "expression": "DIVIDE([True Positives (TP)], [True Positives (TP)] + [False Negatives (FN)], 0)",
            "formatString": "0.0%"
        },
        {
            "name": "Malignant Recall",
            "expression": "[Recall (Sensitivity)]",
            "formatString": "0.0%"
        },
        {
            "name": "Specificity",
            "expression": "DIVIDE([True Negatives (TN)], [True Negatives (TN)] + [False Positives (FP)], 0)",
            "formatString": "0.0%"
        },
        {
            "name": "F1 Score",
            "expression": "DIVIDE(2 * [Precision] * [Recall (Sensitivity)], [Precision] + [Recall (Sensitivity)], 0)",
            "formatString": "0.000"
        },
        {
            "name": "F2 Score",
            "expression": "DIVIDE(5 * [Precision] * [Recall (Sensitivity)], 4 * [Precision] + [Recall (Sensitivity)], 0)",
            "formatString": "0.000"
        },
        {
            "name": "Mean Prediction Confidence",
            "expression": "AVERAGE('Holdout_Patients'[Prediction Confidence (%)])",
            "formatString": "0.0\"%\""
        },
        {
            "name": "Mean Malignancy Probability",
            "expression": "AVERAGE('Holdout_Patients'[Malignancy Probability])",
            "formatString": "0.000"
        },
        {
            "name": "Number of Selected Features",
            "expression": "CALCULATE(COUNTROWS('Feature_Explainability'), 'Feature_Explainability'[Selected] = 1)",
            "formatString": "#,0"
        },
        {
            "name": "Total Evaluated Features",
            "expression": "COUNTROWS('Feature_Explainability')",
            "formatString": "#,0"
        },
        {
            "name": "Best Model Name",
            "expression": "CALCULATE(FIRSTNONBLANK('Model_Benchmark'[Model], 1), 'Model_Benchmark'[Model Rank] = 1)",
            "formatString": ""
        },
        {
            "name": "Best CV ROC-AUC",
            "expression": "CALCULATE(MAX('Model_Benchmark'[CV ROC-AUC]), 'Model_Benchmark'[Model Rank] = 1)",
            "formatString": "0.000"
        },
        {
            "name": "Calibrated Medical Threshold",
            "expression": "0.3786",
            "formatString": "0.0000"
        },
        {
            "name": "Default Decision Threshold",
            "expression": "0.5000",
            "formatString": "0.0000"
        }
    ]

    model_bim = {
        "name": "Explainable_BreastCancer_BI",
        "compatibilityLevel": 1550,
        "model": {
            "culture": "en-US",
            "dataAccessOptions": {
                "legacyRedirects": True,
                "returnErrorValuesAsNull": True
            },
            "defaultPowerBIDataSourceVersion": "powerBI_V3",
            "tables": [
                {
                    "name": "Holdout_Patients",
                    "columns": make_columns(holdout_df),
                    "partitions": [
                        {
                            "name": "Holdout_Patients-Partition",
                            "mode": "import",
                            "source": {
                                "type": "m",
                                "expression": [
                                    f'let Source = Csv.Document(File.Contents("{csv_dir}/holdout_patient_table.csv"), [Delimiter=",", Columns={len(holdout_df.columns)}, Encoding=65001, QuoteStyle=QuoteStyle.None]),',
                                    '#"Promoted Headers" = Table.PromoteHeaders(Source, [PromoteAllScalars=true])',
                                    'in #"Promoted Headers"'
                                ]
                            }
                        }
                    ]
                },
                {
                    "name": "Model_Benchmark",
                    "columns": make_columns(benchmark_df),
                    "partitions": [
                        {
                            "name": "Model_Benchmark-Partition",
                            "mode": "import",
                            "source": {
                                "type": "m",
                                "expression": [
                                    f'let Source = Csv.Document(File.Contents("{csv_dir}/model_benchmark.csv"), [Delimiter=",", Columns={len(benchmark_df.columns)}, Encoding=65001, QuoteStyle=QuoteStyle.None]),',
                                    '#"Promoted Headers" = Table.PromoteHeaders(Source, [PromoteAllScalars=true])',
                                    'in #"Promoted Headers"'
                                ]
                            }
                        }
                    ]
                },
                {
                    "name": "Dimensionality_Benchmark",
                    "columns": make_columns(dim_df),
                    "partitions": [
                        {
                            "name": "Dimensionality_Benchmark-Partition",
                            "mode": "import",
                            "source": {
                                "type": "m",
                                "expression": [
                                    f'let Source = Csv.Document(File.Contents("{csv_dir}/dimensionality_comparison.csv"), [Delimiter=",", Columns={len(dim_df.columns)}, Encoding=65001, QuoteStyle=QuoteStyle.None]),',
                                    '#"Promoted Headers" = Table.PromoteHeaders(Source, [PromoteAllScalars=true])',
                                    'in #"Promoted Headers"'
                                ]
                            }
                        }
                    ]
                },
                {
                    "name": "Feature_Explainability",
                    "columns": make_columns(xai_df),
                    "partitions": [
                        {
                            "name": "Feature_Explainability-Partition",
                            "mode": "import",
                            "source": {
                                "type": "m",
                                "expression": [
                                    f'let Source = Csv.Document(File.Contents("{csv_dir}/selected_feature_xai_ranking.csv"), [Delimiter=",", Columns={len(xai_df.columns)}, Encoding=65001, QuoteStyle=QuoteStyle.None]),',
                                    '#"Promoted Headers" = Table.PromoteHeaders(Source, [PromoteAllScalars=true])',
                                    'in #"Promoted Headers"'
                                ]
                            }
                        }
                    ]
                },
                {
                    "name": "_Clinical_Measures",
                    "columns": [
                        {
                            "name": "Measure_Placeholder",
                            "dataType": "string",
                            "sourceColumn": "Measure_Placeholder",
                            "isHidden": True
                        }
                    ],
                    "measures": measures,
                    "partitions": [
                        {
                            "name": "_Clinical_Measures-Partition",
                            "mode": "import",
                            "source": {
                                "type": "m",
                                "expression": [
                                    'let Source = #table(type table [Measure_Placeholder = text], {{"Measures"}})',
                                    'in Source'
                                ]
                            }
                        }
                    ]
                }
            ],
            "relationships": []
        }
    }

    bim_path = "powerbi/Explainable_BreastCancer_BI.Dataset/model.bim"
    with open(bim_path, "w", encoding="utf-8") as f:
        json.dump(model_bim, f, indent=2)
    print(f"Generated Power BI Tabular Model: {bim_path}")

    # Build report.json (7 Pages with exact metadata and visual layout)
    pages = [
        {
            "name": "ReportSection_Signal",
            "displayName": "1. CLINICAL SIGNAL",
            "filters": "[]",
            "ordinal": 0,
            "width": 1280,
            "height": 720,
            "visualContainers": [
                {
                    "x": 20, "y": 15, "z": 1, "width": 1240, "height": 65,
                    "title": "CLINICAL SIGNAL — Explainable Breast Cancer Classification",
                    "type": "textbox",
                    "config": json.dumps({"text": "CLINICAL SIGNAL | Explainable Breast Cancer Classification (Wisconsin WDBC)\nMethodology: 569 FNA Biopsies → 80/20 Vault Holdout → Collinear Clustering Pruning → RFECV (8 Selected Features) → Soft-Voting Ensemble → Threshold Tuning (0.3786)"})
                },
                {
                    "x": 20, "y": 90, "z": 2, "width": 290, "height": 130,
                    "title": "Holdout Sample Size",
                    "type": "card",
                    "config": json.dumps({"measure": "Total Patients", "subtitle": "114 Patients (Locked Holdout)"})
                },
                {
                    "x": 330, "y": 90, "z": 3, "width": 290, "height": 130,
                    "title": "Dimensionality Reduction",
                    "type": "card",
                    "config": json.dumps({"measure": "Number of Selected Features", "subtitle": "8 Selected / 30 Original (73% Pruned)"})
                },
                {
                    "x": 640, "y": 90, "z": 4, "width": 290, "height": 130,
                    "title": "Champion Model ROC-AUC",
                    "type": "card",
                    "config": json.dumps({"measure": "Best CV ROC-AUC", "subtitle": "Soft-Voting Ensemble (0.9944 Holdout)"})
                },
                {
                    "x": 950, "y": 90, "z": 5, "width": 310, "height": 130,
                    "title": "Calibrated Malignant Recall",
                    "type": "card",
                    "config": json.dumps({"measure": "Malignant Recall", "subtitle": "97.6% Sensitivity (41/42 Cancers Detected)"})
                },
                {
                    "x": 20, "y": 235, "z": 6, "width": 380, "height": 240,
                    "title": "Class Distribution (Actual Holdout)",
                    "type": "pieChart",
                    "config": json.dumps({"category": "Actual Diagnosis", "values": ["Total Patients"]})
                },
                {
                    "x": 420, "y": 235, "z": 7, "width": 460, "height": 240,
                    "title": "Model Performance Across Algorithms (ROC-AUC vs Recall)",
                    "type": "barChart",
                    "config": json.dumps({"category": "Model", "values": ["CV ROC-AUC", "CV Recall (Sensitivity)"]})
                },
                {
                    "x": 900, "y": 235, "z": 8, "width": 360, "height": 240,
                    "title": "Top Morphological Drivers (Mean |SHAP|)",
                    "type": "barChart",
                    "config": json.dumps({"category": "Feature", "values": ["Mean |SHAP|"]})
                },
                {
                    "x": 20, "y": 490, "z": 9, "width": 620, "height": 210,
                    "title": "Clinical Outcomes: Actual vs Calibrated Predictions",
                    "type": "table",
                    "config": json.dumps({"columns": ["Calibrated Outcome", "Total Patients", "Accuracy", "Precision", "Recall (Sensitivity)"]})
                },
                {
                    "x": 660, "y": 490, "z": 10, "width": 600, "height": 210,
                    "title": "Executive Summary & Clinical Takeaways",
                    "type": "textbox",
                    "config": json.dumps({"text": "Key Finding: High correlation (32 pairs with |r| >= 0.85) was eliminated by retaining biological anchors (worst perimeter and mean concave points). Dimensionality was reduced from 30 down to 8 features with ZERO loss in diagnostic power (0.9944 ROC-AUC). Decision threshold tuning to 0.3786 caught 41/42 cancers, cutting false negatives by 67% (from 3 down to 1)."})
                }
            ]
        },
        {
            "name": "ReportSection_FeatureLab",
            "displayName": "2. FEATURE LAB",
            "filters": "[]",
            "ordinal": 1,
            "width": 1280,
            "height": 720,
            "visualContainers": [
                {
                    "x": 20, "y": 15, "z": 1, "width": 1240, "height": 55,
                    "title": "FEATURE LAB — Nuclear Morphology Selection & Ranking",
                    "type": "textbox",
                    "config": json.dumps({"text": "FEATURE LAB: Cellular Morphology Analysis | 8 Elite Native Cytology Traits Selected from 30 Features"})
                },
                {
                    "x": 20, "y": 80, "z": 2, "width": 600, "height": 330,
                    "title": "Ranked Feature Importance (Mean |SHAP| on XGBoost/Trees)",
                    "type": "barChart",
                    "config": json.dumps({"table": "Feature_Explainability", "x": "Feature", "y": "Mean |SHAP|"})
                },
                {
                    "x": 640, "y": 80, "z": 3, "width": 620, "height": 330,
                    "title": "Morphological Correlation with Malignancy (Spearman Rank)",
                    "type": "barChart",
                    "config": json.dumps({"table": "Feature_Explainability", "x": "Feature", "y": "Spearman Correlation"})
                },
                {
                    "x": 20, "y": 425, "z": 4, "width": 1240, "height": 275,
                    "title": "Clinical Feature Catalog & Pruning Breakdown",
                    "type": "table",
                    "config": json.dumps({"table": "Feature_Explainability", "columns": ["Feature Rank", "Feature", "Selected", "Selection Status", "Morphology Category", "Mean |SHAP|", "Spearman Correlation", "Direction & Clinical Meaning"]})
                }
            ]
        },
        {
            "name": "ReportSection_ModelBench",
            "displayName": "3. MODEL BENCH",
            "filters": "[]",
            "ordinal": 2,
            "width": 1280,
            "height": 720,
            "visualContainers": [
                {
                    "x": 20, "y": 15, "z": 1, "width": 1240, "height": 55,
                    "title": "MODEL BENCH — Clinical Model Evaluation & Threshold Tuning",
                    "type": "textbox",
                    "config": json.dumps({"text": "MODEL BENCH: 5 Algorithm Families + Soft-Voting Ensemble | 5-Fold Stratified Cross-Validation & Locked Holdout Benchmark"})
                },
                {
                    "x": 20, "y": 80, "z": 2, "width": 760, "height": 320,
                    "title": "Model Comparison Matrix Across Clinical Metrics",
                    "type": "barChart",
                    "config": json.dumps({"table": "Model_Benchmark", "x": "Model", "metrics": ["CV ROC-AUC", "CV Recall (Sensitivity)", "CV F2-Score", "Holdout Test Accuracy"]})
                },
                {
                    "x": 800, "y": 80, "z": 3, "width": 460, "height": 320,
                    "title": "Decision Threshold Calibration: Default (0.50) vs Calibrated (0.3786)",
                    "type": "table",
                    "config": json.dumps({"rows": [
                        {"Threshold": "Default (0.5000)", "Malignant Recall": "92.86% (39/42)", "Specificity": "98.61% (71/72)", "False Negatives": 3, "False Positives": 1, "Accuracy": "96.49%"},
                        {"Threshold": "Calibrated (0.3786)", "Malignant Recall": "97.62% (41/42)", "Specificity": "97.22% (70/72)", "False Negatives": 1, "False Positives": 2, "Accuracy": "97.37%"}
                    ]})
                },
                {
                    "x": 20, "y": 415, "z": 4, "width": 1240, "height": 285,
                    "title": "Comprehensive Benchmark Results Table",
                    "type": "table",
                    "config": json.dumps({"table": "Model_Benchmark", "columns": ["Model Rank", "Model", "CV ROC-AUC", "CV Recall (Sensitivity)", "CV Precision", "CV F1-Score", "CV F2-Score", "Generalization Gap", "Holdout Test Accuracy", "Holdout TP", "Holdout TN", "Holdout FP", "Holdout FN"]})
                }
            ]
        },
        {
            "name": "ReportSection_Explainability",
            "displayName": "4. WHY THIS PREDICTION?",
            "filters": "[]",
            "ordinal": 3,
            "width": 1280,
            "height": 720,
            "visualContainers": [
                {
                    "x": 20, "y": 15, "z": 1, "width": 1240, "height": 55,
                    "title": "EXPLAINABILITY — SHAP Local & Global Interpretability",
                    "type": "textbox",
                    "config": json.dumps({"text": "WHY THIS PREDICTION? Global TreeSHAP Summary & Patient-Level Local Case Studies"})
                },
                {
                    "x": 20, "y": 80, "z": 2, "width": 300, "height": 620,
                    "title": "Select Patient Case",
                    "type": "slicer",
                    "config": json.dumps({"table": "Holdout_Patients", "field": "Patient ID"})
                },
                {
                    "x": 340, "y": 80, "z": 3, "width": 460, "height": 300,
                    "title": "Patient Diagnosis Receipt",
                    "type": "card",
                    "config": json.dumps({"table": "Holdout_Patients", "fields": ["Patient ID", "Actual Diagnosis", "Predicted Diagnosis (Calibrated)", "Malignancy Probability", "Top Influencing Feature", "Top Feature Impact"]})
                },
                {
                    "x": 820, "y": 80, "z": 4, "width": 440, "height": 300,
                    "title": "SHAP Feature Impact Attribution (Waterfall Drivers)",
                    "type": "barChart",
                    "config": json.dumps({"fields": ["SHAP_worst perimeter", "SHAP_worst texture", "SHAP_mean concave points", "SHAP_area error", "SHAP_worst smoothness", "SHAP_worst symmetry", "SHAP_concavity error", "SHAP_concave points error"]})
                },
                {
                    "x": 340, "y": 395, "z": 5, "width": 920, "height": 305,
                    "title": "Three Benchmark Clinical Archetypes",
                    "type": "table",
                    "config": json.dumps({"archetypes": [
                        {"Case": "Patient A (High Confidence Malignant)", "ID": "PT-033", "Actual": "Malignant", "Prob": "99.8%", "Key Drivers": "worst perimeter (128.0) and mean concave points (0.09) push risk far into positive space."},
                        {"Case": "Patient B (High Confidence Benign)", "ID": "PT-049", "Actual": "Benign", "Prob": "0.1%", "Key Drivers": "Small nuclear perimeter (65.5) and minimal concave indentations (0.01) anchor risk near 0%."},
                        {"Case": "Patient C (Borderline Case)", "ID": "PT-086", "Actual": "Benign", "Prob": "44.2%", "Key Drivers": "Moderate texture (24.1) elevates risk, but small size and low concave points keep it below calibrated threshold."}
                    ]})
                }
            ]
        },
        {
            "name": "ReportSection_Patients",
            "displayName": "5. PATIENT EXPLORER",
            "filters": "[]",
            "ordinal": 4,
            "width": 1280,
            "height": 720,
            "visualContainers": [
                {
                    "x": 20, "y": 15, "z": 1, "width": 1240, "height": 55,
                    "title": "PATIENT EXPLORER — Clinical Cohort Browser",
                    "type": "textbox",
                    "config": json.dumps({"text": "PATIENT EXPLORER: Interactive Investigation of All 114 Holdout Patients with Nuclear Morphology and Probabilities"})
                },
                {
                    "x": 20, "y": 80, "z": 2, "width": 260, "height": 100,
                    "title": "Filter: Actual Diagnosis",
                    "type": "slicer",
                    "config": json.dumps({"table": "Holdout_Patients", "field": "Actual Diagnosis"})
                },
                {
                    "x": 300, "y": 80, "z": 3, "width": 260, "height": 100,
                    "title": "Filter: Prediction Outcome",
                    "type": "slicer",
                    "config": json.dumps({"table": "Holdout_Patients", "field": "Calibrated Outcome"})
                },
                {
                    "x": 580, "y": 80, "z": 4, "width": 260, "height": 100,
                    "title": "Filter: Correctness",
                    "type": "slicer",
                    "config": json.dumps({"table": "Holdout_Patients", "field": "Calibrated Correct"})
                },
                {
                    "x": 860, "y": 80, "z": 5, "width": 400, "height": 100,
                    "title": "Cohort Overview",
                    "type": "card",
                    "config": json.dumps({"measure": "Total Patients", "subtitle": "Interactive Filtered Cohort"})
                },
                {
                    "x": 20, "y": 195, "z": 6, "width": 1240, "height": 505,
                    "title": "Patient Cohort Investigation Table",
                    "type": "table",
                    "config": json.dumps({"table": "Holdout_Patients", "columns": [
                        "Patient ID", "Actual Diagnosis", "Predicted Diagnosis (Calibrated)",
                        "Malignancy Probability", "Prediction Confidence (%)", "Calibrated Outcome",
                        "Top Influencing Feature", "worst perimeter", "mean concave points", "worst texture", "area error"
                    ]})
                }
            ]
        },
        {
            "name": "ReportSection_ErrorMap",
            "displayName": "6. ERROR MAP",
            "filters": "[]",
            "ordinal": 5,
            "width": 1280,
            "height": 720,
            "visualContainers": [
                {
                    "x": 20, "y": 15, "z": 1, "width": 1240, "height": 55,
                    "title": "ERROR MAP — Model Reliability & Misclassification Analysis",
                    "type": "textbox",
                    "config": json.dumps({"text": "ERROR MAP: Evidence-Based Audit of Misclassifications | Transparent Clinical Safety Analysis"})
                },
                {
                    "x": 20, "y": 80, "z": 2, "width": 290, "height": 120,
                    "title": "False Negatives (FN)",
                    "type": "card",
                    "config": json.dumps({"measure": "False Negatives (FN)", "subtitle": "Exactly 1 Missed Cancer (Reduced from 3)"})
                },
                {
                    "x": 330, "y": 80, "z": 3, "width": 290, "height": 120,
                    "title": "False Positives (FP)",
                    "type": "card",
                    "config": json.dumps({"measure": "False Positives (FP)", "subtitle": "2 Unnecessary Biopsies (Safe Trade-off)"})
                },
                {
                    "x": 640, "y": 80, "z": 4, "width": 290, "height": 120,
                    "title": "Malignant Sensitivity",
                    "type": "card",
                    "config": json.dumps({"measure": "Recall (Sensitivity)", "subtitle": "97.6% (41 of 42 Caught)"})
                },
                {
                    "x": 950, "y": 80, "z": 5, "width": 310, "height": 120,
                    "title": "Holdout Error Rate",
                    "type": "card",
                    "config": json.dumps({"value": "2.63%", "subtitle": "3 Total Errors out of 114 Patients"})
                },
                {
                    "x": 20, "y": 215, "z": 6, "width": 520, "height": 240,
                    "title": "Confusion Matrix Heatmap (Calibrated)",
                    "type": "table",
                    "config": json.dumps({
                        "matrix": [
                            {"True Class": "Benign (Actual)", "Pred Benign": 70, "Pred Malignant": 2, "Metric": "97.2% Specificity"},
                            {"True Class": "Malignant (Actual)", "Pred Benign": 1, "Pred Malignant": 41, "Metric": "97.6% Recall"}
                        ]
                    })
                },
                {
                    "x": 560, "y": 215, "z": 7, "width": 700, "height": 240,
                    "title": "Audit of Discrepancy: Myth of 'Zero False Negatives'",
                    "type": "textbox",
                    "config": json.dumps({"text": "CRITICAL CLINICAL AUDIT: Why we do NOT claim Zero False Negatives\nIn early informal notebook drafts, casual comments stated 'zero false negatives'. However, rigorous empirical evaluation of the holdout set reveals exactly 1 False Negative (Patient PT-073) at the optimal calibrated threshold (and 3 at default 0.50 threshold).\nPatient PT-073 presented with an atypical, small-cell malignant tumor where worst perimeter was below typical malignant cutoffs. Claiming zero false negatives in a medical setting is dangerous and scientifically dishonest. Our model catches 41/42 cancers (97.62% sensitivity) with absolute transparency."})
                },
                {
                    "x": 20, "y": 470, "z": 8, "width": 1240, "height": 230,
                    "title": "The 3 Clinical Misclassifications Deep Dive",
                    "type": "table",
                    "config": json.dumps({"table": "Holdout_Patients", "filter": "Calibrated Correct == Incorrect", "columns": [
                        "Patient ID", "Actual Diagnosis", "Predicted Diagnosis (Calibrated)",
                        "Malignancy Probability", "Calibrated Outcome", "Top Influencing Feature",
                        "worst perimeter", "mean concave points", "worst texture", "area error"
                    ]})
                }
            ]
        },
        {
            "name": "ReportSection_Method",
            "displayName": "7. METHOD & DATA",
            "filters": "[]",
            "ordinal": 6,
            "width": 1280,
            "height": 720,
            "visualContainers": [
                {
                    "x": 20, "y": 15, "z": 1, "width": 1240, "height": 55,
                    "title": "METHOD & DATA — Architecture, Governance & Lineage",
                    "type": "textbox",
                    "config": json.dumps({"text": "METHOD & DATA: End-to-End Scientific Architecture, Governance, and Reproducibility Specifications"})
                },
                {
                    "x": 20, "y": 80, "z": 2, "width": 390, "height": 300,
                    "title": "1. Data Origin & Ingestion",
                    "type": "textbox",
                    "config": json.dumps({"text": "• Wisconsin Diagnostic Breast Cancer (WDBC)\n• 569 fine-needle aspirates (FNA) of breast masses\n• Target: 212 Malignant (37.3%), 357 Benign (62.7%)\n• 30 features derived from 10 cellular nuclear characteristics\n• Strict 80/20 train/test stratified holdout sealed in vault."})
                },
                {
                    "x": 430, "y": 80, "z": 3, "width": 420, "height": 300,
                    "title": "2. Two-Stage Feature Selection",
                    "type": "textbox",
                    "config": json.dumps({"text": "• Severe multicollinearity: 32 pairs with |r| >= 0.85, VIF > 1000.\n• Stage 1: Agglomerative hierarchical correlation clustering (cutoff dist = 0.15) pruned redundant twins down to 16 features.\n• Stage 2: RFECV with Random Forest (5-Fold Stratified CV) isolated 8 elite non-redundant traits with peak discriminative power."})
                },
                {
                    "x": 870, "y": 80, "z": 4, "width": 390, "height": 300,
                    "title": "3. Modeling & Threshold Calibration",
                    "type": "textbox",
                    "config": json.dumps({"text": "• 5 model families benchmarked across 5-fold CV.\n• Soft-Voting Ensemble (Logistic Regression + SVM + XGBoost) achieved top AUC (0.9944).\n• Threshold tuning: Shifted decision boundary from default 0.50 down to 0.3786 to maximize F2-score (prioritizing recall 2x over precision)."})
                },
                {
                    "x": 20, "y": 395, "z": 5, "width": 600, "height": 305,
                    "title": "Dimensionality Strategy Comparison",
                    "type": "table",
                    "config": json.dumps({"table": "Dimensionality_Benchmark", "columns": [
                        "Feature Representation", "Dimensionality", "Holdout Test ROC-AUC",
                        "Holdout Recall (Sensitivity)", "Holdout Precision", "Holdout F2-Score", "Clinical Interpretability"
                    ]})
                },
                {
                    "x": 640, "y": 395, "z": 6, "width": 620, "height": 305,
                    "title": "Clinical Governance & Operational Guardrails",
                    "type": "textbox",
                    "config": json.dumps({"text": "CLINICAL RECOMMENDATION FOR ONCOLOGY TEAMS:\n1. Triage Workflow: The model should serve as a Second-Reader Decision Support Tool for pathologists, flagging high-risk aspirates for priority immunohistochemistry.\n2. Borderline Protocol: Any case with probability between 0.30 and 0.60 should trigger secondary re-aspiration or core needle biopsy.\n3. Interpretability Requirement: Every algorithmic score must be accompanied by the TreeSHAP waterfall receipt highlighting nuclear perimeter and concavity values."})
                }
            ]
        }
    ]

    report_json = {
        "version": "1.0",
        "theme": "ClinicalEditorialTheme",
        "sections": pages
    }

    report_path = "powerbi/Explainable_BreastCancer_BI.Report/report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_json, f, indent=2)
    print(f"Generated Power BI Report Layout: {report_path}")

    # Also place root pointer in workspace root
    with open("Explainable_BreastCancer_BI.pbip", "w", encoding="utf-8") as f:
        json.dump({
            "version": "1.0",
            "artifacts": [{"report": {"path": "powerbi/Explainable_BreastCancer_BI.Report"}}],
            "settings": {"enableAutoRecovery": True}
        }, f, indent=2)
    print("Created Explainable_BreastCancer_BI.pbip in workspace root!")

if __name__ == "__main__":
    build_pbi_model()

