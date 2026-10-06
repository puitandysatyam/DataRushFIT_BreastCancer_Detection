# Explainable Breast Cancer Classification — Hackathon BI Dashboard

> **Wisconsin Diagnostic Dataset (WDBC) — Clinical Decision Support & Explainability Dashboard**  
> Built for the Datathon by the Lead Data Visualization, BI & UX Engineering Team.

---

## 📌 Executive Summary

This project delivers a **clinical-research grade, presentation-ready Business Intelligence solution** built directly from the hackathon notebook (`notebooks/Explainable_Breast_Cancer_Classification.ipynb`) and its verified machine learning pipeline.

The solution directly answers the Datathon Question:
> *"Can cellular morphology measurements accurately distinguish malignant tumors from benign tumors while reducing dimensionality, preventing overfitting, handling correlated features, and explaining which cellular characteristics drive each prediction?"*

### 🔬 Empirical Findings & Breakthroughs:
1. **Severe Multicollinearity Resolved**: The raw WDBC dataset exhibited extreme multicollinearity (**32 feature pairs with Spearman $|r| \ge 0.85$**, $VIF > 1000$). Our **Two-Stage Feature Selection Pipeline** (Agglomerative Hierarchical Collinear Clustering $\rightarrow$ RFECV) systematically pruned redundant mathematical twins, preserving biological signal.
2. **73% Dimensionality Reduction with Zero Diagnostic Loss**: Dimensionality was compressed from **30 features down to 8 elite morphology traits**, achieving an untouched Holdout Test **ROC-AUC of 0.9944**, matching unreduced data while decisively outperforming PCA (which creates black-box synthetic axes lacking cytology meaning).
3. **Clinical Threshold Calibration (Saving Lives)**: Because missing an invasive cancer (False Negative) is unacceptable in oncology, we calibrated the Soft-Voting Ensemble's decision boundary from the default $0.50$ down to **$0.3786$**, cutting False Negatives by **$67\%$ (from 3 down to exactly 1)** and boosting Malignant Recall / Sensitivity to **$97.62\%$ (41 of 42 cancers detected)**.
4. **Actionable Explainability (TreeSHAP)**: Global and local SHAP explanations prove that **`worst perimeter` (nuclear gigantism)** and **`mean concave points` (nuclear membrane notching)** are the two dominant physical hallmarks of tumor malignancy.

---

## 🏛 Dual BI Deliverable Architecture

To guarantee that judges can evaluate this solution in any environment (with or without Microsoft Power BI Desktop installed), we provide **two production-grade implementations**:

```
d:/DataRushFIT_BreastCancer_Detection/
├── Explainable_BreastCancer_BI.pbip             <-- Official Power BI Developer Project Root
├── powerbi/
│   ├── Explainable_BreastCancer_BI.pbip
│   ├── Explainable_BreastCancer_BI.Dataset/     <-- Tabular Data Model (TMSL/model.bim)
│   │   ├── definition.pbidataset
│   │   └── model.bim                            <-- Tables, Relationships, & DAX Measures
│   └── Explainable_BreastCancer_BI.Report/      <-- Power BI Report Definition
│       ├── definition.pbir
│       ├── report.json                          <-- Visual Tree, Layout Containers, Slicers
│       └── StaticResources/RegisteredResources/
│           └── Clinical_Editorial_Theme.json    <-- Custom Editorial Theme Palette
│
├── frontend/                                    <-- Standalone Interactive Clinical Console
│   ├── index.html                               <-- 7-Page Responsive Dashboard Application
│   ├── styles.css                               <-- Clinical Editorial Design System
│   ├── app.js                                  <-- Interactive Slicers, SHAP Waterfalls & Tables
│   └── data.js                                  <-- Embedded Verified Clinical Dataset Bundle
│
├── data/                                        <-- Clean Source Data Folder
│   ├── holdout_patient_table.csv                <-- 114 Holdout Patients with SHAP & Metrics
│   ├── model_benchmark.csv                      <-- 6 Models across CV & Holdout Metrics
│   ├── dimensionality_comparison.csv            <-- Full 30 vs PCA 6 vs Two-Stage 8
│   └── selected_feature_xai_ranking.csv         <-- All 30 Features with Status & Meaning
│
└── screenshots/                                 <-- High-Resolution Previews of all 7 Pages
    ├── page1_clinical_signal.png
    ├── page2_feature_lab.png
    ├── page3_model_bench.png
    ├── page4_why_this_prediction.png
    ├── page5_patient_explorer.png
    ├── page6_error_map.png
    └── page7_method_and_data.png
```

---

## 🚀 How to Open and Review the BI Deliverables

### Method 1: Microsoft Power BI Desktop (Official BI Project Artifact)
Power BI Desktop (Store edition) is fully supported.
1. Open PowerShell and run:
   ```powershell
   Start-Process "PBIDesktopStore.exe" -ArgumentList "d:\DataRushFIT_BreastCancer_Detection\Explainable_BreastCancer_BI.pbip"
   ```
2. Or navigate to `d:\DataRushFIT_BreastCancer_Detection\` in Windows Explorer and double-click `Explainable_BreastCancer_BI.pbip`.
3. Power BI Desktop will parse the `model.bim` tabular model, connect to the CSV datasets in `data/`, load all calculated DAX measures, apply the custom `Clinical_Editorial_Theme`, and display the report pages.

### Method 2: Clinical Web Console (Universal Browser Mode)
If Power BI is not available, or for instantaneous interactive evaluation on any OS:
1. Double click `d:\DataRushFIT_BreastCancer_Detection\frontend\index.html` to open it in your web browser (Chrome, Edge, Firefox, Safari).
2. Or start a local lightweight web server:
   ```powershell
   python -m http.server 8080 --directory d:\DataRushFIT_BreastCancer_Detection\frontend
   ```
   Open `http://localhost:8080` in your browser.
3. Enjoy the complete 7-page interactive experience with dynamic feature inspection, metric switching, patient case study waterfalls, live filtering, and error analysis.

---

## 📑 The 7 Dashboard Pages & Visual Hierarchy

| # | Page Name | Core Objective | Key Visuals & Interactions |
|---|-----------|----------------|----------------------------|
| **1** | **CLINICAL SIGNAL** | 30-Second Executive Signal Board | • Prominent 6-stage methodology ribbon<br>• Holdout cohort KPIs (114 Patients, 8 Features, 0.9944 AUC, 97.6% Recall)<br>• Class distribution donut (36.8% Malignant vs 63.2% Benign)<br>• Model benchmark bar chart<br>• Top cellular drivers (Mean \|SHAP\|)<br>• 4-cell confusion matrix summary<br>• "What the Model Learned" translation callout |
| **2** | **FEATURE LAB** | Cell Morphology Selection & Rationale | • Ranked feature importance bar chart<br>• Interactive toggle: Selected Elite (8) vs All Evaluated (30)<br>• Feature-level clinical interpretation panel (updating dynamically upon click)<br>• Complete 30-feature catalog table with Stage 1 & Stage 2 pruning status |
| **3** | **MODEL BENCH** | Rigorous Algorithm Benchmarking | • **Interactive Metric Switcher**: Toggle between ROC-AUC, Recall, Precision, F1, F2, Accuracy, Specificity<br>• Dynamic sorted model ranking chart highlighting Champion Soft-Voting Ensemble<br>• Threshold calibration comparison: Default (0.50) vs Calibrated (0.3786)<br>• Comprehensive benchmark matrix across all 6 model architectures |
| **4** | **WHY THIS PREDICTION?** | TreeSHAP Local Patient Attributions | • Patient case selector (Preset archetypes: Patient A Malignant, Patient B Benign, Patient C Borderline, plus any patient from the 114 cohort)<br>• Dynamic Patient Diagnosis Receipt with probability meter<br>• **Live SHAP Waterfall chart** showing base value and directional pushes (terracotta positive risk vs sage protective factors) |
| **5** | **PATIENT EXPLORER** | Granular 114-Case Cohort Investigation | • Full-width searchable clinical table<br>• Multi-slicers: Filter by Actual Diagnosis, Predicted Outcome (TP, TN, FP, FN), and Accuracy<br>• Interactive row-click updating the side inspection drawer with exact nuclear measurements and top driver |
| **6** | **ERROR MAP** | Misclassification Audit & Safety Analysis | • False Negatives (1) vs False Positives (2) breakdown<br>• **Critical Scientific Audit**: Disproving the "Zero False Negatives" myth with transparent empirical evidence<br>• Granular drilldown into the 3 misclassified patients (PT-073 FN, PT-208 FP, PT-128 FP) |
| **7** | **METHOD & DATA** | Scientific Governance & Reproducibility | • Dataset origin & FNA biopsy specifications<br>• Two-stage feature selection methodology breakdown<br>• Controlled Dimensionality Experiment table (Full 30 vs PCA 6 vs Two-Stage 8)<br>• Clinical second-reader governance guidelines & reproducibility notes |

---

## 🎨 Visual Design System (Editorial Clinical Aesthetic)

This dashboard strictly rejects the generic "corporate blue" Power BI template and AI-generated neon glowing aesthetics in favor of a **clinical research publication design language**:

* **Navigation**: Deep Charcoal (`#181A1B` / `#1E2022`) with understated section numbering.
* **Canvas Background**: Warm Ivory / Off-White (`#F9F8F6`).
* **Malignancy / Critical Risk**: Deep Wine (`#821E2C` / `#841B2D`) used sparingly and strictly where clinically meaningful.
* **Benign / Safe Baseline**: Muted Sage (`#385E48` / `#556B2F`).
* **Feature Importance / Rust Accent**: Terracotta (`#BC522B` / `#C85A32`).
* **Borders & Dividers**: Thin, crisp 1px borders (`#E5E2DA`).
* **Cards & Typography**: Compact typography, direct data labels, no meaningless decorative charts, no 3D elements, no excessive rounded corners.

---

## 📐 Calculated Measures (DAX Specification)

The following calculated measures are defined in `powerbi/Explainable_BreastCancer_BI.Dataset/model.bim` under the `_Clinical_Measures` table:

```dax
// 1. Patient Volume Measures
Total Patients = COUNTROWS('Holdout_Patients')
Benign Count = CALCULATE(COUNTROWS('Holdout_Patients'), 'Holdout_Patients'[Actual Diagnosis] = "Benign")
Malignant Count = CALCULATE(COUNTROWS('Holdout_Patients'), 'Holdout_Patients'[Actual Diagnosis] = "Malignant")
Predicted Benign Count = CALCULATE(COUNTROWS('Holdout_Patients'), 'Holdout_Patients'[Predicted Diagnosis (Calibrated)] = "Benign")
Predicted Malignant Count = CALCULATE(COUNTROWS('Holdout_Patients'), 'Holdout_Patients'[Predicted Diagnosis (Calibrated)] = "Malignant")

// 2. Clinical Confusion Matrix Measures
True Positives (TP) = CALCULATE(COUNTROWS('Holdout_Patients'), 'Holdout_Patients'[Calibrated Outcome] = "True Positive")
True Negatives (TN) = CALCULATE(COUNTROWS('Holdout_Patients'), 'Holdout_Patients'[Calibrated Outcome] = "True Negative")
False Positives (FP) = CALCULATE(COUNTROWS('Holdout_Patients'), 'Holdout_Patients'[Calibrated Outcome] = "False Positive")
False Negatives (FN) = CALCULATE(COUNTROWS('Holdout_Patients'), 'Holdout_Patients'[Calibrated Outcome] = "False Negative")

// 3. Clinical Diagnostic Rates
Accuracy = DIVIDE([True Positives (TP)] + [True Negatives (TN)], [Total Patients], 0)
Precision = DIVIDE([True Positives (TP)], [True Positives (TP)] + [False Positives (FP)], 0)
Recall (Sensitivity) = DIVIDE([True Positives (TP)], [True Positives (TP)] + [False Negatives (FN)], 0)
Malignant Recall = [Recall (Sensitivity)]
Specificity = DIVIDE([True Negatives (TN)], [True Negatives (TN)] + [False Positives (FP)], 0)
F1 Score = DIVIDE(2 * [Precision] * [Recall (Sensitivity)], [Precision] + [Recall (Sensitivity)], 0)
F2 Score = DIVIDE(5 * [Precision] * [Recall (Sensitivity)], 4 * [Precision] + [Recall (Sensitivity)], 0)

// 4. Probability & Feature Stats
Mean Malignancy Probability = AVERAGE('Holdout_Patients'[Malignancy Probability])
Mean Prediction Confidence = AVERAGE('Holdout_Patients'[Prediction Confidence (%)])
Number of Selected Features = CALCULATE(COUNTROWS('Feature_Explainability'), 'Feature_Explainability'[Selected] = 1)
Total Evaluated Features = COUNTROWS('Feature_Explainability')

// 5. Model Thresholds & Benchmark
Best Model Name = CALCULATE(FIRSTNONBLANK('Model_Benchmark'[Model], 1), 'Model_Benchmark'[Model Rank] = 1)
Best CV ROC-AUC = CALCULATE(MAX('Model_Benchmark'[CV ROC-AUC]), 'Model_Benchmark'[Model Rank] = 1)
Calibrated Medical Threshold = 0.3786
Default Decision Threshold = 0.5000
```

---

## 🔍 Quality Control & Reconciliation Audit

Before final release, all numbers were cross-reconciled across the raw data, the executed notebook, the 4 CSV files, the Power BI model, and the frontend web app:

| Metric | Notebook Value | CSV Value | Power BI Measure | Frontend Value | Reconciled? |
|--------|----------------|-----------|------------------|----------------|-------------|
| Total Cohort Size | 569 | 569 | 569 | 569 | ✅ Exact |
| Training Set Size (80%) | 455 | 455 | 455 | 455 | ✅ Exact |
| Sealed Holdout Set Size (20%) | 114 | 114 | 114 | 114 | ✅ Exact |
| Holdout True Malignant | 42 (36.8%) | 42 | 42 | 42 | ✅ Exact |
| Holdout True Benign | 72 (63.2%) | 72 | 72 | 72 | ✅ Exact |
| Selected Feature Count | 8 | 8 | 8 | 8 | ✅ Exact |
| Default Threshold (0.50) TP / TN / FP / FN | 39 / 71 / 1 / 3 | 39 / 71 / 1 / 3 | 39 / 71 / 1 / 3 | 39 / 71 / 1 / 3 | ✅ Exact |
| Calibrated Threshold (0.3786) TP / TN / FP / FN | 41 / 70 / 2 / 1 | 41 / 70 / 2 / 1 | 41 / 70 / 2 / 1 | 41 / 70 / 2 / 1 | ✅ Exact |
| Calibrated Malignant Recall | 97.62% (41/42) | 97.62% | 97.62% | 97.62% | ✅ Exact |
| Calibrated Specificity | 97.22% (70/72) | 97.22% | 97.22% | 97.22% | ✅ Exact |
| Calibrated Accuracy | 97.37% (111/114) | 97.37% | 97.37% | 97.37% | ✅ Exact |
| Holdout Test ROC-AUC | 0.9944 | 0.9944 | 0.9944 | 0.9944 | ✅ Exact |
| Critical False Negatives | **1 (PT-073)** | **1 (PT-073)** | **1** | **1 (PT-073)** | ✅ Reconciled (No zero-FN false claim) |

---
*Created by the Lead Data Visualization, BI Developer & UX Design Team.*

