# Explainable Breast Cancer Classification
## Wisconsin Diagnostic Dataset (WDBC) — Datathon Solution Repository & Clinical BI Platform

### 🎯 Problem Statement
Develop an explainable machine learning model capable of distinguishing benign and malignant breast tumors using quantitative measurements of cell-nuclei characteristics from Fine Needle Aspirates (FNA).

### ❓ The Datathon Question
> *"Can cellular morphology measurements accurately distinguish malignant tumors from benign tumors while reducing dimensionality, preventing overfitting, handling correlated features, and explaining which cellular characteristics drive each prediction?"*

---

## 🏆 Key Results Summary

### 1. Empirical Answer to the Datathon Question
| Datathon Sub-Question | Finding & Evidence |
| :--- | :--- |
| **Accurately Distinguish?** | **YES.** The calibrated Soft-Voting Ensemble achieves a holdout **ROC-AUC of 0.9944** and **Recall of 97.62%** (41/42 malignant tumors detected on the sealed 20% holdout test set). |
| **Handling Correlated Features?** | **YES.** 32 pairs with $\|r\| \ge 0.85$ and $VIF > 1000$ were systematically resolved via **Two-Stage Feature Selection** (Hierarchical Collinear Clustering Pruning $\rightarrow$ RFECV), grouping duplicate metrics and retaining single biological champions. |
| **Reducing Dimensionality?** | **YES.** Dimensionality was reduced by **73.3%** (from 30 features $\rightarrow$ **8 elite native features**) without sacrificing ROC-AUC (0.9944 vs 0.9954) while decisively outperforming PCA in clinical interpretability. |
| **Preventing Overfitting?** | **YES.** Generalization gap between Train and Cross-Validation ROC-AUC was virtually zero (**0.0018** for Logistic Regression, **0.0026** for SVM). |
| **Explaining Predictions?** | **YES.** TreeSHAP proved that **`worst perimeter`** (nuclear gigantism) and **`mean concave points`** (nuclear membrane indentations) are the primary biological drivers of tumor malignancy. |

---

## 📊 Dual Deliverable Business Intelligence Architecture

This repository delivers both an official **Power BI Developer Project (`.pbip`)** and a standalone **Clinical Analytics Web Console (`frontend/`)**:

```
d:/DataRushFIT_BreastCancer_Detection/
├── Explainable_BreastCancer_BI.pbip             <-- Official Power BI Developer Project Root
├── powerbi/                                     <-- Power BI Multi-File Artifact
│   ├── Explainable_BreastCancer_BI.pbip
│   ├── Explainable_BreastCancer_BI.Dataset/     <-- Tabular Data Model (TMSL/model.bim, 24 DAX Measures)
│   └── Explainable_BreastCancer_BI.Report/      <-- Visual Layout (report.json & Editorial Theme)
├── frontend/                                    <-- Standalone Interactive Clinical Console
│   ├── index.html                               <-- 7-Page Full Visual Analytics Application
│   ├── styles.css                               <-- Clinical Editorial Design System
│   ├── app.js                                   <-- Interactive Slicers, SHAP Waterfalls & Tables
│   └── data.js                                  <-- Embedded Verified Clinical Dataset Bundle
├── data/                                        <-- Clean Source Data Folder
│   ├── holdout_patient_table.csv                <-- 114 Holdout Patients with Probabilities & SHAP
│   ├── model_benchmark.csv                      <-- 6 Models across CV & Holdout Metrics
│   ├── dimensionality_comparison.csv            <-- Full 30 vs PCA 6 vs Two-Stage 8
│   └── selected_feature_xai_ranking.csv         <-- All 30 Features with Status & Meaning
├── screenshots/                                 <-- High-Resolution Previews of all 7 Pages
│   ├── page1_clinical_signal.png
│   ├── page2_feature_lab.png
│   ├── page3_model_bench.png
│   ├── page4_why_this_prediction.png
│   ├── page5_patient_explorer.png
│   ├── page6_error_map.png
│   └── page7_method_and_data.png
├── notebooks/
│   └── Explainable_Breast_Cancer_Classification.ipynb  <-- 100% Executed Publication Notebook
├── HACKATHON_DASHBOARD_README.md                <-- BI Project Overview & Setup Guide
├── DASHBOARD_DOCUMENTATION.md                  <-- Technical & Clinical Documentation
├── DATA_DICTIONARY.md                           <-- Complete Field-by-Field Schema Dictionary
└── DEMO_GUIDE.md                                <-- 2-Minute Scripted Presentation Flow for Judges
```

---

## 🚀 How to Launch and Explore

### 1. Power BI Desktop
Double-click `Explainable_BreastCancer_BI.pbip` or execute:
```powershell
Start-Process "PBIDesktopStore.exe" -ArgumentList "d:\DataRushFIT_BreastCancer_Detection\Explainable_BreastCancer_BI.pbip"
```
Loads all 4 data tables, 24 DAX calculated measures, and 7 report pages styled with the custom Clinical Editorial Theme.

### 2. Clinical Web Console (Universal Browser Mode)
Open `d:\DataRushFIT_BreastCancer_Detection\frontend\index.html` in any browser, or start a local server:
```powershell
python -m http.server 8080 --directory d:\DataRushFIT_BreastCancer_Detection\frontend
```
Open [http://localhost:8080](http://localhost:8080) for full interactive exploration across all 7 pages.

---

## 🔬 The 8 Selected Elite Morphology Features
By eliminating collinear twins ($r \ge 0.85$) and running RFECV, the 30 features were condensed into 8 non-redundant, biologically grounded features:

1. **`worst perimeter`** (SHAP = 2.1474) — Represents extreme nuclear boundary size / gigantism.
2. **`worst texture`** (SHAP = 1.0380) — Represents chromatin density and coarseness.
3. **`mean concave points`** (SHAP = 1.0090) — Represents frequency of membrane notches and folds.
4. **`area error`** (SHAP = 0.6055) — Represents nuclear size heterogeneity (anisokaryosis).
5. **`worst smoothness`** (SHAP = 0.3491) — Represents perimeter contour irregularities.
6. **`worst symmetry`** (SHAP = 0.2385) — Represents loss of cellular nuclear symmetry.
7. **`concavity error`** (SHAP = 0.0762) — Represents cell-to-cell variability in membrane indentations.
8. **`concave points error`** (SHAP = 0.0620) — Represents consistency of nuclear notches across cells.

---

## 📈 Model Benchmark Matrix (Stratified 5-Fold CV on 8 Features)
| Model Architecture | CV ROC-AUC | CV Recall (Sensitivity) | CV Precision | CV F2-Score | Holdout Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Support Vector Machine (RBF)** | **0.9944** | 0.9353 | **0.9822** | 0.9443 | 94.74% |
| **Soft-Voting Ensemble (Champion)** | **0.9937** | **0.9471** | **0.9889** | **0.9551** | **96.49% (97.37% Cal)** |
| **Logistic Regression (L2)** | 0.9936 | 0.9412 | 0.9768 | 0.9481 | 95.61% |
| **Random Forest** | 0.9911 | 0.9176 | 0.9636 | 0.9265 | 94.74% |
| **XGBoost** | 0.9906 | 0.9353 | 0.9693 | 0.9419 | 95.61% |
| **LightGBM** | 0.9904 | 0.9412 | 0.9696 | 0.9467 | 94.74% |

---

## ⚖️ Threshold Calibration & Evidence-Based Audit
* **Default Threshold (0.50):** 39 TP, 71 TN, 1 FP, 3 FN $\rightarrow$ Malignant Recall = $92.86\%$, Specificity = $98.61\%$.
* **Calibrated Medical Threshold (0.3786):** 41 TP, 70 TN, 2 FP, 1 FN $\rightarrow$ Malignant Recall = **$97.62\%$**, Specificity = **$97.22\%$**.
* **Clinical Safety Audit:** We explicitly disprove the "Zero False Negatives" claim found in casual notebook drafts. The holdout set contains **exactly 1 False Negative (Patient PT-073, risk = 6.28%)**, a small-cell atypical carcinoma. Transparent clinical disclosure ensures oncologists understand when second-reader reflex review is vital.

---

## 📚 Deliverable Documentation Links
* [HACKATHON_DASHBOARD_README.md](file:///d:/DataRushFIT_BreastCancer_Detection/HACKATHON_DASHBOARD_README.md) — Comprehensive dashboard setup and features.
* [DASHBOARD_DOCUMENTATION.md](file:///d:/DataRushFIT_BreastCancer_Detection/DASHBOARD_DOCUMENTATION.md) — Technical and medical documentation.
* [DATA_DICTIONARY.md](file:///d:/DataRushFIT_BreastCancer_Detection/DATA_DICTIONARY.md) — Field-by-field schema and cytopathology guide.
* [DEMO_GUIDE.md](file:///d:/DataRushFIT_BreastCancer_Detection/DEMO_GUIDE.md) — 2-minute scripted judge walkthrough.
* [Screenshots Preview Directory](file:///d:/DataRushFIT_BreastCancer_Detection/screenshots/) — High-resolution renders of all 7 pages.
