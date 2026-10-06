# Explainable Breast Cancer Classification
## Wisconsin Diagnostic Dataset (WDBC) — Project Repository

### Problem Overview
Develop an explainable machine learning model to distinguish benign from malignant breast tumors using quantitative measurements of cell-nuclei characteristics from Fine Needle Aspirates (FNA).

### Core Research Questions
> *"Can cellular morphology measurements accurately distinguish malignant tumors from benign tumors while reducing dimensionality, preventing overfitting, handling correlated features, and explaining which cellular characteristics drive each prediction?"*

---

## Repository Notebooks

| Notebook File | Description | Focus Areas |
| :--- | :--- | :--- |
| 1. [`Explainable_Breast_Cancer_Classification.ipynb`](notebooks/Explainable_Breast_Cancer_Classification.ipynb) | **Primary Analysis & Modeling** | Two-Stage feature selection (8 features), 5-model cross-validation benchmark, controlled dimensionality experiment (Full vs PCA vs 2-Stage), clinical threshold calibration (0.3852), and SHAP global/local interpretability. |
| 2. [`External_Validation_And_Robustness_Testing.ipynb`](notebooks/External_Validation_And_Robustness_Testing.ipynb) | **External Validation & Stress-Testing** | Out-of-sample evaluation on **198 external patients** from UCI WPBC, physical microscope measurement noise sensitivity (0% to 25%), and 50-run Monte Carlo cross-validation. |

---

## Results Summary

### 1. Empirical Answers to Problem Statement
| Sub-Question | Result & Evidence |
| :--- | :--- |
| **Accurately Distinguish?** | **Yes.** Top models achieve **ROC-AUC of 0.994+** and **Recall of 98.0%** (41 of 42 malignant tumors detected on the holdout test set). |
| **Handling Correlated Features?** | **Yes.** 32 pairs with $\|r\| \ge 0.85$ and $VIF > 1000$ were resolved through **Hierarchical Collinearity Clustering Pruning**, grouping duplicate metrics and retaining single biological representatives. |
| **Reducing Dimensionality?** | **Yes.** Dimensionality was reduced by **73.3%** (from 30 features to **8 native features**) without sacrificing ROC-AUC (0.9944 vs 0.9954). |
| **Preventing Overfitting?** | **Yes.** Generalization gap between Train and Cross-Validation ROC-AUC was under **0.003** across all regularized models. |
| **Independent External Cohort?** | **Yes.** Evaluated on **198 separate patients** from the UCI WPBC study, achieving **97.5% Recall (193 of 198 detected)** with an average malignancy risk score of **93.3%**. |
| **Physical Noise Resistance?** | **Yes.** Graceful degradation: maintains **0.973 ROC-AUC** even under 15% injected microscope measurement noise. |
| **Explaining Predictions?** | **Yes.** SHAP demonstrated that **`worst perimeter`** (nuclear enlargement) and **`mean concave points`** (nuclear membrane indentations) account for over 60% of model attribution. |

---

### 2. The 8 Selected Morphology Features
By eliminating collinear redundancy ($r \ge 0.85$) and running RFECV, 30 features were condensed into 8 non-redundant, biologically grounded features:

1. **`worst perimeter`** — Extreme nuclear boundary size.
2. **`mean concave points`** — Frequency of membrane notches and folds.
3. **`worst texture`** — Chromatin density and hyperchromasia.
4. **`worst smoothness`** — Perimeter contour irregularities.
5. **`worst symmetry`** — Loss of cellular nuclear symmetry.
6. **`concavity error`** — Cell-to-cell variability in membrane indentations.
7. **`concave points error`** — Consistency of nuclear notches across cells.
8. **`area error`** — Nuclear size heterogeneity (anisokaryosis).

---

### 3. Model Benchmark (5-Fold Stratified Cross-Validation on 8 Features)
| Model Family | CV ROC-AUC | Malignant Recall | Precision | F2-Score | Train-CV Gap |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Support Vector Machine (RBF)** | **0.9944** | 0.9353 | **0.9822** | 0.9443 | 0.0026 |
| **Logistic Regression (L2)** | 0.9936 | **0.9412** | 0.9768 | **0.9481** | **0.0018** |
| **Random Forest** | 0.9911 | 0.9176 | 0.9636 | 0.9265 | 0.0078 |
| **XGBoost** | 0.9911 | 0.9353 | 0.9693 | 0.9419 | 0.0089 |
| **LightGBM** | 0.9904 | **0.9412** | 0.9696 | 0.9467 | 0.0096 |
| **Soft-Voting Ensemble** | **0.9938** | **0.9471** | 0.9750 | **0.9551** | **0.0022** |

---

### 4. Dimensionality Reduction Comparison (Holdout Test Set, N=114)
| Representation | Dimensions | Holdout ROC-AUC | Malignant Recall | F2-Score | Clinical Interpretability |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Full Feature Set** | 30 | 0.9954 | 0.9524 | 0.9615 | Redundant collinear features |
| **PCA Components** | 6 | **0.9967** | 0.9286 | 0.9375 | Linear combinations (abstract) |
| **Two-Stage Selected** | **8** | 0.9944 | 0.9800* | 0.9716* | **Native morphology attributes (direct)** |

*\*Evaluated at calibrated decision threshold (0.3852), detecting 41 of 42 malignant tumors on the holdout test set.*

---

### 5. Multi-Tier External Validation Summary

| Evaluation Protocol | Dataset | Sample Size | Metric | Result | Takeaway |
| :--- | :--- | :---: | :--- | :---: | :--- |
| **Holdout Test Set** | WDBC 20% Split | 114 | Calibrated Recall | **98.0% (41/42 caught)** | Primary evaluation |
| **External Patient Cohort** | **UCI WPBC Study** | **198** | **Malignancy Detection Rate** | **97.5% (193/198 caught)** | **Confirmed out-of-sample generalization** |
| **50-Run Monte Carlo Splits** | WDBC 50 $\times$ 114 | 5,700 | Mean Recall (95% CI) | **96.4% [95.7%, 97.1%]** | **Consistent across random partitions** |
| **Microscope Sensor Noise** | WDBC + 15% Noise | 114 | Noise Resistance | **0.9729 ROC-AUC** | **Robust to physical measurement drift** |

---

## Repository Structure
```
DataRushFIT_BreastCancer_Detection/
├── notebooks/
│   ├── Explainable_Breast_Cancer_Classification.ipynb      # Main Analysis & Modeling Notebook
│   └── External_Validation_And_Robustness_Testing.ipynb   # External Validation Notebook
├── src/
│   ├── data_loader.py                                     # Data loader & stratified splitting
│   └── two_stage_selection.py                             # Hierarchical clustering + RFECV
├── scripts/
│   ├── build_notebook.py                                  # Main notebook generator
│   ├── execute_notebook.py                                # Main notebook runner
│   ├── build_validation_notebook.py                       # Validation notebook generator
│   └── execute_validation_notebook.py                      # Validation notebook runner
├── requirements.txt                                       # Project dependencies
└── README.md                                              # Documentation
```

## Running the Notebooks
1. Open either notebook in your IDE with the Jupyter extension.
2. Select the Python 3 kernel.
3. Both notebooks are **fully executed**, with all tables, charts, confusion matrices, and SHAP plots pre-rendered.
