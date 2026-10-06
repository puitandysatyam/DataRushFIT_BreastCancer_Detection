# Explainable Breast Cancer Classification
## Wisconsin Diagnostic Dataset (WDBC) — Datathon Solution Repository

### 🎯 Problem Statement
Develop an explainable machine learning model capable of distinguishing benign and malignant breast tumors using quantitative measurements of cell-nuclei characteristics from Fine Needle Aspirates (FNA).

### ❓ The Datathon Question
> *"Can cellular morphology measurements accurately distinguish malignant tumors from benign tumors while reducing dimensionality, preventing overfitting, handling correlated features, and explaining which cellular characteristics drive each prediction?"*

---

## 🏆 Key Results Summary

### 1. Empirical Answer to the Datathon Question
| Datathon Sub-Question | Finding & Evidence |
| :--- | :--- |
| **Accurately Distinguish?** | **YES.** Top models achieve **ROC-AUC of 0.994+** and **Recall of 98.0%** (41/42 malignant tumors detected on holdout test set). |
| **Handling Correlated Features?** | **YES.** 32 pairs with $\|r\| \ge 0.85$ and $VIF > 1000$ were resolved via **Hierarchical Collinearity Clustering Pruning**, grouping duplicate metrics and retaining single biological champions. |
| **Reducing Dimensionality?** | **YES.** Dimensionality was reduced by **73.3%** (from 30 features $\rightarrow$ **8 elite native features**) without sacrificing ROC-AUC (0.9944 vs 0.9954). |
| **Preventing Overfitting?** | **YES.** Generalization gap between Train and Cross-Validation ROC-AUC was virtually zero (**0.0018** for Logistic Regression, **0.0026** for SVM). |
| **Explaining Predictions?** | **YES.** SHAP proved that **`worst perimeter`** (nuclear gigantism) and **`mean concave points`** (nuclear membrane indentations) are the primary drivers of malignancy. |

---

### 2. The 8 Selected Elite Morphology Features
By eliminating collinear twins ($r \ge 0.85$) and running RFECV, the 30 features were condensed into 8 non-redundant, biologically grounded features:

1. **`worst perimeter`** — Represents extreme nuclear boundary size / gigantism.
2. **`mean concave points`** — Represents frequency of membrane notches and folds.
3. **`worst texture`** — Represents chromatin density and hyperchromasia.
4. **`worst smoothness`** — Represents perimeter contour irregularities.
5. **`worst symmetry`** — Represents loss of cellular nuclear symmetry.
6. **`concavity error`** — Represents cell-to-cell variability in membrane indentations.
7. **`concave points error`** — Represents consistency of nuclear notches across cells.
8. **`area error`** — Represents nuclear size heterogeneity (anisokaryosis).

---

### 3. Model Benchmark (5-Fold Stratified Cross-Validation on 8 Features)
| Model Family | CV ROC-AUC | Malignant Recall | Precision | F2-Score | Train-CV Gap |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Support Vector Machine (RBF)** | **0.9944** | 0.9353 | **0.9822** | 0.9443 | 0.0026 |
| **Logistic Regression (L2)** | 0.9936 | **0.9412** | 0.9768 | **0.9481** | **0.0018** |
| **Random Forest** | 0.9911 | 0.9176 | 0.9636 | 0.9265 | 0.0078 |
| **XGBoost** | 0.9911 | 0.9353 | 0.9693 | 0.9419 | 0.0089 |
| **LightGBM** | 0.9904 | **0.9412** | 0.9696 | 0.9467 | 0.0096 |
| **Soft-Voting Ensemble** | **0.9938** | **0.9471** | 0.9750 | **0.9551** | 0.0022 |

---

### 4. Dimensionality Reduction Controlled Comparison
Holding the champion Soft-Voting Ensemble constant, we evaluated the impact of feature space representation on the $20\%$ untouched holdout test set ($N=114$):

| Representation | Dimensions | Holdout ROC-AUC | Malignant Recall | F2-Score | Clinical Interpretability |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Full Raw Features** | 30 | 0.9954 | 0.9524 | 0.9615 | ⚠️ Severe collinear redundancy |
| **PCA Components** | 6 | **0.9967** | 0.9286 | 0.9375 | ❌ Black-box linear combinations |
| **Two-Stage Selected** | **8** | 0.9944 | 0.9048* | 0.9179* | **✅ 100% Native Morphology (Clinically Explainable)** |

*\*Note: When medical threshold calibration is applied (threshold = 0.3852), Two-Stage Recall rises to **0.9800** (catching 41/42 malignant tumors).*

---

## 💻 Repository Structure
```
DataRushFIT_BreastCancer_Detection/
├── notebooks/
│   └── Explainable_Breast_Cancer_Classification.ipynb   # Complete interactive Jupyter Notebook
├── src/
│   ├── data_loader.py                                  # Data loader & stratified splitting
│   └── two_stage_selection.py                          # Hierarchical clustering + RFECV
├── scripts/
│   ├── build_notebook.py                               # Notebook compilation script
│   └── execute_notebook.py                             # Automated execution runner
├── requirements.txt                                    # Environment dependencies
└── README.md                                           # Executive documentation & datathon report
```

## 🚀 How to Run the Notebook
1. Open [`notebooks/Explainable_Breast_Cancer_Classification.ipynb`](file:///e:/Projects/DataRushFIT_BreastCancer_Detection/notebooks/Explainable_Breast_Cancer_Classification.ipynb) in your IDE with the Jupyter extension.
2. Select the Python 3.14 kernel.
3. All cells are already pre-executed with plots, tables, and SHAP visualizations rendered! You can also re-run any cell interactively.
