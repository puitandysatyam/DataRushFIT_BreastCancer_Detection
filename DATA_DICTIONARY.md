# DATA_DICTIONARY.md
## Explainable Breast Cancer Classification — Comprehensive Data Dictionary

This document provides complete field-by-field definitions, clinical interpretations, data types, and sources for all datasets generated and utilized in the **Explainable Breast Cancer Classification** BI solution.

---

## 1. `holdout_patient_table.csv`
* **File Location:** `d:/DataRushFIT_BreastCancer_Detection/data/holdout_patient_table.csv` (also mirrored in workspace root)  
* **Record Count:** 114 patients (20% Stratified Holdout Test Set sealed from training)  
* **Purpose:** Patient-level clinical outcomes, probabilities, nuclear measurements, and local SHAP feature attributions.

| Column Name | Data Type | Example Value | Description & Clinical Definition |
|-------------|-----------|---------------|-----------------------------------|
| `Patient ID` | String | `PT-033` | De-identified clinical patient identifier assigned from WDBC index. |
| `Row Index` | Integer | `0` | Sequential index (0 to 113) within the holdout partition. |
| `Dataset Index` | Integer | `33` | Original row index within the full 569-patient WDBC dataset. |
| `Actual Diagnosis` | String | `Malignant`, `Benign` | True histological ground-truth diagnosis confirmed by biopsy. |
| `Actual Target` | Integer | `1`, `0` | Binary target mapping: $1 = \text{Malignant}$ (positive risk), $0 = \text{Benign}$. |
| `Predicted Diagnosis (Default 0.50)` | String | `Malignant`, `Benign` | Algorithmic prediction using the standard default 0.50 threshold. |
| `Predicted Target (Default 0.50)` | Integer | `1`, `0` | Binary prediction using default threshold ($P \ge 0.50$). |
| `Default Correct` | String | `Correct`, `Incorrect` | Whether the default prediction matched the true histological diagnosis. |
| `Default Outcome` | String | `True Positive`, `True Negative`, `False Positive`, `False Negative` | Clinical classification category at default 0.50 threshold. |
| `Predicted Diagnosis (Calibrated)` | String | `Malignant`, `Benign` | Algorithmic prediction using the optimal medical decision threshold ($0.3786$). |
| `Predicted Target (Calibrated)` | Integer | `1`, `0` | Binary prediction using calibrated threshold ($P \ge 0.3786$). |
| `Calibrated Correct` | String | `Correct`, `Incorrect` | Whether the calibrated prediction matched true diagnosis (111/114 = 97.37%). |
| `Calibrated Outcome` | String | `True Positive`, `True Negative`, `False Positive`, `False Negative` | Clinical outcome at calibrated threshold (41 TP, 70 TN, 2 FP, 1 FN). |
| `Malignancy Probability` | Float | `0.9982` | Calibrated predicted probability of tumor malignancy from Soft-Voting Ensemble ($0.0$ to $1.0$). |
| `Prediction Confidence (%)` | Float | `99.82` | Percentage certainty of predicted class: $\max(P, 1-P) \times 100$. |
| `Confidence Margin` | Float | `0.9964` | Distance from decision boundary uncertainty: $|P - 0.5| \times 2$ ($0.0$ to $1.0$). |
| `Top Influencing Feature` | String | `worst perimeter` | The morphological feature exerting the largest absolute SHAP push on this patient. |
| `Top Feature Impact` | String | `Elevates Risk`, `Lowers Risk` | Direction of top driver's influence on log-odds of cancer. |
| `Base Value (SHAP Base)` | Float | `-0.7241` | Expected model output (log-odds) before observing patient-specific features. |
| `worst perimeter` | Float | `128.0` | Mean of 3 largest nuclear perimeter measurements ($\mu\text{m}$). |
| `mean concave points` | Float | `0.0890` | Mean number of concave portions / indentations on nuclear contour. |
| `worst texture` | Float | `24.12` | Standard deviation of gray-scale pixel values for the 3 most extreme cells. |
| `concavity error` | Float | `0.0410` | Standard error of the severity of concave contour portions. |
| `concave points error` | Float | `0.0125` | Standard error for number of concave contour portions. |
| `area error` | Float | `72.5` | Standard error for nuclear area ($\mu\text{m}^2$), measuring cell pleomorphism. |
| `worst symmetry` | Float | `0.3210` | Symmetry measurement of the 3 most extreme nuclei. |
| `worst smoothness` | Float | `0.1450` | Local variation in nuclear border radius lengths for extreme cells. |
| `SHAP_worst perimeter` | Float | `+1.4820` | Patient-specific TreeSHAP attribution for `worst perimeter`. |
| `SHAP_mean concave points` | Float | `+0.8920` | Patient-specific TreeSHAP attribution for `mean concave points`. |
| `SHAP_worst texture` | Float | `+0.4210` | Patient-specific TreeSHAP attribution for `worst texture`. |
| `SHAP_concavity error` | Float | `+0.0520` | Patient-specific TreeSHAP attribution for `concavity error`. |
| `SHAP_concave points error` | Float | `+0.0380` | Patient-specific TreeSHAP attribution for `concave points error`. |
| `SHAP_area error` | Float | `+0.3120` | Patient-specific TreeSHAP attribution for `area error`. |
| `SHAP_worst symmetry` | Float | `+0.1140` | Patient-specific TreeSHAP attribution for `worst symmetry`. |
| `SHAP_worst smoothness` | Float | `+0.1820` | Patient-specific TreeSHAP attribution for `worst smoothness`. |

---

## 2. `model_benchmark.csv`
* **File Location:** `d:/DataRushFIT_BreastCancer_Detection/data/model_benchmark.csv`  
* **Record Count:** 6 models evaluated across 5-Fold Stratified CV and Holdout Test  
* **Purpose:** Rigorous benchmark matrix supporting the analytical Metric Switcher.

| Column Name | Data Type | Example Value | Description |
|-------------|-----------|---------------|-------------|
| `Model` | String | `Soft-Voting Ensemble` | Algorithm architecture name. |
| `CV ROC-AUC` | Float | `0.9937` | Mean area under the ROC curve across 5-fold cross-validation. |
| `CV ROC-AUC Std` | Float | `0.0051` | Standard deviation of CV ROC-AUC across the 5 validation folds. |
| `CV Recall (Sensitivity)` | Float | `0.9471` | Mean sensitivity for malignant tumors in cross-validation ($94.7\%$). |
| `CV Precision` | Float | `0.9889` | Mean positive predictive value in cross-validation ($98.9\%$). |
| `CV F1-Score` | Float | `0.9666` | Harmonic mean of precision and recall in cross-validation. |
| `CV F2-Score` | Float | `0.9551` | Clinical utility score weighting recall 2x as heavily as precision. |
| `CV Accuracy` | Float | `0.9758` | Mean percentage of correct predictions in cross-validation. |
| `Train ROC-AUC` | Float | `0.9995` | ROC-AUC evaluated on the training partition. |
| `Generalization Gap` | Float | `0.0058` | Difference between Train AUC and CV AUC ($<0.01$ indicates zero overfitting). |
| `Holdout Test Accuracy` | Float | `0.9649` | Accuracy evaluated on untouched 114 holdout patients at 0.50 threshold. |
| `Holdout Test Precision` | Float | `0.9750` | Precision on untouched holdout set at 0.50 threshold ($39/40$). |
| `Holdout Test Recall` | Float | `0.9286` | Malignant recall on holdout set at default 0.50 threshold ($39/42$). |
| `Holdout Test Specificity` | Float | `0.9861` | Benign specificity on holdout set at default 0.50 threshold ($71/72$). |
| `Holdout Test F1-Score` | Float | `0.9512` | F1-score on holdout test set at default 0.50 threshold. |
| `Holdout Test F2-Score` | Float | `0.9375` | F2-score on holdout test set at default 0.50 threshold. |
| `Holdout Test ROC-AUC` | Float | `0.9944` | Area under ROC curve on holdout test set using continuous probabilities. |
| `Holdout TP` | Integer | `39` | True Positives on holdout set at default threshold. |
| `Holdout TN` | Integer | `71` | True Negatives on holdout set at default threshold. |
| `Holdout FP` | Integer | `1` | False Positives on holdout set at default threshold. |
| `Holdout FN` | Integer | `3` | False Negatives on holdout set at default threshold (reduced to 1 with calibration). |
| `Model Rank` | Integer | `1` | Overall ranking based on cross-validation discriminative capability. |

---

## 3. `dimensionality_comparison.csv`
* **File Location:** `d:/DataRushFIT_BreastCancer_Detection/data/dimensionality_comparison.csv`  
* **Record Count:** 3 controlled feature representations  
* **Purpose:** Controlled experiment demonstrating why Two-Stage Selection outperforms PCA.

| Column Name | Data Type | Example Value | Description |
|-------------|-----------|---------------|-------------|
| `Feature Representation` | String | `Two-Stage Clinical Selection` | Tested feature space (Full 30, PCA 6, or Two-Stage 8). |
| `Dimensionality` | Integer | `8` | Total number of input features fed to the champion ensemble. |
| `Holdout Test ROC-AUC` | Float | `0.9944` | Holdout discriminative performance on this representation. |
| `Holdout Recall (Sensitivity)` | Float | `0.9286` | Malignant sensitivity at default threshold ($97.6\%$ with calibration). |
| `Holdout Precision` | Float | `0.9750` | Positive predictive value on holdout set. |
| `Holdout F2-Score` | Float | `0.9375` | Clinical F2-score ($0.9716$ with calibration). |
| `Holdout Accuracy` | Float | `0.9649` | Overall accuracy on holdout set ($97.37\%$ with calibration). |
| `Clinical Interpretability` | String | `High (Native Cytology)` | Qualitative assessment of physical/pathological interpretability. |

---

## 4. `selected_feature_xai_ranking.csv`
* **File Location:** `d:/DataRushFIT_BreastCancer_Detection/data/selected_feature_xai_ranking.csv`  
* **Record Count:** 30 cellular characteristics evaluated  
* **Purpose:** Complete traceability of every feature across Stage 1 pruning, Stage 2 RFECV, and SHAP.

| Column Name | Data Type | Example Value | Description |
|-------------|-----------|---------------|-------------|
| `Feature` | String | `worst perimeter` | Morphological nuclear characteristic name. |
| `Selected` | Integer | `1`, `0` | Binary indicator ($1 =$ Retained in final elite 8, $0 =$ Pruned). |
| `Selection Status` | String | `Selected (Stage 2 Final Elite)` | Detailed stage-specific status (Stage 1 Pruned, Stage 2 Pruned, or Elite). |
| `Mean \|SHAP\|` | Float | `2.1474` | Global Mean Absolute SHAP impact across the cohort (null for pruned). |
| `Spearman Correlation` | Float | `+0.7936` | Rank correlation coefficient with ground-truth diagnosis ($y$). |
| `Morphology Category` | String | `Size / Gigantism` | Pathological trait category (Size, Contour, Texture, Roughness, Symmetry). |
| `Direction & Clinical Meaning` | String | `Positive (High values...)` | Directional impact on cancer risk and physical cytopathology rationale. |
| `Feature Rank` | Integer | `1` | Overall rank (1 to 30) ordered by SHAP impact then correlation. |

---

## 5. Cytopathology Nuclear Trait Taxonomy

In fine-needle aspiration (FNA) of breast masses, 10 fundamental nuclear characteristics are digitized from microscopic image analysis:

1. **Radius**: Distance from nuclear center to perimeter points.
2. **Texture**: Standard deviation of gray-scale pixel intensities (chromatin coarseness).
3. **Perimeter**: Nuclear boundary circumference ($\mu\text{m}$).
4. **Area**: Nuclear cross-sectional surface area ($\mu\text{m}^2$).
5. **Smoothness**: Local variance of radial lengths (surface regularity).
6. **Compactness**: Computed as $\frac{\text{perimeter}^2}{\text{area}} - 1.0$.
7. **Concavity**: Severity of concave portions of the nuclear contour.
8. **Concave Points**: Number of distinct indentations on the nuclear envelope.
9. **Symmetry**: Nuclear balance along principal axes.
10. **Fractal Dimension**: "Coastline approximation" of boundary roughness (Mandelbrot index).

Each trait is computed across 3 statistics:
* `mean`: The average across all sampled cell nuclei.
* `se`: Standard error (variance among nuclei in the aspirate).
* `worst`: Mean of the 3 most abnormal / extreme cells (most critical in oncology).

