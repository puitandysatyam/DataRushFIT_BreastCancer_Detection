# DEMO_GUIDE.md
## 2-Minute Judge Presentation Script — Explainable Breast Cancer Classification

> **Role:** Lead Data Visualization Engineer & BI Developer  
> **Target Audience:** Hackathon Judges, Clinical Directors, Machine Learning Evaluators  
> **Time Limit:** Exactly 2 Minutes (120 Seconds)  
> **Key Message:** *"We turned an untrusted black-box cancer classifier into an explainable, clinically-calibrated second-reader tool that reduces dimensionality by 73% and catches 97.6% of malignant tumors with zero false-negative obfuscation."*

---

## ⏱️ Step-by-Step Presentation Timeline

### 0:00 – 0:25 | 1. Start on "CLINICAL SIGNAL" (The Problem & The Signal)
* **Screen:** Open **Page 1 (CLINICAL SIGNAL)**.
* **Script:**
  > *"Good afternoon, judges. In breast cancer biopsy analysis, two critical hurdles prevent clinical adoption: extreme feature multicollinearity that inflates model variance, and black-box predictions that pathologists cannot audit.*  
  > *Look at our **Methodology Ribbon** at the top: Starting with 569 fine-needle aspirates, we locked 20% into an untouched holdout vault. Through a Two-Stage Selection pipeline, we reduced 30 features down to **8 core morphology traits** with zero loss in accuracy.*  
  > *Our champion Soft-Voting Ensemble achieved a holdout **ROC-AUC of 0.9944**, detecting **41 of 42 malignant tumors (97.6% sensitivity)** with only 1 false negative."*

---

### 0:25 – 0:45 | 2. Switch to "FEATURE LAB" (Cellular Morphology Drivers)
* **Screen:** Click **Page 2 (FEATURE LAB)**.
* **Action:** Click the top bar for **`worst perimeter`**, then click **`mean concave points`**.
* **Script:**
  > *"Which cellular characteristics actually matter? Raw data had 32 pairs with correlation above 0.85—such as radius, area, and perimeter measuring the same underlying trait.*  
  > *Our Stage 1 clustering pruned redundant mathematical twins, while Stage 2 RFECV isolated 8 elite features.*  
  > *Notice our #1 driver: **`worst perimeter`** (SHAP = 2.15). In pathology, this is **nuclear gigantism**—malignant cells undergo rapid, abnormal nuclear expansion. Our #2 driver is **`mean concave points`**—indentations and notching in the nuclear membrane. These directly mirror microscopic diagnostic criteria."*

---

### 0:45 – 1:05 | 3. Switch to "MODEL BENCH" (Tuning for Human Lives)
* **Screen:** Click **Page 3 (MODEL BENCH)**.
* **Action:** Click the **"Recall (Malignant)"** button on the Metric Switcher, then point to the **Threshold Calibration** panel.
* **Script:**
  > *"On Model Bench, we compare 5 candidate families against our Soft-Voting Ensemble. But in oncology, standard 0.50 thresholding is dangerously naive—it treats missing a cancer identically to an extra biopsy.*  
  > *We calibrated the decision boundary to **0.3786** to maximize clinical $F_2$-score.*  
  > *This single calibration **cut False Negatives by 67%—from 3 down to 1**, catching 2 additional life-threatening cancers at the expense of only 1 extra biopsy referral."*

---

### 1:05 – 1:30 | 4. Switch to "EXPLAINABILITY" (Local Patient Waterfall Receipt)
* **Screen:** Click **Page 4 (WHY THIS PREDICTION?)**.
* **Action:** Click **"Patient A (Malignant)"** (Patient PT-033), then click **"Patient B (Benign)"** (Patient PT-049).
* **Script:**
  > *"Every AI prediction in our system produces an audit receipt. Watch this:*  
  > *For **Patient PT-033**, the model outputs a **99.8% malignancy risk**. Look at the **SHAP Waterfall**: the base value starts at -0.72, and extreme nuclear perimeter (+1.48) and membrane concavity (+0.89) push the score far into malignant space.*  
  > *Now switch to **Patient PT-049**: risk drops to **0.1%**. Small nuclear perimeter and smooth margins pull the score negative. The pathologist sees the physical cytological rationale in seconds."*

---

### 1:30 – 1:50 | 5. Switch to "ERROR MAP" (Scientific Integrity & False Negatives)
* **Screen:** Click **Page 6 (ERROR MAP)**.
* **Action:** Point to the **Audit of Discrepancy** callout and the table row for **PT-073**.
* **Script:**
  > *"Now look at the Error Map. Early informal notebook drafts informally claimed 'zero false negatives'. **We refuse to hide behind misleading claims.***  
  > *Our rigorous holdout audit reveals **exactly 1 False Negative (Patient PT-073)**.*  
  > *This patient had an atypical small-cell carcinoma where nuclear perimeter fell below typical malignant clusters, producing a 6.28% score. We transparently audit all 3 misclassifications so hospital teams know precisely where second-reader review is mandatory."*

---

### 1:50 – 2:00 | 6. Final Takeaway
* **Screen:** Return to **Page 1 (CLINICAL SIGNAL)**.
* **Script:**
  > *"In summary: We delivered a dual-artifact BI platform—a complete Power BI Developer project and an interactive clinical web console—achieving **0.9944 ROC-AUC**, **97.6% sensitivity**, **8 interpretable features**, and **100% transparent explainability**. Thank you!"*

---

## 💡 Quick Tips for the Presenter:
1. **Pacing:** Keep your speech calm and measured. Do not rush.
2. **Key Terms:** Emphasize *"sealed vault holdout"*, *"nuclear gigantism"*, *"threshold calibration"*, and *"transparent audit"*.
3. **If Asked About Zero False Negatives:** Point directly to Page 6 (Error Map) and explain that claiming 0 FN was an unverified narrative discrepancy in earlier drafts; the true holdout set contains exactly 1 FN (PT-073), representing 97.62% sensitivity. Judges will strongly respect this scientific honesty.
4. **If Power BI is requested:** Launch `PBIDesktopStore.exe Explainable_BreastCancer_BI.pbip`. If in a browser or web environment, open `frontend/index.html`. Both represent the same unified data model.

