/**
 * CLINICAL SIGNAL — EXPLAINABLE BREAST CANCER AI CONSOLE
 * Interactivity Engine & Data Bindings
 */

document.addEventListener("DOMContentLoaded", function () {
  const data = CLINICAL_DATA;

  // 1. NAVIGATION ROUTING
  const navItems = document.querySelectorAll(".nav-item");
  const tabContents = document.querySelectorAll(".tab-content");
  const pageTitleDisplay = document.getElementById("page-title-display");
  const pageDescDisplay = document.getElementById("page-desc-display");

  const pageMeta = {
    "tab-signal": { title: "Clinical Signal", desc: "Executive briefing and diagnostic signal board" },
    "tab-featurelab": { title: "Feature Lab", desc: "Nuclear morphology selection & clinical cytology analysis" },
    "tab-modelbench": { title: "Model Bench", desc: "Clinical algorithm evaluation, trade-offs & threshold calibration" },
    "tab-explainability": { title: "Why This Prediction?", desc: "TreeSHAP local feature attribution receipts & global drivers" },
    "tab-patients": { title: "Patient Explorer", desc: "Interactive clinical investigation table for 114 holdout cases" },
    "tab-errormap": { title: "Error Map", desc: "Audit of misclassifications & evidence-based safety evaluation" },
    "tab-method": { title: "Method & Data", desc: "End-to-end scientific methodology, governance & reproducibility" }
  };

  navItems.forEach(item => {
    item.addEventListener("click", () => {
      const targetTab = item.getAttribute("data-tab");

      navItems.forEach(n => n.classList.remove("active"));
      tabContents.forEach(t => t.classList.remove("active"));

      item.classList.add("active");
      const activeTabElem = document.getElementById(targetTab);
      if (activeTabElem) activeTabElem.classList.add("active");

      if (pageMeta[targetTab]) {
        pageTitleDisplay.textContent = pageMeta[targetTab].title;
        pageDescDisplay.textContent = pageMeta[targetTab].desc;
      }

      // Trigger resize for charts
      window.dispatchEvent(new Event("resize"));
    });
  });

  // URL Query Param Support for direct deep-linking & screenshots
  const urlParams = new URLSearchParams(window.location.search);
  const requestedTab = urlParams.get("tab");
  if (requestedTab && document.getElementById(requestedTab)) {
    const navMatch = document.querySelector(`.nav-item[data-tab="${requestedTab}"]`);
    if (navMatch) navMatch.click();
  }

  // 2. PAGE 1: CHARTS (Clinical Signal)
  initSignalCharts();

  // 3. PAGE 2: FEATURE LAB
  initFeatureLab();

  // 4. PAGE 3: MODEL BENCH
  initModelBench();

  // 5. PAGE 4: EXPLAINABILITY (XAI / SHAP)
  initExplainability();

  // 6. PAGE 5: PATIENT EXPLORER
  initPatientExplorer();

  // =========================================================================
  // INITIALIZERS
  // =========================================================================

  function initSignalCharts() {
    // 1. Training Split Graph (Volume, Balance, or Donut)
    const ctxSplit = document.getElementById("chart-split-graph");
    let splitChart = null;
    let splitMode = "volume";

    function renderSplitChart() {
      if (!ctxSplit || typeof Chart === "undefined") return;
      if (splitChart) splitChart.destroy();

      const sp = data.splits || {
        train_total: 455, train_benign: 285, train_malignant: 170,
        holdout_total: 114, holdout_benign: 72, holdout_malignant: 42
      };

      if (splitMode === "volume") {
        splitChart = new Chart(ctxSplit, {
          type: "bar",
          data: {
            labels: ["Train Set (80%)", "Holdout Test (20%)"],
            datasets: [
              {
                label: "Benign (0)",
                data: [sp.train_benign, sp.holdout_benign],
                backgroundColor: "#385E48",
                borderRadius: 2
              },
              {
                label: "Malignant (1)",
                data: [sp.train_malignant, sp.holdout_malignant],
                backgroundColor: "#821E2C",
                borderRadius: 2
              }
            ]
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
              x: { stacked: true, ticks: { font: { size: 11 } } },
              y: { stacked: true, title: { display: true, text: "Patient Biopsies", font: { size: 10 } } }
            },
            plugins: {
              legend: { position: "top", labels: { boxWidth: 10, font: { size: 11 } } }
            }
          }
        });
      } else if (splitMode === "classes") {
        splitChart = new Chart(ctxSplit, {
          type: "bar",
          data: {
            labels: ["Train Partition (N=455)", "Holdout Partition (N=114)"],
            datasets: [
              {
                label: "Benign %",
                data: [62.64, 63.16],
                backgroundColor: "#385E48",
                borderRadius: 2
              },
              {
                label: "Malignant %",
                data: [37.36, 36.84],
                backgroundColor: "#821E2C",
                borderRadius: 2
              }
            ]
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
              x: { ticks: { font: { size: 11 } } },
              y: { min: 0, max: 100, title: { display: true, text: "Prevalence Percentage (%)", font: { size: 10 } } }
            },
            plugins: {
              legend: { position: "top", labels: { boxWidth: 10, font: { size: 11 } } }
            }
          }
        });
      } else if (splitMode === "donut") {
        splitChart = new Chart(ctxSplit, {
          type: "doughnut",
          data: {
            labels: ["Benign (72)", "Malignant (42)"],
            datasets: [{
              data: [sp.holdout_benign, sp.holdout_malignant],
              backgroundColor: ["#385E48", "#821E2C"],
              borderWidth: 1,
              borderColor: "#FFFFFF"
            }]
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
              legend: { position: "bottom", labels: { boxWidth: 12, font: { size: 11 } } }
            },
            cutout: "65%"
          }
        });
      }
    }

    const splitBtns = document.querySelectorAll("#split-chart-switcher .btn-filter");
    splitBtns.forEach(btn => {
      btn.addEventListener("click", () => {
        splitBtns.forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        if (btn.id === "btn-split-volume") splitMode = "volume";
        else if (btn.id === "btn-split-classes") splitMode = "classes";
        else if (btn.id === "btn-split-donut") splitMode = "donut";
        renderSplitChart();
      });
    });

    renderSplitChart();

    // 2. Interactive Confusion Matrix Switcher
    const btnCmCal = document.getElementById("btn-cm-calibrated");
    const btnCmDef = document.getElementById("btn-cm-default");
    const cmSubtitle = document.getElementById("cm-subtitle");

    function setConfusionMatrixView(isCalibrated) {
      const cms = data.confusion_matrices || {
        default: { threshold: 0.50, tp: 39, tn: 71, fp: 1, fn: 3, recall: 92.86, specificity: 98.61, precision: 97.50, f2: 0.9375 },
        calibrated: { threshold: 0.3786, tp: 41, tn: 70, fp: 2, fn: 1, recall: 97.62, specificity: 97.22, precision: 95.35, f2: 0.9716 }
      };

      const m = isCalibrated ? cms.calibrated : cms.default;

      document.getElementById("cm-tn-val").textContent = m.tn;
      document.getElementById("cm-tn-pct").textContent = `${m.specificity.toFixed(1)}% Specificity`;

      document.getElementById("cm-fp-val").textContent = m.fp;
      document.getElementById("cm-fp-pct").textContent = `${(100 - m.specificity).toFixed(1)}% Overcall`;

      document.getElementById("cm-fn-val").textContent = m.fn;
      document.getElementById("cm-fn-pct").textContent = isCalibrated ? "Critical Miss (PT-073)" : "3 Critical Misses (PT-073, PT-385, PT-205)";

      document.getElementById("cm-tp-val").textContent = m.tp;
      document.getElementById("cm-tp-pct").textContent = `${m.recall.toFixed(1)}% Sensitivity`;

      document.getElementById("cm-bar-recall").textContent = `${m.recall.toFixed(2)}%`;
      document.getElementById("cm-bar-spec").textContent = `${m.specificity.toFixed(2)}%`;
      document.getElementById("cm-bar-prec").textContent = `${m.precision.toFixed(2)}%`;
      document.getElementById("cm-bar-f2").textContent = m.f2.toFixed(4);

      if (isCalibrated) {
        cmSubtitle.textContent = "Calibrated Cutoff (0.3786) — 41/42 cancers caught (1 Missed)";
        btnCmCal.classList.add("active");
        btnCmDef.classList.remove("active");
      } else {
        cmSubtitle.textContent = "Default Cutoff (0.5000) — Standard boundary (3 Missed Cancers)";
        btnCmDef.classList.add("active");
        btnCmCal.classList.remove("active");
      }
    }

    if (btnCmCal && btnCmDef) {
      btnCmCal.addEventListener("click", () => setConfusionMatrixView(true));
      btnCmDef.addEventListener("click", () => setConfusionMatrixView(false));
      setConfusionMatrixView(true);
    }

    // Model Performance CV Comparison
    const ctxModels = document.getElementById("chart-signal-models");
    if (ctxModels && typeof Chart !== "undefined") {
      const mLabels = data.models.map(m => m["Model"]);
      const mAuc = data.models.map(m => m["CV ROC-AUC"]);
      const mRec = data.models.map(m => m["CV Recall (Sensitivity)"]);

      new Chart(ctxModels, {
        type: "bar",
        data: {
          labels: mLabels,
          datasets: [
            {
              label: "CV ROC-AUC",
              data: mAuc,
              backgroundColor: "#4E6142",
              borderRadius: 2
            },
            {
              label: "Malignant Recall",
              data: mRec,
              backgroundColor: "#BC522B",
              borderRadius: 2
            }
          ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          indexAxis: "y",
          scales: {
            x: { min: 0.85, max: 1.0, ticks: { font: { size: 10 } } },
            y: { ticks: { font: { size: 10.5 } } }
          },
          plugins: {
            legend: { position: "top", labels: { boxWidth: 10, font: { size: 10.5 } } }
          }
        }
      });
    }

    // Top XAI Drivers
    const ctxXai = document.getElementById("chart-signal-xai");
    if (ctxXai && typeof Chart !== "undefined") {
      const selFeats = data.features.filter(f => f.Selected === 1);
      const fLabels = selFeats.map(f => f.Feature);
      const fShap = selFeats.map(f => f["Mean |SHAP|"]);

      new Chart(ctxXai, {
        type: "bar",
        data: {
          labels: fLabels,
          datasets: [{
            label: "Mean |SHAP| Impact",
            data: fShap,
            backgroundColor: "#BC522B",
            borderRadius: 2
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          indexAxis: "y",
          scales: {
            x: { ticks: { font: { size: 10 } } },
            y: { ticks: { font: { size: 10 } } }
          },
          plugins: {
            legend: { display: false }
          }
        }
      });
    }
  }

  function initFeatureLab() {
    const tbody = document.getElementById("tbody-features");
    const ctxFeat = document.getElementById("chart-featurelab-bar");
    let featChart = null;

    let showOnlySelected = true;

    function renderTable() {
      tbody.innerHTML = "";
      const list = showOnlySelected ? data.features.filter(f => f.Selected === 1) : data.features;

      list.forEach(f => {
        const tr = document.createElement("tr");
        tr.className = f.Selected === 1 ? "feat-row" : "feat-row pruned";
        tr.innerHTML = `
          <td><strong>#${f["Feature Rank"]}</strong></td>
          <td><span style="font-weight: 600; color: var(--text-primary);">${f.Feature}</span></td>
          <td><span class="badge ${f.Selected === 1 ? 'badge-selected' : 'badge-pruned'}">${f["Selection Status"]}</span></td>
          <td><strong style="color: ${f.Selected === 1 ? 'var(--terracotta-accent)' : 'inherit'};">${f["Mean |SHAP|"] != null ? f["Mean |SHAP|"] : '—'}</strong></td>
          <td>${f["Spearman Correlation"] > 0 ? '+' : ''}${f["Spearman Correlation"]}</td>
          <td>${f["Morphology Category"]}</td>
          <td style="font-size: 11.5px; color: var(--text-secondary); max-width: 320px;">${f["Direction & Clinical Meaning"]}</td>
        `;

        tr.addEventListener("click", () => updateFeatureDetail(f));
        tbody.appendChild(tr);
      });
    }

    function renderChart() {
      const list = showOnlySelected ? data.features.filter(f => f.Selected === 1) : data.features.slice(0, 15);
      const labels = list.map(f => f.Feature);
      const values = list.map(f => f["Mean |SHAP|"] || Math.abs(f["Spearman Correlation"]));

      if (featChart) featChart.destroy();

      if (ctxFeat && typeof Chart !== "undefined") {
        featChart = new Chart(ctxFeat, {
          type: "bar",
          data: {
            labels: labels,
            datasets: [{
              label: showOnlySelected ? "Mean |SHAP| Value" : "Feature Impact (|r| or SHAP)",
              data: values,
              backgroundColor: list.map(f => f.Selected === 1 ? "#BC522B" : "#A6ACB2"),
              borderRadius: 2
            }]
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            indexAxis: "y",
            scales: {
              x: { ticks: { font: { size: 10 } } },
              y: { ticks: { font: { size: 10 } } }
            },
            plugins: {
              legend: { display: false }
            },
            onClick: (e, elements) => {
              if (elements.length > 0) {
                const idx = elements[0].index;
                updateFeatureDetail(list[idx]);
              }
            }
          }
        });
      }
    }

    function updateFeatureDetail(f) {
      document.getElementById("fd-name").textContent = f.Feature;
      document.getElementById("fd-category").textContent = `Category: ${f["Morphology Category"]}`;
      document.getElementById("fd-status").textContent = f["Selection Status"];
      document.getElementById("fd-status").className = f.Selected === 1 ? "badge badge-selected" : "badge badge-pruned";
      document.getElementById("fd-rank").textContent = `#${f["Feature Rank"]} of 30`;
      document.getElementById("fd-shap").textContent = f["Mean |SHAP|"] != null ? f["Mean |SHAP|"] : "Pruned (N/A)";
      document.getElementById("fd-corr").textContent = `${f["Spearman Correlation"] > 0 ? '+' : ''}${f["Spearman Correlation"]} (Spearman)`;
      document.getElementById("fd-meaning").textContent = f["Direction & Clinical Meaning"];
    }

    document.getElementById("btn-feat-selected").addEventListener("click", function () {
      this.classList.add("active");
      document.getElementById("btn-feat-all").classList.remove("active");
      showOnlySelected = true;
      renderTable();
      renderChart();
    });

    document.getElementById("btn-feat-all").addEventListener("click", function () {
      this.classList.add("active");
      document.getElementById("btn-feat-selected").classList.remove("active");
      showOnlySelected = false;
      renderTable();
      renderChart();
    });

    renderTable();
    renderChart();
  }

  function initModelBench() {
    const tbody = document.getElementById("tbody-models");
    tbody.innerHTML = "";

    data.models.forEach(m => {
      const tr = document.createElement("tr");
      if (m.Model === "Soft-Voting Ensemble") tr.style.backgroundColor = "var(--olive-light)";
      tr.innerHTML = `
        <td><strong>#${m["Model Rank"]}</strong></td>
        <td><strong>${m.Model}</strong> ${m.Model === "Soft-Voting Ensemble" ? '<span class="badge badge-selected">Champion</span>' : ''}</td>
        <td><strong>${m["CV ROC-AUC"]}</strong> <span style="font-size: 10px; color: var(--text-muted);">±${m["CV ROC-AUC Std"]}</span></td>
        <td style="color: var(--wine-malignant); font-weight: 600;">${(m["CV Recall (Sensitivity)"] * 100).toFixed(1)}%</td>
        <td>${(m["CV Precision"] * 100).toFixed(1)}%</td>
        <td><strong>${m["CV F2-Score"]}</strong></td>
        <td>${m["Generalization Gap"]}</td>
        <td>${(m["Holdout Test Accuracy"] * 100).toFixed(1)}%</td>
        <td>${m["Holdout TP"]}</td>
        <td>${m["Holdout TN"]}</td>
        <td>${m["Holdout FP"]}</td>
        <td style="color: var(--wine-malignant); font-weight: 700;">${m["Holdout FN"]}</td>
      `;
      tbody.appendChild(tr);
    });

    // Dynamic Metric Bar Chart
    const ctxDynamic = document.getElementById("chart-model-dynamic");
    let dynamicChart = null;

    function updateModelChart(metricKey) {
      document.getElementById("model-chart-title").textContent = `Candidate Models Ranked by ${metricKey}`;
      const sortedModels = [...data.models].sort((a, b) => b[metricKey] - a[metricKey]);
      const labels = sortedModels.map(m => m.Model);
      const vals = sortedModels.map(m => m[metricKey]);

      if (dynamicChart) dynamicChart.destroy();

      if (ctxDynamic && typeof Chart !== "undefined") {
        dynamicChart = new Chart(ctxDynamic, {
          type: "bar",
          data: {
            labels: labels,
            datasets: [{
              label: metricKey,
              data: vals,
              backgroundColor: sortedModels.map(m => m.Model === "Soft-Voting Ensemble" ? "#821E2C" : "#4E6142"),
              borderRadius: 2
            }]
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            indexAxis: "y",
            scales: {
              x: { min: Math.max(0, Math.floor(Math.min(...vals) * 10) / 10 - 0.05), max: 1.0 }
            },
            plugins: {
              legend: { display: false }
            }
          }
        });
      }
    }

    const switcherBtns = document.querySelectorAll("#model-metric-switcher .btn-filter");
    switcherBtns.forEach(btn => {
      btn.addEventListener("click", () => {
        switcherBtns.forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        const metric = btn.getAttribute("data-metric");
        updateModelChart(metric);
      });
    });

    updateModelChart("CV ROC-AUC");

    // Validation Loss & Learning Curve Chart
    const ctxLoss = document.getElementById("chart-val-loss");
    let lossChart = null;
    let lossMode = "boosting";

    function renderLossChart() {
      if (!ctxLoss || typeof Chart === "undefined") return;
      if (lossChart) lossChart.destroy();

      const vlData = data.validation_loss || {
        boosting_loss: [
          { iteration: 1, train_loss: 0.5948, val_loss: 0.5983 },
          { iteration: 10, train_loss: 0.2815, val_loss: 0.3164 },
          { iteration: 30, train_loss: 0.0949, val_loss: 0.1402 },
          { iteration: 60, train_loss: 0.0394, val_loss: 0.0961 }
        ],
        learning_curve: [
          { train_samples: 68, train_loss: 0.0826, val_loss: 0.2143 },
          { train_samples: 233, train_loss: 0.0664, val_loss: 0.1068 },
          { train_samples: 455, train_loss: 0.0572, val_loss: 0.0925 }
        ]
      };

      if (lossMode === "boosting") {
        const labels = vlData.boosting_loss.map(b => `Round ${b.iteration}`);
        const tLoss = vlData.boosting_loss.map(b => b.train_loss);
        const vLoss = vlData.boosting_loss.map(b => b.val_loss);

        lossChart = new Chart(ctxLoss, {
          type: "line",
          data: {
            labels: labels,
            datasets: [
              {
                label: "Training Log-Loss",
                data: tLoss,
                borderColor: "#4E6142",
                backgroundColor: "rgba(78, 97, 66, 0.08)",
                fill: true,
                tension: 0.3,
                borderWidth: 2,
                pointRadius: 3
              },
              {
                label: "Validation Log-Loss (Holdout Test)",
                data: vLoss,
                borderColor: "#BC522B",
                backgroundColor: "transparent",
                borderDash: [5, 4],
                tension: 0.3,
                borderWidth: 2,
                pointRadius: 3
              }
            ]
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
              x: { ticks: { font: { size: 10 } } },
              y: { title: { display: true, text: "Cross-Entropy Loss (Log-Loss)", font: { size: 10 } } }
            },
            plugins: {
              legend: { position: "top", labels: { boxWidth: 12, font: { size: 11 } } }
            }
          }
        });
      } else {
        const labels = vlData.learning_curve.map(c => `N=${c.train_samples}`);
        const tLoss = vlData.learning_curve.map(c => c.train_loss);
        const vLoss = vlData.learning_curve.map(c => c.val_loss);

        lossChart = new Chart(ctxLoss, {
          type: "line",
          data: {
            labels: labels,
            datasets: [
              {
                label: "Training Score (CV)",
                data: tLoss,
                borderColor: "#4E6142",
                backgroundColor: "rgba(78, 97, 66, 0.08)",
                fill: true,
                tension: 0.3,
                borderWidth: 2,
                pointRadius: 4
              },
              {
                label: "5-Fold Cross-Validation Loss",
                data: vLoss,
                borderColor: "#821E2C",
                backgroundColor: "transparent",
                borderDash: [4, 4],
                tension: 0.3,
                borderWidth: 2,
                pointRadius: 4
              }
            ]
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
              x: { title: { display: true, text: "Training Cohort Size", font: { size: 10 } }, ticks: { font: { size: 10 } } },
              y: { title: { display: true, text: "5-Fold CV Log-Loss", font: { size: 10 } } }
            },
            plugins: {
              legend: { position: "top", labels: { boxWidth: 12, font: { size: 11 } } }
            }
          }
        });
      }
    }

    const lossBtns = document.querySelectorAll("#loss-chart-switcher .btn-filter");
    lossBtns.forEach(btn => {
      btn.addEventListener("click", () => {
        lossBtns.forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        lossMode = btn.id === "btn-loss-boosting" ? "boosting" : "samples";
        renderLossChart();
      });
    });

    renderLossChart();
  }

  function initExplainability() {
    const pSelect = document.getElementById("patient-xai-select");
    const ctxWaterfall = document.getElementById("chart-xai-waterfall");
    let waterfallChart = null;

    // Populate dropdown with all holdout patients
    data.holdout.forEach(p => {
      // Check if not already in select
      let exists = false;
      for (let i = 0; i < pSelect.options.length; i++) {
        if (pSelect.options[i].value === p["Patient ID"]) {
          exists = true;
          break;
        }
      }
      if (!exists) {
        const opt = document.createElement("option");
        opt.value = p["Patient ID"];
        opt.textContent = `${p["Patient ID"]} (${p["Actual Diagnosis"]} — ${(p["Malignancy Probability"] * 100).toFixed(1)}%)`;
        pSelect.appendChild(opt);
      }
    });

    function loadPatientXai(pid) {
      const patient = data.holdout.find(p => p["Patient ID"] === pid) || data.holdout[0];

      document.getElementById("xai-receipt-id").textContent = patient["Patient ID"];
      const isMalignant = patient["Actual Diagnosis"] === "Malignant";
      const badge = document.getElementById("xai-receipt-badge");
      badge.textContent = patient["Actual Diagnosis"];
      badge.className = isMalignant ? "badge badge-malignant" : "badge badge-benign";

      const prob = (patient["Malignancy Probability"] * 100).toFixed(1);
      document.getElementById("xai-receipt-prob").textContent = `${prob}%`;
      document.getElementById("xai-receipt-prob").style.color = isMalignant ? "var(--wine-malignant)" : "var(--sage-benign)";

      const meter = document.getElementById("xai-meter-prob");
      meter.style.width = `${prob}%`;
      meter.className = isMalignant ? "meter-fill fill-wine" : "meter-fill fill-sage";

      document.getElementById("xai-receipt-actual").textContent = patient["Actual Diagnosis"];
      document.getElementById("xai-receipt-pred").textContent = patient["Predicted Diagnosis (Calibrated)"];
      document.getElementById("xai-receipt-driver").textContent = `${patient["Top Influencing Feature"]} (${patient["Top Feature Impact"]})`;

      // Dynamic narrative
      const narr = document.getElementById("xai-receipt-narrative");
      if (patient["Calibrated Outcome"] === "False Negative") {
        narr.innerHTML = `<strong>Diagnostic Alert:</strong> This is a misclassified case (False Negative). While the true histology is Malignant, cellular measurements fell below the cutoffs for worst perimeter (${patient["worst perimeter"]}) and concavity, generating an algorithmic risk of ${prob}%.`;
        narr.className = "clinical-callout wine";
      } else if (isMalignant) {
        narr.innerHTML = `High nuclear perimeter (${patient["worst perimeter"]} μm) and concavity (${patient["mean concave points"]}) push log-odds significantly into malignant territory. Recommendation: Urgent histopathology referral.`;
        narr.className = "clinical-callout wine";
      } else {
        narr.innerHTML = `Low nuclear dimensions (${patient["worst perimeter"]} μm) and intact smooth cell membrane margins anchor diagnostic risk near zero. Algorithmic prediction matches benign cytology.`;
        narr.className = "clinical-callout";
      }

      // Render Waterfall Chart
      const shapKeys = [
        "worst perimeter", "worst texture", "mean concave points",
        "area error", "worst smoothness", "worst symmetry",
        "concavity error", "concave points error"
      ];

      const shapValues = shapKeys.map(k => patient[`SHAP_${k}`] || 0);

      if (waterfallChart) waterfallChart.destroy();

      if (ctxWaterfall && typeof Chart !== "undefined") {
        waterfallChart = new Chart(ctxWaterfall, {
          type: "bar",
          data: {
            labels: shapKeys,
            datasets: [{
              label: "SHAP Impact on Malignancy Log-Odds",
              data: shapValues,
              backgroundColor: shapValues.map(v => v >= 0 ? "#BC522B" : "#4E6142"),
              borderRadius: 2
            }]
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            indexAxis: "y",
            scales: {
              x: {
                title: { display: true, text: "← Towards Benign (-SHAP)  |  Towards Malignant (+SHAP) →", font: { size: 10 } }
              },
              y: { ticks: { font: { size: 10.5 } } }
            },
            plugins: {
              legend: { display: false }
            }
          }
        });
      }
    }

    pSelect.addEventListener("change", (e) => loadPatientXai(e.target.value));

    document.getElementById("btn-preset-a").addEventListener("click", () => {
      pSelect.value = "PT-250";
      loadPatientXai("PT-250");
    });
    document.getElementById("btn-preset-b").addEventListener("click", () => {
      pSelect.value = "PT-345";
      loadPatientXai("PT-345");
    });
    document.getElementById("btn-preset-c").addEventListener("click", () => {
      pSelect.value = "PT-385";
      loadPatientXai("PT-385");
    });

    loadPatientXai("PT-250");
  }

  function initPatientExplorer() {
    const tbody = document.getElementById("tbody-pe");
    const countDisplay = document.getElementById("pe-count-display");
    const searchInput = document.getElementById("pe-search");
    const filterActual = document.getElementById("pe-filter-actual");
    const filterOutcome = document.getElementById("pe-filter-outcome");
    const filterCorrect = document.getElementById("pe-filter-correct");

    function getFilteredList() {
      const q = searchInput.value.trim().toUpperCase();
      const fAct = filterActual.value;
      const fOut = filterOutcome.value;
      const fCor = filterCorrect.value;

      return data.holdout.filter(p => {
        if (q && !p["Patient ID"].includes(q)) return false;
        if (fAct !== "ALL" && p["Actual Diagnosis"] !== fAct) return false;
        if (fOut !== "ALL" && p["Calibrated Outcome"] !== fOut) return false;
        if (fCor !== "ALL" && p["Calibrated Correct"] !== fCor) return false;
        return true;
      });
    }

    function renderPeTable() {
      const list = getFilteredList();
      countDisplay.textContent = list.length;
      tbody.innerHTML = "";

      list.forEach(p => {
        const tr = document.createElement("tr");
        const isMal = p["Actual Diagnosis"] === "Malignant";
        const prob = (p["Malignancy Probability"] * 100).toFixed(1);

        let outBadge = "badge-tp";
        if (p["Calibrated Outcome"] === "True Negative") outBadge = "badge-tn";
        else if (p["Calibrated Outcome"] === "False Positive") outBadge = "badge-fp";
        else if (p["Calibrated Outcome"] === "False Negative") outBadge = "badge-fn";

        tr.innerHTML = `
          <td><strong>${p["Patient ID"]}</strong></td>
          <td><span class="badge ${isMal ? 'badge-malignant' : 'badge-benign'}">${p["Actual Diagnosis"]}</span></td>
          <td>${p["Predicted Diagnosis (Calibrated)"]}</td>
          <td style="font-weight: 600; color: ${prob > 38 ? 'var(--wine-malignant)' : 'var(--sage-benign)'};">${prob}%</td>
          <td>${p["Prediction Confidence (%)"]}%</td>
          <td><span class="badge ${outBadge}">${p["Calibrated Outcome"]}</span></td>
          <td style="font-size: 11px;">${p["Top Influencing Feature"]}</td>
        `;

        tr.addEventListener("click", () => {
          document.querySelectorAll("#tbody-pe tr").forEach(r => r.classList.remove("selected"));
          tr.classList.add("selected");
          showPatientDetail(p);
        });

        tbody.appendChild(tr);
      });

      if (list.length > 0) showPatientDetail(list[0]);
    }

    function showPatientDetail(p) {
      document.getElementById("pe-sel-id").textContent = p["Patient ID"];
      document.getElementById("pe-sel-meta").textContent = `Holdout Index #${p["Row Index"]} | Actual: ${p["Actual Diagnosis"]}`;

      const badge = document.getElementById("pe-sel-badge");
      badge.textContent = p["Calibrated Outcome"];
      badge.className = p["Calibrated Outcome"] === "False Negative" ? "badge badge-fn" : (p["Calibrated Outcome"] === "False Positive" ? "badge badge-fp" : (p["Calibrated Outcome"] === "True Negative" ? "badge badge-tn" : "badge badge-tp"));

      const prob = (p["Malignancy Probability"] * 100).toFixed(1);
      document.getElementById("pe-sel-prob").textContent = `${prob}%`;
      document.getElementById("pe-sel-conf").textContent = `${p["Prediction Confidence (%)"]}%`;

      document.getElementById("pe-feat-perim").textContent = `${p["worst perimeter"]} μm`;
      document.getElementById("pe-feat-conc").textContent = p["mean concave points"];
      document.getElementById("pe-feat-text").textContent = p["worst texture"];
      document.getElementById("pe-feat-area").textContent = p["area error"];
      document.getElementById("pe-feat-smooth").textContent = p["worst smoothness"];

      const callout = document.getElementById("pe-sel-callout");
      if (p["Calibrated Outcome"] === "False Negative") {
        callout.innerHTML = `<strong>Critical Misclassification:</strong> True malignant tumor classified as benign. Primary cellular driver was atypical low nuclear perimeter (${p["worst perimeter"]}).`;
        callout.className = "clinical-callout wine";
      } else if (p["Actual Diagnosis"] === "Malignant") {
        callout.innerHTML = `Aggressive cytology: ${p["Top Influencing Feature"]} drove malignant probability to ${prob}%. Precision calibrated diagnosis.`;
        callout.className = "clinical-callout wine";
      } else {
        callout.innerHTML = `Benign mass correctly confirmed. Uniform cell dimensions and low concavity anchor prediction.`;
        callout.className = "clinical-callout";
      }
    }

    searchInput.addEventListener("input", renderPeTable);
    filterActual.addEventListener("change", renderPeTable);
    filterOutcome.addEventListener("change", renderPeTable);
    filterCorrect.addEventListener("change", renderPeTable);

    renderPeTable();
  }
});
