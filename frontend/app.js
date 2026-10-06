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
    // Class Distribution Donut
    const ctxClass = document.getElementById("chart-class-dist");
    if (ctxClass && typeof Chart !== "undefined") {
      new Chart(ctxClass, {
        type: "doughnut",
        data: {
          labels: ["Benign (72)", "Malignant (42)"],
          datasets: [{
            data: [72, 42],
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
          cutout: "68%"
        }
      });
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
      pSelect.value = "PT-033";
      loadPatientXai("PT-033");
    });
    document.getElementById("btn-preset-b").addEventListener("click", () => {
      pSelect.value = "PT-049";
      loadPatientXai("PT-049");
    });
    document.getElementById("btn-preset-c").addEventListener("click", () => {
      pSelect.value = "PT-086";
      loadPatientXai("PT-086");
    });

    loadPatientXai("PT-033");
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
      badge.className = p["Calibrated Outcome"] === "False Negative" ? "badge badge-fn" : (p["Calibrated Outcome"] === "False Positive" ? "badge badge-fp" : "badge badge-tp");

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
