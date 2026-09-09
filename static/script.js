// Register Chart.js Data Labels Plugin
Chart.register(ChartDataLabels);

/* ==========================================================================
   1. State & Constants Configuration
   ========================================================================== */
const PRESETS = {
  scholar: {
    gender: "Male",
    study_time_hours: 5.9,
    attendance_percent: 100.0,
    sleep_hours: 7.3,
    parental_education: "Bachelors",
    internet_access: "Yes",
    extracurricular_activities: "Yes",
    part_time_job: "No",
    previous_grade: 89.8,
  },
  turnaround: {
    gender: "Male",
    study_time_hours: 4.6,
    attendance_percent: 100.0,
    sleep_hours: 5.8,
    parental_education: "Masters",
    internet_access: "Yes",
    extracurricular_activities: "Yes",
    part_time_job: "No",
    previous_grade: 37.5,
  },
  at_risk: {
    gender: "Female",
    study_time_hours: 1.9,
    attendance_percent: 69.0,
    sleep_hours: 6.0,
    parental_education: "High School",
    internet_access: "Yes",
    extracurricular_activities: "No",
    part_time_job: "Yes",
    previous_grade: 34.5,
  },
  multitasker: {
    gender: "Female",
    study_time_hours: 6.3,
    attendance_percent: 100.0,
    sleep_hours: 5.7,
    parental_education: "High School",
    internet_access: "Yes",
    extracurricular_activities: "Yes",
    part_time_job: "Yes",
    previous_grade: 75.5,
  },
  passive: {
    gender: "Male",
    study_time_hours: 1.5,
    attendance_percent: 100.0,
    sleep_hours: 5.0,
    parental_education: "High School",
    internet_access: "Yes",
    extracurricular_activities: "Yes",
    part_time_job: "Yes",
    previous_grade: 58.6,
  },
  baseline: {
    gender: "Male",
    study_time_hours: 3.6,
    attendance_percent: 85.0,
    sleep_hours: 6.8,
    parental_education: "High School",
    internet_access: "Yes",
    extracurricular_activities: "No",
    part_time_job: "No",
    previous_grade: 69.7,
  },
};

const BEST_MODEL_NAME = "Bayesian Ridge";
let predictionChart = null;
let explorerChart = null;
let cachedMetadata = null;
let currentPredictions = [];
let currentBatchData = null;
let rowRadarChart = null;
const tooltipContainer = document.getElementById("rowRadarTooltip");

// Model Theme Color Mappings
const MODEL_PALETTE = {
  "Bayesian Ridge": {
    bg: "#0d6efd",
    border: "#0a58ca",
    secondaryBg: "#6ea8fe",
  },
  CatBoost: { bg: "#fd7e14", border: "#ca6510", secondaryBg: "#fecba1" },
  LightGBM: { bg: "#198754", border: "#146c43", secondaryBg: "#75b798" },
  XGBoost: { bg: "#d63384", border: "#ab296a", secondaryBg: "#e685b5" },
  "Voting Reg.": { bg: "#6f42c1", border: "#59359a", secondaryBg: "#a98eda" },
  "Random Forest": { bg: "#20c997", border: "#1aa179", secondaryBg: "#7ee2c4" },
  "Linear Regression": {
    bg: "#0dcaf0",
    border: "#0aa2c0",
    secondaryBg: "#6edff6",
  },
};

const FALLBACK_COLORS = [
  { bg: "#0d6efd", border: "#0a58ca", secondaryBg: "#6ea8fe" },
  { bg: "#fd7e14", border: "#ca6510", secondaryBg: "#fecba1" },
  { bg: "#198754", border: "#146c43", secondaryBg: "#75b798" },
  { bg: "#d63384", border: "#ab296a", secondaryBg: "#e685b5" },
  { bg: "#6f42c1", border: "#59359a", secondaryBg: "#a98eda" },
];

/* ==========================================================================
   2. Helper Utilities & Event Binding
   ========================================================================== */
function getModelColors(modelName, index = 0) {
  if (MODEL_PALETTE[modelName]) return MODEL_PALETTE[modelName];
  for (const key in MODEL_PALETTE) {
    if (modelName.toLowerCase().includes(key.toLowerCase())) {
      return MODEL_PALETTE[key];
    }
  }
  return FALLBACK_COLORS[index % FALLBACK_COLORS.length];
}

function updateLabel(id, text) {
  const badge = document.getElementById(id + "_val");
  if (badge) badge.innerText = text;
}

function getRadioValue(name) {
  const checked = document.querySelector(`input[name="${name}"]:checked`);
  return checked ? checked.value : "";
}

function setRadioValue(name, value) {
  const target = document.querySelector(
    `input[name="${name}"][value="${value}"]`,
  );
  if (target) target.checked = true;
}

// Global Dark/Light Mode Theme Toggle
function toggleTheme() {
  const htmlTag = document.documentElement;
  const currentTheme = htmlTag.getAttribute("data-bs-theme");
  const newTheme = currentTheme === "dark" ? "light" : "dark";

  htmlTag.setAttribute("data-bs-theme", newTheme);
  document.getElementById("themeToggleBtn").innerText =
    newTheme === "dark" ? "☀️ Light Mode" : "🌙 Dark Mode";

  if (currentPredictions.length) renderChart(currentPredictions);
  if (cachedMetadata) updateExplorerChart();
}

// Applies archetypal input presets
function applyPreset(presetKey) {
  const preset = PRESETS[presetKey];
  if (!preset) return;

  setRadioValue("gender", preset.gender);
  setRadioValue("parental_education", preset.parental_education);
  setRadioValue("internet_access", preset.internet_access);
  setRadioValue(
    "extracurricular_activities",
    preset.extracurricular_activities,
  );
  setRadioValue("part_time_job", preset.part_time_job);

  document.getElementById("study_time_hours").value = preset.study_time_hours;
  document.getElementById("attendance_percent").value =
    preset.attendance_percent;
  document.getElementById("sleep_hours").value = preset.sleep_hours;
  document.getElementById("previous_grade").value = preset.previous_grade;

  updateLabel("study_time_hours", preset.study_time_hours + " hrs");
  updateLabel("attendance_percent", preset.attendance_percent);
  updateLabel("sleep_hours", preset.sleep_hours + " hrs");
  updateLabel("previous_grade", preset.previous_grade);

  runPrediction();
}

/* ==========================================================================
   3. Single Record Prediction Engine
   ========================================================================== */
async function runPrediction() {
  const errorAlert = document.getElementById("errorAlert");
  const loadingIndicator = document.getElementById("loadingIndicator");

  loadingIndicator.classList.remove("d-none");
  errorAlert.classList.add("d-none");

  const payload = {
    gender: getRadioValue("gender"),
    study_time_hours:
      parseFloat(document.getElementById("study_time_hours").value) || 0.0,
    attendance_percent:
      parseFloat(document.getElementById("attendance_percent").value) || 0.0,
    sleep_hours:
      parseFloat(document.getElementById("sleep_hours").value) || 0.0,
    parental_education: getRadioValue("parental_education"),
    internet_access: getRadioValue("internet_access"),
    extracurricular_activities: getRadioValue("extracurricular_activities"),
    part_time_job: getRadioValue("part_time_job"),
    previous_grade:
      parseFloat(document.getElementById("previous_grade").value) || 0.0,
  };

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const result = await response.json();

    if (result.success && result.predictions) {
      updateDashboardSummary(result.predictions);
      renderChart(result.predictions);
    } else {
      errorAlert.innerText = result.error || "Failed to generate predictions.";
      errorAlert.classList.remove("d-none");
    }
  } catch (err) {
    errorAlert.innerText =
      "Server error encountered. Verify Flask server logs.";
    errorAlert.classList.remove("d-none");
  } finally {
    loadingIndicator.classList.add("d-none");
  }
}

function updateDashboardSummary(predictions) {
  const tableBody = document.getElementById("predictionsTableBody");
  tableBody.innerHTML = "";

  const bestModelData =
    predictions.find((p) => p.model === BEST_MODEL_NAME) || predictions[0];
  const highestScoreModelData = predictions.reduce((prev, current) => {
    const prevScore = typeof prev.score === "number" ? prev.score : -1;
    const currScore = typeof current.score === "number" ? current.score : -1;
    return currScore > prevScore ? current : prev;
  }, predictions[0]);

  document.getElementById("bestModelName").innerText = bestModelData.model;
  document.getElementById("bestModelScore").innerText =
    bestModelData.score !== "N/A" ? `${bestModelData.score}` : "N/A";
  const bestBadge = document.getElementById("bestModelGrade");
  bestBadge.className = `grade-badge grade-${bestModelData.grade}`;
  bestBadge.innerText = bestModelData.grade;

  document.getElementById("highestScoreModelName").innerText =
    highestScoreModelData.model;
  document.getElementById("highestScoreVal").innerText =
    highestScoreModelData.score !== "N/A"
      ? `${highestScoreModelData.score}`
      : "N/A";
  const highestBadge = document.getElementById("highestScoreGrade");
  highestBadge.className = `grade-badge grade-${highestScoreModelData.grade}`;
  highestBadge.innerText = highestScoreModelData.grade;

  predictions.forEach((item) => {
    const row = document.createElement("tr");
    const badgeClass = `grade-${item.grade}`;
    const displayScore = item.score !== "N/A" ? `${item.score} / 100` : "N/A";

    row.innerHTML = `
            <td class="fw-semibold">${item.model}</td>
            <td>${displayScore}</td>
            <td><span class="grade-badge ${badgeClass}">${item.grade}</span></td>
        `;
    tableBody.appendChild(row);
  });
}

function renderChart(predictions) {
  currentPredictions = predictions;
  const canvas = document.getElementById("predictionChart");
  const ctx = canvas.getContext("2d");

  const isDark =
    document.documentElement.getAttribute("data-bs-theme") === "dark";
  const tickColor = isDark ? "#94a3b8" : "#6c757d";
  const gridColor = isDark ? "#334155" : "#e9ecef";
  const labelColor = isDark ? "#f8fafc" : "#212529";

  const labels = predictions.map((p) => p.model);
  const scores = predictions.map((p) =>
    typeof p.score === "number" ? p.score : 0,
  );
  const grades = predictions.map((p) => p.grade || "N/A");

  const minVal = Math.min(...scores);
  const maxVal = Math.max(...scores);
  const rMin = Math.max(0, Math.floor(minVal - 6));
  const rMax = Math.min(105, Math.ceil(maxVal + 6));

  if (predictionChart) predictionChart.destroy();

  predictionChart = new Chart(ctx, {
    type: "radar",
    data: {
      labels: labels,
      datasets: [
        {
          label: "Predicted Score",
          data: scores,
          backgroundColor: isDark
            ? "rgba(13, 110, 253, 0.35)"
            : "rgba(13, 110, 253, 0.2)",
          borderColor: "#0d6efd",
          borderWidth: 2,
          pointBackgroundColor: labels.map(
            (name, idx) => getModelColors(name, idx).bg,
          ),
          pointBorderColor: labels.map(
            (name, idx) => getModelColors(name, idx).border,
          ),
          pointHoverBackgroundColor: "#ffffff",
          pointHoverBorderColor: "#0d6efd",
          pointRadius: 6,
          pointHoverRadius: 8,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 500 },
      scales: {
        r: {
          min: rMin,
          max: rMax,
          ticks: {
            stepSize: Math.ceil((rMax - rMin) / 4) || 5,
            font: { size: 11 },
            color: tickColor,
            backdropColor: "transparent",
          },
          grid: { color: gridColor },
          angleLines: { color: gridColor },
          pointLabels: {
            font: { size: 12, weight: "600" },
            color: labelColor,
          },
        },
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: function (context) {
              return ` Score: ${context.raw.toFixed(1)} | Grade: ${grades[context.dataIndex]}`;
            },
          },
        },
        datalabels: {
          anchor: "end",
          align: "top",
          offset: 2,
          font: { weight: "bold", size: 12 },
          color: isDark ? "#6ea8fe" : "#0d6efd",
          formatter: (val) => val.toFixed(1),
        },
      },
    },
  });
}

/* ==========================================================================
   4. CSV Batch Import & Tooltip Handling
   ========================================================================== */
async function handleCSVUpload(event) {
  const file = event.target.files[0];
  if (!file) return;

  const errorAlert = document.getElementById("errorAlert");
  const loadingIndicator = document.getElementById("loadingIndicator");

  loadingIndicator.classList.remove("d-none");
  errorAlert.classList.add("d-none");

  const formData = new FormData();
  formData.append("file", file);

  try {
    const response = await fetch("/upload_csv", {
      method: "POST",
      body: formData,
    });
    const result = await response.json();

    if (result.success) {
      currentBatchData = result;
      renderBatchResultsTable(result);
      if (result.aggregated_predictions) {
        updateDashboardSummary(result.aggregated_predictions);
        renderChart(result.aggregated_predictions);
      }
    } else {
      let errMsg = result.error || "Failed to evaluate CSV dataset.";
      if (result.details?.length)
        errMsg += " Details: " + result.details.join(" | ");
      errorAlert.innerText = errMsg;
      errorAlert.classList.remove("d-none");
    }
  } catch (err) {
    errorAlert.innerText =
      "Error submitting file to server. Verify server connectivity.";
    errorAlert.classList.remove("d-none");
  } finally {
    loadingIndicator.classList.add("d-none");
    event.target.value = "";
  }
}

function renderBatchResultsTable(data) {
  const resultsSection = document.getElementById("csvResultsSection");
  const summaryText = document.getElementById("csvSummaryText");
  const headerRow = document.getElementById("csvTableHeaderRow");
  const tbody = document.getElementById("csvTableBody");

  summaryText.innerText = `Evaluated ${data.total_records} rows from dataset. Displaying predictions for sample records:`;

  if (!data.sample_row_predictions?.length) return;

  const firstSample = data.sample_row_predictions[0];
  const modelNames = Object.keys(firstSample.predictions);

  headerRow.innerHTML = `
        <th scope="col" class="text-nowrap">#</th>
        <th scope="col" class="text-nowrap">Gender</th>
        <th scope="col" class="text-nowrap">Study Hrs</th>
        <th scope="col" class="text-nowrap">Attendance (%)</th>
        <th scope="col" class="text-nowrap">Sleep Hrs</th>
        <th scope="col" class="text-nowrap">Parental Edu.</th>
        <th scope="col" class="text-nowrap">Internet Access</th>
        <th scope="col" class="text-nowrap">Extracurriculars</th>
        <th scope="col" class="text-nowrap">Part-Time Job</th>
        <th scope="col" class="text-nowrap">Prev Grade</th>
    `;

  modelNames.forEach((model) => {
    headerRow.appendChild(
      createTableHeader(`${model} Score`, "text-primary text-nowrap"),
    );
    headerRow.appendChild(
      createTableHeader(`${model} Grade`, "text-primary text-nowrap"),
    );
  });

  tbody.innerHTML = "";
  data.sample_row_predictions.forEach((item) => {
    const tr = document.createElement("tr");
    tr.style.cursor = "pointer";
    const inp = item.inputs;

    let rowHtml = `
            <td class="fw-bold">${item.row_index}</td>
            <td>${inp.gender || "-"}</td>
            <td>${inp.study_time_hours ?? "-"}</td>
            <td>${inp.attendance_percent ?? "-"}</td>
            <td>${inp.sleep_hours ?? "-"}</td>
            <td>${inp.parental_education || "-"}</td>
            <td>${inp.internet_access || "-"}</td>
            <td>${inp.extracurricular_activities || "-"}</td>
            <td>${inp.part_time_job || "-"}</td>
            <td>${inp.previous_grade ?? "-"}</td>
        `;

    modelNames.forEach((model) => {
      const pred = item.predictions[model];
      if (pred && pred.score !== "N/A") {
        rowHtml += `<td class="fw-semibold text-nowrap">${pred.score}</td>
                            <td><span class="grade-badge grade-${pred.grade}">${pred.grade}</span></td>`;
      } else {
        rowHtml += `<td><span class="badge bg-secondary">N/A</span></td>
                            <td><span class="badge bg-secondary">N/A</span></td>`;
      }
    });

    tr.innerHTML = rowHtml;

    // Hover Radar Tooltip Listeners
    tr.addEventListener("mouseenter", (e) => {
      renderRowRadarTooltip(item.predictions);
      tooltipContainer.style.display = "block";
      updateTooltipPosition(e);
    });

    tr.addEventListener("mousemove", updateTooltipPosition);
    tr.addEventListener(
      "mouseleave",
      () => (tooltipContainer.style.display = "none"),
    );

    tbody.appendChild(tr);
  });

  resultsSection.classList.remove("d-none");
  resultsSection.scrollIntoView({ behavior: "smooth" });
}

function createTableHeader(text, className) {
  const th = document.createElement("th");
  th.scope = "col";
  th.className = className;
  th.innerText = text;
  return th;
}

// Hover Radar Chart inside Tooltip Box
function renderRowRadarTooltip(modelPredictions) {
  const canvas = document.getElementById("rowRadarCanvas");
  const ctx = canvas.getContext("2d");
  const isDark =
    document.documentElement.getAttribute("data-bs-theme") === "dark";

  const labels = Object.keys(modelPredictions);
  const scores = labels.map((m) =>
    typeof modelPredictions[m]?.score === "number"
      ? modelPredictions[m].score
      : 0,
  );

  const minVal = Math.min(...scores);
  const maxVal = Math.max(...scores);
  const rMin = Math.max(0, Math.floor(minVal - 5));
  const rMax = Math.min(100, Math.ceil(maxVal + 5));

  if (rowRadarChart) rowRadarChart.destroy();

  rowRadarChart = new Chart(ctx, {
    type: "radar",
    data: {
      labels: labels,
      datasets: [
        {
          label: "Prediction",
          data: scores,
          backgroundColor: isDark
            ? "rgba(13, 110, 253, 0.35)"
            : "rgba(13, 110, 253, 0.2)",
          borderColor: "#0d6efd",
          borderWidth: 2,
          pointBackgroundColor: labels.map(
            (name, idx) => getModelColors(name, idx).bg,
          ),
          pointRadius: 4,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      scales: {
        r: {
          min: rMin,
          max: rMax,
          ticks: {
            display: false,
            stepSize: Math.ceil((rMax - rMin) / 3) || 5,
          },
          grid: { color: isDark ? "#334155" : "#e9ecef" },
          pointLabels: {
            font: { size: 9, weight: "600" },
            color: isDark ? "#f8fafc" : "#212529",
          },
        },
      },
      plugins: {
        legend: { display: false },
        datalabels: {
          display: true,
          align: "top",
          anchor: "end",
          offset: 2,
          color: isDark ? "#ffffff" : "#0d6efd",
          font: { size: 10, weight: "bold" },
          formatter: (val) => (typeof val === "number" ? val.toFixed(1) : val),
        },
      },
    },
  });
}

function updateTooltipPosition(e) {
  const tooltipWidth = 280;
  const tooltipHeight = 280;
  const offset = 15;

  let left = e.pageX + offset;
  let top = e.pageY + offset;

  if (left + tooltipWidth > window.innerWidth + window.scrollX) {
    left = e.pageX - tooltipWidth - offset;
  }
  if (top + tooltipHeight > window.innerHeight + window.scrollY) {
    top = e.pageY - tooltipHeight - offset;
  }

  tooltipContainer.style.left = `${left}px`;
  tooltipContainer.style.top = `${top}px`;
}

function exportBatchTableToCSV(filename = "batch_analysis_results.csv") {
  if (!currentBatchData?.sample_row_predictions?.length) {
    alert("No batch dataset available to export.");
    return;
  }

  const records =
    currentBatchData.full_row_predictions ||
    currentBatchData.sample_row_predictions;
  const firstSample = records[0];
  const modelNames = Object.keys(firstSample.predictions || {});

  let csvRows = [];
  const inputColumns = [
    { header: "Gender", key: "gender" },
    { header: "Study Hrs", key: "study_time_hours" },
    { header: "Attendance", key: "attendance_percent" },
    { header: "Sleep Hrs", key: "sleep_hours" },
    { header: "Parental Education", key: "parental_education" },
    { header: "Internet Access", key: "internet_access" },
    { header: "Extracurricular Activities", key: "extracurricular_activities" },
    { header: "Part Time Job", key: "part_time_job" },
    { header: "Prev Score", key: "previous_grade" },
  ];

  const headers = ["# Row", ...inputColumns.map((c) => c.header)];
  modelNames.forEach((m) => {
    headers.push(`${m} Score`, `${m} Grade`);
  });
  csvRows.push(
    headers.map((h) => `"${String(h).replace(/"/g, '""')}"`).join(","),
  );

  records.forEach((item) => {
    const inp = item.inputs || {};
    const rowData = [item.row_index];

    inputColumns.forEach((col) => rowData.push(inp[col.key] ?? ""));
    modelNames.forEach((m) => {
      const pred = item.predictions?.[m];
      if (pred && pred.score !== "N/A" && pred.score !== undefined) {
        rowData.push(pred.score, pred.grade);
      } else {
        rowData.push("N/A", "N/A");
      }
    });

    csvRows.push(
      rowData.map((v) => `"${String(v).replace(/"/g, '""')}"`).join(","),
    );
  });

  const blob = new Blob([csvRows.join("\n")], {
    type: "text/csv;charset=utf-8;",
  });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");

  link.href = url;
  link.setAttribute("download", filename);
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

/* ==========================================================================
   5. Offline Model Metrics Explorer
   ========================================================================== */
async function toggleExplorerSection() {
  const explorerSec = document.getElementById("explorerSection");
  const btnIcon = document.getElementById("explorerBtnIcon");
  const btnText = document.getElementById("explorerBtnText");

  if (explorerSec.classList.contains("d-none")) {
    explorerSec.classList.remove("d-none");
    btnIcon.innerText = "🔒";
    btnText.innerText = "Hide Model Evaluation Metrics";

    if (!cachedMetadata) {
      await loadModelMetadata();
    } else {
      updateExplorerChart();
    }
  } else {
    explorerSec.classList.add("d-none");
    btnIcon.innerText = "🔍";
    btnText.innerText = "Explore Model Evaluation Metrics";
  }
}

async function loadModelMetadata() {
  try {
    const response = await fetch("/metadata");
    const data = await response.json();
    if (data.success && data.models) {
      cachedMetadata = data.models;
      updateExplorerChart();
    }
  } catch (err) {
    console.error("Failed to load metadata:", err);
  }
}

function updateExplorerChart() {
  if (!cachedMetadata?.length) return;

  const isDark =
    document.documentElement.getAttribute("data-bs-theme") === "dark";
  const tickColor = isDark ? "#94a3b8" : "#6c757d";
  const gridColor = isDark ? "#334155" : "#e9ecef";
  const labelColor = isDark ? "#f8fafc" : "#212529";

  const metric1Key = document.getElementById("metricSelect1").value;
  const metric2Key = document.getElementById("metricSelect2").value;

  const labels = cachedMetadata.map((m) => m.model_name);
  const metric1Data = cachedMetadata.map((m) => m.metrics?.[metric1Key] ?? 0);

  const datasets = [
    {
      label: metric1Key,
      data: metric1Data,
      backgroundColor: labels.map((name, idx) => getModelColors(name, idx).bg),
      borderColor: labels.map((name, idx) => getModelColors(name, idx).border),
      borderWidth: 1.5,
      borderRadius: 6,
      maxBarThickness: metric2Key !== "none" ? 30 : 45,
    },
  ];

  if (metric2Key !== "none") {
    datasets.push({
      label: metric2Key,
      data: cachedMetadata.map((m) => m.metrics?.[metric2Key] ?? 0),
      backgroundColor: labels.map(
        (name, idx) => getModelColors(name, idx).secondaryBg,
      ),
      borderColor: labels.map((name, idx) => getModelColors(name, idx).border),
      borderWidth: 1.5,
      borderRadius: 6,
      maxBarThickness: 30,
    });
  }

  let allActiveValues = [...metric1Data];
  if (metric2Key !== "none") {
    allActiveValues = allActiveValues.concat(
      cachedMetadata.map((m) => m.metrics?.[metric2Key] || 0),
    );
  }
  const maxVal = Math.max(...allActiveValues);

  const canvas = document.getElementById("explorerChart");
  const ctx = canvas.getContext("2d");

  if (explorerChart) explorerChart.destroy();

  explorerChart = new Chart(ctx, {
    type: "bar",
    data: { labels, datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 400 },
      layout: { padding: { top: 25 } },
      scales: {
        x: {
          grid: { display: false },
          ticks: { font: { size: 12, weight: "600" }, color: labelColor },
        },
        y: {
          suggestedMax: maxVal > 0 ? maxVal * 1.15 : 1,
          grid: { color: gridColor, strokeDash: [4, 4] },
          ticks: { font: { size: 11 }, color: tickColor },
        },
      },
      plugins: {
        legend: {
          display: true,
          position: "top",
          labels: { font: { size: 12, weight: "600" }, color: labelColor },
        },
        datalabels: {
          anchor: "end",
          align: "top",
          offset: 2,
          font: { weight: "bold", size: 11 },
          color: labelColor,
          formatter: (val) =>
            typeof val !== "number"
              ? "N/A"
              : val < 1
                ? val.toFixed(4)
                : val.toFixed(2),
        },
      },
    },
  });
}

/* ==========================================================================
   6. DOM Ready Listener
   ========================================================================== */
document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll(".live-input").forEach((input) => {
    if (input.type === "range") {
      input.addEventListener("input", (e) => {
        const id = e.target.id;
        const suffix =
          id === "study_time_hours" || id === "sleep_hours" ? " hrs" : "";
        updateLabel(id, e.target.value + suffix);
      });
    }
    input.addEventListener("change", runPrediction);
  });

  const radarTabBtn = document.getElementById("radar-tab");
  if (radarTabBtn) {
    radarTabBtn.addEventListener("shown.bs.tab", () => {
      if (predictionChart) predictionChart.resize();
    });
  }

  runPrediction();
});
