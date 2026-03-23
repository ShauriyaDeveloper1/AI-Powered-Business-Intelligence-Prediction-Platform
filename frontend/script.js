const API_BASE = "http://127.0.0.1:8000";

let token = localStorage.getItem("ai_token") || "";
let currentUser = localStorage.getItem("ai_user") || "";
let latestUploadId = null;
let previousUploadId = null;
let lastSelectedFile = null;
let trainingTimerId = null;
let trainingTimerStartedAt = null;
let trainingEstimatedTotalSeconds = null;
let authMode = "login";
let isHistoryMode = false;
let predictionHistoryCache = [];
let predictionHistoryCurrentUploadId = null;
let predictionHistoryFilters = {
    task: "all",
    datasetQuery: "",
    last30DaysOnly: false,
};

const TRAINING_ESTIMATE_STORAGE_KEY = "ai_training_eta_v1";
const MB_BYTES = 1024 * 1024;

const DATASET_VALIDATION_GUIDE = {
    churn: {
        required: ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen", "Churn"],
        rules: [
            "Minimum rows: 50",
            "Target column: Churn (last column)",
            "No missing values in required columns",
            "Churn must contain only 0 or 1",
        ],
    },
    sales: {
        required: ["Quantity", "Discount", "Profit", "Sales"],
        rules: [
            "Minimum rows: 50",
            "Target column: Sales (last column)",
            "All required columns must be numeric",
            "Sales cannot be negative",
        ],
    },
    segmentation: {
        required: ["tenure", "MonthlyCharges", "TotalCharges"],
        rules: [
            "Minimum rows: 50",
            "No target column required",
            "Only numeric values in required columns",
            "No missing values in required columns",
        ],
    },
};

function renderDatasetRequirements(analysisType) {
    const guide = DATASET_VALIDATION_GUIDE[analysisType] || DATASET_VALIDATION_GUIDE.churn;
    const columnsNode = document.getElementById("requirementsColumns");
    const rulesNode = document.getElementById("requirementsRules");

    if (columnsNode) {
        columnsNode.textContent = `Required columns: ${guide.required.join(", ")}`;
    }

    if (rulesNode) {
        rulesNode.innerHTML = guide.rules.map((rule) => `<li>${rule}</li>`).join("");
    }
}

let featureChart = null;
let clusterDistributionChart = null;
let clusterScatterChart = null;
let modelMetricChart = null;
let churnChart = null;
let salesTrendChart = null;

const featurePalette = [
    "linear-gradient(90deg, #22c1c3, #17a4db)",
    "linear-gradient(90deg, #7ed957, #42b883)",
    "linear-gradient(90deg, #7f5af0, #5a49b8)",
    "linear-gradient(90deg, #ff9f1c, #f47f32)",
    "linear-gradient(90deg, #00c2ff, #2f80ed)",
];

const loginSection = document.getElementById("loginSection");
const dashboardSection = document.getElementById("dashboardSection");
const logoutBtn = document.getElementById("logoutBtn");
const welcomeUserNode = document.getElementById("welcomeUser");
const analysisTypeSelect = document.getElementById("analysisType");
const topbar = document.getElementById("topbar");
const toastContainer = document.getElementById("toastContainer");
const exitHistoryBtn = document.getElementById("exitHistoryBtn");

function getActiveAnalysisType() {
    return analysisTypeSelect?.value || "churn";
}

function analysisTypeToTaskType(analysisType) {
    if (analysisType === "sales") {
        return "regression";
    }
    return "classification";
}

function taskTypeToAnalysisType(taskType) {
    if (taskType === "regression") {
        return "sales";
    }
    return "churn";
}

function setNodeVisibility(id, visible) {
    const node = document.getElementById(id);
    if (!node) {
        return;
    }
    node.classList.toggle("hidden", !visible);
}

function updateFeatureSectionTitles(mode) {
    const listTitle = document.getElementById("featureListTitle");
    const chartTitle = document.getElementById("featureChartTitle");

    if (!listTitle || !chartTitle) {
        return;
    }

    if (mode === "sales") {
        listTitle.textContent = "Top Sales Drivers";
        chartTitle.textContent = "Sales Driver Chart";
        return;
    }

    if (mode === "segmentation") {
        listTitle.textContent = "Segment Drivers";
        chartTitle.textContent = "Segment Driver Chart";
        return;
    }

    listTitle.textContent = "Feature Importance";
    chartTitle.textContent = "Feature Importance Chart";
}

function renderClusterSummary(summary = []) {
    const box = document.getElementById("clusterSummaryWrap");
    if (!box) {
        return;
    }

    if (!summary.length) {
        box.className = "summary-list muted";
        box.innerHTML = "Run training to view segment summary.";
        return;
    }

    box.className = "summary-list";
    box.innerHTML = summary
        .map((item) => `<div class="summary-item"><strong>${item.label}</strong><span>${formatNumber(item.count)}</span></div>`)
        .join("");
}

function configureModuleView() {
    const mode = getActiveAnalysisType();

    const isChurn = mode === "churn";
    const isSales = mode === "sales";
    const isSegmentation = mode === "segmentation";
    const showFeatureImportance = isChurn || isSales || isSegmentation;

    setNodeVisibility("predictionPanel", !isSegmentation);
    setNodeVisibility("featureListPanel", showFeatureImportance);
    setNodeVisibility("featureChartPanel", showFeatureImportance);
    setNodeVisibility("churnChartPanel", isChurn);
    setNodeVisibility("salesTrendSection", isSales);
    setNodeVisibility("modelMetricsPanel", !isSegmentation);
    setNodeVisibility("clusterScatterPanel", isSegmentation);
    updateFeatureSectionTitles(mode);

    const label1 = document.getElementById("kpiLabel1");
    const label2 = document.getElementById("kpiLabel2");
    const label3 = document.getElementById("kpiLabel3");
    const label4 = document.getElementById("kpiLabel4");

    if (isSales) {
        label1.textContent = "MSE Score";
        label2.textContent = "Processed Records";
        label3.textContent = "Best Model";
        label4.textContent = "Predictions Made";
    } else if (isSegmentation) {
        label1.textContent = "Total Records";
        label2.textContent = "Segments";
        label3.textContent = "Largest Segment";
        label4.textContent = "Predictions Made";
    } else {
        label1.textContent = "Model Accuracy";
        label2.textContent = "Precision";
        label3.textContent = "Recall";
        label4.textContent = "Predictions Made";
    }

    const modeText = isSales
        ? "Sales Forecast mode active"
        : (isSegmentation ? "Customer Segmentation mode active" : "Customer Churn mode active");
    renderDatasetRequirements(mode);
    setStatus("uploadStatus", modeText, "success");
}

function setStatus(id, text, type = "") {
    const node = document.getElementById(id);
    node.textContent = text;
    node.className = `status ${type}`;
}

function setHistoryMode(active, uploadId = null) {
    isHistoryMode = active;

    const uploadPanel = document.getElementById("uploadPanel");
    const historyBanner = document.getElementById("historyBanner");
    const historyBannerLabel = document.getElementById("historyBannerLabel");

    if (uploadPanel) uploadPanel.classList.toggle("hidden", active);
    if (historyBanner) historyBanner.classList.toggle("hidden", !active);

    if (active && uploadId !== null && uploadId !== undefined && historyBannerLabel) {
        historyBannerLabel.textContent = `Viewing historical analysis from run #${uploadId}`;
    }
}

function resetDashboardForNewUpload() {
    latestUploadId = null;

    document.getElementById("statAccuracy").textContent = "-";
    document.getElementById("statPrecision").textContent = "-";
    document.getElementById("statRecall").textContent = "-";
    document.getElementById("statRecords").textContent = "0";
    document.getElementById("statChurn").textContent = "-";
    document.getElementById("statBestModel").textContent = "-";

    renderTable("previewWrap", []);
    renderTopFeatureList([]);
    renderClusterSummary([]);
    document.getElementById("comparisonBox").textContent = "No training run yet.";
    document.getElementById("insightText").textContent = "Upload and train a new dataset to view analytics.";

    destroyChart(featureChart);
    featureChart = null;
    destroyChart(clusterDistributionChart);
    clusterDistributionChart = null;
    destroyChart(clusterScatterChart);
    clusterScatterChart = null;
    destroyChart(modelMetricChart);
    modelMetricChart = null;
    destroyChart(churnChart);
    churnChart = null;
    destroyChart(salesTrendChart);
    salesTrendChart = null;

    updateTrainingTimerBadge(null, false);
    updateRestoreButtonState();
}

function safeParseNumber(value) {
    const num = Number(value);
    return Number.isFinite(num) ? num : null;
}

function loadEtaStats() {
    try {
        const raw = localStorage.getItem(TRAINING_ESTIMATE_STORAGE_KEY);
        if (!raw) {
            return {};
        }
        const parsed = JSON.parse(raw);
        return parsed && typeof parsed === "object" ? parsed : {};
    } catch (_error) {
        return {};
    }
}

function saveEtaStats(stats) {
    try {
        localStorage.setItem(TRAINING_ESTIMATE_STORAGE_KEY, JSON.stringify(stats));
    } catch (_error) {
        // Ignore storage failures silently.
    }
}

function getTaskEtaModel(taskType) {
    const allStats = loadEtaStats();
    const byTask = allStats[taskType];
    if (!byTask || typeof byTask !== "object") {
        return null;
    }

    const secondsPerMb = safeParseNumber(byTask.secondsPerMb);
    const fixedOverhead = safeParseNumber(byTask.fixedOverhead);
    if (secondsPerMb === null || fixedOverhead === null) {
        return null;
    }

    return {
        secondsPerMb: Math.max(0.1, secondsPerMb),
        fixedOverhead: Math.max(0.5, fixedOverhead),
    };
}

function estimateTrainingTotalSeconds(fileSizeBytes, taskType) {
    const fileMb = Math.max(0.05, (fileSizeBytes || 0) / MB_BYTES);
    const model = getTaskEtaModel(taskType);

    if (model) {
        return (fileMb * model.secondsPerMb) + model.fixedOverhead;
    }

    // Initial fallback before any real measurements are learned.
    return 2 + (fileMb * 4.5);
}

function recordTrainingEtaSample(taskType, fileSizeBytes, actualSeconds) {
    if (!taskType || !fileSizeBytes || !actualSeconds || actualSeconds <= 0) {
        return;
    }

    const fileMb = Math.max(0.05, fileSizeBytes / MB_BYTES);
    const sampleSecondsPerMb = Math.max(0.1, actualSeconds / fileMb);
    const sampleOverhead = Math.max(0.5, Math.min(6, actualSeconds * 0.12));
    const stats = loadEtaStats();
    const current = getTaskEtaModel(taskType);

    const nextSecondsPerMb = current
        ? (current.secondsPerMb * 0.7) + (sampleSecondsPerMb * 0.3)
        : sampleSecondsPerMb;
    const nextOverhead = current
        ? (current.fixedOverhead * 0.7) + (sampleOverhead * 0.3)
        : sampleOverhead;

    stats[taskType] = {
        secondsPerMb: Number(nextSecondsPerMb.toFixed(4)),
        fixedOverhead: Number(nextOverhead.toFixed(4)),
    };

    saveEtaStats(stats);
}

function updateTrainingTimerBadge(seconds = null, active = false, finalLabel = "", estimatedTotalSeconds = null) {
    const badge = document.getElementById("trainingTimeBadge");
    if (!badge) {
        return;
    }

    if (finalLabel) {
        badge.textContent = finalLabel;
    } else if (active && typeof seconds === "number") {
        if (typeof estimatedTotalSeconds === "number" && estimatedTotalSeconds > 0) {
            const estimated = Math.max(seconds, estimatedTotalSeconds);
            const remaining = Math.max(0, estimated - seconds);
            badge.textContent = `Elapsed: ${seconds.toFixed(1)}s | Est total: ${estimated.toFixed(1)}s | Remaining: ${remaining.toFixed(1)}s`;
        } else {
            badge.textContent = `Elapsed time: ${seconds.toFixed(1)}s | Estimating total...`;
        }
    } else {
        badge.textContent = "Elapsed time will appear here while training.";
    }

    badge.classList.toggle("active", active || Boolean(finalLabel));
}

function stopTrainingTimer(finalSeconds = null) {
    if (trainingTimerId !== null) {
        window.clearInterval(trainingTimerId);
        trainingTimerId = null;
    }

    if (typeof finalSeconds === "number" && finalSeconds > 0) {
        updateTrainingTimerBadge(finalSeconds, false, `Completed in ${finalSeconds.toFixed(2)}s`);
    } else {
        updateTrainingTimerBadge(null, false);
    }

    trainingTimerStartedAt = null;
    trainingEstimatedTotalSeconds = null;
}

function startTrainingTimer(fileSizeBytes, taskType) {
    stopTrainingTimer();
    trainingTimerStartedAt = Date.now();
    trainingEstimatedTotalSeconds = estimateTrainingTotalSeconds(fileSizeBytes, taskType);
    updateTrainingTimerBadge(0, true, "", trainingEstimatedTotalSeconds);
    trainingTimerId = window.setInterval(() => {
        const elapsedSeconds = (Date.now() - trainingTimerStartedAt) / 1000;
        updateTrainingTimerBadge(elapsedSeconds, true, "", trainingEstimatedTotalSeconds);
    }, 200);
}

function updateRestoreButtonState() {
    const restoreBtn = document.getElementById("restorePreviousBtn");
    if (!restoreBtn) {
        return;
    }

    const canRestore = previousUploadId !== null && previousUploadId !== undefined;
    restoreBtn.disabled = !canRestore;
}

function showToast(message, type = "success") {
    if (!toastContainer || !message) {
        return;
    }

    const toast = document.createElement("div");
    toast.className = `toast ${type}`;
    toast.textContent = message;
    toastContainer.appendChild(toast);

    window.setTimeout(() => {
        toast.style.animation = "toastOut 0.18s ease forwards";
        window.setTimeout(() => toast.remove(), 220);
    }, 2600);
}

function setTableActionBusy(button, busyText, isBusy) {
    if (!(button instanceof HTMLButtonElement)) {
        return;
    }

    if (isBusy) {
        if (!button.dataset.originalText) {
            button.dataset.originalText = button.textContent;
        }
        button.disabled = true;
        button.textContent = busyText;
        button.classList.add("is-busy");
        return;
    }

    button.disabled = false;
    button.textContent = button.dataset.originalText || button.textContent;
    button.classList.remove("is-busy");
}

    function setButtonBusy(button, busyText, isBusy) {
        if (!button) {
            return;
        }

        if (isBusy) {
            if (!button.dataset.originalText) {
                button.dataset.originalText = button.textContent;
            }
            button.disabled = true;
            button.textContent = busyText;
            button.classList.add("is-busy");
            return;
        }

        button.disabled = false;
        button.textContent = button.dataset.originalText || button.textContent;
        button.classList.remove("is-busy");
    }

function summarizeColumnMatch(columnMatch = {}) {
    const mappedCount = Object.keys(columnMatch.mapped_fields || {}).length;
    const missingCount = (columnMatch.missing_filled || []).length;
    const ignoredCount = (columnMatch.ignored_fields || []).length;
    return `mapped:${mappedCount}, filled-missing:${missingCount}, ignored:${ignoredCount}`;
}

function authHeaders() {
    return {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${token}`,
    };
}

function handleUnauthorized(message = "Session expired. Please login again.") {
    logout();
    setStatus("loginStatus", message, "error");
}

async function fetchWithAuth(url, options = {}) {
    const headers = {
        ...(options.headers || {}),
        "Authorization": `Bearer ${token}`,
    };

    const response = await fetch(url, { ...options, headers });
    if (response.status === 401) {
        handleUnauthorized();
        throw new Error("Unauthorized");
    }
    return response;
}

function formatPct(value) {
    if (value === null || value === undefined || Number.isNaN(Number(value))) {
        return "-";
    }
    return `${(Number(value) * 100).toFixed(2)}%`;
}

function formatNumber(value) {
    if (value === null || value === undefined || Number.isNaN(Number(value))) {
        return "-";
    }
    return Number(value).toLocaleString();
}

function formatMetricsSummary(taskType, performanceMetrics = {}) {
    if (!performanceMetrics || typeof performanceMetrics !== "object") {
        return "-";
    }

    if (taskType === "classification") {
        const acc = formatPct(performanceMetrics.accuracy);
        const f1 = formatPct(performanceMetrics.f1);
        return `Accuracy ${acc}, F1 ${f1}`;
    }

    if (taskType === "regression") {
        return `MSE ${formatNumber(performanceMetrics.mse)}`;
    }

    return Object.entries(performanceMetrics)
        .slice(0, 2)
        .map(([key, val]) => `${key}: ${typeof val === "number" ? formatNumber(val) : String(val)}`)
        .join(", ");
}

function formatPredictionSummary(predictionSummary = {}) {
    const total = formatNumber(predictionSummary.total_predictions || 0);
    const avgProb = predictionSummary.average_probability === null || predictionSummary.average_probability === undefined
        ? "-"
        : formatPct(predictionSummary.average_probability);
    return `${total} predictions, avg probability ${avgProb}`;
}

function formatClusterSummary(clusters = []) {
    if (!Array.isArray(clusters) || clusters.length === 0) {
        return "-";
    }

    const largest = clusters.reduce((max, item) => (item.count > max.count ? item : max), clusters[0]);
    return `${clusters.length} clusters, largest ${largest.label} (${formatNumber(largest.count)})`;
}

function attachSlideNavigation(containerId) {
    const wrap = document.getElementById(containerId);
    if (!wrap) {
        return;
    }

    wrap.addEventListener(
        "wheel",
        (event) => {
            // Keep native wheel/touchpad scrolling, with optional deltaX support from touchpads.
            if (Math.abs(event.deltaX) > 0) {
                wrap.scrollLeft += event.deltaX;
            }
        },
        { passive: true }
    );
}

function extractFilenameFromDisposition(disposition = "") {
    const quoted = disposition.match(/filename="([^"]+)"/i);
    if (quoted && quoted[1]) {
        return quoted[1];
    }
    const plain = disposition.match(/filename=([^;]+)/i);
    if (plain && plain[1]) {
        return plain[1].trim();
    }
    return "ai_training_report.md";
}

function updateAuthView() {
    const isLoggedIn = Boolean(token);
    topbar.classList.toggle("hidden", !isLoggedIn);
    loginSection.classList.toggle("hidden", isLoggedIn);
    dashboardSection.classList.toggle("hidden", !isLoggedIn);
    logoutBtn.classList.toggle("hidden", !isLoggedIn);
    welcomeUserNode.classList.toggle("hidden", !isLoggedIn);
    welcomeUserNode.textContent = isLoggedIn ? `Welcome, ${currentUser || "User"}` : "";
}

function clearAuthState() {
    token = "";
    currentUser = "";
    latestUploadId = null;
    lastSelectedFile = null;
    localStorage.removeItem("ai_token");
    localStorage.removeItem("ai_user");
}

async function validateStoredSession() {
    if (!token) {
        return false;
    }

    try {
        const response = await fetch(`${API_BASE}/dashboard`, {
            headers: { "Authorization": `Bearer ${token}` },
        });

        return response.ok;
    } catch (error) {
        return false;
    }
}

function setAuthMode(mode) {
    authMode = mode;
    const isSignup = mode === "signup";
    const authTitle = document.getElementById("authTitle");
    const authSubTitle = document.getElementById("authSubTitle");
    const showLoginBtn = document.getElementById("showLoginBtn");
    const showSignupBtn = document.getElementById("showSignupBtn");

    if (authTitle) {
        authTitle.textContent = isSignup ? "Sign Up" : "Sign In";
    }
    if (authSubTitle) {
        authSubTitle.textContent = isSignup
            ? "Create your account to get started."
            : "Use your account to continue.";
    }

    showLoginBtn?.classList.toggle("active", !isSignup);
    showSignupBtn?.classList.toggle("active", isSignup);

    document.getElementById("confirmPasswordField").classList.toggle("hidden", !isSignup);
    document.getElementById("signupBtn").classList.toggle("hidden", !isSignup);
    document.getElementById("loginBtn").classList.toggle("hidden", isSignup);
    document.getElementById("helperLoginPrompt").classList.toggle("hidden", !isSignup);
    document.getElementById("helperSignupPrompt").classList.toggle("hidden", isSignup);
    setStatus("loginStatus", "", "");
}

function renderTable(containerId, rows) {
    const wrap = document.getElementById(containerId);
    if (!rows || rows.length === 0) {
        wrap.innerHTML = '<p class="muted">No data available.</p>';
        return;
    }

    const columns = Object.keys(rows[0]);
    const thead = `<thead><tr>${columns.map((col) => `<th>${col}</th>`).join("")}</tr></thead>`;
    const tbody = `<tbody>${rows
        .map((row) => `<tr>${columns.map((col) => `<td>${String(row[col] ?? "")}</td>`).join("")}</tr>`)
        .join("")}</tbody>`;

    wrap.innerHTML = `<table>${thead}${tbody}</table>`;
}

function formatInputPreview(input) {
    const entries = Object.entries(input || {});
    if (!entries.length) {
        return "-";
    }
    return entries
        .slice(0, 4)
        .map(([key, value]) => `${key}: ${value}`)
        .join(" | ");
}

function applyPredictionHistoryFilters(rows = []) {
    const taskFilter = (predictionHistoryFilters.task || "all").toLowerCase();
    const datasetQuery = (predictionHistoryFilters.datasetQuery || "").trim().toLowerCase();
    const last30DaysOnly = Boolean(predictionHistoryFilters.last30DaysOnly);
    const now = Date.now();
    const thirtyDaysMs = 30 * 24 * 60 * 60 * 1000;

    return rows.filter((item) => {
        if (taskFilter !== "all" && String(item.task_type || "").toLowerCase() !== taskFilter) {
            return false;
        }

        if (datasetQuery) {
            const datasetName = String(item.dataset_name || "Ad-hoc Predictions").toLowerCase();
            if (!datasetName.includes(datasetQuery)) {
                return false;
            }
        }

        if (last30DaysOnly) {
            const createdAtMs = Date.parse(item.created_at || "");
            if (!Number.isFinite(createdAtMs) || (now - createdAtMs) > thirtyDaysMs) {
                return false;
            }
        }

        return true;
    });
}

function renderFilteredPredictionHistorySummary() {
    const filteredRows = applyPredictionHistoryFilters(predictionHistoryCache);
    renderPredictionHistorySummary(filteredRows);
}

function renderPredictionHistorySummary(rows = []) {
    if (!rows.length) {
        renderTable("historyWrap", []);
        predictionHistoryCurrentUploadId = null;
        return;
    }

    const grouped = new Map();
    for (const item of rows) {
        const uploadKey = item.upload_id === null || item.upload_id === undefined ? `adhoc_${item.id}` : `upload_${item.upload_id}`;
        if (!grouped.has(uploadKey)) {
            grouped.set(uploadKey, {
                key: uploadKey,
                upload_id: item.upload_id,
                dataset_name: item.dataset_name || "Ad-hoc Predictions",
                task_type: item.task_type,
                total_predictions: 0,
                last_prediction_at: item.created_at,
            });
        }
        const group = grouped.get(uploadKey);
        group.total_predictions += 1;
        if (new Date(item.created_at) > new Date(group.last_prediction_at)) {
            group.last_prediction_at = item.created_at;
        }
    }

    const summaryRows = Array.from(grouped.values())
        .sort((a, b) => new Date(b.last_prediction_at) - new Date(a.last_prediction_at))
        .map((group) => {
            const canDeleteGroup = group.upload_id !== null && group.upload_id !== undefined;
            const loadAction = `<button class="table-action load-prediction-history-btn" data-upload-id="${group.upload_id ?? ""}" data-group-key="${group.key}">Load</button>`;
            const deleteAction = canDeleteGroup
                ? `<button class="table-action delete-prediction-upload-btn" data-upload-id="${group.upload_id}" data-dataset-name="${String(group.dataset_name || "").replace(/"/g, "&quot;")}">Delete</button>`
                : "";
            return {
                dataset_name: group.dataset_name,
                task_type: group.task_type,
                total_predictions: formatNumber(group.total_predictions),
                last_prediction_at: new Date(group.last_prediction_at).toLocaleString(),
                actions: `${loadAction}${deleteAction}`,
            };
        });

    predictionHistoryCurrentUploadId = null;
    renderTable("historyWrap", summaryRows);
}

function renderPredictionHistoryDetails(uploadId, groupKey = "") {
    const rows = predictionHistoryCache.filter((item) => {
        if (uploadId === null || uploadId === undefined || Number.isNaN(uploadId)) {
            return item.upload_id === null || item.upload_id === undefined;
        }
        return Number(item.upload_id) === Number(uploadId);
    });

    const detailRows = rows.map((item) => ({
        id: item.id,
        input_data: formatInputPreview(item.input),
        output: item.output_value,
        probability: item.probability_score === null || item.probability_score === undefined ? "-" : formatPct(item.probability_score),
        model: item.model_used,
        created_at: new Date(item.created_at).toLocaleString(),
        action: `<button class="table-action delete-prediction-btn" data-prediction-id="${item.id}">Delete</button>`,
    }));

    predictionHistoryCurrentUploadId = uploadId;
    renderTable("historyWrap", detailRows);

    const wrap = document.getElementById("historyWrap");
    if (wrap) {
        const backBtn = `<div style="margin-bottom:8px;"><button class="table-action back-prediction-history-btn">Back to Dataset List</button></div>`;
        wrap.insertAdjacentHTML("afterbegin", backBtn);
    }
}

async function deletePredictionItem(predictionId, targetButton) {
    try {
        if (targetButton) targetButton.disabled = true;
        const response = await fetchWithAuth(`${API_BASE}/history/predictions/${predictionId}`, {
            method: "DELETE",
        });
        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.detail || "Failed to delete prediction");
        }

        predictionHistoryCache = predictionHistoryCache.filter((item) => Number(item.id) !== Number(predictionId));

        if (predictionHistoryCurrentUploadId !== null && predictionHistoryCurrentUploadId !== undefined) {
            renderPredictionHistoryDetails(predictionHistoryCurrentUploadId);
        } else {
            renderFilteredPredictionHistorySummary();
        }

        setStatus("uploadStatus", "Prediction deleted", "success");
        showToast("Prediction deleted", "success");
    } catch (error) {
        setStatus("uploadStatus", error.message || "Failed to delete prediction", "error");
        showToast(error.message || "Delete failed", "error");
        if (targetButton) targetButton.disabled = false;
    }
}

async function deletePredictionUploadGroup(uploadId, datasetName, targetButton) {
    const ok = window.confirm(`Delete all prediction history for ${datasetName}?`);
    if (!ok) {
        return;
    }

    try {
        if (targetButton) targetButton.disabled = true;
        const response = await fetchWithAuth(`${API_BASE}/history/upload/${uploadId}`, {
            method: "DELETE",
        });
        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.detail || "Failed to delete prediction history");
        }

        predictionHistoryCache = predictionHistoryCache.filter((item) => Number(item.upload_id) !== Number(uploadId));
        renderFilteredPredictionHistorySummary();
        setStatus("uploadStatus", `Deleted ${data.deleted} predictions`, "success");
        showToast("Prediction history deleted", "success");
    } catch (error) {
        setStatus("uploadStatus", error.message || "Failed to delete prediction history", "error");
        showToast(error.message || "Delete failed", "error");
        if (targetButton) targetButton.disabled = false;
    }
}

function destroyChart(instance) {
    if (instance) {
        instance.destroy();
    }
}

function renderTopFeatureList(items = []) {
    const box = document.getElementById("topFeaturesList");
    if (!box) {
        return;
    }

    if (!items.length) {
        box.innerHTML = '<p class="muted">No feature importance yet.</p>';
        return;
    }

    const top = items.slice(0, 5);
    box.innerHTML = top
        .map((item, idx) => {
            const pct = Math.max(6, Math.round((Number(item.importance || 0) * 100)));
            const gradient = featurePalette[idx % featurePalette.length];
            return `
                <div class="feature-item">
                    <div class="feature-track">
                        <div class="feature-bar" style="width:${pct}%; background:${gradient};">${item.feature}</div>
                    </div>
                    <div class="feature-value">${Number(item.importance || 0).toFixed(2)}</div>
                </div>
            `;
        })
        .join("");
}

function renderFeatureChart(items = []) {
    destroyChart(featureChart);
    const ctx = document.getElementById("featureChart").getContext("2d");

    featureChart = new Chart(ctx, {
        type: "bar",
        data: {
            labels: items.map((item) => item.feature),
            datasets: [
                {
                    label: "Importance",
                    data: items.map((item) => item.importance),
                    backgroundColor: "rgba(79, 124, 255, 0.7)",
                    borderColor: "rgba(79, 124, 255, 1)",
                    borderWidth: 1,
                },
            ],
        },
        options: {
            responsive: true,
            plugins: {
                legend: { display: false },
            },
            scales: {
                x: { ticks: { color: "#dbe4f7" } },
                y: { ticks: { color: "#dbe4f7" }, beginAtZero: true },
            },
        },
    });
}

function renderClusterDistribution(summary = []) {
    destroyChart(clusterDistributionChart);
    const ctx = document.getElementById("clusterDistributionChart").getContext("2d");

    clusterDistributionChart = new Chart(ctx, {
        type: "doughnut",
        data: {
            labels: summary.map((item) => item.label),
            datasets: [
                {
                    data: summary.map((item) => item.count),
                    backgroundColor: ["#4f7cff", "#f25f5c", "#1fba7a", "#f4c95d"],
                },
            ],
        },
        options: {
            plugins: {
                legend: { labels: { color: "#dbe4f7" } },
            },
        },
    });
}

function renderClusterScatter(points = []) {
    destroyChart(clusterScatterChart);
    const ctx = document.getElementById("clusterScatterChart").getContext("2d");

    const grouped = {};
    points.forEach((point) => {
        const key = `Cluster ${point.cluster + 1}`;
        if (!grouped[key]) {
            grouped[key] = [];
        }
        grouped[key].push({ x: point.x, y: point.y });
    });

    const palette = ["#4f7cff", "#f25f5c", "#1fba7a", "#f4c95d"];
    const datasets = Object.keys(grouped).map((key, idx) => ({
        label: key,
        data: grouped[key],
        backgroundColor: palette[idx % palette.length],
        pointRadius: 4,
    }));

    clusterScatterChart = new Chart(ctx, {
        type: "scatter",
        data: { datasets },
        options: {
            scales: {
                x: { ticks: { color: "#dbe4f7" } },
                y: { ticks: { color: "#dbe4f7" } },
            },
            plugins: {
                legend: { labels: { color: "#dbe4f7" } },
            },
        },
    });
}

function renderModelMetrics(metrics = {}, taskType = "classification") {
    destroyChart(modelMetricChart);
    const ctx = document.getElementById("modelMetricChart").getContext("2d");

    let labels = [];
    let values = [];

    if (taskType === "classification" && metrics.classification) {
        const lr = metrics.classification.logistic_regression || {};
        const rf = metrics.classification.random_forest || {};
        labels = ["LR Accuracy", "LR Precision", "RF Recall", "RF F1"];
        values = [lr.accuracy || 0, lr.precision || 0, rf.recall || 0, rf.f1 || 0].map((v) => Number(v) * 100);
    } else if (metrics.regression) {
        const lr = metrics.regression.linear_regression || {};
        const rf = metrics.regression.random_forest_regressor || {};
        labels = ["Linear MSE", "RF Regressor MSE"];
        values = [lr.mse || 0, rf.mse || 0].map((v) => Number(v));
    }

    modelMetricChart = new Chart(ctx, {
        type: "bar",
        data: {
            labels,
            datasets: [
                {
                    label: taskType === "classification" ? "Score (%)" : "MSE",
                    data: values,
                    backgroundColor: "rgba(79, 124, 255, 0.72)",
                    borderColor: "rgba(79, 124, 255, 1)",
                    borderWidth: 1,
                },
            ],
        },
        options: {
            plugins: {
                legend: { display: false },
            },
            scales: {
                x: { ticks: { color: "#dbe4f7" } },
                y: { ticks: { color: "#dbe4f7" }, beginAtZero: true },
            },
        },
    });
}

function renderChurnChart(churnRate) {
    destroyChart(churnChart);
    const ctx = document.getElementById("churnChart").getContext("2d");
    const churn = Number(churnRate || 0);
    const retained = Math.max(0, 1 - churn);

    churnChart = new Chart(ctx, {
        type: "pie",
        data: {
            labels: ["Churn", "Retained"],
            datasets: [
                {
                    data: [churn, retained],
                    backgroundColor: ["#f25f5c", "#1fba7a"],
                },
            ],
        },
        options: {
            plugins: {
                legend: { labels: { color: "#dbe4f7" } },
            },
        },
    });
}

function renderSalesTrend(previewRows = [], targetColumn = "") {
    destroyChart(salesTrendChart);
    const canvas = document.getElementById("salesTrendChart");
    if (!canvas) {
        return;
    }

    if (!previewRows.length) {
        const ctx = canvas.getContext("2d");
        salesTrendChart = new Chart(ctx, {
            type: "line",
            data: { labels: [], datasets: [{ label: "Revenue", data: [] }] },
            options: { plugins: { legend: { labels: { color: "#dbe4f7" } } } },
        });
        return;
    }

    const sample = previewRows[0] || {};
    const keys = Object.keys(sample);
    const numericKeys = keys.filter((key) => typeof sample[key] === "number");
    const selectedTarget = targetColumn && numericKeys.includes(targetColumn)
        ? targetColumn
        : (numericKeys[0] || keys[keys.length - 1]);

    const labels = previewRows.map((_, idx) => `Row ${idx + 1}`);
    const values = previewRows.map((row) => Number(row[selectedTarget]) || 0);

    const ctx = canvas.getContext("2d");
    salesTrendChart = new Chart(ctx, {
        type: "line",
        data: {
            labels,
            datasets: [
                {
                    label: `Trend (${selectedTarget})`,
                    data: values,
                    borderColor: "#32a3ff",
                    backgroundColor: "rgba(50,163,255,0.2)",
                    fill: true,
                    tension: 0.25,
                },
            ],
        },
        options: {
            plugins: { legend: { labels: { color: "#dbe4f7" } } },
            scales: {
                x: { ticks: { color: "#dbe4f7" } },
                y: { ticks: { color: "#dbe4f7" }, beginAtZero: true },
            },
        },
    });
}

function updateComparison(metrics, taskType = "classification") {
    const box = document.getElementById("comparisonBox");

    if (taskType === "classification" && metrics.classification) {
        const lr = metrics.classification.logistic_regression;
        const rf = metrics.classification.random_forest;
        box.innerHTML = `
            <strong>Logistic Regression</strong> → Accuracy ${formatPct(lr.accuracy)}, Precision ${formatPct(lr.precision)}, Recall ${formatPct(lr.recall)}, F1 ${formatPct(lr.f1)}<br>
            <strong>Random Forest</strong> → Accuracy ${formatPct(rf.accuracy)}, Precision ${formatPct(rf.precision)}, Recall ${formatPct(rf.recall)}, F1 ${formatPct(rf.f1)}<br>
            <strong>Winner:</strong> ${metrics.model_comparison?.winner || "-"}
        `;
        return;
    }

    if (metrics.regression) {
        const lr = metrics.regression.linear_regression;
        const rf = metrics.regression.random_forest_regressor;
        box.innerHTML = `
            <strong>Linear Regression</strong> → MSE ${formatNumber(lr.mse)}<br>
            <strong>Random Forest Regressor</strong> → MSE ${formatNumber(rf.mse)}<br>
            <strong>Winner:</strong> ${metrics.model_comparison?.winner || "-"}
        `;
        return;
    }

    box.textContent = "No training run yet.";
}

function applyAnalysisToUI(data, fallbackTaskType = "classification") {
    const taskType = data.task_type || fallbackTaskType;
    const analysisType = getActiveAnalysisType();
    const showFeatureImportance = analysisType === "churn" || analysisType === "sales" || analysisType === "segmentation";

    if (data.upload_id !== null && data.upload_id !== undefined) {
        latestUploadId = data.upload_id;
    }

    renderTable("previewWrap", data.preview || []);
    updateComparison(data.metrics || {}, taskType);
    if (showFeatureImportance) {
        renderFeatureChart(data.feature_importance || []);
        renderTopFeatureList(data.feature_importance || []);
    } else {
        renderFeatureChart([]);
        renderTopFeatureList([]);
    }
    renderClusterDistribution(data.clusters?.summary || []);
    renderClusterSummary(data.clusters?.summary || []);
    renderClusterScatter(data.clusters?.plot?.points || []);
    renderModelMetrics(data.metrics || {}, taskType || "classification");

    if (analysisType === "churn" && (taskType || "classification") === "classification") {
        renderChurnChart(data.metrics?.dashboard?.churn_rate || 0);
    } else {
        destroyChart(churnChart);
        churnChart = null;
    }

    if (analysisType === "sales") {
        renderSalesTrend(data.preview || [], data.target_column || "");
    } else {
        destroyChart(salesTrendChart);
        salesTrendChart = null;
    }

    document.getElementById("insightText").textContent = data.insight || "No insight available yet.";
}

async function loadTrainingHistory() {
    if (!token) {
        return;
    }

    try {
        const response = await fetchWithAuth(`${API_BASE}/analytics-history?limit=20`, {});
        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.detail || "Training history fetch failed");
        }

        const rows = (data.items || []).map((item) => ({
            id: item.id,
            dataset_name: item.dataset_name,
            model_used: item.model_used,
            training_date: new Date(item.training_date).toLocaleString(),
            performance_metrics: formatMetricsSummary(item.task_type, item.performance_metrics),
            prediction_summary: formatPredictionSummary(item.prediction_summary),
            clustering_results: formatClusterSummary(item.clustering_results),
            action:
                `<button class="table-action load-analysis-btn" data-upload-id="${item.id}">Load</button> ` +
                `<button class="table-action delete-dataset-btn" data-upload-id="${item.id}" data-dataset-name="${String(item.dataset_name || "").replace(/"/g, "&quot;")}">Delete</button>`,
        }));

        renderTable("trainingHistoryWrap", rows);
    } catch (error) {
        console.error(error);
    }
}

async function deleteDataset(uploadId, datasetName = "dataset", actionButton = null) {
    if (!uploadId) {
        return;
    }

    const confirmed = window.confirm(
        `Delete ${datasetName} and all related analytics, clustering, and prediction logs? This cannot be undone.`
    );
    if (!confirmed) {
        return;
    }

    try {
        setTableActionBusy(actionButton, "Deleting...", true);
        setStatus("uploadStatus", `Deleting ${datasetName}...`, "");
        const response = await fetchWithAuth(`${API_BASE}/uploads/${uploadId}`, {
            method: "DELETE",
            headers: {},
        });
        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.detail || "Dataset deletion failed");
        }

        if (latestUploadId === uploadId) {
            latestUploadId = null;
        }

        setStatus(
            "uploadStatus",
            `${data.message}. Removed prediction logs: ${formatNumber(data.deleted_predictions)}. File deleted: ${data.file_deleted ? "yes" : "no"}`,
            "success"
        );
        showToast("Deleted successfully", "success");

        resetDashboardForNewUpload();
        await loadHistory();
        await loadTrainingHistory();
    } catch (error) {
        setStatus("uploadStatus", error.message, "error");
        showToast(error.message || "Delete failed", "error");
    } finally {
        setTableActionBusy(actionButton, "Deleting...", false);
    }
}

async function downloadTrainingReport() {
    const reportButton = document.getElementById("downloadReportBtn");
    if (!token) {
        setStatus("uploadStatus", "Please login first", "error");
        return;
    }

    let uploadId = latestUploadId;
    if (uploadId === null || uploadId === undefined) {
        try {
            const latestResponse = await fetchWithAuth(`${API_BASE}/latest-upload`, {});
            const latestData = await latestResponse.json();
            if (!latestResponse.ok) {
                throw new Error(latestData.detail || "Could not fetch latest training run");
            }
            uploadId = latestData?.item?.id;
            if (uploadId !== null && uploadId !== undefined) {
                latestUploadId = uploadId;
            }
        } catch (error) {
            setStatus("uploadStatus", error.message || "No training run available for report", "error");
            showToast(error.message || "No report available", "error");
            return;
        }
    }

    if (uploadId === null || uploadId === undefined) {
        setStatus("uploadStatus", "Train a model or load a historical run before downloading report", "error");
        showToast("No training report available", "error");
        return;
    }

    try {
        setButtonBusy(reportButton, "Preparing Report...", true);
        setStatus("uploadStatus", "Preparing report download...", "");
        const activeMode = getActiveAnalysisType();
        const reportQuery = new URLSearchParams({ report_mode: activeMode });
        const response = await fetchWithAuth(`${API_BASE}/uploads/${uploadId}/report?${reportQuery.toString()}`, {});
        if (!response.ok) {
            let message = "Report download failed";
            try {
                const err = await response.json();
                message = err.detail || message;
            } catch (_e) {
                message = response.statusText || message;
            }
            throw new Error(message);
        }

        const blob = await response.blob();
        const disposition = response.headers.get("Content-Disposition") || "";
        const fileName = extractFilenameFromDisposition(disposition);

        const url = window.URL.createObjectURL(blob);
        const anchor = document.createElement("a");
        anchor.href = url;
        anchor.download = fileName;
        document.body.appendChild(anchor);
        anchor.click();
        anchor.remove();
        window.URL.revokeObjectURL(url);

        setStatus("uploadStatus", "Report downloaded successfully", "success");
        showToast("Report downloaded", "success");
    } catch (error) {
        setStatus("uploadStatus", error.message, "error");
        showToast(error.message || "Report download failed", "error");
    } finally {
        setButtonBusy(reportButton, "Preparing Report...", false);
    }
}

async function loadUploadAnalysis(uploadId) {
    try {
        setStatus("uploadStatus", `Loading analysis for run #${uploadId}...`, "");
        const response = await fetchWithAuth(`${API_BASE}/uploads/${uploadId}`, {});
        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.detail || "Analysis load failed");
        }

        const item = data.item || {};
        const mappedMode = taskTypeToAnalysisType(item.task_type || "classification");
        if (analysisTypeSelect) {
            analysisTypeSelect.value = mappedMode;
        }
        configureModuleView();

        applyAnalysisToUI(
            {
                upload_id: item.id,
                task_type: item.task_type,
                target_column: item.target_column,
                preview: item.preview || [],
                metrics: item.metrics || {},
                clusters: item.clusters || { summary: [] },
                feature_importance: item.feature_importance || [],
                insight: item.insight || `Historical analysis loaded from ${item.filename || "dataset"}`,
            },
            item.task_type || "classification"
        );

        setHistoryMode(true, uploadId);
        await refreshDashboard();
    } catch (error) {
        setStatus("uploadStatus", error.message, "error");
    }
}

async function exitHistoryMode() {
    setHistoryMode(false);

    if (!token) {
        return;
    }

    try {
        const response = await fetchWithAuth(`${API_BASE}/latest-upload`, {});
        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.detail || "Unable to return to dashboard");
        }

        const latest = data.item;
        if (latest && analysisTypeSelect) {
            analysisTypeSelect.value = taskTypeToAnalysisType(latest.task_type || "classification");
        }

        configureModuleView();
        resetDashboardForNewUpload();
        setStatus("uploadStatus", "Returned to dashboard. Upload new dataset to generate fresh graphs.", "success");
    } catch (error) {
        setStatus("uploadStatus", error.message || "Unable to return to dashboard", "error");
    }
}

async function login() {
    const username = document.getElementById("username").value.trim();
    const password = document.getElementById("password").value;

    if (!username || !password) {
        setStatus("loginStatus", "Username and password are required", "error");
        return;
    }

    try {
        const response = await fetch(`${API_BASE}/auth/login`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ username, password }),
        });

        const data = await response.json();
        if (!response.ok) {
            let errorMsg = data.detail || "Login failed";
            
            if (response.status === 404 || errorMsg.toLowerCase().includes("not found") || errorMsg.toLowerCase().includes("does not exist")) {
                errorMsg = "Account not found. Please sign up first.";
            } else if (response.status === 401 || errorMsg.toLowerCase().includes("incorrect") || errorMsg.toLowerCase().includes("invalid")) {
                errorMsg = "Invalid credentials. Please check your username and password.";
            }
            
            throw new Error(errorMsg);
        }

        token = data.token;
        currentUser = data.user?.username || username;
        localStorage.setItem("ai_token", token);
        localStorage.setItem("ai_user", currentUser);
        setStatus("loginStatus", "Login successful", "success");
        updateAuthView();
        resetDashboardForNewUpload();
        await loadHistory();
        await loadTrainingHistory();
    } catch (error) {
        clearAuthState();
        updateAuthView();
        setStatus("loginStatus", error.message, "error");
    }
}

async function signup() {
    const username = document.getElementById("username").value.trim();
    const password = document.getElementById("password").value;
    const confirmPassword = document.getElementById("confirmPassword").value;

    if (!username || !password) {
        setStatus("loginStatus", "Username and password are required", "error");
        return;
    }

    if (password !== confirmPassword) {
        setStatus("loginStatus", "Passwords do not match. Please try again.", "error");
        return;
    }

    if (password.length < 6) {
        setStatus("loginStatus", "Password must be at least 6 characters long", "error");
        return;
    }

    try {
        const response = await fetch(`${API_BASE}/auth/signup`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ username, password, confirm_password: confirmPassword }),
        });

        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.detail || "Signup failed");
        }

        token = "";
        currentUser = "";
        localStorage.removeItem("ai_token");
        localStorage.removeItem("ai_user");
        const signedUpUsername = username;
        document.getElementById("password").value = "";
        document.getElementById("confirmPassword").value = "";
        setAuthMode("login");
        setStatus("loginStatus", "Account created! You can now sign in.", "success");
        document.getElementById("username").value = signedUpUsername;
        document.getElementById("password").focus();
        updateAuthView();
    } catch (error) {
        setStatus("loginStatus", error.message, "error");
    }
}

function submitAuthFromKeyboard(event) {
    if (event.key !== "Enter") {
        return;
    }

    event.preventDefault();

    const fieldId = event.target?.id;

    if (authMode === "signup") {
        // Guide user through each field in order before submitting
        if (fieldId === "username") {
            document.getElementById("password").focus();
            return;
        }
        if (fieldId === "password") {
            document.getElementById("confirmPassword").focus();
            return;
        }
        // On confirmPassword or any other field — submit
        signup();
        return;
    }

    login();
}

function togglePasswordVisibility(event) {
    const button = event.currentTarget;
    const targetId = button.dataset.target;
    const input = document.getElementById(targetId);
    
    if (!input) {
        return;
    }
    
    const isPassword = input.type === "password";
    input.type = isPassword ? "text" : "password";
    button.classList.toggle("visible", isPassword);
}

async function uploadAndTrain() {
    const uploadButton = document.getElementById("uploadBtn");
    const fileInput = document.getElementById("fileInput");
    const currentFile = fileInput.files[0];
    const file = currentFile || lastSelectedFile;
    previousUploadId = latestUploadId;
    const analysisType = getActiveAnalysisType();
    const taskType = analysisTypeToTaskType(analysisType);
    const targetColumn = document.getElementById("targetColumn").value.trim();

    if (!file) {
        setStatus("uploadStatus", "Please choose a CSV file at least once", "error");
        return;
    }

    setButtonBusy(uploadButton, "Training...", true);
    startTrainingTimer(file.size, taskType);

    const formData = new FormData();
    formData.append("file", file);

    setStatus("uploadStatus", "Validating dataset structure...", "");

    const query = new URLSearchParams({ task_type: taskType, analysis_mode: analysisType });
    if (targetColumn) {
        query.set("target_column", targetColumn);
    }

    try {
        setHistoryMode(false);
        setStatus(
            "uploadStatus",
            currentFile
                ? "Running pipeline: upload → preprocess → train → cluster..."
                : "Running pipeline using the last selected dataset...",
            ""
        );

        const response = await fetchWithAuth(`${API_BASE}/upload?${query.toString()}`, {
            method: "POST",
            headers: {},
            body: formData,
        });

        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.detail || "Upload failed");
        }

        latestUploadId = data.upload_id;
        lastSelectedFile = file;
        const duplicateInfo = data.cleaning_report?.duplicates_removed ?? 0;
        const missingTargetInfo = data.cleaning_report?.rows_removed_missing_target ?? 0;
        const invalidNumericTargetInfo = data.cleaning_report?.rows_removed_invalid_numeric_target ?? 0;
        const trainingSecs = typeof data.training_time_seconds === "number" && data.training_time_seconds > 0
            ? data.training_time_seconds
            : null;
        if (trainingSecs !== null) {
            recordTrainingEtaSample(taskType, file.size, trainingSecs);
        }
        stopTrainingTimer(trainingSecs);
        updateRestoreButtonState();
        setStatus(
            "uploadStatus",
            `${data.message} (${data.processed_records}/${data.total_records} records used, auto-cleaned: duplicates ${duplicateInfo}, missing target ${missingTargetInfo}, invalid numeric target ${invalidNumericTargetInfo})`,
            "success"
        );
        applyAnalysisToUI(data, taskType);
        await refreshDashboard();
        await loadHistory();
        await loadTrainingHistory();
    } catch (error) {
        stopTrainingTimer();
        setStatus("uploadStatus", error.message, "error");
    } finally {
        setButtonBusy(uploadButton, "Training...", false);
    }
}

async function runPredictionUpload() {
    const taskType = analysisTypeToTaskType(getActiveAnalysisType());
    const fileInput = document.getElementById("predictFileInput");
    const predictUploadBtn = document.getElementById("predictUploadBtn");
    const file = fileInput.files?.[0];

    if (!file) {
        setStatus("predictionUploadStatus", "Please choose a CSV file for batch prediction", "error");
        return;
    }

    const formData = new FormData();
    formData.append("file", file);

    const query = new URLSearchParams({ task_type: taskType });
    if (latestUploadId !== null && latestUploadId !== undefined) {
        query.set("upload_id", String(latestUploadId));
    }

    try {
        setButtonBusy(predictUploadBtn, "Predicting...", true);
        setStatus("predictionUploadStatus", "Running batch prediction...", "");

        const response = await fetchWithAuth(`${API_BASE}/predict-upload?${query.toString()}`, {
            method: "POST",
            headers: {},
            body: formData,
        });

        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.detail || "Batch prediction failed");
        }

        const rows = (data.preview || []).map((item) => ({
            row: item.row,
            prediction: item.prediction,
            probability: item.probability === null || item.probability === undefined ? "-" : formatPct(item.probability),
            model: item.model,
        }));

        renderTable("predictionUploadWrap", rows);
        setStatus(
            "predictionUploadStatus",
            `Batch prediction complete: ${data.predicted_rows} rows (showing ${data.preview_rows})`,
            "success"
        );

        if (data.insight) {
            document.getElementById("insightText").textContent = data.insight;
        }

        await loadHistory();
    } catch (error) {
        setStatus("predictionUploadStatus", error.message, "error");
    } finally {
        setButtonBusy(predictUploadBtn, "Predicting...", false);
    }
}

async function refreshDashboard() {
    if (!token) {
        return;
    }

    try {
        const response = await fetchWithAuth(`${API_BASE}/dashboard`, {});

        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.detail || "Dashboard fetch failed");
        }

        const mode = getActiveAnalysisType();
        const isChurn = mode === "churn";
        const isSales = mode === "sales";
        const isSegmentation = mode === "segmentation";
        const showFeatureImportance = isChurn || isSales || isSegmentation;

        document.getElementById("statRecords").textContent = formatNumber(data.total_records);

        if (isChurn) {
            document.getElementById("statAccuracy").textContent = formatPct(data.model_accuracy);
            document.getElementById("statPrecision").textContent = formatPct(data.metrics?.classification?.random_forest?.precision);
            document.getElementById("statRecall").textContent = formatPct(data.metrics?.classification?.random_forest?.recall);
        }

        if (isSales) {
            const mse = data.metrics?.regression?.random_forest_regressor?.mse;
            document.getElementById("statAccuracy").textContent = formatNumber(mse);
            document.getElementById("statPrecision").textContent = formatNumber(data.processed_records);
            document.getElementById("statRecall").textContent = data.model_comparison?.winner || "-";
        }

        if (isSegmentation) {
            const summary = data.cluster_distribution || [];
            const largest = summary.reduce((max, item) => (item.count > max.count ? item : max), { label: "-", count: 0 });
            document.getElementById("statAccuracy").textContent = formatNumber(data.total_records);
            document.getElementById("statPrecision").textContent = formatNumber(summary.length);
            document.getElementById("statRecall").textContent = `${largest.label} (${formatNumber(largest.count)})`;
        }

        document.getElementById("statChurn").textContent = formatPct(data.churn_rate);
        document.getElementById("statBestModel").textContent = data.model_comparison?.winner || "-";
        const classificationMetrics = data.metrics?.classification?.random_forest || {};
        if (!isChurn) {
            document.getElementById("statPrecision").textContent = document.getElementById("statPrecision").textContent || "-";
            document.getElementById("statRecall").textContent = document.getElementById("statRecall").textContent || "-";
        }
        document.getElementById("insightText").textContent = data.insight || "No insight available yet.";

        if (showFeatureImportance) {
            renderFeatureChart(data.feature_importance || []);
            renderTopFeatureList(data.feature_importance || []);
        } else {
            renderFeatureChart([]);
            renderTopFeatureList([]);
        }
        renderClusterDistribution(data.cluster_distribution || []);
        renderClusterSummary(data.cluster_distribution || []);
        renderModelMetrics(data.metrics || {}, data.task_type || "classification");

        if (isSales) {
            const latest = await fetchWithAuth(`${API_BASE}/latest-upload`, {});
            const latestData = await latest.json();
            const previewRows = latestData?.item?.preview || [];
            renderSalesTrend(previewRows, latestData?.item?.target_column || "");
        }

        if (isChurn && (data.task_type || "classification") === "classification") {
            renderChurnChart(data.churn_rate || 0);
        } else {
            destroyChart(churnChart);
            churnChart = null;
        }

        if (!isSales) {
            destroyChart(salesTrendChart);
            salesTrendChart = null;
        }
    } catch (error) {
        console.error(error);
    }
}

async function loadHistory() {
    if (!token) {
        return;
    }

    try {
        const response = await fetchWithAuth(`${API_BASE}/history?limit=20`, {});

        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.detail || "History fetch failed");
        }

        predictionHistoryCache = data.items || [];

        const predictionCountNode = document.getElementById("statPredictions");
        if (predictionCountNode) {
            predictionCountNode.textContent = `${predictionHistoryCache.length}+`;
        }

        renderFilteredPredictionHistorySummary();
    } catch (error) {
        console.error(error);
    }
}

function logout() {
    setHistoryMode(false);
    clearAuthState();
    updateAuthView();
    setAuthMode("login");
}

async function initializeApp() {
    attachSlideNavigation("previewWrap");
    attachSlideNavigation("trainingHistoryWrap");
    attachSlideNavigation("historyWrap");

    configureModuleView();
    setAuthMode("login");

    if (token) {
        const isValidSession = await validateStoredSession();
        if (!isValidSession) {
            clearAuthState();
            setStatus("loginStatus", "Please login to continue", "");
        }
    }

    updateAuthView();

    if (token) {
        resetDashboardForNewUpload();
        await loadHistory();
        await loadTrainingHistory();
    }
}

document.getElementById("loginBtn").addEventListener("click", login);
document.getElementById("signupBtn").addEventListener("click", signup);
document.getElementById("showLoginBtn").addEventListener("click", () => setAuthMode("login"));
document.getElementById("showSignupBtn").addEventListener("click", () => setAuthMode("signup"));
document.getElementById("helperLoginBtn").addEventListener("click", () => setAuthMode("login"));
document.getElementById("helperSignupBtn").addEventListener("click", () => setAuthMode("signup"));
document.getElementById("username").addEventListener("keydown", submitAuthFromKeyboard);
document.getElementById("password").addEventListener("keydown", submitAuthFromKeyboard);
document.getElementById("confirmPassword").addEventListener("keydown", submitAuthFromKeyboard);
document.querySelectorAll(".password-toggle").forEach(button => {
    button.addEventListener("click", togglePasswordVisibility);
});
document.getElementById("uploadBtn").addEventListener("click", uploadAndTrain);
document.getElementById("restorePreviousBtn").addEventListener("click", async () => {
    if (previousUploadId === null) return;
    await loadUploadAnalysis(previousUploadId);
    updateRestoreButtonState();
});
document.getElementById("exitHistoryBtn").addEventListener("click", exitHistoryMode);
document.getElementById("downloadReportBtn").addEventListener("click", downloadTrainingReport);
document.getElementById("downloadReportBannerBtn").addEventListener("click", downloadTrainingReport);
analysisTypeSelect.addEventListener("change", async () => {
    configureModuleView();
    if (latestUploadId) {
        await refreshDashboard();
    }
});
document.getElementById("fileInput").addEventListener("change", (event) => {
    const selected = event.target?.files?.[0];
    if (selected) {
        lastSelectedFile = selected;
    }
});
document.getElementById("predictUploadBtn").addEventListener("click", runPredictionUpload);
document.getElementById("trainingHistoryWrap").addEventListener("click", (event) => {
    const target = event.target;
    if (!(target instanceof HTMLElement)) {
        return;
    }

    if (target.classList.contains("load-analysis-btn")) {
        const uploadId = Number(target.dataset.uploadId);
        if (!Number.isNaN(uploadId) && uploadId > 0) {
            loadUploadAnalysis(uploadId);
        }
        return;
    }

    if (target.classList.contains("delete-dataset-btn")) {
        const uploadId = Number(target.dataset.uploadId);
        const datasetName = target.dataset.datasetName || "dataset";
        if (!Number.isNaN(uploadId) && uploadId > 0) {
            deleteDataset(uploadId, datasetName, target);
        }
    }
});
document.getElementById("historyWrap").addEventListener("click", (event) => {
    const target = event.target;
    if (!(target instanceof HTMLElement)) {
        return;
    }

    if (target.classList.contains("back-prediction-history-btn")) {
        renderFilteredPredictionHistorySummary();
        return;
    }

    if (target.classList.contains("load-prediction-history-btn")) {
        const uploadRaw = target.dataset.uploadId;
        const uploadId = uploadRaw === "" || uploadRaw === undefined ? null : Number(uploadRaw);
        renderPredictionHistoryDetails(uploadId, target.dataset.groupKey || "");
        return;
    }

    if (target.classList.contains("delete-prediction-upload-btn")) {
        const uploadId = Number(target.dataset.uploadId);
        const datasetName = target.dataset.datasetName || "dataset";
        if (!Number.isNaN(uploadId) && uploadId > 0) {
            deletePredictionUploadGroup(uploadId, datasetName, target);
        }
        return;
    }

    if (target.classList.contains("delete-prediction-btn")) {
        const predictionId = Number(target.dataset.predictionId);
        if (!Number.isNaN(predictionId) && predictionId > 0) {
            deletePredictionItem(predictionId, target);
        }
    }
});

const historyTaskFilterNode = document.getElementById("historyTaskFilter");
const historyDatasetFilterNode = document.getElementById("historyDatasetFilter");
const historyLast30FilterNode = document.getElementById("historyLast30Filter");
const historyFilterResetBtn = document.getElementById("historyFilterResetBtn");

if (historyTaskFilterNode) {
    historyTaskFilterNode.addEventListener("change", () => {
        predictionHistoryFilters.task = historyTaskFilterNode.value || "all";
        renderFilteredPredictionHistorySummary();
    });
}

if (historyDatasetFilterNode) {
    historyDatasetFilterNode.addEventListener("input", () => {
        predictionHistoryFilters.datasetQuery = historyDatasetFilterNode.value || "";
        renderFilteredPredictionHistorySummary();
    });
}

if (historyLast30FilterNode) {
    historyLast30FilterNode.addEventListener("change", () => {
        predictionHistoryFilters.last30DaysOnly = Boolean(historyLast30FilterNode.checked);
        renderFilteredPredictionHistorySummary();
    });
}

if (historyFilterResetBtn) {
    historyFilterResetBtn.addEventListener("click", () => {
        predictionHistoryFilters = { task: "all", datasetQuery: "", last30DaysOnly: false };
        if (historyTaskFilterNode) {
            historyTaskFilterNode.value = "all";
        }
        if (historyDatasetFilterNode) {
            historyDatasetFilterNode.value = "";
        }
        if (historyLast30FilterNode) {
            historyLast30FilterNode.checked = false;
        }
        renderFilteredPredictionHistorySummary();
    });
}

logoutBtn.addEventListener("click", logout);

initializeApp();
