/**
 * VoicePrint Studio Frontend Application
 * Zero-Inference Clipboard Rewriter & Voice DNA Studio
 * 100% Vanilla JS — Zero CDN Dependencies
 */

(function () {
  "use strict";

  // =========================================================================
  // Application State
  // =========================================================================
  const appState = {
    settings: {},
    activeContextMode: "casual_chat",
    intensity: "deep",
    dnaByMode: {},
    activeDna: {},
    samples: [],
    rules: [],
    editPairs: [],
    history: [],
    pangramScans: [],
    lastPangramResult: null,
    maturity: {
      score: 0,
      label: "Seedling",
      total_samples: 0,
      gold_edits: 0,
      rules_count: 0,
    },
    lastAiDraft: "",
    lastOriginalInput: "",
    activeTab: "studio",
    isRewriting: false,
    clearKeyFlags: {
      gemini: false,
      openai: false,
      anthropic: false,
      pangram: false,
    },
  };

  const SAMPLE_AI_PARAGRAPH =
    "In today's fast-paced digital age, it is important to note that artificial intelligence has become a multifaceted tapestry of innovation. By leveraging cutting-edge solutions, organizations can seamlessly elevate their workflows and foster unprecedented synergy. Furthermore, delving into this transformative landscape serves as a testament to human ingenuity. Needless to say, not only does this empower teams, but it also optimizes efficiency across the board.";

  const SAMPLE_INSPECTOR_SUBMISSION =
    "In today's fast-paced digital era, artificial intelligence has fundamentally revolutionized the landscape of contemporary work. By seamlessly integrating cutting-edge machine learning models into everyday workflows, enterprises can unlock unprecedented levels of productivity and innovation. Furthermore, it is important to remember that human-in-the-loop oversight serves as a vital safeguard against hallucinations and ethical pitfalls. Ultimately, embracing this transformative paradigm shift empowers individuals to transcend routine tasks and delve into creative problem-solving across diverse industries.";

  const SAMPLE_ACADEMIC_DRAFT =
    "In modern society, many scholars argue that neural language models exhibit profound transformative effects on cognitive workflows (Vaswani et al., 2017). Due to the fact that deep architectures capture intricate statistical regularities across massive corpora, organizations are enabled to streamline repetitive authoring. However, research suggests that algorithmic outputs frequently perpetuate latent historical biases without authorial intentionality. In this paper, we investigate the empirical trade-offs between automated drafting and human voice fidelity.";

  const SAMPLE_ACADEMIC_SOURCE =
    "Neural language models exhibit profound transformative effects on cognitive workflows by capturing intricate statistical regularities across massive text corpora. Although organizations can streamline repetitive drafting, algorithmic outputs frequently perpetuate latent historical biases without explicit human intent. We demonstrate the practical necessity of continuous authorial calibration.";

  let sourceDebounceTimer = null;
  let outputDebounceTimer = null;
  let searchDebounceTimer = null;

  // =========================================================================
  // DOM Element Selectors Cache
  // =========================================================================
  const dom = {};

  function cacheDomElements() {
    // Header & Brand
    dom.maturityLabel = document.getElementById("maturity-label");
    dom.maturityScore = document.getElementById("maturity-score");
    dom.maturityBar = document.getElementById("maturity-bar");
    dom.maturitySamples = document.getElementById("maturity-samples-count");
    dom.maturityGold = document.getElementById("maturity-gold-count");
    dom.btnOpenSettings = document.getElementById("btn-open-settings");

    // Mode pills
    dom.modePills = document.querySelectorAll(".mode-pill");

    // Tabs
    dom.tabButtons = document.querySelectorAll(".tab-btn");
    dom.tabPanes = document.querySelectorAll(".tab-pane");

    // Studio: Source
    dom.sourceInput = document.getElementById("source-input");
    dom.sourceWordCount = document.getElementById("source-word-count");
    dom.sourceCharCount = document.getElementById("source-char-count");
    dom.btnGrabClipboard = document.getElementById("btn-grab-clipboard");
    dom.btnSampleAi = document.getElementById("btn-sample-ai");
    dom.btnClearSource = document.getElementById("btn-clear-source");
    dom.analysisSpinner = document.getElementById("analysis-spinner");

    // Studio: Metrics & Tells
    dom.valHumanScore = document.getElementById("val-human-score");
    dom.barHumanScore = document.getElementById("bar-human-score");
    dom.hintHumanScore = document.getElementById("hint-human-score");
    dom.valAiRisk = document.getElementById("val-ai-risk");
    dom.barAiRisk = document.getElementById("bar-ai-risk");
    dom.valBurstiness = document.getElementById("val-burstiness");
    dom.barBurstiness = document.getElementById("bar-burstiness");
    dom.sourceCadenceStats = document.getElementById("source-cadence-stats");
    dom.sourceSparkline = document.getElementById("source-sparkline");
    dom.tellsCountBadge = document.getElementById("tells-count-badge");
    dom.aiTellsContainer = document.getElementById("ai-tells-container");

    // Studio: Action Bar
    dom.intensityButtons = document.querySelectorAll(".seg-btn");
    dom.btnRewrite = document.getElementById("btn-rewrite");
    dom.btnRewriteText = document.getElementById("btn-rewrite-text");
    dom.btnQuickRewrite = document.getElementById("btn-quick-rewrite");

    // Studio: Output
    dom.outputText = document.getElementById("output-text");
    dom.outputWordCount = document.getElementById("output-word-count");
    dom.outputCharCount = document.getElementById("output-char-count");
    dom.btnCopyOnly = document.getElementById("btn-copy-only");
    dom.diffBadge = document.getElementById("diff-badge");
    dom.providerPill = document.getElementById("provider-pill");
    dom.providerName = document.getElementById("provider-name");

    // Studio: Comparison & Learn
    dom.compHumanBefore = document.getElementById("comp-human-before");
    dom.compHumanAfter = document.getElementById("comp-human-after");
    dom.compHumanDelta = document.getElementById("comp-human-delta");
    dom.compBurstBefore = document.getElementById("comp-burst-before");
    dom.compBurstAfter = document.getElementById("comp-burst-after");
    dom.compBurstDelta = document.getElementById("comp-burst-delta");
    dom.compDnaMatch = document.getElementById("comp-dna-match");
    dom.compSamplesRef = document.getElementById("comp-samples-ref");
    dom.outputCadenceStats = document.getElementById("output-cadence-stats");
    dom.outputSparkline = document.getElementById("output-sparkline");
    dom.btnCopyLearn = document.getElementById("btn-copy-learn");
    dom.learnedHabitsBanner = document.getElementById("learned-habits-banner");
    dom.learnedHabitsList = document.getElementById("learned-habits-list");

    // Tab 2: Samples & Import
    dom.importForm = document.getElementById("import-form");
    dom.importFormat = document.getElementById("import-format");
    dom.importMode = document.getElementById("import-mode");
    dom.importSender = document.getElementById("import-sender");
    dom.importDropzone = document.getElementById("import-dropzone");
    dom.filePicker = document.getElementById("file-picker");
    dom.btnBrowseFile = document.getElementById("btn-browse-file");
    dom.importText = document.getElementById("import-text");
    dom.importFeedback = document.getElementById("import-feedback");
    dom.btnSeedDemo = document.getElementById("btn-seed-demo");
    dom.btnClearSamples = document.getElementById("btn-clear-samples");
    dom.samplesSubtitle = document.getElementById("samples-subtitle");
    dom.samplesSearch = document.getElementById("samples-search");
    dom.samplesListContainer = document.getElementById("samples-list-container");

    // Tab 3: DNA & Rules
    dom.dnaActiveModeTitle = document.getElementById("dna-active-mode-title");
    dom.dnaAvgLen = document.getElementById("dna-avg-len");
    dom.dnaStdLen = document.getElementById("dna-std-len");
    dom.dnaContractionRate = document.getElementById("dna-contraction-rate");
    dom.dnaContractionHint = document.getElementById("dna-contraction-hint");
    dom.dnaLowerStart = document.getElementById("dna-lower-start");
    dom.dnaLowerI = document.getElementById("dna-lower-i");
    dom.dnaNoPeriod = document.getElementById("dna-no-period");
    dom.dnaPunctEmdash = document.getElementById("dna-punct-emdash");
    dom.dnaPunctComma = document.getElementById("dna-punct-comma");
    dom.dnaPunctExcl = document.getElementById("dna-punct-excl");
    dom.dnaOpenersCloud = document.getElementById("dna-openers-cloud");
    dom.dnaSlangCloud = document.getElementById("dna-slang-cloud");
    dom.dnaDirectivesList = document.getElementById("dna-directives-list");
    dom.addRuleForm = document.getElementById("add-rule-form");
    dom.ruleAvoid = document.getElementById("rule-avoid");
    dom.rulePrefer = document.getElementById("rule-prefer");
    dom.rulesTableBody = document.getElementById("rules-table-body");
    dom.editPairsContainer = document.getElementById("edit-pairs-container");
    dom.historyContainer = document.getElementById("history-container");

    // Tab 4: macOS Hotkey & CLI
    dom.btnInstallHotkey = document.getElementById("btn-install-hotkey");
    dom.hotkeyInstallStatus = document.getElementById("hotkey-install-status");
    dom.copySnippetButtons = document.querySelectorAll(".btn-copy-snippet");

    // Settings Modal
    dom.settingsModalBackdrop = document.getElementById("modal-settings-backdrop");
    dom.btnCloseSettings = document.getElementById("btn-close-settings");
    dom.btnCancelSettings = document.getElementById("btn-cancel-settings");
    dom.settingsForm = document.getElementById("settings-form");
    dom.settingProvider = document.getElementById("setting-provider");
    dom.settingGeminiKey = document.getElementById("setting-gemini-key");
    dom.settingGeminiModel = document.getElementById("setting-gemini-model");
    dom.badgeGemini = document.getElementById("badge-gemini");
    dom.btnClearGemini = document.getElementById("btn-clear-gemini");
    dom.settingOpenaiKey = document.getElementById("setting-openai-key");
    dom.settingOpenaiModel = document.getElementById("setting-openai-model");
    dom.badgeOpenai = document.getElementById("badge-openai");
    dom.btnClearOpenai = document.getElementById("btn-clear-openai");
    dom.settingAnthropicKey = document.getElementById("setting-anthropic-key");
    dom.settingAnthropicModel = document.getElementById("setting-anthropic-model");
    dom.badgeAnthropic = document.getElementById("badge-anthropic");
    dom.btnClearAnthropic = document.getElementById("btn-clear-anthropic");
    dom.settingOllamaUrl = document.getElementById("setting-ollama-url");
    dom.settingOllamaModel = document.getElementById("setting-ollama-model");
    dom.badgePangram = document.getElementById("badge-pangram");
    dom.settingPangramKey = document.getElementById("setting-pangram-key");
    dom.btnClearPangram = document.getElementById("btn-clear-pangram");
    dom.btnTestPangram = document.getElementById("btn-test-pangram");
    dom.settingPangramModel = document.getElementById("setting-pangram-model");
    dom.settingPangramPublicLink = document.getElementById("setting-pangram-public-link");

    // Tab 5: AI & Originality Inspector
    dom.btnNavInspector = document.getElementById("tab-nav-inspector");
    dom.paneInspector = document.getElementById("pane-inspector");
    dom.inspectorInput = document.getElementById("inspector-input");
    dom.inspectorWordCount = document.getElementById("inspector-word-count");
    dom.inspectorCharCount = document.getElementById("inspector-char-count");
    dom.inspectorReadinessPill = document.getElementById("inspector-readiness-pill");
    dom.inspectorCostEst = document.getElementById("inspector-cost-est");
    dom.btnInspectorClipboard = document.getElementById("btn-inspector-clipboard");
    dom.btnInspectorUpload = document.getElementById("btn-inspector-upload");
    dom.fileInspectorUpload = document.getElementById("file-inspector-upload");
    dom.btnInspectorSample = document.getElementById("btn-inspector-sample");
    dom.btnInspectorSettings = document.getElementById("btn-inspector-settings");
    dom.inspectorModelSelect = document.getElementById("inspector-model-select");
    dom.inspectorShareableLink = document.getElementById("inspector-shareable-link");
    dom.inspectorDocTitle = document.getElementById("inspector-doc-title");
    dom.btnRunFreePrescan = document.getElementById("btn-run-free-prescan");
    dom.btnRunPlagiarismScan = document.getElementById("btn-run-plagiarism-scan");
    dom.btnRunPangramScan = document.getElementById("btn-run-pangram-scan");
    dom.inspectorPrescanBox = document.getElementById("inspector-prescan-box");
    dom.prescanSummaryText = document.getElementById("prescan-summary-text");
    dom.btnClosePrescan = document.getElementById("btn-close-prescan");
    dom.prescanHumanScore = document.getElementById("prescan-human-score");
    dom.prescanAiRisk = document.getElementById("prescan-ai-risk");
    dom.prescanBurstiness = document.getElementById("prescan-burstiness");
    dom.prescanFlagsCount = document.getElementById("prescan-flags-count");
    dom.prescanFlagsList = document.getElementById("prescan-flags-list");
    dom.inspectorEmptyState = document.getElementById("inspector-empty-state");
    dom.inspectorActiveResults = document.getElementById("inspector-active-results");
    dom.inspectorHeadlineBadge = document.getElementById("inspector-headline-badge");
    dom.inspectorHeadlineTitle = document.getElementById("inspector-headline-title");
    dom.inspectorHeadlinePrediction = document.getElementById("inspector-headline-prediction");
    dom.inspectorModelBadge = document.getElementById("inspector-model-badge");
    dom.inspectorDashboardLink = document.getElementById("inspector-dashboard-link");
    dom.lblMeterAi = document.getElementById("lbl-meter-ai");
    dom.lblMeterAssisted = document.getElementById("lbl-meter-assisted");
    dom.lblMeterHuman = document.getElementById("lbl-meter-human");
    dom.meterSegmentAi = document.getElementById("meter-segment-ai");
    dom.meterSegmentAssisted = document.getElementById("meter-segment-assisted");
    dom.meterSegmentHuman = document.getElementById("meter-segment-human");
    dom.cardValAi = document.getElementById("card-val-ai");
    dom.cardSubAi = document.getElementById("card-sub-ai");
    dom.cardValAssisted = document.getElementById("card-val-assisted");
    dom.cardSubAssisted = document.getElementById("card-sub-assisted");
    dom.cardValHuman = document.getElementById("card-val-human");
    dom.cardSubHuman = document.getElementById("card-sub-human");
    dom.cardValHumanizer = document.getElementById("card-val-humanizer");
    dom.cardSubHumanizer = document.getElementById("card-sub-humanizer");
    dom.inspectorHighlightedText = document.getElementById("inspector-highlighted-text");
    dom.inspectorWindowsCount = document.getElementById("inspector-windows-count");
    dom.inspectorWindowsList = document.getElementById("inspector-windows-list");
    dom.inspectorPlagiarismSection = document.getElementById("inspector-plagiarism-section");
    dom.inspectorPlagiarismScoreBadge = document.getElementById("inspector-plagiarism-score-badge");
    dom.inspectorPlagiarismStatus = document.getElementById("inspector-plagiarism-status");
    dom.inspectorPlagiarismMatches = document.getElementById("inspector-plagiarism-matches");
    dom.inspectorHistoryCount = document.getElementById("inspector-history-count");
    dom.inspectorHistoryTbody = document.getElementById("inspector-history-tbody");

    // Tab 6: Source & Revision Coach
    dom.btnNavAcademic = document.getElementById("tab-nav-academic");
    dom.paneAcademic = document.getElementById("pane-academic");
    dom.btnAcadSample = document.getElementById("btn-acad-sample");
    dom.btnAcadClipboard = document.getElementById("btn-acad-clipboard");
    dom.btnAcadClear = document.getElementById("btn-acad-clear");
    dom.acadDraftInput = document.getElementById("acad-draft-input");
    dom.acadDraftWordCount = document.getElementById("acad-draft-word-count");
    dom.btnRunCoach = document.getElementById("btn-run-coach");
    dom.acadSourceInput = document.getElementById("acad-source-input");
    dom.acadSourceWordCount = document.getElementById("acad-source-word-count");
    dom.btnRunCitationCheck = document.getElementById("btn-run-citation-check");
    dom.btnRunBothAcad = document.getElementById("btn-run-both-acad");
    dom.acadCitationReport = document.getElementById("acad-citation-report");
    dom.acadStatusBadge = document.getElementById("acad-status-badge");
    dom.acadOverlapBadge = document.getElementById("acad-overlap-badge");
    dom.acadCitationEmpty = document.getElementById("acad-citation-empty");
    dom.acadCitationBody = document.getElementById("acad-citation-body");
    dom.acadStatusLabel = document.getElementById("acad-status-label");
    dom.acadQuotedCount = document.getElementById("acad-quoted-count");
    dom.acadMatchesCount = document.getElementById("acad-matches-count");
    dom.acadAnnotatedDraft = document.getElementById("acad-annotated-draft");
    dom.acadMatchesList = document.getElementById("acad-matches-list");
    dom.acadCoachReport = document.getElementById("acad-coach-report");
    dom.acadClarityScore = document.getElementById("acad-clarity-score");
    dom.acadCoachEmpty = document.getElementById("acad-coach-empty");
    dom.acadCoachBody = document.getElementById("acad-coach-body");
    dom.acadCountPassive = document.getElementById("acad-count-passive");
    dom.acadCountUnattributed = document.getElementById("acad-count-unattributed");
    dom.acadCountFiller = document.getElementById("acad-count-filler");
    dom.acadCountOpener = document.getElementById("acad-count-opener");
    dom.acadCountLong = document.getElementById("acad-count-long");
    dom.acadPriorityActions = document.getElementById("acad-priority-actions");
    dom.acadCoachList = document.getElementById("acad-coach-list");

    // Toast Container
    dom.toastContainer = document.getElementById("toast-container");
  }

  // =========================================================================
  // API Fetch Wrappers
  // =========================================================================
  async function apiRequest(endpoint, method = "GET", body = null) {
    const options = {
      method,
      headers: {
        Accept: "application/json",
      },
    };
    if (body !== null) {
      options.headers["Content-Type"] = "application/json";
      options.body = JSON.stringify(body);
    }
    const response = await fetch(endpoint, options);
    if (!response.ok) {
      let errDetail = response.statusText;
      try {
        const errJson = await response.json();
        if (errJson && errJson.error) errDetail = errJson.error;
      } catch (_) {}
      throw new Error(`Server error (${response.status}): ${errDetail}`);
    }
    return await response.json();
  }

  // =========================================================================
  // Toast Notifications
  // =========================================================================
  function showToast(title, message = "", type = "info", duration = 3500) {
    if (!dom.toastContainer) return;

    const toast = document.createElement("div");
    toast.className = `toast toast-${type}`;

    let iconSvg = "";
    if (type === "success") {
      iconSvg =
        '<svg viewBox="0 0 20 20" fill="currentColor" class="toast-icon"><path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clip-rule="evenodd"/></svg>';
    } else if (type === "error") {
      iconSvg =
        '<svg viewBox="0 0 20 20" fill="currentColor" class="toast-icon"><path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clip-rule="evenodd"/></svg>';
    } else {
      iconSvg =
        '<svg viewBox="0 0 20 20" fill="currentColor" class="toast-icon"><path fill-rule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clip-rule="evenodd"/></svg>';
    }

    toast.innerHTML = `
      ${iconSvg}
      <div class="toast-content">
        <div class="toast-title">${escapeHtml(title)}</div>
        ${message ? `<div class="toast-message">${escapeHtml(message)}</div>` : ""}
      </div>
    `;

    dom.toastContainer.appendChild(toast);

    setTimeout(() => {
      toast.style.opacity = "0";
      toast.style.transform = "translateX(20px)";
      setTimeout(() => {
        if (toast.parentNode) toast.parentNode.removeChild(toast);
      }, 200);
    }, duration);
  }

  // =========================================================================
  // Text Utilities
  // =========================================================================
  function escapeHtml(str) {
    if (!str) return "";
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  function countWords(str) {
    const trimmed = (str || "").trim();
    if (!trimmed) return 0;
    return trimmed.split(/\s+/).length;
  }

  function getModeDisplayLabel(mode) {
    const labels = {
      casual_chat: "Casual Chat",
      work_email: "Work Email",
      long_form: "Long-Form",
      social_post: "Social / DM",
    };
    return labels[mode] || mode;
  }

  // =========================================================================
  // Sparkline SVG Generator
  // =========================================================================
  function renderSparklineSvg(container, lengths, isOutput = false) {
    if (!container) return;
    if (!lengths || lengths.length === 0) {
      container.innerHTML = `<div class="sparkline-placeholder">${
        isOutput ? "Output sentence rhythm will appear here" : "Type or paste text to visualize sentence rhythm"
      }</div>`;
      return;
    }

    const width = 460;
    const height = 54;
    const padding = 4;
    const n = lengths.length;

    const maxLen = Math.max(16, ...lengths);
    const sum = lengths.reduce((acc, val) => acc + val, 0);
    const avg = sum / n;

    // Bar dimensions
    const gap = Math.max(2, Math.min(8, Math.floor(width / (n * 3))));
    const barWidth = Math.max(3, Math.min(22, (width - (n - 1) * gap) / n));
    const totalBarsWidth = n * barWidth + (n - 1) * gap;
    const startX = Math.max(0, (width - totalBarsWidth) / 2);

    let barsHtml = "";
    lengths.forEach((len, idx) => {
      const barH = Math.max(4, Math.round((len / maxLen) * (height - padding * 2)));
      const x = startX + idx * (barWidth + gap);
      const y = height - barH - padding;
      const barClass = isOutput ? "spark-bar spark-bar-output" : "spark-bar";
      barsHtml += `<rect class="${barClass}" x="${x}" y="${y}" width="${barWidth}" height="${barH}"><title>Sentence ${
        idx + 1
      }: ${len} words</title></rect>`;
    });

    const avgY = Math.round(height - (avg / maxLen) * (height - padding * 2) - padding);
    const meanLineHtml = `<line class="spark-mean-line" x1="${startX}" y1="${avgY}" x2="${
      startX + totalBarsWidth
    }" y2="${avgY}"><title>Mean length: ${avg.toFixed(1)} words</title></line>`;

    container.innerHTML = `
      <svg class="sparkline-svg" viewBox="0 0 ${width} ${height}" preserveAspectRatio="none">
        ${barsHtml}
        ${meanLineHtml}
      </svg>
    `;
  }

  // =========================================================================
  // State Synchronization & Rendering
  // =========================================================================
  function applyState(data) {
    if (!data) return;
    if (data.settings) appState.settings = data.settings;
    if (data.active_context_mode) appState.activeContextMode = data.active_context_mode;
    if (data.dna_by_mode) appState.dnaByMode = data.dna_by_mode;
    if (data.active_dna) appState.activeDna = data.active_dna;
    if (data.samples) appState.samples = data.samples;
    if (data.learned_replacements) appState.rules = data.learned_replacements;
    if (data.edit_pairs) appState.editPairs = data.edit_pairs;
    if (data.history) appState.history = data.history;
    if (data.maturity) appState.maturity = data.maturity;
    if (data.pangram_scans) appState.pangramScans = data.pangram_scans;

    if (appState.settings.intensity) {
      appState.intensity = appState.settings.intensity;
    }

    renderHeader();
    renderModePills();
    renderIntensitySelector();
    renderDnaTab();
    renderSamplesTab();
    renderRulesTab();
    renderHistoryAndPairs();
    renderPangramHistory();
    renderSettingsModal();
  }

  function renderHeader() {
    const mat = appState.maturity;
    if (dom.maturityLabel) dom.maturityLabel.textContent = mat.label || "Seedling";
    if (dom.maturityScore) dom.maturityScore.textContent = `${mat.score || 0}%`;
    if (dom.maturityBar) dom.maturityBar.style.width = `${Math.min(100, Math.max(0, mat.score || 0))}%`;
    if (dom.maturitySamples) dom.maturitySamples.textContent = `${mat.total_samples || 0} samples`;
    if (dom.maturityGold) dom.maturityGold.textContent = `${mat.gold_edits || 0} gold edits`;
  }

  function renderModePills() {
    dom.modePills.forEach((pill) => {
      const mode = pill.getAttribute("data-mode");
      if (mode === appState.activeContextMode) {
        pill.classList.add("active");
      } else {
        pill.classList.remove("active");
      }
    });

    if (dom.importMode) dom.importMode.value = appState.activeContextMode;
    if (dom.dnaActiveModeTitle) {
      dom.dnaActiveModeTitle.textContent = getModeDisplayLabel(appState.activeContextMode);
    }
    if (dom.samplesSubtitle) {
      dom.samplesSubtitle.textContent = `Showing stored writing samples for ${getModeDisplayLabel(
        appState.activeContextMode
      )}`;
    }
  }

  function renderIntensitySelector() {
    dom.intensityButtons.forEach((btn) => {
      const val = btn.getAttribute("data-intensity");
      if (val === appState.intensity) {
        btn.classList.add("active");
      } else {
        btn.classList.remove("active");
      }
    });
  }

  function renderAnalysisData(analysis, targetDna) {
    if (!analysis) return;

    // Human Score
    const humanScore = analysis.human_score ?? 0;
    if (dom.valHumanScore) {
      dom.valHumanScore.textContent = humanScore;
      if (humanScore >= 75) {
        dom.valHumanScore.className = "metric-value font-mono text-emerald";
        dom.hintHumanScore.textContent = "High human alignment";
      } else if (humanScore >= 50) {
        dom.valHumanScore.className = "metric-value font-mono text-amber";
        dom.hintHumanScore.textContent = "Moderate AI characteristics";
      } else {
        dom.valHumanScore.className = "metric-value font-mono text-rose";
        dom.hintHumanScore.textContent = "Heavy synthetic clichés";
      }
    }
    if (dom.barHumanScore) {
      dom.barHumanScore.style.width = `${humanScore}%`;
    }

    // AI Risk Score
    const aiRisk = analysis.ai_risk_score ?? 0;
    if (dom.valAiRisk) dom.valAiRisk.textContent = aiRisk;
    if (dom.barAiRisk) dom.barAiRisk.style.width = `${aiRisk}%`;

    // Burstiness Score
    const burstiness = analysis.burstiness_score ?? 0;
    if (dom.valBurstiness) dom.valBurstiness.textContent = burstiness;
    if (dom.barBurstiness) dom.barBurstiness.style.width = `${burstiness}%`;

    // Sentence Length Sparkline
    const lengths = analysis.sentence_lengths || [];
    const mean = analysis.metrics?.mean_sentence_len || 0;
    const std = analysis.metrics?.stdev_sentence_len || 0;
    if (dom.sourceCadenceStats) {
      dom.sourceCadenceStats.textContent = `Mean: ${mean.toFixed(1)}w • StdDev: ${std.toFixed(1)}w (${
        lengths.length
      } sentences)`;
    }
    renderSparklineSvg(dom.sourceSparkline, lengths, false);

    // AI Tells Chips
    const flags = analysis.flags || [];
    if (dom.tellsCountBadge) {
      dom.tellsCountBadge.textContent = `${flags.length} flag${flags.length === 1 ? "" : "s"}`;
    }
    if (dom.aiTellsContainer) {
      if (flags.length === 0) {
        dom.aiTellsContainer.innerHTML = '<span class="tells-empty">No synthetic clichés detected</span>';
      } else {
        dom.aiTellsContainer.innerHTML = flags
          .map(
            (flag) => `
          <div class="tell-chip" title="Category: ${escapeHtml(flag.category || "AI Tell")}">
            <span class="tell-phrase">${escapeHtml(flag.phrase)}</span>
            ${flag.suggestion ? `<span class="tell-suggest">→ ${escapeHtml(flag.suggestion)}</span>` : ""}
          </div>
        `
          )
          .join("");
      }
    }
  }

  function renderOutputComparison(result) {
    if (!result) return;
    const before = result.before || {};
    const after = result.after || {};

    if (dom.compHumanBefore) dom.compHumanBefore.textContent = `${before.human_score ?? "--"}%`;
    if (dom.compHumanAfter) dom.compHumanAfter.textContent = `${after.human_score ?? "--"}%`;
    if (dom.compHumanDelta) {
      const delta = (after.human_score || 0) - (before.human_score || 0);
      dom.compHumanDelta.textContent = delta >= 0 ? `+${delta}` : `${delta}`;
      dom.compHumanDelta.className = delta >= 0 ? "comp-badge badge-emerald" : "comp-badge";
    }

    if (dom.compBurstBefore) dom.compBurstBefore.textContent = before.burstiness_score ?? "--";
    if (dom.compBurstAfter) dom.compBurstAfter.textContent = after.burstiness_score ?? "--";
    if (dom.compBurstDelta) {
      const bDelta = (after.burstiness_score || 0) - (before.burstiness_score || 0);
      dom.compBurstDelta.textContent = bDelta >= 0 ? `+${bDelta}` : `${bDelta}`;
    }

    if (dom.compDnaMatch) {
      dom.compDnaMatch.textContent = `${after.style_match_score ?? 85}%`;
    }
    if (dom.compSamplesRef) {
      dom.compSamplesRef.textContent = result.samples_referenced ?? 0;
    }

    // Output sparkline
    const outLengths = after.sentence_lengths || [];
    const outMean = after.metrics?.mean_sentence_len || 0;
    const outStd = after.metrics?.stdev_sentence_len || 0;
    if (dom.outputCadenceStats) {
      dom.outputCadenceStats.textContent = `Mean: ${outMean.toFixed(1)}w • StdDev: ${outStd.toFixed(1)}w (${
        outLengths.length
      } sentences)`;
    }
    renderSparklineSvg(dom.outputSparkline, outLengths, true);

    // Provider pill
    if (dom.providerPill && dom.providerName) {
      dom.providerPill.style.display = "flex";
      dom.providerName.textContent = result.provider_used || "deterministic_local";
    }
  }

  function renderDnaTab() {
    const dna = appState.activeDna || {};
    if (dom.dnaAvgLen) dom.dnaAvgLen.textContent = dna.avg_sentence_length?.toFixed(1) || "13.0";
    if (dom.dnaStdLen) dom.dnaStdLen.textContent = `${dna.sentence_length_std?.toFixed(1) || "5.5"} words`;
    if (dom.dnaContractionRate) {
      const cr = Math.round((dna.contraction_rate || 0.65) * 100);
      dom.dnaContractionRate.textContent = `${cr}%`;
    }
    if (dom.dnaLowerStart) {
      const lsr = Math.round((dna.lowercase_start_rate || 0) * 100);
      dom.dnaLowerStart.textContent = `${lsr}%`;
    }
    if (dom.dnaLowerI) {
      const lir = Math.round((dna.lowercase_i_rate || 0) * 100);
      dom.dnaLowerI.textContent = `${lir}%`;
    }
    if (dom.dnaNoPeriod) {
      const npr = Math.round((dna.no_trailing_period_rate || 0) * 100);
      dom.dnaNoPeriod.textContent = `${npr}%`;
    }

    const punct = dna.punctuation_profile || {};
    if (dom.dnaPunctEmdash) dom.dnaPunctEmdash.textContent = punct.em_dash?.toFixed(1) || "0.0";
    if (dom.dnaPunctComma) dom.dnaPunctComma.textContent = punct.comma?.toFixed(1) || "0.0";
    if (dom.dnaPunctExcl) dom.dnaPunctExcl.textContent = punct.exclamation?.toFixed(1) || "0.0";

    // Openers
    if (dom.dnaOpenersCloud) {
      const openers = dna.common_openers || [];
      if (openers.length === 0) {
        dom.dnaOpenersCloud.innerHTML = '<span class="text-muted text-sm">No frequent openers detected yet</span>';
      } else {
        dom.dnaOpenersCloud.innerHTML = openers
          .map((op) => `<span class="dna-tag font-mono">${escapeHtml(op)}</span>`)
          .join("");
      }
    }

    // Slang & Fillers
    if (dom.dnaSlangCloud) {
      const slang = dna.common_fillers_and_slang || [];
      if (slang.length === 0) {
        dom.dnaSlangCloud.innerHTML = '<span class="text-muted text-sm">No slang or filler markers detected yet</span>';
      } else {
        dom.dnaSlangCloud.innerHTML = slang
          .map((s) => `<span class="dna-tag text-emerald font-mono">${escapeHtml(s)}</span>`)
          .join("");
      }
    }

    // Directives
    if (dom.dnaDirectivesList) {
      const directives = dna.directives_summary || [];
      if (directives.length === 0) {
        dom.dnaDirectivesList.innerHTML =
          '<li>Using default natural human cadence. Upload samples to personalize!</li>';
      } else {
        dom.dnaDirectivesList.innerHTML = directives
          .map((dir) => `<li>${escapeHtml(dir)}</li>`)
          .join("");
      }
    }
  }

  function renderSamplesTab(filterKeyword = "") {
    if (!dom.samplesListContainer) return;
    const samples = appState.samples || [];
    const query = filterKeyword.trim().toLowerCase();

    const filtered = query
      ? samples.filter((s) => (s.content || "").toLowerCase().includes(query))
      : samples;

    if (filtered.length === 0) {
      dom.samplesListContainer.innerHTML = `
        <div class="samples-empty">
          ${query ? `No samples matching "${escapeHtml(query)}"` : "No samples uploaded for this mode yet. Upload chats or click 'Load Demo Voice Samples' above!"}
        </div>
      `;
      return;
    }

    dom.samplesListContainer.innerHTML = filtered
      .map((s) => {
        const isGold = parseFloat(s.weight || 1.0) > 1.5;
        const dateStr = s.created_at ? s.created_at.replace("T", " ").substring(0, 16) : "";
        return `
        <div class="sample-item-card ${isGold ? "gold-sample" : ""}" data-sample-id="${s.id}">
          <div class="sample-item-head">
            <div class="sample-badges-group">
              <span class="source-badge">${escapeHtml(s.source_type || "manual")}</span>
              ${
                isGold
                  ? '<span class="weight-badge-gold">★ Gold Edit 2.5x</span>'
                  : '<span class="weight-badge-normal">Uploaded 1.0x</span>'
              }
            </div>
            <button type="button" class="btn btn-ghost btn-sm text-rose btn-delete-sample" data-id="${
              s.id
            }" title="Delete sample">
              Delete
            </button>
          </div>
          <div class="sample-content" title="Click to expand/collapse">${escapeHtml(s.content)}</div>
          <div class="sample-meta-row">
            <span>${s.word_count || countWords(s.content)} words</span>
            <span>${dateStr}</span>
          </div>
        </div>
      `;
      })
      .join("");
  }

  function renderRulesTab() {
    if (!dom.rulesTableBody) return;
    const rules = appState.rules || [];

    if (rules.length === 0) {
      dom.rulesTableBody.innerHTML = `
        <tr>
          <td colspan="5" class="text-muted text-center" style="padding: 24px; text-align: center;">
            No learned or custom replacement rules yet. Add one above or edit drafts in the Studio!
          </td>
        </tr>
      `;
      return;
    }

    dom.rulesTableBody.innerHTML = rules
      .map((r) => {
        let sourceClass = "source-tag-auto";
        if (r.source === "manual") sourceClass = "source-tag-manual";
        else if (r.source === "auto_diff_cut") sourceClass = "source-tag-cut";

        return `
        <tr data-rule-id="${r.id}">
          <td><strong class="text-rose font-mono">${escapeHtml(r.avoid_text)}</strong></td>
          <td><span class="text-emerald font-mono">${escapeHtml(r.prefer_text || "— (trimmed)")}</span></td>
          <td><span class="source-tag ${sourceClass}">${escapeHtml(r.source || "auto")}</span></td>
          <td class="font-mono">${r.times_seen || 1}</td>
          <td class="text-right">
            <button type="button" class="btn btn-ghost btn-sm text-rose btn-delete-rule" data-id="${
              r.id
            }" title="Delete rule">
              Delete
            </button>
          </td>
        </tr>
      `;
      })
      .join("");
  }

  function renderHistoryAndPairs() {
    // Edit pairs
    if (dom.editPairsContainer) {
      const pairs = appState.editPairs || [];
      if (pairs.length === 0) {
        dom.editPairsContainer.innerHTML =
          '<div class="text-muted text-sm" style="padding: 16px;">No final-edit learning pairs recorded yet. Edit a rewritten draft in the Studio and click "Copy to Clipboard & Learn from My Edits"!</div>';
      } else {
        dom.editPairsContainer.innerHTML = pairs
          .map((p) => {
            let extracted = [];
            try {
              extracted = JSON.parse(p.extracted_rules_json || "[]");
            } catch (_) {}
            const dateStr = p.created_at ? p.created_at.replace("T", " ").substring(0, 16) : "";

            return `
            <div class="edit-pair-item">
              <div style="font-size: 11px; color: var(--text-faint); margin-bottom: 4px;">${dateStr}</div>
              <div style="font-size: 12px; margin-bottom: 4px;">
                <span class="text-muted">Draft:</span> <span style="text-decoration: line-through; color: #f87171;">${escapeHtml(
                  p.ai_draft
                )}</span>
              </div>
              <div style="font-size: 12px; margin-bottom: 6px;">
                <span class="text-emerald font-bold">Your Edit:</span> <span>${escapeHtml(
                  p.user_final_edit
                )}</span>
              </div>
              <div>
                ${extracted
                  .map((rule) => `<span class="edit-pair-rules-chip">✓ ${escapeHtml(rule)}</span>`)
                  .join("")}
              </div>
            </div>
          `;
          })
          .join("");
      }
    }

    // History
    if (dom.historyContainer) {
      const hist = appState.history || [];
      if (hist.length === 0) {
        dom.historyContainer.innerHTML =
          '<div class="text-muted text-sm" style="padding: 16px;">No recent rewrites recorded yet.</div>';
      } else {
        dom.historyContainer.innerHTML = hist
          .map((h) => {
            const dateStr = h.created_at ? h.created_at.replace("T", " ").substring(0, 16) : "";
            const before = h.before_human_score ?? "--";
            const after = h.after_human_score ?? "--";
            return `
            <div class="history-item">
              <div style="display: flex; justify-content: space-between; font-size: 11px; margin-bottom: 4px;">
                <span class="text-muted">${getModeDisplayLabel(h.context_mode)} • ${escapeHtml(
                  h.provider_used || "local"
                )}</span>
                <span class="text-faint">${dateStr}</span>
              </div>
              <div style="display: flex; align-items: baseline; gap: 8px; font-size: 12px; margin-bottom: 4px;">
                <span>Human Score:</span>
                <strong class="font-mono">${before}% → <span class="text-emerald">${after}%</span></strong>
              </div>
              <div style="font-size: 12px; color: var(--text-main); line-height: 1.4; max-height: 48px; overflow: hidden; text-overflow: ellipsis;">
                ${escapeHtml(h.output_text)}
              </div>
            </div>
          `;
          })
          .join("");
      }
    }
  }

  function renderSettingsModal() {
    const s = appState.settings || {};
    if (dom.settingProvider) dom.settingProvider.value = s.provider || "auto";

    // Gemini
    if (dom.settingGeminiModel) dom.settingGeminiModel.value = s.gemini_model || "gemini-2.5-flash";
    if (dom.badgeGemini) {
      if (s.gemini_api_key_configured) {
        dom.badgeGemini.textContent = `Configured (${s.gemini_api_key_preview || "••••"})`;
        dom.badgeGemini.className = "provider-status-badge badge-configured";
      } else {
        dom.badgeGemini.textContent = "Not Configured";
        dom.badgeGemini.className = "provider-status-badge badge-unconfigured";
      }
    }

    // OpenAI
    if (dom.settingOpenaiModel) dom.settingOpenaiModel.value = s.openai_model || "gpt-4o-mini";
    if (dom.badgeOpenai) {
      if (s.openai_api_key_configured) {
        dom.badgeOpenai.textContent = `Configured (${s.openai_api_key_preview || "••••"})`;
        dom.badgeOpenai.className = "provider-status-badge badge-configured";
      } else {
        dom.badgeOpenai.textContent = "Not Configured";
        dom.badgeOpenai.className = "provider-status-badge badge-unconfigured";
      }
    }

    // Anthropic
    if (dom.settingAnthropicModel)
      dom.settingAnthropicModel.value = s.anthropic_model || "claude-3-5-haiku-latest";
    if (dom.badgeAnthropic) {
      if (s.anthropic_api_key_configured) {
        dom.badgeAnthropic.textContent = `Configured (${s.anthropic_api_key_preview || "••••"})`;
        dom.badgeAnthropic.className = "provider-status-badge badge-configured";
      } else {
        dom.badgeAnthropic.textContent = "Not Configured";
        dom.badgeAnthropic.className = "provider-status-badge badge-unconfigured";
      }
    }

    // Ollama
    if (dom.settingOllamaUrl) dom.settingOllamaUrl.value = s.ollama_url || "http://localhost:11434";
    if (dom.settingOllamaModel) dom.settingOllamaModel.value = s.ollama_model || "llama3.2";

    // Pangram Labs (Pay-As-You-Go)
    if (dom.settingPangramModel) {
      dom.settingPangramModel.value = s.pangram_model || "pangram-4";
    }
    if (dom.settingPangramPublicLink) {
      dom.settingPangramPublicLink.checked = s.pangram_public_link === "true" || s.pangram_public_link === true;
    }
    if (dom.badgePangram) {
      if (s.pangram_api_key_configured) {
        dom.badgePangram.textContent = `Configured (${s.pangram_api_key_preview || "••••"})`;
        dom.badgePangram.className = "provider-status-badge badge-configured";
      } else {
        dom.badgePangram.textContent = "Not Configured";
        dom.badgePangram.className = "provider-status-badge badge-unconfigured";
      }
    }
  }

  // =========================================================================
  // Live Analysis & Rewrite Interactions
  // =========================================================================
  function scheduleSourceAnalysis(delayMs = 300) {
    if (sourceDebounceTimer) clearTimeout(sourceDebounceTimer);
    sourceDebounceTimer = setTimeout(runSourceAnalysis, delayMs);
  }

  async function runSourceAnalysis() {
    const text = (dom.sourceInput?.value || "").trim();
    const countW = countWords(text);
    const countC = text.length;

    if (dom.sourceWordCount) dom.sourceWordCount.textContent = `${countW} words`;
    if (dom.sourceCharCount) dom.sourceCharCount.textContent = `${countC} chars`;

    if (!text) {
      renderAnalysisData({
        word_count: 0,
        char_count: 0,
        human_score: 0,
        ai_risk_score: 0,
        burstiness_score: 0,
        sentence_lengths: [],
        flags: [],
        metrics: { mean_sentence_len: 0, stdev_sentence_len: 0 },
      });
      return;
    }

    if (dom.analysisSpinner) dom.analysisSpinner.style.display = "flex";

    try {
      const res = await apiRequest("/api/analyze", "POST", {
        text,
        context_mode: appState.activeContextMode,
      });
      renderAnalysisData(res.analysis, appState.activeDna);
    } catch (err) {
      console.warn("Analysis failed:", err);
    } finally {
      if (dom.analysisSpinner) dom.analysisSpinner.style.display = "none";
    }
  }

  function handleOutputChanged() {
    const text = (dom.outputText?.value || "").trim();
    if (dom.outputWordCount) dom.outputWordCount.textContent = `${countWords(text)} words`;
    if (dom.outputCharCount) dom.outputCharCount.textContent = `${text.length} chars`;

    // Check if user edited compared to lastAiDraft
    const differs = appState.lastAiDraft && text !== appState.lastAiDraft.trim() && text.length > 0;
    if (dom.diffBadge) {
      dom.diffBadge.style.display = differs ? "flex" : "none";
    }

    // Debounce re-analysis of output
    if (outputDebounceTimer) clearTimeout(outputDebounceTimer);
    outputDebounceTimer = setTimeout(async () => {
      if (!text) return;
      try {
        const res = await apiRequest("/api/analyze", "POST", {
          text,
          context_mode: appState.activeContextMode,
        });
        const outLengths = res.analysis.sentence_lengths || [];
        const outMean = res.analysis.metrics?.mean_sentence_len || 0;
        const outStd = res.analysis.metrics?.stdev_sentence_len || 0;
        if (dom.outputCadenceStats) {
          dom.outputCadenceStats.textContent = `Mean: ${outMean.toFixed(1)}w • StdDev: ${outStd.toFixed(
            1
          )}w (${outLengths.length} sentences)`;
        }
        renderSparklineSvg(dom.outputSparkline, outLengths, true);
      } catch (_) {}
    }, 400);
  }

  async function handleGrabClipboard() {
    try {
      const data = await apiRequest(`/api/clipboard?mode=${encodeURIComponent(appState.activeContextMode)}`);
      if (dom.sourceInput) {
        dom.sourceInput.value = data.text || "";
        scheduleSourceAnalysis(10);
      }
      showToast("Clipboard Grabbed", `Imported ${countWords(data.text || "")} words from macOS clipboard`, "success");
    } catch (err) {
      showToast("Clipboard Error", err.message, "error");
    }
  }

  function handleTrySampleAiText() {
    if (dom.sourceInput) {
      dom.sourceInput.value = SAMPLE_AI_PARAGRAPH;
      scheduleSourceAnalysis(10);
      showToast("Sample Loaded", "AI-cliché heavy draft loaded for testing", "info");
    }
  }

  function handleClearSource() {
    if (dom.sourceInput) {
      dom.sourceInput.value = "";
      scheduleSourceAnalysis(10);
    }
  }

  async function handleRewrite() {
    const text = (dom.sourceInput?.value || "").trim();
    if (!text) {
      showToast("Input Required", "Please enter or paste text to rewrite", "error");
      return;
    }
    if (appState.isRewriting) return;

    appState.isRewriting = true;
    if (dom.btnRewrite) dom.btnRewrite.disabled = true;
    if (dom.btnRewriteText) dom.btnRewriteText.textContent = "Rewriting...";

    try {
      const payload = {
        text,
        context_mode: appState.activeContextMode,
        intensity: appState.intensity,
      };
      const res = await apiRequest("/api/rewrite", "POST", payload);

      if (dom.outputText) {
        dom.outputText.value = res.rewritten || "";
      }
      appState.lastAiDraft = res.rewritten || "";
      appState.lastOriginalInput = res.original || text;

      if (dom.diffBadge) dom.diffBadge.style.display = "none";
      if (dom.learnedHabitsBanner) dom.learnedHabitsBanner.style.display = "none";

      renderOutputComparison(res);
      handleOutputChanged();

      showToast("VoicePrint Humanized", `Rewritten using engine: ${res.provider_used || "local"}`, "success");

      // Refresh history & state
      refreshStateSilently();
    } catch (err) {
      showToast("Rewrite Failed", err.message, "error");
    } finally {
      appState.isRewriting = false;
      if (dom.btnRewrite) dom.btnRewrite.disabled = false;
      if (dom.btnRewriteText) dom.btnRewriteText.textContent = "Rewrite in My Voice";
    }
  }

  async function handleQuickRewrite() {
    if (appState.isRewriting) return;
    appState.isRewriting = true;
    if (dom.btnQuickRewrite) dom.btnQuickRewrite.disabled = true;

    try {
      const res = await apiRequest("/api/rewrite", "POST", {
        from_clipboard: true,
        auto_copy: true,
        context_mode: appState.activeContextMode,
        intensity: appState.intensity,
      });

      if (dom.sourceInput) dom.sourceInput.value = res.original || "";
      if (dom.outputText) dom.outputText.value = res.rewritten || "";

      appState.lastAiDraft = res.rewritten || "";
      appState.lastOriginalInput = res.original || "";

      scheduleSourceAnalysis(10);
      renderOutputComparison(res);
      handleOutputChanged();

      // Client clipboard backup
      if (res.rewritten && navigator.clipboard) {
        try {
          await navigator.clipboard.writeText(res.rewritten);
        } catch (_) {}
      }

      showToast("1-Click Complete", "Grabbed clipboard, humanized in your voice, and copied back!", "success");
      refreshStateSilently();
    } catch (err) {
      showToast("1-Click Failed", err.message, "error");
    } finally {
      appState.isRewriting = false;
      if (dom.btnQuickRewrite) dom.btnQuickRewrite.disabled = false;
    }
  }

  async function handleCopyAndLearn() {
    const finalEdit = (dom.outputText?.value || "").trim();
    if (!finalEdit) {
      showToast("Output Empty", "Nothing to copy or learn from", "error");
      return;
    }

    try {
      // 1. Write to clipboard via server API
      const payload = {
        context_mode: appState.activeContextMode,
        original_input: appState.lastOriginalInput || dom.sourceInput?.value || "",
        ai_draft: appState.lastAiDraft || finalEdit,
        user_final_edit: finalEdit,
        copy: true,
      };

      const res = await apiRequest("/api/learn-edit", "POST", payload);

      // Client clipboard backup
      if (navigator.clipboard) {
        try {
          await navigator.clipboard.writeText(finalEdit);
        } catch (_) {}
      }

      // Update state
      if (res.state) {
        applyState(res.state);
      }

      // Show habits banner
      const habits = res.learned_habits || [];
      if (dom.learnedHabitsBanner && dom.learnedHabitsList) {
        dom.learnedHabitsBanner.style.display = "block";
        dom.learnedHabitsList.innerHTML = habits.map((h) => `<li>${escapeHtml(h)}</li>`).join("");
      }

      if (dom.diffBadge) dom.diffBadge.style.display = "none";
      appState.lastAiDraft = finalEdit;

      showToast(
        "Copied & Learned!",
        `Saved as 2.5x gold sample • Learned ${res.rules_added || 0} new rules`,
        "success"
      );
    } catch (err) {
      showToast("Learning Failed", err.message, "error");
    }
  }

  async function handleCopyOnly() {
    const text = dom.outputText?.value || "";
    if (!text) return;
    try {
      await apiRequest("/api/clipboard", "POST", { text });
      if (navigator.clipboard) {
        await navigator.clipboard.writeText(text);
      }
      showToast("Copied to Clipboard", `${countWords(text)} words copied`, "success");
    } catch (err) {
      showToast("Copy Failed", err.message, "error");
    }
  }

  // =========================================================================
  // Context Mode & Tab Switching
  // =========================================================================
  async function switchContextMode(newMode) {
    if (newMode === appState.activeContextMode) return;
    appState.activeContextMode = newMode;
    renderModePills();

    try {
      const res = await apiRequest("/api/settings", "POST", {
        active_context_mode: newMode,
      });
      applyState(res.state);
      // Re-analyze source input under new mode DNA
      scheduleSourceAnalysis(10);
      showToast("Mode Switched", `Active context mode is now ${getModeDisplayLabel(newMode)}`, "info");
    } catch (err) {
      showToast("Mode Switch Failed", err.message, "error");
    }
  }

  function switchTab(tabName) {
    appState.activeTab = tabName;
    dom.tabButtons.forEach((btn) => {
      if (btn.getAttribute("data-tab") === tabName) {
        btn.classList.add("active");
      } else {
        btn.classList.remove("active");
      }
    });

    dom.tabPanes.forEach((pane) => {
      if (pane.id === `pane-${tabName}`) {
        pane.classList.add("active");
      } else {
        pane.classList.remove("active");
      }
    });
  }

  function handleIntensityClick(intensity) {
    appState.intensity = intensity;
    renderIntensitySelector();
    apiRequest("/api/settings", "POST", { intensity }).catch(() => {});
  }

  // =========================================================================
  // Grow With Me (Tab 2) Importer & Samples
  // =========================================================================
  function setupFileDropzone() {
    const dropzone = dom.importDropzone;
    const filePicker = dom.filePicker;
    if (!dropzone || !filePicker) return;

    dom.btnBrowseFile?.addEventListener("click", () => filePicker.click());
    dropzone.addEventListener("click", (e) => {
      if (e.target !== dom.btnBrowseFile) filePicker.click();
    });

    ["dragenter", "dragover"].forEach((eventName) => {
      dropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropzone.classList.add("dragover");
      });
    });

    ["dragleave", "drop"].forEach((eventName) => {
      dropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropzone.classList.remove("dragover");
      });
    });

    dropzone.addEventListener("drop", (e) => {
      const dt = e.dataTransfer;
      const files = dt.files;
      if (files && files[0]) handleUploadedFile(files[0]);
    });

    filePicker.addEventListener("change", (e) => {
      if (filePicker.files && filePicker.files[0]) {
        handleUploadedFile(filePicker.files[0]);
      }
    });
  }

  function handleUploadedFile(file) {
    const reader = new FileReader();
    reader.onload = function (evt) {
      const content = evt.target.result;
      if (dom.importText) dom.importText.value = content;
      showToast("File Loaded", `Read ${file.name} (${Math.round(file.size / 1024)} KB)`, "info");
    };
    reader.onerror = function () {
      showToast("File Read Error", "Could not read uploaded file", "error");
    };
    reader.readAsText(file);
  }

  async function handleImportSubmit(e) {
    e.preventDefault();
    const raw_text = (dom.importText?.value || "").trim();
    if (!raw_text) {
      showToast("Import Empty", "Please paste content or drop a file to import", "error");
      return;
    }

    const payload = {
      raw_text,
      source_format: dom.importFormat?.value || "auto",
      sender_name: dom.importSender?.value || "",
      context_mode: dom.importMode?.value || appState.activeContextMode,
    };

    try {
      const res = await apiRequest("/api/samples/import", "POST", payload);
      applyState(res.state);
      if (dom.importText) dom.importText.value = "";

      showToast(
        "Import Successful",
        `Parsed ${res.parsed_count} messages, saved ${res.inserted_count} new writing samples`,
        "success"
      );
    } catch (err) {
      showToast("Import Failed", err.message, "error");
    }
  }

  async function handleSeedDemo() {
    try {
      const res = await apiRequest("/api/samples/seed-demo", "POST", {
        context_mode: appState.activeContextMode,
      });
      applyState(res.state);
      showToast("Demo Samples Loaded", "Authentic conversational & work writing samples seeded", "success");
    } catch (err) {
      showToast("Seed Failed", err.message, "error");
    }
  }

  async function handleClearSamples() {
    const modeLabel = getModeDisplayLabel(appState.activeContextMode);
    if (!confirm(`Are you sure you want to clear all samples for ${modeLabel}?`)) return;

    try {
      const res = await apiRequest("/api/samples/clear", "POST", {
        context_mode: appState.activeContextMode,
      });
      applyState(res.state);
      showToast("Samples Cleared", `Removed ${res.removed} samples from ${modeLabel}`, "info");
    } catch (err) {
      showToast("Clear Failed", err.message, "error");
    }
  }

  async function handleDeleteSample(sampleId) {
    if (!confirm("Delete this writing sample?")) return;
    try {
      const res = await apiRequest(
        `/api/samples/${sampleId}?mode=${encodeURIComponent(appState.activeContextMode)}`,
        "DELETE"
      );
      applyState(res.state);
      showToast("Sample Deleted", "Removed sample from VoicePrint", "info");
    } catch (err) {
      showToast("Delete Failed", err.message, "error");
    }
  }

  // =========================================================================
  // Rules (Tab 3) Handlers
  // =========================================================================
  async function handleAddRule(e) {
    e.preventDefault();
    const avoid = (dom.ruleAvoid?.value || "").trim();
    const prefer = (dom.rulePrefer?.value || "").trim();
    if (!avoid) {
      showToast("Field Required", "Please specify the cliché or phrase to avoid", "error");
      return;
    }

    try {
      const res = await apiRequest("/api/rules", "POST", {
        context_mode: appState.activeContextMode,
        avoid_text: avoid,
        prefer_text: prefer,
      });
      applyState(res.state);
      if (dom.ruleAvoid) dom.ruleAvoid.value = "";
      if (dom.rulePrefer) dom.rulePrefer.value = "";
      showToast("Rule Added", `Replaces "${avoid}" with "${prefer || "(trimmed)"}"`, "success");
    } catch (err) {
      showToast("Add Rule Failed", err.message, "error");
    }
  }

  async function handleDeleteRule(ruleId) {
    if (!confirm("Delete this replacement rule?")) return;
    try {
      const res = await apiRequest(
        `/api/rules/${ruleId}?mode=${encodeURIComponent(appState.activeContextMode)}`,
        "DELETE"
      );
      applyState(res.state);
      showToast("Rule Deleted", "Replacement rule removed", "info");
    } catch (err) {
      showToast("Delete Failed", err.message, "error");
    }
  }

  // =========================================================================
  // macOS Hotkey (Tab 4) Handlers
  // =========================================================================
  async function handleInstallHotkey() {
    if (dom.btnInstallHotkey) dom.btnInstallHotkey.disabled = true;
    try {
      const res = await apiRequest("/api/macos/install-hotkey", "POST", {});
      if (dom.hotkeyInstallStatus) {
        dom.hotkeyInstallStatus.style.display = "inline-flex";
        dom.hotkeyInstallStatus.textContent = "✓ Installed in macOS Services";
      }
      showToast(
        "Quick Action Installed",
        "Assigned to macOS Services! Configure shortcut in System Settings.",
        "success",
        5000
      );
    } catch (err) {
      showToast("Installation Failed", err.message, "error");
    } finally {
      if (dom.btnInstallHotkey) dom.btnInstallHotkey.disabled = false;
    }
  }

  function handleCopySnippet(btn) {
    const text = btn.getAttribute("data-copy") || "";
    if (!text) return;
    navigator.clipboard.writeText(text).then(() => {
      const orig = btn.textContent;
      btn.textContent = "Copied!";
      setTimeout(() => {
        btn.textContent = orig;
      }, 1500);
    });
  }

  // =========================================================================
  // Settings Modal Handlers
  // =========================================================================
  function openSettingsModal() {
    renderSettingsModal();
    if (dom.settingsModalBackdrop) dom.settingsModalBackdrop.style.display = "flex";
  }

  function closeSettingsModal() {
    if (dom.settingsModalBackdrop) dom.settingsModalBackdrop.style.display = "none";
    appState.clearKeyFlags = { gemini: false, openai: false, anthropic: false, pangram: false };
  }

  async function handleSaveSettings(e) {
    e.preventDefault();
    const payload = {
      provider: dom.settingProvider?.value || "auto",
      gemini_model: dom.settingGeminiModel?.value || "",
      openai_model: dom.settingOpenaiModel?.value || "",
      anthropic_model: dom.settingAnthropicModel?.value || "",
      ollama_url: dom.settingOllamaUrl?.value || "",
      ollama_model: dom.settingOllamaModel?.value || "",
      pangram_model: dom.settingPangramModel?.value || "pangram-4",
      pangram_public_link: dom.settingPangramPublicLink?.checked ? "true" : "false",
    };

    // API Keys (only sent if typed or cleared)
    const geminiKey = dom.settingGeminiKey?.value || "";
    if (geminiKey) payload.gemini_api_key = geminiKey;
    if (appState.clearKeyFlags.gemini) payload.clear_gemini_api_key = true;

    const openaiKey = dom.settingOpenaiKey?.value || "";
    if (openaiKey) payload.openai_api_key = openaiKey;
    if (appState.clearKeyFlags.openai) payload.clear_openai_api_key = true;

    const anthropicKey = dom.settingAnthropicKey?.value || "";
    if (anthropicKey) payload.anthropic_api_key = anthropicKey;
    if (appState.clearKeyFlags.anthropic) payload.clear_anthropic_api_key = true;

    const pangramKey = dom.settingPangramKey?.value || "";
    if (pangramKey) payload.pangram_api_key = pangramKey;
    if (appState.clearKeyFlags.pangram) payload.clear_pangram_api_key = true;

    try {
      const res = await apiRequest("/api/settings", "POST", payload);
      applyState(res.state);
      closeSettingsModal();
      showToast("Settings Saved", "AI Engine configuration updated", "success");
    } catch (err) {
      showToast("Save Settings Failed", err.message, "error");
    }
  }

  // =========================================================================
  // Silent State Refresh
  // =========================================================================
  async function refreshStateSilently() {
    try {
      const data = await apiRequest(`/api/state?mode=${encodeURIComponent(appState.activeContextMode)}`);
      applyState(data);
    } catch (_) {}
  }

  // =========================================================================
  // Pangram Labs & Originality Inspector Handlers
  // =========================================================================
  async function handleTestPangramKey() {
    const typedKey = dom.settingPangramKey?.value?.trim() || "";
    let url = "/api/pangram/models";
    if (typedKey) {
      url += `?api_key=${encodeURIComponent(typedKey)}`;
    }
    const origText = dom.btnTestPangram?.textContent;
    if (dom.btnTestPangram) {
      dom.btnTestPangram.textContent = "Testing...";
      dom.btnTestPangram.disabled = true;
    }
    try {
      const res = await apiRequest(url);
      if (res && res.ok) {
        const modelsList = (res.models || []).join(", ") || "pangram-4";
        showToast("Pangram API Connected", `Valid key. Available models: ${modelsList}`, "success");
      } else {
        showToast("Pangram Key Error", res.error || "Unable to verify API key with Pangram", "error");
      }
    } catch (err) {
      showToast("Pangram Test Failed", err.message, "error");
    } finally {
      if (dom.btnTestPangram) {
        dom.btnTestPangram.textContent = origText || "Test Key";
        dom.btnTestPangram.disabled = false;
      }
    }
  }

  function updateInspectorCounts() {
    const text = dom.inspectorInput?.value || "";
    const words = countWords(text);
    const chars = text.length;
    const strippedChars = text.trim().length;

    if (dom.inspectorWordCount) {
      dom.inspectorWordCount.textContent = `${words} word${words === 1 ? "" : "s"}`;
    }
    if (dom.inspectorCharCount) {
      dom.inspectorCharCount.textContent = `${chars} char${chars === 1 ? "" : "s"}`;
    }

    if (dom.inspectorReadinessPill) {
      if (strippedChars >= 50) {
        dom.inspectorReadinessPill.textContent = `Ready (${strippedChars} chars)`;
        dom.inspectorReadinessPill.className = "inspector-pill-readiness pill-ready";
      } else {
        dom.inspectorReadinessPill.textContent = `Requires ≥50 chars (${strippedChars}/50)`;
        dom.inspectorReadinessPill.className = "inspector-pill-readiness pill-warning";
      }
    }

    if (dom.inspectorCostEst) {
      const cost = (words / 100.0) * 0.05;
      dom.inspectorCostEst.textContent = `Est. ~$${cost.toFixed(3)} USD`;
    }
  }

  function handleGrabInspectorClipboard() {
    if (!navigator.clipboard?.readText) {
      showToast("Clipboard Unavailable", "Clipboard API permission not granted", "error");
      return;
    }
    navigator.clipboard
      .readText()
      .then((clip) => {
        if (!clip) {
          showToast("Clipboard Empty", "macOS clipboard had no text content", "info");
          return;
        }
        if (dom.inspectorInput) dom.inspectorInput.value = clip;
        updateInspectorCounts();
        showToast("Clipboard Pasted", `Pasted ${countWords(clip)} words into Inspector`, "success");
      })
      .catch((err) => {
        showToast("Clipboard Error", err.message, "error");
      });
  }

  function handleLoadInspectorSample() {
    if (dom.inspectorInput) dom.inspectorInput.value = SAMPLE_INSPECTOR_SUBMISSION;
    if (dom.inspectorDocTitle) dom.inspectorDocTitle.value = "AI & Humanizer Evaluation Sample";
    updateInspectorCounts();
    showToast("Sample Loaded", "Sample submission loaded into inspector.", "info");
  }

  async function handleInspectorUpload(e) {
    const file = e.target.files?.[0];
    if (!file) return;

    showToast("Uploading File", `Reading ${file.name}...`, "info");
    const reader = new FileReader();

    reader.onload = async () => {
      try {
        const rawResult = reader.result;
        let b64 = "";
        if (typeof rawResult === "string") {
          const parts = rawResult.split(",");
          b64 = parts.length > 1 ? parts[1] : parts[0];
        }
        const preferP4 = (dom.inspectorModelSelect?.value || "pangram-4") === "pangram-4";
        const pubLink = dom.inspectorShareableLink?.checked || false;

        const res = await apiRequest("/api/pangram/file", "POST", {
          filename: file.name,
          content_base64: b64,
          prefer_pangram_4: preferP4,
          public_dashboard_link: pubLink,
        });

        if (res && res.result) {
          if (dom.inspectorInput) dom.inspectorInput.value = res.result.text || "";
          if (dom.inspectorDocTitle) dom.inspectorDocTitle.value = file.name;
          updateInspectorCounts();
          renderPangramResult(res.result);
          if (res.state) applyState(res.state);
          showToast("File Scanned", `Pangram Labs completed analysis for ${file.name}`, "success");
        }
      } catch (err) {
        showToast("File Scan Failed", err.message, "error");
      } finally {
        if (dom.fileInspectorUpload) dom.fileInspectorUpload.value = "";
      }
    };

    reader.onerror = () => {
      showToast("File Read Error", "Failed to read chosen file", "error");
      if (dom.fileInspectorUpload) dom.fileInspectorUpload.value = "";
    };

    reader.readAsDataURL(file);
  }

  async function handleRunFreePrescan() {
    const text = dom.inspectorInput?.value || "";
    if (!text.trim()) {
      showToast("Empty Input", "Please enter or paste text to run the free local pre-scan.", "error");
      return;
    }

    try {
      const res = await apiRequest("/api/analyze", "POST", {
        text,
        context_mode: appState.activeContextMode,
      });

      if (dom.inspectorPrescanBox) dom.inspectorPrescanBox.style.display = "block";
      if (dom.prescanSummaryText) {
        dom.prescanSummaryText.textContent = `Analyzed ${res.word_count || countWords(text)} words locally ($0 cost)`;
      }
      if (dom.prescanHumanScore) {
        dom.prescanHumanScore.textContent = `${res.human_score ?? 0}%`;
      }
      if (dom.prescanAiRisk) {
        dom.prescanAiRisk.textContent = `${res.ai_risk_score ?? 0}%`;
      }
      if (dom.prescanBurstiness) {
        dom.prescanBurstiness.textContent = `${res.burstiness_score ?? 0}`;
      }
      if (dom.prescanFlagsCount) {
        dom.prescanFlagsCount.textContent = (res.flags || []).length;
      }
      if (dom.prescanFlagsList) {
        const flags = res.flags || [];
        if (flags.length === 0) {
          dom.prescanFlagsList.innerHTML = '<span class="text-xs text-emerald font-mono">No clichés or AI markers detected in local heuristic pass.</span>';
        } else {
          dom.prescanFlagsList.innerHTML = flags
            .map((f) => `<span class="prescan-flag-pill font-mono">${escapeHtml(f.phrase || f.type || "cliché")}</span>`)
            .join("");
        }
      }
      showToast("Pre-Scan Complete", "Local heuristic and Voice DNA analysis completed ($0 cost)", "info");
    } catch (err) {
      showToast("Pre-Scan Failed", err.message, "error");
    }
  }

  async function handleRunPangramScan() {
    const text = dom.inspectorInput?.value || "";
    if (text.trim().length < 50) {
      showToast("Too Short for Pangram", "Pangram Labs requires at least 50 characters of text to run an enterprise scan.", "error");
      return;
    }

    const origText = dom.btnRunPangramScan?.textContent;
    if (dom.btnRunPangramScan) {
      dom.btnRunPangramScan.textContent = "Scanning with Pangram 4...";
      dom.btnRunPangramScan.disabled = true;
    }

    try {
      const model = dom.inspectorModelSelect?.value || "pangram-4";
      const pubLink = dom.inspectorShareableLink?.checked || false;
      const title = dom.inspectorDocTitle?.value?.trim() || "";

      const res = await apiRequest("/api/pangram/scan", "POST", {
        text,
        model,
        public_dashboard_link: pubLink,
        title,
      });

      if (res && res.result) {
        renderPangramResult(res.result);
        if (res.state) applyState(res.state);
        showToast("Pangram 4 Scan Complete", `Verdict: ${res.result.headline || res.result.prediction_short || "Finished"}`, "success");
      }
    } catch (err) {
      showToast("Pangram Scan Failed", err.message, "error");
    } finally {
      if (dom.btnRunPangramScan) {
        dom.btnRunPangramScan.textContent = origText || "Scan with Pangram 4";
        dom.btnRunPangramScan.disabled = false;
      }
    }
  }

  async function handleRunPlagiarismScan() {
    const text = dom.inspectorInput?.value || "";
    if (text.trim().length < 50) {
      showToast("Too Short for Plagiarism Check", "Plagiarism scanning requires at least 50 characters of text.", "error");
      return;
    }

    const origText = dom.btnRunPlagiarismScan?.textContent;
    if (dom.btnRunPlagiarismScan) {
      dom.btnRunPlagiarismScan.textContent = "Scanning Plagiarism...";
      dom.btnRunPlagiarismScan.disabled = true;
    }

    try {
      const title = dom.inspectorDocTitle?.value?.trim() || "";
      const res = await apiRequest("/api/pangram/plagiarism", "POST", {
        text,
        title,
      });

      if (res && res.result) {
        renderPangramResult(res.result);
        if (res.state) applyState(res.state);
        const score = Math.round((res.result.plagiarism_score || 0) * 100);
        showToast("Plagiarism Scan Complete", `Web overlap score: ${score}%`, "success");
      }
    } catch (err) {
      showToast("Plagiarism Scan Failed", err.message, "error");
    } finally {
      if (dom.btnRunPlagiarismScan) {
        dom.btnRunPlagiarismScan.textContent = origText || "Web Plagiarism Check";
        dom.btnRunPlagiarismScan.disabled = false;
      }
    }
  }

  function renderHighlightedSubmission(text, windows) {
    if (!text) return "";
    if (!windows || windows.length === 0) return escapeHtml(text);

    // Normalize and filter valid windows safely
    const validWindows = windows
      .filter((w) => typeof w.start_index === "number" && typeof w.end_index === "number" && w.start_index < w.end_index)
      .map((w) => ({
        start: Math.max(0, Math.min(text.length, w.start_index)),
        end: Math.max(0, Math.min(text.length, w.end_index)),
        label: String(w.label || "HUMAN").toUpperCase(),
        isHumanized: Boolean(w.is_humanized || (w.humanizer_score && w.humanizer_score >= 0.5)),
        confidence: w.confidence || 0,
        humanizerScore: w.humanizer_score || 0,
      }))
      .sort((a, b) => a.start - b.start);

    let html = "";
    let cursor = 0;

    for (const w of validWindows) {
      if (w.start > cursor) {
        html += escapeHtml(text.substring(cursor, w.start));
      }
      if (w.end > w.start && w.start >= cursor) {
        const segmentText = text.substring(w.start, w.end);
        let cls = "span-human";
        if (w.label === "AI") {
          cls = "span-ai";
        } else if (w.label === "AI_ASSISTED" || w.label === "ASSISTED") {
          cls = "span-assisted";
        }
        const humanizerClass = w.isHumanized ? " span-humanizer" : "";
        const badge = w.isHumanized ? '<span class="tag-humanizer-scrubbed">Scrubbed</span>' : "";
        const titleAttr = `${escapeHtml(w.label)} (${Math.round(w.confidence * 100)}%)${w.isHumanized ? " • Humanizer scrubbed" : ""}`;
        html += `<span class="${cls}${humanizerClass}" title="${titleAttr}">${escapeHtml(segmentText)}${badge}</span>`;
        cursor = w.end;
      }
    }

    if (cursor < text.length) {
      html += escapeHtml(text.substring(cursor));
    }
    return html;
  }

  function renderWindowBreakdown(windows) {
    if (!dom.inspectorWindowsList) return;
    if (!windows || windows.length === 0) {
      if (dom.inspectorWindowsCount) dom.inspectorWindowsCount.textContent = "0 windows analyzed";
      dom.inspectorWindowsList.innerHTML = '<div class="text-xs text-muted" style="padding: 12px;">No sliding window segments recorded for this scan.</div>';
      return;
    }

    if (dom.inspectorWindowsCount) {
      dom.inspectorWindowsCount.textContent = `${windows.length} window segments analyzed`;
    }

    dom.inspectorWindowsList.innerHTML = windows
      .map((w, idx) => {
        const label = String(w.label || "HUMAN").toUpperCase();
        let labelClass = "text-emerald";
        let badgeBg = "rgba(34, 197, 94, 0.15)";
        if (label === "AI") {
          labelClass = "text-rose";
          badgeBg = "rgba(239, 68, 68, 0.15)";
        } else if (label === "AI_ASSISTED" || label === "ASSISTED") {
          labelClass = "text-amber";
          badgeBg = "rgba(245, 158, 11, 0.15)";
        }

        const isHum = Boolean(w.is_humanized || (w.humanizer_score && w.humanizer_score >= 0.5));
        const confPct = Math.round((w.confidence || 0) * 100);
        const humScore = typeof w.humanizer_score === "number" ? (w.humanizer_score * 100).toFixed(0) : "0";

        return `
          <div class="window-card">
            <div class="window-card-head">
              <div style="display: flex; align-items: center; gap: 8px;">
                <span class="font-mono text-xs text-muted">#${idx + 1}</span>
                <span style="font-weight: 600; font-size: 11px; padding: 2px 8px; border-radius: var(--radius-pill); background: ${badgeBg};" class="${labelClass}">
                  ${escapeHtml(label)} (${confPct}%)
                </span>
                ${isHum ? '<span class="tag-humanizer-scrubbed">Scrubbed (' + humScore + '%)</span>' : ''}
              </div>
              <div class="text-xs text-muted font-mono">
                ${w.word_count || 0} words • ${w.token_length || 0} tokens
              </div>
            </div>
            <div class="window-card-body font-mono">
              "${escapeHtml(w.text || "")}"
            </div>
          </div>
        `;
      })
      .join("");
  }

  function renderPlagiarismMatches(result) {
    if (!dom.inspectorPlagiarismMatches) return;
    const matches = result.matches || [];
    const score = result.plagiarism_score ?? 0;
    const scorePct = Math.round(score * 100);

    if (dom.inspectorPlagiarismScoreBadge) {
      dom.inspectorPlagiarismScoreBadge.textContent = `${scorePct}% Overlap`;
      dom.inspectorPlagiarismScoreBadge.className = scorePct > 15 ? "badge-neutral font-mono text-rose" : "badge-neutral font-mono text-emerald";
    }

    if (dom.inspectorPlagiarismStatus) {
      dom.inspectorPlagiarismStatus.textContent = result.plagiarism_detected ? "Plagiarism Detected" : "No Critical Plagiarism Detected";
    }

    if (matches.length === 0) {
      dom.inspectorPlagiarismMatches.innerHTML = '<div class="text-xs text-muted" style="padding: 12px;">No matching web sources found. High textual originality.</div>';
      return;
    }

    dom.inspectorPlagiarismMatches.innerHTML = matches
      .map((m, idx) => {
        const url = m.url || m.source || "";
        const title = m.title || m.name || url || `Match #${idx + 1}`;
        const pct = m.percentage != null ? `${Math.round(m.percentage)}% match` : "";
        const snippet = m.snippet || m.matched_text || "";

        return `
          <div class="plagiarism-match-card">
            <div class="plagiarism-match-head">
              <strong style="color: #ffffff;">${escapeHtml(title)}</strong>
              ${pct ? `<span class="font-mono text-rose text-xs">${escapeHtml(pct)}</span>` : ""}
            </div>
            ${url ? `<div class="text-xs text-muted mb-2"><a href="${escapeHtml(url)}" target="_blank" rel="noopener noreferrer">${escapeHtml(url)}</a></div>` : ""}
            ${snippet ? `<div class="text-xs text-muted font-mono" style="background: rgba(255,255,255,0.03); padding: 8px; border-radius: 4px;">"${escapeHtml(snippet)}"</div>` : ""}
          </div>
        `;
      })
      .join("");
  }

  function renderPangramResult(result) {
    if (!result) return;
    appState.lastPangramResult = result;

    if (dom.inspectorEmptyState) dom.inspectorEmptyState.style.display = "none";
    if (dom.inspectorActiveResults) dom.inspectorActiveResults.style.display = "block";

    // Headline and prediction
    const headline = result.headline || result.prediction_short || "Completed";
    if (dom.inspectorHeadlineTitle) dom.inspectorHeadlineTitle.textContent = headline;
    if (dom.inspectorHeadlinePrediction) dom.inspectorHeadlinePrediction.textContent = result.prediction || "";

    const ai = result.fraction_ai || 0;
    const assisted = result.fraction_ai_assisted || 0;
    const human = result.fraction_human || 0;

    if (dom.inspectorHeadlineBadge) {
      if (result.scan_type === "plagiarism") {
        const pScore = Math.round((result.plagiarism_score || 0) * 100);
        dom.inspectorHeadlineBadge.textContent = `Plagiarism ${pScore}%`;
        dom.inspectorHeadlineBadge.className = pScore > 15 ? "inspector-verdict-badge verdict-ai" : "inspector-verdict-badge verdict-human";
      } else if (ai >= 0.65) {
        dom.inspectorHeadlineBadge.textContent = `Likely AI (${Math.round(ai * 100)}%)`;
        dom.inspectorHeadlineBadge.className = "inspector-verdict-badge verdict-ai";
      } else if (assisted >= 0.35 || ai >= 0.35) {
        dom.inspectorHeadlineBadge.textContent = `AI-Assisted (${Math.round(assisted * 100)}%)`;
        dom.inspectorHeadlineBadge.className = "inspector-verdict-badge verdict-assisted";
      } else {
        dom.inspectorHeadlineBadge.textContent = `Likely Human (${Math.round(human * 100)}%)`;
        dom.inspectorHeadlineBadge.className = "inspector-verdict-badge verdict-human";
      }
    }

    if (dom.inspectorModelBadge) {
      dom.inspectorModelBadge.textContent = result.model || "pangram-4";
    }

    if (dom.inspectorDashboardLink) {
      if (result.dashboard_link) {
        dom.inspectorDashboardLink.href = result.dashboard_link;
        dom.inspectorDashboardLink.style.display = "inline-flex";
      } else {
        dom.inspectorDashboardLink.style.display = "none";
      }
    }

    // 3-Segment Stacked Meter
    const aiPct = Math.min(100, Math.max(0, ai * 100)).toFixed(1);
    const assistedPct = Math.min(100, Math.max(0, assisted * 100)).toFixed(1);
    const humanPct = Math.min(100, Math.max(0, human * 100)).toFixed(1);

    if (dom.lblMeterAi) dom.lblMeterAi.textContent = `${aiPct}%`;
    if (dom.lblMeterAssisted) dom.lblMeterAssisted.textContent = `${assistedPct}%`;
    if (dom.lblMeterHuman) dom.lblMeterHuman.textContent = `${humanPct}%`;

    if (dom.meterSegmentAi) dom.meterSegmentAi.style.width = `${aiPct}%`;
    if (dom.meterSegmentAssisted) dom.meterSegmentAssisted.style.width = `${assistedPct}%`;
    if (dom.meterSegmentHuman) dom.meterSegmentHuman.style.width = `${humanPct}%`;

    // 4 Stat Cards
    if (dom.cardValAi) dom.cardValAi.textContent = `${aiPct}%`;
    if (dom.cardSubAi) dom.cardSubAi.textContent = `${result.num_ai_segments || 0} segments`;

    if (dom.cardValAssisted) dom.cardValAssisted.textContent = `${assistedPct}%`;
    if (dom.cardSubAssisted) dom.cardSubAssisted.textContent = `${result.num_ai_assisted_segments || 0} segments`;

    if (dom.cardValHuman) dom.cardValHuman.textContent = `${humanPct}%`;
    if (dom.cardSubHuman) dom.cardSubHuman.textContent = `${result.num_human_segments || 0} segments`;

    const hasHumanizer = Boolean(result.has_humanized_segments || (result.max_humanizer_score && result.max_humanizer_score >= 0.5));
    if (dom.cardValHumanizer) {
      if (hasHumanizer) {
        const score = typeof result.max_humanizer_score === "number" ? Math.round(result.max_humanizer_score * 100) : 100;
        dom.cardValHumanizer.textContent = `Detected (${score}%)`;
        dom.cardValHumanizer.className = "stat-card-value font-mono text-rose";
        if (dom.cardSubHumanizer) dom.cardSubHumanizer.textContent = "Scrubber / bypass signature flagged";
      } else {
        dom.cardValHumanizer.textContent = "Clean";
        dom.cardValHumanizer.className = "stat-card-value font-mono text-emerald";
        if (dom.cardSubHumanizer) dom.cardSubHumanizer.textContent = `Max score: ${(result.max_humanizer_score || 0).toFixed(2)}`;
      }
    }

    // Highlighted Submission Viewer
    const textToHighlight = result.text || dom.inspectorInput?.value || "";
    if (dom.inspectorHighlightedText) {
      dom.inspectorHighlightedText.innerHTML = renderHighlightedSubmission(textToHighlight, result.windows);
    }

    // Windows list
    renderWindowBreakdown(result.windows);

    // Web Plagiarism Section
    if (result.scan_type === "plagiarism" || (result.matches && result.matches.length > 0)) {
      if (dom.inspectorPlagiarismSection) dom.inspectorPlagiarismSection.style.display = "block";
      renderPlagiarismMatches(result);
    } else {
      if (dom.inspectorPlagiarismSection) dom.inspectorPlagiarismSection.style.display = "none";
    }
  }

  function renderPangramHistory() {
    if (!dom.inspectorHistoryTbody) return;
    const scans = appState.pangramScans || [];

    if (dom.inspectorHistoryCount) {
      dom.inspectorHistoryCount.textContent = `${scans.length} scan${scans.length === 1 ? "" : "s"} saved`;
    }

    if (scans.length === 0) {
      dom.inspectorHistoryTbody.innerHTML = `
        <tr>
          <td colspan="6" class="text-center text-muted" style="padding: 24px;">No saved Pangram scans yet. Run a scan to build history.</td>
        </tr>
      `;
      return;
    }

    dom.inspectorHistoryTbody.innerHTML = scans
      .map((scan) => {
        let dateStr = "";
        try {
          dateStr = new Date(scan.created_at).toLocaleString([], {
            month: "short",
            day: "numeric",
            hour: "2-digit",
            minute: "2-digit",
          });
        } catch (_) {
          dateStr = scan.created_at || "";
        }

        const typeLabel = scan.scan_type === "plagiarism" ? "Plagiarism" : (scan.scan_type === "file" ? "File Scan" : "Pangram 4");

        let summaryText = "";
        if (scan.scan_type === "plagiarism") {
          const resObj = typeof scan.result === "string" ? JSON.parse(scan.result) : (scan.result || {});
          const pScore = Math.round((resObj.plagiarism_score || 0) * 100);
          summaryText = `<span class="${pScore > 15 ? 'text-rose' : 'text-emerald'}">${pScore}% web overlap</span>`;
        } else {
          const aiPct = Math.round((scan.fraction_ai || 0) * 100);
          const humPct = Math.round((scan.fraction_human || 0) * 100);
          const scrubbed = (scan.max_humanizer_score || 0) >= 0.5 ? '<span class="tag-humanizer-scrubbed" style="font-size: 8px;">Scrubbed</span>' : '';
          summaryText = `<span class="text-rose">AI: ${aiPct}%</span> • <span class="text-emerald">Human: ${humPct}%</span> ${scrubbed}`;
        }

        const costStr = scan.estimated_cost_usd ? `$${scan.estimated_cost_usd.toFixed(3)}` : "$0.00";

        return `
          <tr>
            <td class="font-mono text-xs text-muted">${escapeHtml(dateStr)}</td>
            <td>
              <strong style="color: #ffffff; display: block; max-width: 280px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">
                ${escapeHtml(scan.title || "Untitled Scan")}
              </strong>
              <span class="text-xs text-muted font-mono">${scan.word_count || 0} words</span>
            </td>
            <td>
              <span class="badge-neutral text-xs font-mono">${escapeHtml(typeLabel)}</span>
            </td>
            <td class="text-xs font-mono">
              ${summaryText}
            </td>
            <td class="font-mono text-xs text-muted">${escapeHtml(costStr)}</td>
            <td style="text-align: right;">
              <button type="button" class="btn btn-secondary btn-sm btn-load-scan" data-id="${scan.id}" title="Load results into inspector view ($0 cost)">
                Load Result
              </button>
              <button type="button" class="btn btn-ghost btn-sm text-rose btn-del-scan" data-id="${scan.id}" title="Delete scan record">
                ×
              </button>
            </td>
          </tr>
        `;
      })
      .join("");
  }

  function handleLoadSavedScan(scanId) {
    const numericId = parseInt(scanId, 10);
    const item = (appState.pangramScans || []).find((s) => s.id === numericId);
    if (!item) return;

    if (dom.inspectorInput) {
      dom.inspectorInput.value = item.input_text || item.result?.text || "";
    }
    if (dom.inspectorDocTitle) {
      dom.inspectorDocTitle.value = item.title || "";
    }
    updateInspectorCounts();

    const resultObj = typeof item.result === "string" ? JSON.parse(item.result) : (item.result || {});
    renderPangramResult(resultObj);
    showToast("Loaded Result", `Restored scan from ${item.title || "saved history"} ($0 cost)`, "info");
  }

  async function handleDeleteSavedScan(scanId) {
    if (!confirm("Delete this saved Pangram scan from your local history?")) return;
    try {
      const res = await apiRequest(`/api/pangram/scans/${scanId}`, "DELETE");
      applyState(res.state);
      showToast("Scan Deleted", "Removed scan record from history", "success");
    } catch (err) {
      showToast("Delete Failed", err.message, "error");
    }
  }

  // =========================================================================
  // Source & Revision Coach Handlers (Academic Integrity)
  // =========================================================================
  function updateAcademicCounts() {
    if (dom.acadDraftWordCount) {
      dom.acadDraftWordCount.textContent = `${countWords(dom.acadDraftInput?.value || "")} words`;
    }
    if (dom.acadSourceWordCount) {
      dom.acadSourceWordCount.textContent = `${countWords(dom.acadSourceInput?.value || "")} words`;
    }
  }

  function handleLoadAcademicSample() {
    if (dom.acadDraftInput) dom.acadDraftInput.value = SAMPLE_ACADEMIC_DRAFT;
    if (dom.acadSourceInput) dom.acadSourceInput.value = SAMPLE_ACADEMIC_SOURCE;
    updateAcademicCounts();
    showToast("Sample Coursework Loaded", "Sample student draft and reference source loaded.", "info");
  }

  function handleGrabAcademicClipboard() {
    if (!navigator.clipboard?.readText) {
      showToast("Clipboard Unavailable", "Clipboard API permission not granted", "error");
      return;
    }
    navigator.clipboard
      .readText()
      .then((clip) => {
        if (!clip) {
          showToast("Clipboard Empty", "macOS clipboard had no text content", "info");
          return;
        }
        if (dom.acadDraftInput) dom.acadDraftInput.value = clip;
        updateAcademicCounts();
        showToast("Clipboard Pasted", `Pasted ${countWords(clip)} words into Student Draft`, "success");
      })
      .catch((err) => {
        showToast("Clipboard Error", err.message, "error");
      });
  }

  function handleClearAcademic() {
    if (dom.acadDraftInput) dom.acadDraftInput.value = "";
    if (dom.acadSourceInput) dom.acadSourceInput.value = "";
    updateAcademicCounts();
    if (dom.acadCitationEmpty) dom.acadCitationEmpty.style.display = "block";
    if (dom.acadCitationBody) dom.acadCitationBody.style.display = "none";
    if (dom.acadCoachEmpty) dom.acadCoachEmpty.style.display = "block";
    if (dom.acadCoachBody) dom.acadCoachBody.style.display = "none";
  }

  async function handleRunCitationCheck() {
    const draftText = dom.acadDraftInput?.value?.trim() || "";
    const sourceText = dom.acadSourceInput?.value?.trim() || "";
    if (!draftText || !sourceText) {
      showToast("Missing Inputs", "Please enter both Student Draft and Source Text to check overlap & citations.", "error");
      return;
    }

    const origText = dom.btnRunCitationCheck?.textContent;
    if (dom.btnRunCitationCheck) {
      dom.btnRunCitationCheck.textContent = "Checking Overlap...";
      dom.btnRunCitationCheck.disabled = true;
    }

    try {
      const res = await apiRequest("/api/academic/citation-check", "POST", {
        draft_text: draftText,
        source_text: sourceText,
      });
      renderCitationReport(res);
      showToast("Source Overlap Checked", `Attribution status: ${res.status_label || "Completed"}`, "success");
    } catch (err) {
      showToast("Citation Check Failed", err.message, "error");
    } finally {
      if (dom.btnRunCitationCheck) {
        dom.btnRunCitationCheck.textContent = origText || "Check Source Overlap & Citations";
        dom.btnRunCitationCheck.disabled = false;
      }
    }
  }

  async function handleRunCoach() {
    const draftText = dom.acadDraftInput?.value?.trim() || "";
    if (!draftText) {
      showToast("Missing Draft", "Please enter your student draft to run revision coaching diagnostics.", "error");
      return;
    }

    const origText = dom.btnRunCoach?.textContent;
    if (dom.btnRunCoach) {
      dom.btnRunCoach.textContent = "Analyzing Clarity...";
      dom.btnRunCoach.disabled = true;
    }

    try {
      const res = await apiRequest("/api/academic/coach", "POST", {
        draft_text: draftText,
      });
      renderAcademicCoachReport(res);
      showToast("Academic Coaching Diagnostics Ready", `Clarity score: ${res.clarity_score}/100`, "success");
    } catch (err) {
      showToast("Coaching Failed", err.message, "error");
    } finally {
      if (dom.btnRunCoach) {
        dom.btnRunCoach.textContent = origText || "Run Academic Revision Coach";
        dom.btnRunCoach.disabled = false;
      }
    }
  }

  async function handleRunBothAcademic() {
    const draftText = dom.acadDraftInput?.value?.trim() || "";
    const sourceText = dom.acadSourceInput?.value?.trim() || "";
    if (!draftText) {
      showToast("Missing Draft", "Please enter your student draft.", "error");
      return;
    }

    const origText = dom.btnRunBothAcad?.textContent;
    if (dom.btnRunBothAcad) {
      dom.btnRunBothAcad.textContent = "Running Both Checks...";
      dom.btnRunBothAcad.disabled = true;
    }

    try {
      const coachPromise = apiRequest("/api/academic/coach", "POST", { draft_text: draftText });
      const citationPromise = sourceText
        ? apiRequest("/api/academic/citation-check", "POST", { draft_text: draftText, source_text: sourceText })
        : Promise.resolve(null);

      const [coachRes, citationRes] = await Promise.all([coachPromise, citationPromise]);

      if (coachRes) renderAcademicCoachReport(coachRes);
      if (citationRes) renderCitationReport(citationRes);

      showToast("Academic Checks Completed", "Both revision diagnostics and citation checks updated.", "success");
    } catch (err) {
      showToast("Checks Failed", err.message, "error");
    } finally {
      if (dom.btnRunBothAcad) {
        dom.btnRunBothAcad.textContent = origText || "Run Both Checks";
        dom.btnRunBothAcad.disabled = false;
      }
    }
  }

  function renderCitationReport(data) {
    if (!data) return;
    if (dom.acadCitationEmpty) dom.acadCitationEmpty.style.display = "none";
    if (dom.acadCitationBody) dom.acadCitationBody.style.display = "block";

    const overlap = data.overall_overlap_pct ?? 0;
    if (dom.acadOverlapBadge) {
      dom.acadOverlapBadge.textContent = `${overlap.toFixed(1)}%`;
      if (overlap < 8.0) {
        dom.acadOverlapBadge.className = "font-mono text-emerald";
      } else if (overlap <= 25.0) {
        dom.acadOverlapBadge.className = "font-mono text-amber";
      } else {
        dom.acadOverlapBadge.className = "font-mono text-rose";
      }
    }

    if (dom.acadStatusBadge) {
      dom.acadStatusBadge.textContent = data.status_label || "Completed";
    }
    if (dom.acadStatusLabel) {
      dom.acadStatusLabel.textContent = data.status_label || "Clean Attribution";
      if (overlap < 8.0) {
        dom.acadStatusLabel.className = "text-emerald";
      } else if (overlap <= 25.0) {
        dom.acadStatusLabel.className = "text-amber";
      } else {
        dom.acadStatusLabel.className = "text-rose";
      }
    }
    if (dom.acadQuotedCount) {
      dom.acadQuotedCount.textContent = data.quoted_cited_count ?? 0;
    }
    if (dom.acadMatchesCount) {
      dom.acadMatchesCount.textContent = (data.matches || []).length;
    }

    // Annotated Draft Segments
    if (dom.acadAnnotatedDraft) {
      const segs = data.annotated_segments || [];
      dom.acadAnnotatedDraft.innerHTML = segs
        .map((seg) => {
          if (seg.type === "quoted_cited") {
            return `<span class="annotated-quoted" title="Properly Quoted & Cited">${escapeHtml(seg.text)}</span>`;
          }
          if (seg.type === "verbatim") {
            return `<span class="annotated-verbatim" title="Verbatim Unquoted Overlap">${escapeHtml(seg.text)}</span>`;
          }
          if (seg.type === "paraphrase") {
            return `<span class="annotated-paraphrase" title="Close Paraphrase">${escapeHtml(seg.text)}</span>`;
          }
          return `<span>${escapeHtml(seg.text)}</span>`;
        })
        .join("");
    }

    // Matches List
    if (dom.acadMatchesList) {
      const matches = data.matches || [];
      if (matches.length === 0) {
        dom.acadMatchesList.innerHTML = `
          <div class="text-xs text-muted" style="padding: 12px; background: var(--bg-subtle); border-radius: 6px;">
            No verbatim or close paraphrase overlap detected outside of properly quoted citations.
          </div>
        `;
      } else {
        dom.acadMatchesList.innerHTML = matches
          .map((m, idx) => {
            const isVerbatim = m.type === "verbatim_unquoted";
            const typeTitle = isVerbatim ? "Verbatim Unquoted Phrasing" : "Close Paraphrase";
            const sevClass = m.severity === "high" ? "text-rose" : "text-amber";
            const citedStatus = m.has_citation_nearby ? "Cited nearby" : "No citation found";
            const citedClass = m.has_citation_nearby ? "text-emerald" : "text-rose";

            return `
              <div class="acad-match-item">
                <div class="acad-match-head">
                  <div style="display: flex; align-items: center; gap: 8px;">
                    <span class="font-mono text-xs text-muted">#${idx + 1}</span>
                    <strong class="${sevClass}" style="font-size: 12px;">${escapeHtml(typeTitle)}</strong>
                    <span class="font-mono text-xs text-muted">(${m.word_length || 0} words)</span>
                  </div>
                  <span class="font-mono text-xs ${citedClass}">${escapeHtml(citedStatus)}</span>
                </div>
                <div class="acad-match-excerpts font-mono">
                  <div class="excerpt-draft">
                    <span class="text-xs text-muted" style="display: block; margin-bottom: 2px;">Draft Excerpt:</span>
                    "${escapeHtml(m.draft_excerpt || "")}"
                  </div>
                  <div class="excerpt-source">
                    <span class="text-xs text-muted" style="display: block; margin-bottom: 2px;">Source Reference:</span>
                    "${escapeHtml(m.source_excerpt || "")}"
                  </div>
                </div>
                <div class="acad-guidance-box">
                  <strong>Revision Guidance:</strong> ${escapeHtml(m.guidance || "")}
                </div>
              </div>
            `;
          })
          .join("");
      }
    }
  }

  function renderAcademicCoachReport(data) {
    if (!data) return;
    if (dom.acadCoachEmpty) dom.acadCoachEmpty.style.display = "none";
    if (dom.acadCoachBody) dom.acadCoachBody.style.display = "block";

    const score = data.clarity_score ?? 0;
    if (dom.acadClarityScore) {
      dom.acadClarityScore.textContent = `${score}/100`;
      if (score >= 80) {
        dom.acadClarityScore.className = "font-mono text-emerald";
      } else if (score >= 60) {
        dom.acadClarityScore.className = "font-mono text-amber";
      } else {
        dom.acadClarityScore.className = "font-mono text-rose";
      }
    }

    const counts = data.counts || {};
    if (dom.acadCountPassive) dom.acadCountPassive.textContent = counts.passive_voice || 0;
    if (dom.acadCountUnattributed) dom.acadCountUnattributed.textContent = counts.unattributed_claim || 0;
    if (dom.acadCountFiller) dom.acadCountFiller.textContent = counts.vague_filler || 0;
    if (dom.acadCountOpener) dom.acadCountOpener.textContent = counts.repetitive_opener || 0;
    if (dom.acadCountLong) dom.acadCountLong.textContent = counts.long_sentence || 0;

    // Priority actions
    if (dom.acadPriorityActions) {
      const actions = data.priority_actions || [];
      dom.acadPriorityActions.innerHTML = actions.map((act) => `<li>${escapeHtml(act)}</li>`).join("");
    }

    // Sentence diagnostics list
    if (dom.acadCoachList) {
      const feedback = data.sentence_feedback || [];
      if (feedback.length === 0) {
        dom.acadCoachList.innerHTML = `
          <div class="text-xs text-emerald" style="padding: 12px; background: rgba(34,197,94,0.08); border-radius: 6px;">
            ✓ Excellent scholarly prose! Zero clarity flags detected across analyzed sentences.
          </div>
        `;
      } else {
        dom.acadCoachList.innerHTML = feedback
          .map((item) => {
            const issuesHtml = (item.issues || [])
              .map(
                (iss) => `
                <div class="acad-issue-tag">
                  <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 2px;">
                    <strong class="text-amber" style="font-size: 11px;">${escapeHtml(iss.category)}:</strong>
                    <span class="font-mono text-muted text-xs">"${escapeHtml(iss.phrase)}"</span>
                  </div>
                  <div class="text-xs text-muted">${escapeHtml(iss.tip)}</div>
                </div>
              `
              )
              .join("");

            return `
              <div class="acad-sentence-item">
                <div class="acad-sentence-head">
                  <span>Sentence #${(item.sentence_index ?? 0) + 1}</span>
                  <span class="font-mono">${item.word_count || 0} words</span>
                </div>
                <div class="acad-sentence-text font-mono">
                  "${escapeHtml(item.sentence || "")}"
                </div>
                <div class="acad-issues-list">
                  ${issuesHtml}
                </div>
              </div>
            `;
          })
          .join("");
      }
    }
  }

  // =========================================================================
  // Event Listeners Binding
  // =========================================================================
  function bindEventListeners() {
    // Mode switcher pills
    dom.modePills.forEach((pill) => {
      pill.addEventListener("click", () => {
        const mode = pill.getAttribute("data-mode");
        if (mode) switchContextMode(mode);
      });
    });

    // Navigation tabs
    dom.tabButtons.forEach((btn) => {
      btn.addEventListener("click", () => {
        const tab = btn.getAttribute("data-tab");
        if (tab) switchTab(tab);
      });
    });

    // Studio: Source actions
    dom.btnGrabClipboard?.addEventListener("click", handleGrabClipboard);
    dom.btnSampleAi?.addEventListener("click", handleTrySampleAiText);
    dom.btnClearSource?.addEventListener("click", handleClearSource);

    // Studio: Live text input debounce
    dom.sourceInput?.addEventListener("input", () => scheduleSourceAnalysis(250));

    // Studio: Action Bar
    dom.intensityButtons.forEach((btn) => {
      btn.addEventListener("click", () => {
        const val = btn.getAttribute("data-intensity");
        if (val) handleIntensityClick(val);
      });
    });

    dom.btnRewrite?.addEventListener("click", handleRewrite);
    dom.btnQuickRewrite?.addEventListener("click", handleQuickRewrite);

    // Keyboard shortcut: Cmd+Enter / Ctrl+Enter to trigger rewrite
    window.addEventListener("keydown", (e) => {
      if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
        if (appState.activeTab === "studio") {
          e.preventDefault();
          handleRewrite();
        }
      }
    });

    // Studio: Output
    dom.outputText?.addEventListener("input", handleOutputChanged);
    dom.btnCopyOnly?.addEventListener("click", handleCopyOnly);
    dom.btnCopyLearn?.addEventListener("click", handleCopyAndLearn);

    // Tab 2: Samples & Import
    dom.importForm?.addEventListener("submit", handleImportSubmit);
    dom.btnSeedDemo?.addEventListener("click", handleSeedDemo);
    dom.btnClearSamples?.addEventListener("click", handleClearSamples);
    setupFileDropzone();

    // Live search filter for samples
    dom.samplesSearch?.addEventListener("input", (e) => {
      const kw = e.target.value;
      if (searchDebounceTimer) clearTimeout(searchDebounceTimer);
      searchDebounceTimer = setTimeout(() => renderSamplesTab(kw), 150);
    });

    // Sample list item actions: expand text and delete button
    dom.samplesListContainer?.addEventListener("click", (e) => {
      const deleteBtn = e.target.closest(".btn-delete-sample");
      if (deleteBtn) {
        const sampleId = deleteBtn.getAttribute("data-id");
        if (sampleId) handleDeleteSample(sampleId);
        return;
      }

      const contentBox = e.target.closest(".sample-content");
      if (contentBox) {
        contentBox.classList.toggle("expanded");
      }
    });

    // Tab 3: Rules
    dom.addRuleForm?.addEventListener("submit", handleAddRule);
    dom.rulesTableBody?.addEventListener("click", (e) => {
      const deleteBtn = e.target.closest(".btn-delete-rule");
      if (deleteBtn) {
        const ruleId = deleteBtn.getAttribute("data-id");
        if (ruleId) handleDeleteRule(ruleId);
      }
    });

    // Tab 4: macOS Hotkey & CLI
    dom.btnInstallHotkey?.addEventListener("click", handleInstallHotkey);
    dom.copySnippetButtons.forEach((btn) => {
      btn.addEventListener("click", () => handleCopySnippet(btn));
    });

    // Settings Modal
    dom.btnOpenSettings?.addEventListener("click", openSettingsModal);
    dom.btnCloseSettings?.addEventListener("click", closeSettingsModal);
    dom.btnCancelSettings?.addEventListener("click", closeSettingsModal);
    dom.settingsForm?.addEventListener("submit", handleSaveSettings);

    dom.settingsModalBackdrop?.addEventListener("click", (e) => {
      if (e.target === dom.settingsModalBackdrop) closeSettingsModal();
    });

    window.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && dom.settingsModalBackdrop?.style.display === "flex") {
        closeSettingsModal();
      }
    });

    // Clear key handlers in settings
    dom.btnClearGemini?.addEventListener("click", () => {
      appState.clearKeyFlags.gemini = true;
      if (dom.badgeGemini) dom.badgeGemini.textContent = "Will be cleared on save";
      if (dom.settingGeminiKey) dom.settingGeminiKey.value = "";
    });
    dom.btnClearOpenai?.addEventListener("click", () => {
      appState.clearKeyFlags.openai = true;
      if (dom.badgeOpenai) dom.badgeOpenai.textContent = "Will be cleared on save";
      if (dom.settingOpenaiKey) dom.settingOpenaiKey.value = "";
    });
    dom.btnClearAnthropic?.addEventListener("click", () => {
      appState.clearKeyFlags.anthropic = true;
      if (dom.badgeAnthropic) dom.badgeAnthropic.textContent = "Will be cleared on save";
      if (dom.settingAnthropicKey) dom.settingAnthropicKey.value = "";
    });

    // Pangram Labs Settings Clear & Test
    dom.btnClearPangram?.addEventListener("click", () => {
      appState.clearKeyFlags.pangram = true;
      if (dom.badgePangram) {
        dom.badgePangram.textContent = "Will be cleared on save";
        dom.badgePangram.className = "provider-status-badge badge-unconfigured";
      }
      if (dom.settingPangramKey) dom.settingPangramKey.value = "";
    });
    dom.btnTestPangram?.addEventListener("click", handleTestPangramKey);

    // Inspector Tab Listeners
    dom.inspectorInput?.addEventListener("input", updateInspectorCounts);
    dom.btnInspectorClipboard?.addEventListener("click", handleGrabInspectorClipboard);
    dom.btnInspectorUpload?.addEventListener("click", () => dom.fileInspectorUpload?.click());
    dom.fileInspectorUpload?.addEventListener("change", handleInspectorUpload);
    dom.btnInspectorSample?.addEventListener("click", handleLoadInspectorSample);
    dom.btnInspectorSettings?.addEventListener("click", openSettingsModal);
    dom.btnRunFreePrescan?.addEventListener("click", handleRunFreePrescan);
    dom.btnClosePrescan?.addEventListener("click", () => {
      if (dom.inspectorPrescanBox) dom.inspectorPrescanBox.style.display = "none";
    });
    dom.btnRunPangramScan?.addEventListener("click", handleRunPangramScan);
    dom.btnRunPlagiarismScan?.addEventListener("click", handleRunPlagiarismScan);

    // Saved Pangram Scan History Delegated Actions
    dom.inspectorHistoryTbody?.addEventListener("click", (e) => {
      const loadBtn = e.target.closest(".btn-load-scan");
      if (loadBtn) {
        const scanId = loadBtn.getAttribute("data-id");
        if (scanId) handleLoadSavedScan(scanId);
        return;
      }
      const delBtn = e.target.closest(".btn-del-scan");
      if (delBtn) {
        const scanId = delBtn.getAttribute("data-id");
        if (scanId) handleDeleteSavedScan(scanId);
      }
    });

    // Academic Tab Listeners
    dom.acadDraftInput?.addEventListener("input", updateAcademicCounts);
    dom.acadSourceInput?.addEventListener("input", updateAcademicCounts);
    dom.btnAcadSample?.addEventListener("click", handleLoadAcademicSample);
    dom.btnAcadClipboard?.addEventListener("click", handleGrabAcademicClipboard);
    dom.btnAcadClear?.addEventListener("click", handleClearAcademic);
    dom.btnRunCoach?.addEventListener("click", handleRunCoach);
    dom.btnRunCitationCheck?.addEventListener("click", handleRunCitationCheck);
    dom.btnRunBothAcad?.addEventListener("click", handleRunBothAcademic);
  }

  // =========================================================================
  // Application Bootstrap
  // =========================================================================
  async function init() {
    cacheDomElements();
    bindEventListeners();

    try {
      const state = await apiRequest("/api/state");
      applyState(state);
    } catch (err) {
      console.error("Failed to load initial state:", err);
      showToast("Connection Error", "Could not reach VoicePrint server daemon at 127.0.0.1:8765", "error");
    }
  }

  // Run on DOM ready
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
