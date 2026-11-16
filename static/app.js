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
    },
  };

  const SAMPLE_AI_PARAGRAPH =
    "In today's fast-paced digital age, it is important to note that artificial intelligence has become a multifaceted tapestry of innovation. By leveraging cutting-edge solutions, organizations can seamlessly elevate their workflows and foster unprecedented synergy. Furthermore, delving into this transformative landscape serves as a testament to human ingenuity. Needless to say, not only does this empower teams, but it also optimizes efficiency across the board.";

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
    appState.clearKeyFlags = { gemini: false, openai: false, anthropic: false };
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
