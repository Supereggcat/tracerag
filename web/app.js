/* TraceRAG web client. No build step: plain modules, plain fetch. */

(function () {
  "use strict";

  var DECISION_ICONS = { descend: "arrow-down-right", select: "check", prune: "x" };

  var state = {
    docs: [],
    docId: "",
    tree: null,
    mode: "hybrid",
    topK: 5,
    config: null,
    hits: [],
  };

  var dom = {};

  function $(id) { return document.getElementById(id); }

  function esc(value) {
    return String(value == null ? "" : value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function icons() {
    if (window.lucide && typeof window.lucide.createIcons === "function") {
      window.lucide.createIcons();
    }
  }

  function toast(message, kind) {
    dom.toast.textContent = message;
    dom.toast.style.background = kind === "error" ? "var(--danger)" : "";
    dom.toast.classList.add("is-visible");
    window.clearTimeout(toast._timer);
    toast._timer = window.setTimeout(function () {
      dom.toast.classList.remove("is-visible");
    }, 3600);
  }

  async function api(path, options) {
    var response = await fetch(path, options);
    var payload = null;
    try { payload = await response.json(); } catch (error) { payload = null; }
    if (!response.ok) {
      var detail = payload && payload.detail ? payload.detail : response.statusText;
      throw new Error(detail || ("HTTP " + response.status));
    }
    return payload;
  }

  function formatInt(value) {
    return new Intl.NumberFormat().format(value || 0);
  }

  /* ---------- status ---------- */

  function renderStatus() {
    var config = state.config || {};
    if (config.llm_enabled) {
      dom.statusPill.className = "pill pill-ok";
      dom.statusPill.innerHTML = '<i data-lucide="cpu"></i><span id="status-text">' + esc(config.model) + "</span>";
    } else {
      dom.statusPill.className = "pill pill-off";
      dom.statusPill.innerHTML = '<i data-lucide="wifi-off"></i><span id="status-text">Offline BM25</span>';
    }
    icons();
  }

  /* ---------- library ---------- */

  function renderDocs() {
    dom.docCount.textContent = String(state.docs.length);
    dom.docSelect.innerHTML = "";

    if (!state.docs.length) {
      dom.docList.innerHTML = "";
      dom.docSelect.innerHTML = '<option value="">No document</option>';
      return;
    }

    state.docs.forEach(function (doc) {
      var option = document.createElement("option");
      option.value = doc.doc_id;
      option.textContent = doc.source_name;
      if (doc.doc_id === state.docId) { option.selected = true; }
      dom.docSelect.appendChild(option);
    });

    dom.docList.innerHTML = "";
    state.docs.forEach(function (doc) {
      var row = document.createElement("li");
      var button = document.createElement("button");
      button.type = "button";
      button.className = "doc-item" + (doc.doc_id === state.docId ? " is-active" : "");
      button.dataset.docId = doc.doc_id;
      button.innerHTML =
        '<span class="doc-main">' +
          '<span class="doc-name">' + esc(doc.source_name) + "</span>" +
          '<span class="doc-meta">' + doc.nodes + " nodes · " + formatInt(doc.estimated_tokens) + " tok · depth " + doc.max_depth + "</span>" +
        "</span>" +
        '<span class="btn btn-icon" role="button" tabindex="0" title="Remove document" data-remove="' + esc(doc.doc_id) + '">' +
          '<i data-lucide="trash-2"></i>' +
        "</span>";
      row.appendChild(button);
      dom.docList.appendChild(row);
    });
    icons();
  }

  function renderOutline() {
    var tree = state.tree;
    if (!tree) {
      dom.outline.innerHTML = "";
      dom.outlineCount.textContent = "0";
      return;
    }
    var total = 0;
    var html = [];
    (function walk(node) {
      (node.children || []).forEach(function (child) {
        if (total >= 400) { return; }
        total += 1;
        var depth = Math.max(child.level - 1, 0);
        var hit = state.hits.indexOf(child.id) !== -1;
        html.push(
          '<div class="outline-item' + (hit ? " is-hit" : "") + '" data-node-id="' + esc(child.id) + '" style="--depth:' + depth + '">' +
            '<span class="outline-id">' + esc(child.id) + "</span>" +
            '<span class="outline-title" title="' + esc(child.title) + '">' + esc(child.title) + "</span>" +
          "</div>"
        );
        walk(child);
      });
    })(tree);
    dom.outline.innerHTML = html.join("");
    dom.outlineCount.textContent = String(total);
  }

  /* ---------- answer ---------- */

  function renderAnswerText(text, citations) {
    var known = {};
    citations.forEach(function (citation) { known[citation.ref] = true; });
    var escaped = esc(text).replace(/\[(\d+)\]/g, function (match, value) {
      return known[Number(value)]
        ? '<button type="button" class="cite-chip" data-ref="' + value + '" title="Citation ' + value + '">' + value + "</button>"
        : "";
    });
    return escaped
      .split(/\n{2,}/)
      .map(function (part) { return "<p>" + part.trim() + "</p>"; })
      .join("");
  }

  function renderLedger(entries) {
    dom.ledgerCount.textContent = String(entries.length);
    if (!entries.length) {
      dom.ledger.innerHTML = '<li class="ledger-row is-bad"><span class="ledger-mark"><i data-lucide="alert-triangle"></i></span><span class="ledger-body">No sentences to audit.</span><span class="ledger-score"></span></li>';
      icons();
      return;
    }
    dom.ledger.innerHTML = entries.map(function (entry) {
      var ok = entry.supported;
      var refs = entry.citation_refs && entry.citation_refs.length
        ? entry.citation_refs.map(function (ref) { return "[" + ref + "]"; }).join(" ")
        : "—";
      var percent = Math.max(0, Math.min(100, Math.round((entry.support || 0) * 100)));
      return (
        '<li class="ledger-row ' + (ok ? "is-ok" : "is-bad") + '">' +
          '<span class="ledger-mark"><i data-lucide="' + (ok ? "check-circle-2" : "alert-triangle") + '"></i></span>' +
          '<span class="ledger-body">' + esc(entry.sentence) +
            (entry.note ? '<span class="ledger-note">' + esc(entry.note) + "</span>" : "") +
          "</span>" +
          '<span class="ledger-score">' +
            '<span class="ledger-refs">' + refs + "</span>" +
            '<span class="meter"><span style="width:' + percent + '%"></span></span>' +
          "</span>" +
        "</li>"
      );
    }).join("");
    icons();
  }

  function renderCitations(citations) {
    dom.citationCount.textContent = String(citations.length);
    if (!citations.length) {
      dom.citations.innerHTML = "";
      return;
    }
    dom.citations.innerHTML = citations.map(function (citation) {
      var pages = citation.page_start === citation.page_end
        ? "p. " + citation.page_start
        : "pp. " + citation.page_start + "–" + citation.page_end;
      return (
        '<li class="citation" data-citation-ref="' + citation.ref + '" id="citation-' + citation.ref + '">' +
          '<span class="citation-head">' +
            '<span class="citation-ref">[' + citation.ref + "]</span>" +
            '<span class="citation-title" title="' + esc(citation.title) + '">' + esc(citation.title) + "</span>" +
          "</span>" +
          '<span class="citation-meta">' + esc(citation.node_id) + " · " + pages + " · score " + Number(citation.score).toFixed(3) + "</span>" +
          '<span class="citation-snippet">' + esc(citation.snippet) + "</span>" +
        "</li>"
      );
    }).join("");
  }

  function renderTrace(steps) {
    dom.traceCount.textContent = String(steps.length);
    if (!steps.length) {
      dom.traceList.innerHTML = "";
      return;
    }
    dom.traceList.innerHTML = steps.map(function (step) {
      var icon = DECISION_ICONS[step.decision] || "circle";
      return (
        '<li class="trace-item" data-decision="' + esc(step.decision) + '" data-node-id="' + esc(step.node_id) + '">' +
          '<span class="trace-decision"><i data-lucide="' + icon + '"></i>' + esc(step.decision) + "</span>" +
          '<span class="trace-title" title="' + esc(step.title) + '">' + esc(step.title) + "</span>" +
          '<span class="trace-score">' + Number(step.score).toFixed(2) + "</span>" +
          (step.rationale ? '<span class="trace-reason">' + esc(step.rationale) + "</span>" : "") +
        "</li>"
      );
    }).join("");
    icons();
  }

  function renderUsage(usage) {
    if (!usage) {
      dom.usage.innerHTML = "";
      return;
    }
    var parts = [
      usage.calls > 0 ? (usage.model || "model") : "offline",
      usage.calls + " calls",
      formatInt(usage.total_tokens) + " tok",
      "$" + Number(usage.cost_usd || 0).toFixed(6),
      usage.latency_ms + " ms",
    ];
    dom.usage.innerHTML = parts.map(function (part) {
      return "<span>" + esc(part) + "</span>";
    }).join("");
  }

  function renderAnswer(payload) {
    dom.emptyState.hidden = true;
    dom.answerRegion.hidden = false;

    dom.answerMode.textContent = payload.mode;
    var support = Math.round((payload.supported_ratio || 0) * 100);
    dom.answerSupport.className = "pill " + (support >= 80 ? "pill-ok" : "pill-warn");
    dom.answerSupport.innerHTML = '<i data-lucide="shield-check"></i><span>' + support + "% supported</span>";
    dom.answerText.innerHTML = renderAnswerText(payload.text, payload.citations || []);

    renderLedger(payload.ledger || []);
    renderCitations(payload.citations || []);
    renderTrace(payload.trace || []);
    renderUsage(payload.usage);

    state.hits = payload.selected || [];
    renderOutline();
    icons();
  }

  /* ---------- actions ---------- */

  function setBusy(busy) {
    dom.askBtn.disabled = busy;
    dom.askBtn.innerHTML = busy
      ? '<i data-lucide="loader-circle" class="spin"></i><span>Working</span>'
      : '<i data-lucide="corner-down-left"></i><span>Ask</span>';
    icons();
  }

  async function loadDocs() {
    state.docs = await api("/api/docs");
    if (!state.docId || !state.docs.some(function (doc) { return doc.doc_id === state.docId; })) {
      state.docId = state.docs.length ? state.docs[0].doc_id : "";
    }
    renderDocs();
    if (state.docId) { await loadTree(state.docId); } else { state.tree = null; renderOutline(); }
    updateEmptyState();
  }

  async function loadTree(docId) {
    state.tree = await api("/api/docs/" + encodeURIComponent(docId) + "/tree");
    state.hits = [];
    renderOutline();
  }

  function updateEmptyState() {
    dom.emptyText.textContent = state.docs.length ? "No answer yet" : "No document indexed";
  }

  async function selectDoc(docId) {
    if (!docId || docId === state.docId) { return; }
    state.docId = docId;
    renderDocs();
    try {
      await loadTree(docId);
    } catch (error) {
      toast("Could not load that document: " + error.message, "error");
    }
  }

  async function ask(event) {
    if (event) { event.preventDefault(); }
    var question = dom.question.value.trim();
    if (!question) { dom.question.focus(); return; }
    if (!state.docId) { toast("Upload a document first", "error"); return; }

    setBusy(true);
    try {
      var payload = await api("/api/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: question,
          doc_id: state.docId,
          mode: state.mode,
          top_k: state.topK,
        }),
      });
      renderAnswer(payload);
    } catch (error) {
      toast("Query failed: " + error.message, "error");
    } finally {
      setBusy(false);
    }
  }

  async function upload(file) {
    var body = new FormData();
    body.append("file", file, file.name);
    dom.uploadBtn.disabled = true;
    try {
      var meta = await api("/api/docs", { method: "POST", body: body });
      state.docId = meta.doc_id;
      await loadDocs();
      toast("Indexed " + meta.source_name + " · " + meta.nodes + " nodes");
    } catch (error) {
      toast("Upload failed: " + error.message, "error");
    } finally {
      dom.uploadBtn.disabled = false;
    }
  }

  async function removeDoc(docId) {
    try {
      await api("/api/docs/" + encodeURIComponent(docId), { method: "DELETE" });
      if (state.docId === docId) { state.docId = ""; }
      await loadDocs();
      toast("Removed");
    } catch (error) {
      toast("Could not remove: " + error.message, "error");
    }
  }

  function flashCitation(ref) {
    var node = document.getElementById("citation-" + ref);
    if (!node) { return; }
    node.classList.add("is-flash");
    node.scrollIntoView({ behavior: "smooth", block: "center" });
    window.setTimeout(function () { node.classList.remove("is-flash"); }, 1400);
  }

  /* ---------- wiring ---------- */

  function bind() {
    dom.question = $("question");
    dom.askBtn = $("ask-btn");
    dom.askForm = $("ask-form");
    dom.docSelect = $("doc-select");
    dom.docList = $("doc-list");
    dom.docCount = $("doc-count");
    dom.outline = $("outline");
    dom.outlineCount = $("outline-count");
    dom.emptyState = $("empty-state");
    dom.emptyText = $("empty-text");
    dom.answerRegion = $("answer-region");
    dom.answerMode = $("answer-mode");
    dom.answerSupport = $("answer-support");
    dom.answerText = $("answer-text");
    dom.ledger = $("ledger");
    dom.ledgerCount = $("ledger-count");
    dom.citations = $("citations");
    dom.citationCount = $("citation-count");
    dom.traceList = $("trace-list");
    dom.traceCount = $("trace-count");
    dom.usage = $("usage");
    dom.toast = $("toast");
    dom.statusPill = $("status-pill");
    dom.uploadBtn = $("upload-btn");
    dom.fileInput = $("file-input");
    dom.topK = $("top-k");

    dom.askForm.addEventListener("submit", ask);

    dom.docSelect.addEventListener("change", function () { selectDoc(dom.docSelect.value); });

    dom.docList.addEventListener("click", function (event) {
      var remove = event.target.closest("[data-remove]");
      if (remove) {
        event.stopPropagation();
        removeDoc(remove.dataset.remove);
        return;
      }
      var item = event.target.closest("[data-doc-id]");
      if (item) { selectDoc(item.dataset.docId); }
    });

    dom.docList.addEventListener("keydown", function (event) {
      if (event.key !== "Enter" && event.key !== " ") { return; }
      var remove = event.target.closest("[data-remove]");
      if (remove) { event.preventDefault(); removeDoc(remove.dataset.remove); }
    });

    $("mode-group").addEventListener("click", function (event) {
      var button = event.target.closest("[data-mode]");
      if (!button) { return; }
      state.mode = button.dataset.mode;
      Array.prototype.forEach.call(this.querySelectorAll(".seg"), function (seg) {
        var active = seg === button;
        seg.classList.toggle("is-active", active);
        seg.setAttribute("aria-pressed", active ? "true" : "false");
      });
    });

    dom.topK.addEventListener("change", function () {
      var value = parseInt(dom.topK.value, 10);
      state.topK = Math.max(1, Math.min(12, isNaN(value) ? 5 : value));
      dom.topK.value = String(state.topK);
    });

    dom.uploadBtn.addEventListener("click", function () { dom.fileInput.click(); });
    dom.fileInput.addEventListener("change", function () {
      if (dom.fileInput.files && dom.fileInput.files[0]) { upload(dom.fileInput.files[0]); }
      dom.fileInput.value = "";
    });

    dom.answerText.addEventListener("click", function (event) {
      var chip = event.target.closest(".cite-chip");
      if (chip) { flashCitation(chip.dataset.ref); }
    });
  }

  async function boot() {
    bind();
    icons();
    try {
      state.config = await api("/api/config");
      renderStatus();
      await loadDocs();
    } catch (error) {
      toast("Startup failed: " + error.message, "error");
    }
  }

  document.addEventListener("DOMContentLoaded", boot);
})();
