"use strict";

// ------------------------------------------------------------------ state
const state = {
  health: null,
  docs: [],
  records: [],
  evidence: new Map(), // "filename\u0000field" -> {quote, page, verified}
  edits: new Map(), // "filename\u0000field" -> {value, original, edited_at}
  pinned: null, // filename the next question is restricted to
  busy: false,
};

const $ = (sel) => document.querySelector(sel);
const el = (tag, attrs = {}, ...children) => {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === undefined || v === null || v === false) continue;
    if (k === "class") n.className = v;
    else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
    else if (k === "html") n.innerHTML = v;
    else n.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat()) if (c !== null && c !== undefined && c !== false) n.append(c.nodeType ? c : String(c));
  return n;
};

const ROUTES = {
  sql: "From the extracted table",
  document: "From the whole document",
  search: "From document search",
  not_found: "Not in the documents",
  error: "Error",
};

async function api(path, opts = {}) {
  const r = await fetch(path, opts);
  if (!r.ok) {
    let msg = r.statusText;
    try { msg = (await r.json()).detail || msg; } catch {}
    throw new Error(msg);
  }
  return r.headers.get("content-type")?.includes("json") ? r.json() : r;
}

// Read a newline-delimited JSON stream, calling onEvent for every object.
async function readStream(response, onEvent) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let i;
    while ((i = buf.indexOf("\n")) >= 0) {
      const line = buf.slice(0, i).trim();
      buf = buf.slice(i + 1);
      if (line) onEvent(JSON.parse(line));
    }
  }
}

function banner(text, kind = "warn") {
  const b = $("#banner");
  b.hidden = !text;
  b.textContent = text || "";
  b.className = "banner" + (kind === "error" ? " error" : "");
}

const short = (s, n = 60) => (s && s.length > n ? s.slice(0, n - 1) + "…" : s || "");
const fmtS = (s) => (s < 10 ? s.toFixed(1) : Math.round(s)) + " s";
const label = (name) => name.replace(/_/g, " ");

function docTitle(filename) {
  const rec = state.records.find((r) => r.filename === filename);
  const hf = state.health?.header_fields || [];
  if (rec) {
    const title = rec[hf[0]];
    const parties = hf.slice(1).map((f) => rec[f]).filter(Boolean);
    if (title || parties.length) return { title: title || filename, sub: parties.join(" · ") };
  }
  return { title: filename, sub: "" };
}

// ------------------------------------------------------------------ loading
async function loadAll() {
  const [health, profiles] = await Promise.all([api("/health"), api("/profiles")]);
  state.health = health;
  renderProfiles(profiles);
  $("#schema-name").textContent = `${health.schema} schema · ${health.fields.length} fields`;
  banner(health.index_problem || "");
  await refreshLibrary();
  renderExamples();
  loadConcurrency();
}

// ------------------------------------------------------------------ concurrency
let conc = null;

async function loadConcurrency() {
  conc = await api("/concurrency");
  $("#conc-value").textContent = conc.value;
  $("#conc-btn").classList.toggle("over", conc.local && conc.value > conc.recommended);
}

function renderConcurrency(value) {
  const v = Math.max(1, Math.min(conc.max, Math.round(+value || 1)));
  $("#conc-range").value = v;
  $("#conc-number").value = v;
  $("#conc-help").textContent = conc.local
    ? "How many requests the local model server works on at once: several documents of an upload, and the parts of a long document, are read side by side."
    : "How many requests are sent to the cloud API at once: several documents of an upload, and the parts of a long document, are read side by side. Up to 10.";
  $("#conc-rec").innerHTML = "";
  $("#conc-rec").append(el("strong", {}, `Recommended: ${conc.recommended}`), " – ", conc.why);
  const hw = conc.hardware || {};
  $("#conc-hw").textContent = conc.local
    ? `This machine: ${hw.cpu_cores ?? "?"} CPU threads · ${hw.ram_gb ?? "?"} GB RAM · GPU: ${hw.gpu || "none found"}${hw.vram_gb ? ` (${hw.vram_gb} GB)` : ""}`
    : "";
  const warn = $("#conc-warn");
  warn.hidden = !(conc.local && v > conc.recommended);
  warn.textContent = `Above the recommendation for this machine: requests will queue on the local model server and each answer gets slower; raising it also needs "Max concurrent predictions" in LM Studio to be at least ${v}.`;
}

$("#conc-btn").addEventListener("click", async (e) => {
  e.stopPropagation();
  const pop = $("#conc-pop");
  if (!pop.hidden) { pop.hidden = true; return; }
  await loadConcurrency();
  renderConcurrency(conc.value);
  pop.hidden = false;
});
$("#conc-pop").addEventListener("click", (e) => e.stopPropagation());
document.addEventListener("click", () => ($("#conc-pop").hidden = true));
$("#conc-range").addEventListener("input", (e) => renderConcurrency(e.target.value));
$("#conc-number").addEventListener("input", (e) => renderConcurrency(e.target.value));
$("#conc-use-rec").addEventListener("click", () => renderConcurrency(conc.recommended));
$("#conc-apply").addEventListener("click", async () => {
  try {
    conc = await api("/concurrency", { method: "PUT", headers: { "content-type": "application/json" }, body: JSON.stringify({ value: +$("#conc-number").value }) });
    $("#conc-value").textContent = conc.value;
    $("#conc-btn").classList.toggle("over", conc.local && conc.value > conc.recommended);
    $("#conc-pop").hidden = true;
  } catch (err) {
    alert(err.message);
  }
});

async function refreshLibrary() {
  const [docs, records, evidence, edits] = await Promise.all([api("/documents"), api("/records"), api("/evidence"), api("/edits")]);
  state.docs = docs;
  state.records = records;
  state.evidence = new Map(evidence.map((e) => [e.filename + "\u0000" + e.field, e]));
  state.edits = new Map(edits.map((e) => [e.filename + "\u0000" + e.field, e]));
  $("#doc-count").textContent = `${docs.length} document${docs.length === 1 ? "" : "s"}`;
  $("#records-count").textContent = records.length || "";
  renderDocs();
  renderGrid();
}

// Sequence number (order processed) and when / how long, for a document.
function docOrder(filename) {
  // a document keeps its number for good: its id in the library, not its position in the list
  const d = state.docs.find((x) => x.filename === filename);
  return d ? d.id : null;
}
function fmtWhen(ts) {
  if (!ts) return "";
  const d = new Date(ts.replace(" ", "T") + "Z"); // SQLite CURRENT_TIMESTAMP is UTC
  if (isNaN(d)) return ts;
  const pad = (n) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
}
function processedLine(d) {
  return `Processed ${fmtWhen(d.ingested_at)}${d.ingest_seconds ? ` in ${fmtDur(d.ingest_seconds)}` : ""}`;
}

// Which models processed a document: its vectors, and its extracted row (a chat model).
function modelLine(d) {
  return `Vectors: ${d.embed_model || "not recorded"} · Fields extracted by: ${d.extract_model || "not recorded"}`;
}

// ------------------------------------------------------------------ profiles
function renderProfiles({ active, profiles }) {
  const sel = $("#profile-select");
  sel.innerHTML = "";
  for (const p of profiles) {
    const name = `${p.label}${p.available ? "" : " (not configured)"}`;
    sel.append(el("option", { value: p.name, selected: p.name === active, disabled: !p.available }, name));
  }
  const cur = profiles.find((p) => p.name === active);
  const badge = $("#profile-badge");
  const where = cur.chat_local && cur.embed_local ? "On this machine"
    : !cur.chat_local && !cur.embed_local ? "Cloud API"
    : cur.chat_local ? "Local chat · cloud embeddings" : "Cloud chat · local embeddings";
  badge.textContent = where;
  badge.className = "badge " + (cur.local ? "local" : cur.chat_local || cur.embed_local ? "mixed" : "cloud");
  badge.title = `Chat: ${cur.chat_model} (${cur.chat_local ? "this machine" : "cloud"})\n` +
    `Embeddings: ${cur.embed_model} (${cur.embed_local ? "this machine" : "cloud"})\nLibrary: ${cur.library}`;
  sel.title = badge.title;
  sel.parentElement.parentElement.hidden = profiles.length < 2 && false;
}

$("#profile-select").addEventListener("change", async (e) => {
  try {
    await api("/profiles/active", { method: "PUT", headers: { "content-type": "application/json" }, body: JSON.stringify({ name: e.target.value }) });
    state.pinned = null;
    renderPin();
    await loadAll();
  } catch (err) {
    banner(err.message, "error");
    loadAll();
  }
});

// ------------------------------------------------------------------ library
function renderDocs() {
  const q = $("#doc-filter").value.toLowerCase();
  const ul = $("#docs");
  ul.innerHTML = "";
  state.docs.forEach((d) => {
    const { title, sub } = docTitle(d.filename);
    if (q && !(d.filename + title + sub).toLowerCase().includes(q)) return;
    const li = el(
      "li",
      { class: d.filename === state.pinned ? "pinned" : "", title: `#${d.id} ${d.filename}\n${processedLine(d)}\n${modelLine(d)}`, onclick: () => openViewer(d.filename) },
      el("div", { class: "doc-title" }, el("span", { class: "seq" }, `#${d.id}`), title),
      el("div", { class: "doc-file" }, sub || d.filename),
      el("div", { class: "doc-when" }, processedLine(d)),
      el(
        "div",
        { class: "doc-meta" },
        el("span", { class: "pill" + (d.ocr_pages ? " ocr" : "") }, d.pages ? `${d.pages} p${d.ocr_pages ? " · OCR" : ""}` : "text"),
        el(
          "div",
          { class: "doc-actions" },
          el("button", { class: "icon-btn", title: "Ask about this document only", onclick: (e) => { e.stopPropagation(); pin(d.filename); } }, "Ask"),
          el("button", { class: "icon-btn danger", title: "Remove from the library", onclick: (e) => { e.stopPropagation(); removeDoc(d.filename); } }, "✕"),
        ),
      ),
    );
    ul.append(li);
  });
  if (!state.docs.length) ul.append(el("li", { class: "muted" }, "No documents yet."));
}

$("#doc-filter").addEventListener("input", renderDocs);

async function removeDoc(filename) {
  if (!confirm(`Remove ${filename} from the library?`)) return;
  await api(`/documents/${encodeURIComponent(filename)}`, { method: "DELETE" });
  if (state.pinned === filename) { state.pinned = null; renderPin(); }
  await refreshLibrary();
}

// ------------------------------------------------------------------ upload with progress
const INGEST_STAGES = [
  ["parse", "Read"],
  ["extract", "Extract"],
  ["verify", "Check quotes"],
  ["chunk", "Split"],
  ["embed", "Index"],
  ["save", "Save"],
];

function stageDetail(ev) {
  const { stage, done, total } = ev;
  if (stage === "parse") return total ? `Reading page ${done} of ${total}${ev.ocr ? " (OCR – scanned page)" : ""}` : "Opening the file";
  if (stage === "extract") return total > 1 ? `Model is reading part ${done} of ${total} and filling in the fields` : "Model is reading the document and filling in the fields";
  if (stage === "verify") return ev.quoted !== undefined ? `${ev.verified} of ${ev.quoted} values backed by a quote found in the text` : "Looking up each quote in the document";
  if (stage === "chunk") return `${total} passages`;
  if (stage === "embed") return `Indexing passages ${done} of ${total}`;
  if (stage === "save") return "Saving";
  return stage;
}

// Rough share of a file's time per stage, so the bar moves evenly.
const STAGE_WEIGHT = { parse: 0.1, extract: 0.65, verify: 0.03, chunk: 0.02, embed: 0.17, save: 0.03 };

function fileFraction(ev) {
  let f = 0;
  for (const [s] of INGEST_STAGES) {
    if (s === ev.stage) {
      const part = ev.total ? ev.done / ev.total : 0;
      // the extract stage reports when a part *starts*; count it as half done
      const inner = s === "extract" && ev.total ? (ev.done - 0.5) / ev.total : part;
      return f + STAGE_WEIGHT[s] * Math.max(0, inner);
    }
    f += STAGE_WEIGHT[s];
  }
  return f;
}

async function upload(files) {
  if (!files.length) return;
  const fd = new FormData();
  for (const f of files) fd.append("files", f);
  const job = el("div", { class: "job" });
  const head = el("div", { class: "job-head" }, el("span", {}, `Adding ${files.length} document${files.length > 1 ? "s" : ""}`), el("span", { class: "muted" }, "Uploading…"));
  const total = el("div", { class: "bar" }, el("i", { style: "width:2%" }));
  const fileName = el("div", { class: "job-file muted" }, " ");
  const stages = el("div", { class: "stages" }, INGEST_STAGES.map(([s, l]) => el("div", { class: "stage", "data-stage": s }, l)));
  const fileBar = el("div", { class: "bar thin" }, el("i", { style: "width:0" }));
  const detail = el("div", { class: "stage-detail" });
  const results = el("div", { class: "job-results" });
  job.append(head, total, fileName, stages, fileBar, detail, results);
  $("#jobs").prepend(job);

  let started;
  try {
    started = await api("/ingest-jobs", { method: "POST", body: fd });
  } catch (err) {
    head.lastChild.textContent = "Failed";
    detail.textContent = err.message;
    return;
  }
  const n = started.files.length;
  const t0 = Date.now();
  const tick = setInterval(() => (head.lastChild.textContent = fmtS((Date.now() - t0) / 1000)), 500);
  let doneFiles = 0;

  const setStage = (stage) => {
    let passed = true;
    for (const node of stages.children) {
      const s = node.dataset.stage;
      if (s === stage) passed = false;
      node.className = "stage" + (s === stage ? " active" : passed ? " done" : "");
    }
  };

  const response = await fetch(`/ingest-jobs/${started.id}/events`);
  await readStream(response, (ev) => {
    if (ev.type === "file") {
      fileName.textContent = `${ev.index + 1} / ${n} · ${ev.filename}`;
      fileName.title = ev.filename;
      setStage("parse");
      fileBar.firstChild.style.width = "0";
    } else if (ev.type === "step") {
      setStage(ev.stage);
      detail.textContent = stageDetail(ev);
      const frac = fileFraction(ev);
      fileBar.firstChild.style.width = `${Math.round(frac * 100)}%`;
      total.firstChild.style.width = `${Math.max(2, Math.round(((ev.index + frac) / n) * 100))}%`;
    } else if (ev.type === "result") {
      doneFiles = ev.index + 1;
      total.firstChild.style.width = `${Math.round((doneFiles / n) * 100)}%`;
      const ok = ev.status === "ingested";
      const text = ok
        ? `✓ ${ev.filename} · ${ev.pages ?? "–"} p${ev.ocr_pages ? ` (${ev.ocr_pages} OCR)` : ""} · ${ev.seconds}s`
        : ev.status === "unchanged" ? `= ${ev.filename} · already in the library` : `✕ ${ev.filename} · ${ev.error}`;
      results.prepend(el("div", { class: ok ? "ok" : ev.status === "unchanged" ? "same" : "err", title: ev.error || "" }, text));
      if (ok) refreshLibrary();
    } else if (ev.type === "done") {
      clearInterval(tick);
      for (const node of stages.children) node.className = "stage done";
      total.classList.add("done");
      fileBar.parentNode && fileBar.remove();
      const failed = (ev.results || []).filter((r) => r.status === "error").length;
      head.firstChild.textContent = failed ? `Done · ${failed} failed` : "Done";
      head.lastChild.textContent = fmtS((Date.now() - t0) / 1000);
      detail.textContent = "";
      fileName.textContent = `${n} file${n > 1 ? "s" : ""}`;
      refreshLibrary();
      setTimeout(() => { if (!failed) job.remove(); }, 15000);
    }
  });
}

const drop = $("#drop");
$("#file-input").addEventListener("change", (e) => { upload([...e.target.files]); e.target.value = ""; });
drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
drop.addEventListener("dragleave", () => drop.classList.remove("over"));
drop.addEventListener("drop", (e) => { e.preventDefault(); drop.classList.remove("over"); upload([...e.dataTransfer.files]); });

// ------------------------------------------------------------------ examples
function renderExamples() {
  const box = $("#examples");
  box.innerHTML = "";
  for (const ex of state.health.examples || []) {
    const shows = ex.shows || "";
    box.append(
      el(
        "button",
        { class: "example", onclick: () => ask(ex.question) },
        shows ? el("span", { class: `route ${shows}` }, ROUTES[shows] || shows) : null,
        el("span", { class: "q" }, ex.question),
      ),
    );
  }
}

// ------------------------------------------------------------------ pin
function pin(filename) {
  state.pinned = filename;
  renderPin();
  renderDocs();
  closeViewer();
  showTab("ask");
  $("#question").focus();
}

function renderPin() {
  $("#pin").hidden = !state.pinned;
  if (state.pinned) {
    const { title, sub } = docTitle(state.pinned);
    $("#pin-name").textContent = short(sub ? `${title} – ${sub}` : title, 80);
    $("#pin-name").title = state.pinned;
  }
}

$("#pin-clear").addEventListener("click", () => { state.pinned = null; renderPin(); renderDocs(); });

// ------------------------------------------------------------------ ask with progress
const ASK_STEPS = {
  match: "Look for documents named in the question",
  route: "Decide how to answer",
  sql: "Query the extracted table",
  relax: "No rows – check the filter against the stored values",
  read: "Read the named document",
  search: "Search all documents",
  answer: "Write the answer",
};

function stepInfo(ev) {
  if (ev.stage === "match") {
    const docs = Object.entries(ev.documents || {});
    if (!docs.length) return "none named";
    return docs.map(([f, how]) => `${short(docTitle(f).sub || docTitle(f).title, 40)}${how === "partial" ? " (short name)" : ""}`).join(", ");
  }
  if (ev.stage === "route" && ev.route) return { sql: "table query", document: "one document", search: "search" }[ev.route] || ev.route;
  if (ev.stage === "sql" && ev.rows !== undefined) return `${ev.rows} row${ev.rows === 1 ? "" : "s"}`;
  if (ev.stage === "relax" && ev.changed !== undefined) return ev.changed ? "filter loosened" : "filter is right – nothing matches";
  if (ev.stage === "read") return short(docTitle(ev.document).title, 50) + (ev.pages ? ` · ${ev.pages} pages` : "");
  if (ev.stage === "search" && ev.excerpts !== undefined) return `${ev.excerpts} passages`;
  return "";
}

async function ask(question) {
  question = question.trim();
  if (!question || state.busy) return;
  state.busy = true;
  $("#send").disabled = true;
  $("#welcome")?.remove();
  const thread = $("#thread");
  const document_ = state.pinned;

  thread.append(
    el("div", { class: "msg-q" }, question, document_ ? el("span", { class: "pinned-to" }, `Only in ${short(docTitle(document_).title, 60)}`) : null),
  );
  const steps = el("div", { class: "steps" });
  const box = el("div", { class: "msg-a" }, steps);
  thread.append(box);
  box.scrollIntoView({ behavior: "smooth", block: "end" });

  const rows = new Map();
  const t0 = Date.now();
  let current = null;
  const startStep = (stage, at) => {
    if (current && current !== stage) finishStep(current, at);
    if (!rows.has(stage)) {
      const row = el("div", { class: "step active" }, el("span", { class: "ico" }), el("span", {}, ASK_STEPS[stage] || stage, " ", el("span", { class: "info" })), el("span", { class: "time" }));
      row.started = at;
      rows.set(stage, row);
      steps.append(row);
    }
    current = stage;
    return rows.get(stage);
  };
  const finishStep = (stage, at) => {
    const row = rows.get(stage);
    if (!row || row.classList.contains("done")) return;
    row.className = "step done";
    row.querySelector(".time").textContent = fmtS(Math.max(0, at - row.started));
  };
  const timer = setInterval(() => {
    const row = current && rows.get(current);
    if (row && !row.classList.contains("done")) row.querySelector(".time").textContent = fmtS((Date.now() - t0) / 1000 - row.started);
  }, 200);

  let answer = null;
  try {
    const response = await fetch("/ask/stream", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ question, document: document_ }),
    });
    if (!response.ok) throw new Error((await response.json()).detail || response.statusText);
    await readStream(response, (ev) => {
      if (ev.type === "step") {
        const row = startStep(ev.stage, ev.t);
        const info = stepInfo(ev);
        if (info) row.querySelector(".info").textContent = "· " + info;
        if (ev.total && ev.done >= ev.total && ev.stage !== "answer") finishStep(ev.stage, ev.t);
      } else if (ev.type === "answer") {
        answer = ev;
      } else if (ev.type === "error") {
        throw new Error(ev.error);
      }
    });
  } catch (err) {
    answer = { route: "error", answer: err.message, seconds: (Date.now() - t0) / 1000 };
  }
  clearInterval(timer);
  if (current) finishStep(current, answer?.seconds ?? (Date.now() - t0) / 1000);
  renderAnswer(box, steps, answer);
  state.busy = false;
  $("#send").disabled = false;
  box.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function renderAnswer(box, steps, a) {
  // collapse the steps into a one-line summary that can be reopened
  const n = steps.children.length;
  const summary = el("div", { class: "steps-summary", onclick: () => (steps.hidden = !steps.hidden) }, `${n} step${n === 1 ? "" : "s"} · ${fmtS(a.seconds || 0)} — show steps`);
  steps.hidden = true;
  box.prepend(summary);

  const meta = el("div", { class: "answer-meta" }, el("span", { class: `route ${a.route}` }, ROUTES[a.route] || a.route));
  if (a.model) meta.append(el("span", {}, a.model));
  if (a.usage) {
    const tokens = (a.usage.input || 0) + (a.usage.output || 0);
    if (tokens) meta.append(el("span", { class: "dot" }, `${tokens.toLocaleString()} tokens`));
    meta.append(el("span", { class: "dot" }, a.usage.usd ? `$${a.usage.usd.toFixed(4)}` : state.health?.profile?.local ? "no API cost" : "cost unknown"));
  }
  box.append(meta);

  const text = el("div", { class: "answer-text" });
  // [n] citations become clickable chips; **bold** from the model is shown as bold
  const parts = String(a.answer || "").split(/(\[\d+\]|\*\*[^*\n]+\*\*)/g);
  const cited = new Map((a.sources || []).map((s) => [s.n, s]));
  for (const p of parts) {
    const m = p.match(/^\[(\d+)\]$/);
    if (m && cited.has(+m[1])) {
      const s = cited.get(+m[1]);
      text.append(el("span", { class: "cite", title: `${s.filename}${s.page ? `, page ${s.page}` : ""}`, onclick: () => openViewer(s.filename, { page: s.page, text: s.text }) }, m[1]));
    } else if (/^\*\*[^*\n]+\*\*$/.test(p)) text.append(el("strong", {}, p.slice(2, -2)));
    else text.append(p);
  }
  box.append(text);

  if (a.rows && a.columns) {
    box.append(
      el("details", { class: "block", open: a.rows.length > 0 && a.rows.length <= 12 },
        el("summary", {}, `Matching rows (${a.rows.length})`),
        el("div", { class: "block-body table-wrap" }, resultTable(a.columns, a.rows, a.evidence || []))),
    );
  }
  if (a.evidence && a.evidence.length) {
    const unverified = a.evidence.filter((e) => e.value !== null && e.value !== false && !e.verified).length;
    box.append(
      el("details", { class: "block" },
        el("summary", {}, `Where each value comes from (${a.evidence.length})`, unverified ? el("span", { class: "tag-warn" }, `${unverified} to check`) : null),
        el("div", { class: "block-body evidence-list" }, a.evidence.map(evidenceRow))),
    );
  }
  if (a.sql) {
    box.append(el("details", { class: "block" }, el("summary", {}, "SQL"), el("div", { class: "block-body" }, el("pre", { class: "sql" }, a.sql))));
  }
  if (a.sources && a.sources.length) {
    box.append(
      el("div", { class: "sources" },
        a.sources.map((s) =>
          el("div", { class: "source", "data-n": s.n, onclick: () => openViewer(s.filename, { page: s.page, text: s.text }) },
            el("span", { class: "n" }, `[${s.n}]`),
            el("span", { class: "where", title: s.filename }, `${docTitle(s.filename).title}${s.page ? ` · page ${s.page}` : ""}`),
            el("span", { class: "snip" }, s.text)))),
    );
  }
}

function evidenceRow(e) {
  const t = docTitle(e.filename);
  return el("div", { class: "ev-row" },
    el("span", { class: "who", title: e.filename, onclick: () => openViewer(e.filename, { quote: e.quote, page: e.page }) }, t.sub || t.title),
    el("span", { class: "what" },
      el("strong", {}, label(e.field)), ": ", fmtValue(e.value), " ",
      e.quote ? el("q", {}, e.quote) : el("span", { class: "muted" }, "no quote"),
      e.page ? el("span", { class: "pg" }, `p. ${e.page}`) : null,
      e.quote && !e.verified ? el("span", { class: "tag-warn" }, "not found in text") : null));
}

function fmtValue(v) {
  if (v === null || v === undefined) return el("span", { class: "null" }, "—");
  if (v === true || v === 1) return el("span", { class: "yes" }, "Yes");
  if (v === false || v === 0) return el("span", { class: "no" }, "No");
  if (typeof v === "number") return Number.isInteger(v) ? String(v) : String(+v.toFixed(2));
  return String(v);
}

function resultTable(columns, rows, evidence) {
  const ev = new Map(evidence.map((e) => [e.filename + "\u0000" + e.field, e]));
  const fileCol = columns.indexOf("filename");
  return el("table", { class: "grid" },
    el("thead", {}, el("tr", {}, columns.map((c) => el("th", {}, label(c))))),
    el("tbody", {}, rows.map((r) => el("tr", {}, r.map((v, i) => {
      const fn = fileCol >= 0 ? r[fileCol] : null;
      if (i === fileCol) return el("td", {}, el("span", { class: "doclink", title: v, onclick: () => openViewer(v) }, short(v, 48)));
      const e = fn && ev.get(fn + "\u0000" + columns[i]);
      return valueCell(v, e, fn);
    })))));
}

function valueCell(v, e, filename) {
  const td = el("td", {}, e && e.quote ? el("span", { class: "ev " + (e.verified ? "ok" : "warn") }) : null, fmtValue(v));
  if (e && e.quote) {
    td.classList.add("v");
    td.addEventListener("mouseenter", () => showPopover(td, e));
    td.addEventListener("mouseleave", hidePopover);
    td.addEventListener("click", () => openViewer(filename, { quote: e.quote, page: e.page }));
  }
  return td;
}

// ------------------------------------------------------------------ extracted data tab
const grid = { sort: { key: "#", dir: 1 }, selected: new Set() };

function rowFlags(r) {
  const fields = state.health.fields.map((f) => f.name);
  const ev = fields.map((f) => state.evidence.get(r.filename + "\u0000" + f)).filter(Boolean);
  return {
    check: ev.some((e) => e.quote && !e.verified),
    edited: fields.some((f) => state.edits.has(r.filename + "\u0000" + f)),
    nosource: ev.length === 0,
  };
}

function sortValue(r, key) {
  if (key === "#") return docOrder(r.filename) ?? 0;
  if (key === "processed") return (state.docs.find((d) => d.filename === r.filename) || {}).ingested_at || "";
  if (key === "document") return r.filename.toLowerCase();
  const v = r[key];
  if (v === null || v === undefined) return null;
  return typeof v === "string" ? v.toLowerCase() : +v;
}

function visibleRecords() {
  const q = $("#grid-search").value.trim().toLowerCase();
  const filter = $("#grid-filter").value;
  let rows = state.records.filter((r) => {
    if (filter !== "all" && !rowFlags(r)[filter]) return false;
    return !q || Object.values(r).some((v) => v !== null && String(v).toLowerCase().includes(q));
  });
  const { key, dir } = grid.sort;
  rows = rows.slice().sort((a, b) => {
    const x = sortValue(a, key), y = sortValue(b, key);
    if (x === y) return 0;
    if (x === null) return 1; // empty values last, either direction
    if (y === null) return -1;
    return (x < y ? -1 : 1) * dir;
  });
  return rows;
}

function header(key, text, title) {
  const active = grid.sort.key === key;
  return el("th", {
    class: "sortable" + (active ? " sorted" : ""), title: title || "Click to sort",
    onclick: () => { grid.sort = { key, dir: active ? -grid.sort.dir : 1 }; renderGrid(); },
  }, text, active ? el("span", { class: "arrow" }, grid.sort.dir > 0 ? " ▲" : " ▼") : null);
}

function editableCell(r, f) {
  const key = r.filename + "\u0000" + f.name;
  const td = valueCell(r[f.name], state.evidence.get(key), r.filename);
  const edit = state.edits.get(key);
  if (edit) {
    td.classList.add("edited");
    td.title = `Edited by hand ${fmtWhen(edit.edited_at)} – was: ${edit.original ?? "empty"}`;
  }
  td.append(el("button", { class: "pen", title: "Edit this value", onclick: (e) => { e.stopPropagation(); startEdit(td, r, f); } }, "✎"));
  return td;
}

function startEdit(td, r, f) {
  hidePopover();
  const key = r.filename + "\u0000" + f.name;
  const cur = r[f.name];
  td.innerHTML = "";
  td.classList.add("editing");
  let input;
  if (f.type === "boolean") {
    input = el("select", {}, ["", "true", "false"].map((v) => el("option", { value: v, selected: String(cur ?? "") === v }, v === "" ? "—" : v === "true" ? "Yes" : "No")));
  } else {
    input = el("input", { type: f.type === "date" ? "date" : f.type === "number" ? "number" : "text", value: cur ?? "", step: "any" });
  }
  const save = async () => {
    const raw = input.value;
    const value = f.type === "boolean" ? (raw === "" ? null : raw === "true") : raw === "" ? null : raw;
    try {
      await api("/records/value", { method: "PUT", headers: { "content-type": "application/json" }, body: JSON.stringify({ filename: r.filename, field: f.name, value }) });
      await refreshLibrary();
    } catch (err) {
      alert(err.message);
      renderGrid();
    }
  };
  const actions = el("span", { class: "edit-actions" },
    el("button", { class: "mini ok", title: "Save (Enter)", onclick: save }, "✓"),
    el("button", { class: "mini", title: "Cancel (Esc)", onclick: () => renderGrid() }, "✕"),
    state.edits.has(key) ? el("button", { class: "mini", title: "Restore the value the model extracted", onclick: async () => {
      await api(`/records/value?filename=${encodeURIComponent(r.filename)}&field=${encodeURIComponent(f.name)}`, { method: "DELETE" });
      await refreshLibrary();
    } }, "↺") : null);
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") save();
    if (e.key === "Escape") { e.stopPropagation(); renderGrid(); }
  });
  td.append(input, actions);
  input.focus();
}

function renderSelection() {
  const n = grid.selected.size;
  $("#selection-actions").hidden = n === 0;
  $("#selection-count").textContent = `${n} selected`;
}

function renderGrid() {
  const table = $("#grid");
  table.innerHTML = "";
  for (const f of [...grid.selected]) if (!state.docs.some((d) => d.filename === f)) grid.selected.delete(f);
  renderSelection();
  if (!state.records.length) {
    table.append(el("tr", {}, el("td", { class: "muted" }, "No extracted data yet – add documents.")));
    $("#grid-count").textContent = "";
    return;
  }
  const fields = state.health.fields;
  const rows = visibleRecords();
  $("#grid-count").textContent = rows.length === state.records.length ? `${rows.length} rows` : `${rows.length} of ${state.records.length} rows`;
  const allOn = rows.length > 0 && rows.every((r) => grid.selected.has(r.filename));
  const all = el("input", { type: "checkbox", title: "Select all shown rows", onchange: (e) => {
    for (const r of rows) e.target.checked ? grid.selected.add(r.filename) : grid.selected.delete(r.filename);
    renderGrid();
  } });
  all.checked = allOn;
  table.append(
    el("thead", {}, el("tr", {}, el("th", { class: "check-cell" }, all), header("#", "#"), header("document", "document"), header("processed", "processed"),
      fields.map((f) => header(f.name, label(f.name), `${f.description} – click to sort`)))),
    el("tbody", {}, rows.map((r) => {
      const d = state.docs.find((x) => x.filename === r.filename) || {};
      const box = el("input", { type: "checkbox", onchange: (e) => {
        e.target.checked ? grid.selected.add(r.filename) : grid.selected.delete(r.filename);
        e.target.closest("tr").classList.toggle("selected", e.target.checked);
        renderSelection();
      } });
      box.checked = grid.selected.has(r.filename);
      return el("tr", { class: box.checked ? "selected" : "" },
        el("td", { class: "check-cell" }, box),
        el("td", { class: "seq-cell" }, docOrder(r.filename) ?? ""),
        el("td", {}, el("span", { class: "doclink", title: r.filename, onclick: () => openViewer(r.filename) }, short(r.filename, 40))),
        el("td", { class: "when-cell", title: modelLine(d) }, fmtWhen(d.ingested_at), d.ingest_seconds ? el("span", { class: "muted" }, ` · ${fmtDur(d.ingest_seconds)}`) : null),
        fields.map((f) => editableCell(r, f)));
    })),
  );
}

$("#grid-search").addEventListener("input", renderGrid);
$("#grid-filter").addEventListener("change", renderGrid);
$("#sel-clear").addEventListener("click", () => { grid.selected.clear(); renderGrid(); });
$("#sel-delete").addEventListener("click", async () => {
  const names = [...grid.selected];
  if (!confirm(`Delete ${names.length} document${names.length > 1 ? "s" : ""} from the library?\n\nTheir text, index, extracted values and edits are removed. This cannot be undone.`)) return;
  await api("/documents/delete", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ filenames: names }) });
  if (names.includes(state.pinned)) { state.pinned = null; renderPin(); }
  grid.selected.clear();
  await refreshLibrary();
});
$("#sel-reprocess").addEventListener("click", async () => {
  const names = [...grid.selected];
  const p = state.health.profile;
  const cost = p.chat_local && p.embed_local ? "Runs on this machine – no API cost." : "Uses a cloud API – this costs money.";
  const mixed = state.docs.some((d) => !names.includes(d.filename) && d.embed_model && d.embed_model !== p.embed_model);
  if (mixed) {
    alert(`Other documents in this library were indexed with a different embedding model than ${p.embed_model}. Select all documents to re-index the whole library, or switch the model preset.`);
    return;
  }
  if (!confirm(`Re-process ${names.length} document${names.length > 1 ? "s" : ""} with ${p.chat_model} + ${p.embed_model}?\n\nFields are extracted again (hand edits are kept) and the text is re-indexed. ${cost}\nProgress and a Stop button appear under Background work.`)) return;
  try {
    await api("/documents/reprocess", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ filenames: names }) });
    grid.selected.clear();
    renderGrid();
    pollActivity();
  } catch (err) {
    alert(err.message);
  }
});

// ------------------------------------------------------------------ popover
function showPopover(anchor, e) {
  const p = $("#popover");
  p.innerHTML = "";
  p.append(el("q", {}, e.quote), el("span", { class: "pg" }, `${e.page ? `Page ${e.page} · ` : ""}${e.verified ? "quote found in the document" : "quote NOT found in the document – check it"} · click to open`));
  p.hidden = false;
  const r = anchor.getBoundingClientRect();
  const top = r.bottom + 6 + p.offsetHeight > innerHeight ? r.top - p.offsetHeight - 6 : r.bottom + 6;
  p.style.top = `${top}px`;
  p.style.left = `${Math.min(r.left, innerWidth - p.offsetWidth - 12)}px`;
}
function hidePopover() { $("#popover").hidden = true; }

// ------------------------------------------------------------------ document viewer
let viewerFile = null;

async function openViewer(filename, focus = {}) {
  hidePopover();
  viewerFile = filename;
  const v = $("#viewer");
  v.hidden = false;
  $("#viewer-backdrop").hidden = false;
  const { title, sub } = docTitle(filename);
  $("#viewer-title").textContent = title;
  $("#viewer-sub").textContent = sub ? `${sub} · ${filename}` : filename;
  const doc = state.docs.find((d) => d.filename === filename);
  $("#viewer-models").textContent = doc ? `#${docOrder(filename)} · ${processedLine(doc)} · ${modelLine(doc)}` : "";
  const body = $("#viewer-body");
  body.innerHTML = '<div class="muted">Loading…</div>';
  let data;
  try {
    data = await api(`/documents/${encodeURIComponent(filename)}/pages`);
  } catch (err) {
    body.textContent = err.message;
    return;
  }
  if (viewerFile !== filename) return;
  body.innerHTML = "";

  if (data.record) {
    const ev = new Map(data.evidence.map((e) => [e.field, e]));
    const rec = el("div", { class: "record" });
    for (const f of state.health.fields) {
      const e = ev.get(f.name);
      rec.append(
        el("span", { class: "k" }, label(f.name)),
        el("span", { class: "val" + (e?.quote ? " has" : ""), onclick: e?.quote ? () => focusQuote(e.quote, e.page) : null },
          e?.quote ? el("span", { class: "ev " + (e.verified ? "ok" : "warn") }) : null, " ", fmtValue(data.record[f.name])),
      );
    }
    body.append(rec);
  }
  for (const p of data.pages) {
    body.append(el("div", { class: "page", "data-page": p.page ?? "" },
      el("div", { class: "page-no" }, p.page ? `Page ${p.page}` : "Text"),
      el("div", { class: "page-text" }, p.text)));
  }
  if (focus.quote) focusQuote(focus.quote, focus.page);
  else if (focus.text) focusQuote(focus.text, focus.page, true);
  else if (focus.page) body.querySelector(`[data-page="${focus.page}"]`)?.scrollIntoView({ block: "start" });
}

// Highlight a quote (or a cited passage) in the page text, tolerating whitespace differences.
function focusQuote(quote, page, passage = false) {
  const body = $("#viewer-body");
  body.querySelectorAll("mark").forEach((m) => m.replaceWith(...m.childNodes));
  body.querySelectorAll(".page-text").forEach((n) => n.normalize());
  const pieces = String(quote).split(/\.\.\.|…/).map((s) => s.trim()).filter((s) => s.length >= 4);
  const pages = [...body.querySelectorAll(".page")];
  const ordered = page ? [...pages.filter((p) => p.dataset.page == page), ...pages.filter((p) => p.dataset.page != page)] : pages;
  let first = null;
  for (const piece of passage ? [String(quote).slice(0, 160)] : pieces) {
    const words = piece.split(/\s+/).filter(Boolean).map((w) => w.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
    if (!words.length) continue;
    const re = new RegExp(words.join("[\\s\\W]*"), "i");
    for (const pg of ordered) {
      const node = pg.querySelector(".page-text");
      const m = node.textContent.match(re);
      if (!m) continue;
      const mark = wrapRange(node, m.index, m.index + m[0].length);
      if (mark && !first) first = mark;
      break;
    }
  }
  if (first) {
    first.classList.add("focus");
    first.scrollIntoView({ block: "center" });
  } else if (page) {
    body.querySelector(`[data-page="${page}"]`)?.scrollIntoView({ block: "start" });
  }
}

function wrapRange(node, start, end) {
  const text = node.firstChild;
  if (!text || text.nodeType !== Node.TEXT_NODE) return null;
  const range = document.createRange();
  range.setStart(text, start);
  range.setEnd(text, end);
  const mark = document.createElement("mark");
  range.surroundContents(mark);
  return mark;
}

function closeViewer() { $("#viewer").hidden = true; $("#viewer-backdrop").hidden = true; viewerFile = null; }
$("#viewer-backdrop").addEventListener("click", closeViewer);
$("#viewer-close").addEventListener("click", closeViewer);
$("#viewer-ask").addEventListener("click", () => viewerFile && pin(viewerFile));
document.addEventListener("keydown", (e) => { if (e.key === "Escape") { closeViewer(); hidePopover(); } });

// ------------------------------------------------------------------ tabs & composer
function showTab(name) {
  for (const t of document.querySelectorAll(".tab")) t.classList.toggle("active", t.dataset.tab === name);
  $("#panel-ask").hidden = name !== "ask";
  $("#panel-data").hidden = name !== "data";
}
for (const t of document.querySelectorAll(".tab")) t.addEventListener("click", () => showTab(t.dataset.tab));

const input = $("#question");
input.addEventListener("input", () => { input.style.height = "auto"; input.style.height = `${input.scrollHeight}px`; });
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); $("#composer").requestSubmit(); }
});
$("#composer").addEventListener("submit", (e) => {
  e.preventDefault();
  const q = input.value;
  if (!q.trim() || state.busy) return;
  input.value = "";
  input.style.height = "auto";
  ask(q);
});

// ------------------------------------------------------------------ background activity
const KIND = { ingest: "Import", upload: "Upload", reprocess: "Re-process", eval: "Evaluation", question: "Question" };

async function stopTask(t) {
  if (!confirm(`Stop "${t.title}"?\n\nIt stops after the step it is on; nothing half-done is saved.`)) return;
  try {
    await api(`/activity/${encodeURIComponent(t.id)}/stop`, { method: "POST" });
  } catch (err) {
    alert(err.message);
  }
  lastActivity = "";
  pollActivity();
}

const fmtDur = (s) => {
  if (s === null || s === undefined) return "";
  s = Math.round(s);
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ${String(s % 60).padStart(2, "0")}s`;
  return `${Math.floor(m / 60)}h ${String(m % 60).padStart(2, "0")}m`;
};

let lastActivity = "";
async function pollActivity() {
  let data;
  try {
    data = await api("/activity");
  } catch {
    return;
  }
  // questions from this page already show their own steps; list only other work
  const tasks = data.tasks.filter((t) => t.status !== "done" && t.status !== "cancelled" && (t.kind !== "question" || t.status === "running"));
  const key = JSON.stringify(tasks.map((t) => [t.id, t.done, t.step, t.status, t.detail, t.stopping, Math.round(t.elapsed / 5)]));
  const running = tasks.filter((t) => t.status === "running");
  $("#activity").hidden = tasks.length === 0;
  $("#activity-count").textContent = running.length ? `${running.length} running` : "all finished";

  const profile = state.health?.profile;
  const others = running.filter((t) => t.local_model && t.kind !== "question");
  const note = $("#busy-note");
  note.hidden = !(profile?.local && others.length);
  if (!note.hidden) {
    note.textContent = `The local model is busy with ${others.map((t) => KIND[t.kind].toLowerCase()).join(" and ")} – a question will wait for it` +
      " (shown under Background work). A cloud preset answers right away.";
  }

  if (key === lastActivity) return;
  lastActivity = key;
  const list = $("#activity-list");
  list.innerHTML = "";
  for (const t of tasks) {
    const pct = t.total ? Math.min(100, Math.round((t.done / t.total) * 100)) : 0;
    const status = t.status === "running" ? (t.stopping ? "stopping after the current step…" : "") : t.status === "done" ? "finished" : t.status === "stopped" ? "stopped (process ended)" : "failed";
    list.append(
      el("div", { class: `task ${t.status}${t.status === "running" && !t.total ? " indeterminate" : ""}` },
        el("div", { class: "task-top" },
          el("span", { class: "task-title", title: t.title }, el("span", { class: "task-kind" }, KIND[t.kind] || t.kind), t.title),
          t.total ? el("span", { class: "task-num" }, `${t.done} / ${t.total} ${t.unit}`) : null,
          t.status === "running" && t.kind !== "question"
            ? el("button", { class: "stop-btn", title: "Stop this task", disabled: t.stopping, onclick: () => stopTask(t) }, t.stopping ? "Stopping" : "Stop")
            : null),
        el("div", { class: "bar thin" }, el("i", { style: `width:${t.status === "done" ? 100 : pct}%` })),
        el("div", { class: "task-step" }, status || t.step || "working…"),
        t.detail ? el("div", { class: "task-detail", title: t.detail }, t.detail) : null,
        el("div", { class: "task-foot" },
          el("span", {}, t.model ? el("span", { class: "model-chip" + (t.local_model ? " local" : "") }, t.model) : ""),
          el("span", {}, t.status === "running" ? `${fmtDur(t.elapsed)} elapsed${t.eta ? ` · ~${fmtDur(t.eta)} left` : ""}` : fmtDur(t.elapsed)))),
    );
  }
  // a finished import elsewhere may have added documents
  if (data.tasks.some((t) => t.status === "done" && (t.kind === "ingest" || t.kind === "upload") && Date.now() / 1000 - (t.finished || 0) < 5)) refreshLibrary();
}
setInterval(pollActivity, 2000);

loadAll().catch((err) => banner(`Cannot reach the server: ${err.message}`, "error"));
pollActivity();
