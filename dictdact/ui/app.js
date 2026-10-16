/* DictDact dashboard — vanilla JS, talks to the localhost API with the per-launch token. */
(() => {
  const TOKEN = new URLSearchParams(location.search).get("t") || "";
  const $ = (s) => document.querySelector(s);
  const $$ = (s) => Array.from(document.querySelectorAll(s));
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));

  async function api(path, body) {
    const opts = body ? {method: "POST", headers: {"Content-Type": "application/json", "X-DictDact-Token": TOKEN}, body: JSON.stringify(body)}
                      : {headers: {"X-DictDact-Token": TOKEN}};
    const r = await fetch(path, opts);
    if (!r.ok) { let e = {}; try { e = await r.json(); } catch (_) {} throw new Error(e.error || r.statusText); }
    return r.json();
  }
  let toastT;
  function toast(msg) { const t = $("#toast"); t.textContent = msg; t.classList.remove("hidden"); clearTimeout(toastT); toastT = setTimeout(() => t.classList.add("hidden"), 2200); }

  // ---- tabs -------------------------------------------------------------------------
  let tab = "live";
  let micTesting = false;   // declared before showTab() runs at load with a ?tab= deep link (TDZ bug)
  let vbTick = 0, vbExportSeen = "", vbOn = false;   // same reason: showTab("voicebank") can run at load
  function showTab(name) {
    tab = name;
    $$(".nav").forEach((a) => a.classList.toggle("active", a.dataset.tab === name));
    $$(".tab").forEach((s) => s.classList.toggle("active", s.id === "tab-" + name));
    if (name === "dictations") { loadHistory(); loadStats(); }
    if (name === "mic") loadDevices();
    if (name === "vocab") { loadVocab(); loadSuggestions(); }
    if (name === "settings") loadConfig();
    if (name === "voicebank") loadVoiceBank();
    if (name !== "mic" && micTesting) stopMicTest();
  }
  $$(".nav").forEach((a) => a.addEventListener("click", () => showTab(a.dataset.tab)));
  window.dictdactNavigate = (p) => { const m = /tab=(\w+)/.exec(p || ""); if (m) showTab(m[1]); };
  window.dictdactNavigate(location.search);

  // ---- waveform ---------------------------------------------------------------------
  const wave = $("#wave"), ctx = wave.getContext("2d");
  const NB = 64, bars = new Array(NB).fill(0);
  let waveState = "idle", phase = 0;
  function norm(db) { return Math.pow(Math.min(1, Math.max(0, (db + 62) / 44)), 0.7); }
  function drawWave() {
    const dpr = window.devicePixelRatio || 1, W = wave.clientWidth, H = 140;
    if (wave.width !== W * dpr) { wave.width = W * dpr; wave.height = H * dpr; }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, W, H);
    const gap = 3, bw = Math.max(2, (W - 24 - gap * (NB - 1)) / NB), maxH = H - 24;
    phase += 0.25;
    for (let i = 0; i < NB; i++) {
      let v = bars[i];
      if (waveState === "processing") { const c = (Math.sin(phase * 0.6) + 1) / 2 * (NB - 1); v = Math.max(0.05, Math.exp(-((i - c) ** 2) / 40)) * 0.6; }
      const h = Math.max(2, v * maxH), x = 12 + i * (bw + gap), y = (H - h) / 2;
      const a = waveState === "idle" ? 0.18 : 0.35 + 0.65 * (i + 1) / NB;
      ctx.fillStyle = waveState === "processing" ? `rgba(89,160,255,${0.3 + 0.6 * v})` : waveState === "locked" ? `rgba(245,166,35,${a})` : `rgba(232,233,238,${a})`;
      ctx.beginPath(); ctx.roundRect(x, y, bw, h, 2); ctx.fill();
    }
    requestAnimationFrame(drawWave);
  }
  requestAnimationFrame(drawWave);
  function pushLevel(db) { bars.push(norm(db)); bars.shift(); }

  // ---- words per day ------------------------------------------------------------------
  let chartDays = 30;
  const nf = (n) => Number(n || 0).toLocaleString();
  const dayLabel = (iso) => new Date(iso + "T00:00:00").toLocaleDateString(undefined, {day: "numeric", month: "short"});
  const dur = (sec) => (sec >= 3600 ? (sec / 3600).toFixed(1) + " h" : Math.round(sec / 60) + " min");

  async function loadStats() {
    try { renderChart(await api("/api/stats?days=" + chartDays)); }
    catch (e) { $("#chart-sum").textContent = "Stats unavailable: " + e.message; }
  }
  function renderChart(s) {
    const days = s.days || [], t = s.totals || {}, todayISO = (s.today || {}).date;
    const peak = Math.max(1, ...days.map((d) => d.words));
    $("#chart").innerHTML = days.length
      ? days.map((d) => {
          const h = d.words ? Math.max(3, Math.round((d.words / peak) * 100)) : 0;
          const cls = "bar" + (d.words ? "" : " zero") + (d.date === todayISO ? " today" : "");
          return `<div class="${cls}" data-date="${esc(d.date)}" data-words="${d.words}" data-takes="${d.takes}" data-seconds="${Math.round(d.seconds)}"`
               + ` title="${esc(dayLabel(d.date))} · ${nf(d.words)} words"><i style="height:${h}%"></i></div>`;
        }).join("")
      : `<div class="chart-empty">Nothing dictated yet — hold Ctrl and say something.</div>`;
    $("#chart-sum").textContent = t.first_day
      ? `${nf(t.words)} words in ${t.days_since_first} day${t.days_since_first === 1 ? "" : "s"} since ${dayLabel(t.first_day)}`
        + ` · ${nf(t.words_per_day)} a day · ${dur(t.seconds)} dictating`
        + (t.best ? ` · best ${nf(t.best.words)} on ${dayLabel(t.best.date)}` : "")
      : "Nothing recorded yet.";
    $("#chart").scrollLeft = $("#chart").scrollWidth;      // a long range opens on today, not on last year
    $("#chart-from").textContent = days.length ? dayLabel(days[0].date) : "";
    $("#chart-to").textContent = days.length ? dayLabel(days[days.length - 1].date) : "";
    $("#chart-hover").textContent = days.length ? `peak ${nf(peak)} words` : "";
    hoverDefault = $("#chart-hover").textContent;
  }
  let hoverDefault = "";
  $("#chart").addEventListener("mouseover", (e) => {
    const b = e.target.closest(".bar"); if (!b) return;
    const w = +b.dataset.words, n = +b.dataset.takes;
    $("#chart-hover").textContent = w
      ? `${dayLabel(b.dataset.date)} · ${nf(w)} words · ${n} take${n === 1 ? "" : "s"} · ${dur(+b.dataset.seconds)}`
      : `${dayLabel(b.dataset.date)} · nothing dictated`;
  });
  $("#chart").addEventListener("mouseleave", () => { $("#chart-hover").textContent = hoverDefault; });
  $("#chart-range").addEventListener("click", (e) => {
    const btn = e.target.closest("button"); if (!btn) return;
    chartDays = +btn.dataset.days;
    $$("#chart-range button").forEach((x) => x.classList.toggle("active", x === btn));
    loadStats();
  });

  // ---- state polling ----------------------------------------------------------------
  const LABEL = {idle: "Ready", recording: "Listening…", locked: "Locked — tap to finish", processing: "Transcribing…", done: "Done ✓", error: "Problem"};
  let lastId = null, lastState = null;
  function renderState(s) {
    $("#ver").textContent = "v" + s.version;
    const st = s.ready ? s.state : "idle";
    const badge = $("#state-badge"); badge.className = "state-badge " + st;
    $("#state-text").textContent = s.ready ? (st === "processing" && /Structuring/.test(s.status) ? "Structuring…" : LABEL[st] || st) : "Loading models…";
    $("#status-line").textContent = s.status || "";
    $("#mic-now").textContent = s.mic.name || "system default";
    $("#engine-now").textContent = s.engine; $("#fmt-now").textContent = s.formatter;
    $("#hotkey-hint").innerHTML = `Hold <kbd>${esc(s.hotkey.toUpperCase())}</kbd> and talk · tap-tap to lock · <kbd>Ctrl</kbd>+<kbd>Alt</kbd>+<kbd>V</kbd> re-pastes`;
    const c = s.counts;
    $("#st-today").textContent = c.today; $("#st-words").textContent = nf(c.words_today);
    $("#st-wpd").textContent = nf(c.words_per_day); $("#st-all").textContent = nf(c.words_total);
    $("#st-since").textContent = c.first_day
      ? `${dur(c.seconds_today)} today · ${dur(c.seconds_total)} in ${c.days_since_first} day${c.days_since_first === 1 ? "" : "s"} since ${dayLabel(c.first_day)}`
      : "";
    const p = s.perms;
    $("#perms").innerHTML = [["Mic", p.microphone === "granted" || p.microphone === "unknown"], ["Accessibility", !!p.accessibility]]
      .map(([n, ok]) => `<span class="p"><span class="pdot ${ok ? "ok" : "bad"}"></span>${n}</span>`).join("");
    $("#perm-list").innerHTML = [["Microphone", p.microphone === "granted" || p.microphone === "unknown", p.microphone], ["Accessibility (hotkey + paste)", !!p.accessibility, p.accessibility ? "granted" : "missing"]]
      .map(([n, ok, v]) => `<div class="perm"><span class="pdot ${ok ? "ok" : "bad"}"></span>${n} <span class="muted">· ${esc(v)}</span></div>`).join("");
    if (st !== lastState) { waveState = st === "recording" ? "recording" : st === "locked" ? "locked" : st === "processing" ? "processing" : "idle"; if (waveState === "idle") bars.fill(0); lastState = st; }
    if (s.last && s.last.id !== lastId) { lastId = s.last.id; renderLast(s.last); if (tab === "dictations") { loadHistory(); loadStats(); } }
    $("#paths").textContent = `${s.paths.config} · ${s.paths.history}`;
    const as = $("#autostart"); as.disabled = !s.autostart.available; as.checked = !!s.autostart.enabled; as.parentElement.style.opacity = s.autostart.available ? 1 : 0.5;
  }
  function fmtMeta(b) {
    const pasted = b.pasted ? `<span class="badge">pasted ✓</span>` : `<span class="badge np">not pasted</span>`;
    return `<span>${esc(b.ts.replace("T", " ").slice(0, 16))}</span><span>·</span><span>${esc(b.app || "")}</span><span>·</span><span>${Math.round(b.duration_s)} s</span><span class="badge ${esc(b.mode)}">${esc(b.mode)}</span>${pasted}`;
  }
  function renderLast(b) {
    $("#last-empty").classList.add("hidden"); $("#last").classList.remove("hidden");
    $("#last-meta").innerHTML = fmtMeta(b); $("#last-text").textContent = b.text; $("#last-raw").textContent = b.raw;
    const t = b.timings || {};
    $("#last-timings").textContent = `transcribe ${t.transcribe ?? "?"} s · structure ${t.format ?? "?"} s · ${b.engine} · ${b.formatter}`;
    $("#last .actions").dataset.id = b.id;
  }
  $("#last .actions").addEventListener("click", async (e) => {
    const act = e.target.dataset.act; if (!act) return;
    const id = $("#last .actions").dataset.id;
    if (act === "fix") return startFix(id);
    if (act === "fix-save") return saveFix(id);
    if (act === "fix-cancel") return cancelFix();
    await action(act, id);
  });
  async function action(act, id) {
    try { await api("/api/" + act, {id}); toast(act === "copy" ? "Copied" : "Pasting…"); } catch (e) { toast(e.message); }
  }

  // ---- fix & learn --------------------------------------------------------------------
  function autoGrowFix(ta) { ta.style.height = "auto"; ta.style.height = ta.scrollHeight + "px"; }
  function fixButtons() { return `<button class="primary" data-act="fix-save">Save &amp; learn</button><button data-act="fix-cancel">Cancel</button>`; }
  function lastButtons() { return `<button data-act="copy">Copy</button><button data-act="paste" class="primary">Paste again</button><button data-act="fix">Fix</button>`; }
  function toastLearned(learned) { toast(learned && learned.length ? "Learned: " + learned.map((l) => `${l.heard} → ${l.term}`).join(", ") : "Saved"); }

  function startFix(id) {
    const textEl = $("#last-text"); if (!textEl) return;
    const ta = document.createElement("textarea");
    ta.id = "last-fix-area"; ta.className = "fix-area"; ta.rows = 4;
    ta.value = textEl.textContent; ta.dataset.original = textEl.textContent;
    textEl.replaceWith(ta);
    autoGrowFix(ta);
    ta.addEventListener("input", () => autoGrowFix(ta));
    ta.addEventListener("keydown", (e) => {
      if (e.key === "Escape") { e.preventDefault(); cancelFix(); }
      else if ((e.metaKey || e.ctrlKey) && e.key === "Enter") { e.preventDefault(); saveFix(id); }
    });
    $("#last .actions").innerHTML = fixButtons();
    ta.focus();
  }
  function exitFixUI() {
    const ta = $("#last-fix-area");
    if (ta) { const d = document.createElement("div"); d.className = "text"; d.id = "last-text"; ta.replaceWith(d); }
    $("#last .actions").innerHTML = lastButtons();
  }
  function cancelFix() {
    const ta = $("#last-fix-area"); const original = ta ? ta.dataset.original : "";
    exitFixUI();
    const d = $("#last-text"); if (d) d.textContent = original;
  }
  async function saveFix(id) {
    const ta = $("#last-fix-area"); if (!ta) return;
    try {
      const {learned, block} = await api("/api/fix", {id, text: ta.value});
      exitFixUI();
      renderLast(block);
      await action("copy", block.id || id);
      toastLearned(learned);
    } catch (e) { toast(e.message); }
  }

  async function pollState() { try { renderState(await api("/api/state")); } catch (e) { $("#status-line").textContent = "Backend unreachable: " + e.message; } }
  setInterval(pollState, 1000); pollState();

  // ---- level polling (fast, only when it matters) -----------------------------------------
  async function pollLevel() {
    if (!(tab === "live" || micTesting)) return;
    try {
      const l = await api("/api/level");
      const db = l.source === "none" ? -120 : l.db;
      if (tab === "live") { if (l.recording) pushLevel(db); $("#meter-fill").style.width = (norm(db) * 100) + "%"; $("#meter-db").textContent = l.source === "none" ? "—" : db.toFixed(0) + " dB"; }
      if (micTesting) {
        $("#test-fill").style.width = (norm(db) * 100) + "%"; $("#test-db").textContent = l.source === "none" ? "—" : db.toFixed(0) + " dB";
        const v = $("#test-verdict");
        if (l.source === "none") { if (++noneStreak > 20) { v.textContent = "Test ended."; micTesting = false; $("#mic-test").textContent = "Test selected mic"; } }
        else { noneStreak = 0; v.textContent = db < -70 ? "Silence — say something (right mic? lid closed?)" : db < -40 ? "Whisper level — fine for whispering, low for normal speech (raise the gain?)" : db < -12 ? "Good" : "Hot — lower the gain"; }
      }
    } catch (_) {}
  }
  setInterval(pollLevel, 50);

  // ---- history ------------------------------------------------------------------------
  let q = "", histLimit = 300;
  async function loadHistory() {
    const {items, total} = await api("/api/history?q=" + encodeURIComponent(q) + "&limit=" + histLimit);
    $("#count").textContent = (items.length < total ? `${items.length} of ${total}` : `${total}`) + (q ? " matching" : " dictations");
    const more = items.length < total ? `<div class="actions"><button class="ghost" id="show-all">Show all ${total}</button></div>` : "";
    $("#list").innerHTML = items.map((b) => `<div class="item" data-id="${esc(b.id)}"><div class="meta">${fmtMeta(b)}</div><div class="text">${esc(b.text)}</div>${b.raw && b.raw !== b.text ? `<details><summary>Raw</summary><div class="raw">${esc(b.raw)}</div></details>` : ""}<div class="actions">${itemButtons()}</div></div>`).join("")
      || `<div class="muted">Nothing here yet — hold Ctrl and talk.</div>`;
    $("#list").insertAdjacentHTML("beforeend", more);
    const sa = $("#show-all"); if (sa) sa.addEventListener("click", () => { histLimit = total; loadHistory(); });
  }
  function itemButtons() { return `<button data-act="copy">Copy</button><button data-act="paste" class="primary">Paste</button><button data-act="fix">Fix</button>`; }
  $("#list").addEventListener("click", async (e) => {
    const act = e.target.dataset.act; if (!act) return;
    const item = e.target.closest(".item"); const id = item.dataset.id;
    if (act === "fix") return startFixItem(item, id);
    if (act === "fix-save") return saveFixItem(item, id);
    if (act === "fix-cancel") return cancelFixItem(item);
    action(act, id);
  });
  function startFixItem(item, id) {
    const textEl = item.querySelector(".text"); if (!textEl) return;
    const ta = document.createElement("textarea");
    ta.className = "fix-area"; ta.rows = 4;
    ta.value = textEl.textContent; ta.dataset.original = textEl.textContent;
    textEl.replaceWith(ta);
    autoGrowFix(ta);
    ta.addEventListener("input", () => autoGrowFix(ta));
    ta.addEventListener("keydown", (e) => {
      if (e.key === "Escape") { e.preventDefault(); cancelFixItem(item); }
      else if ((e.metaKey || e.ctrlKey) && e.key === "Enter") { e.preventDefault(); saveFixItem(item, id); }
    });
    item.querySelector(".actions").innerHTML = fixButtons();
    ta.focus();
  }
  function cancelFixItem(item) {
    const ta = item.querySelector(".fix-area"); const original = ta ? ta.dataset.original : "";
    if (ta) { const d = document.createElement("div"); d.className = "text"; d.textContent = original; ta.replaceWith(d); }
    item.querySelector(".actions").innerHTML = itemButtons();
  }
  async function saveFixItem(item, id) {
    const ta = item.querySelector(".fix-area"); if (!ta) return;
    try {
      const {learned, block} = await api("/api/fix", {id, text: ta.value});
      await loadHistory();
      if (block.id === lastId) renderLast(block);
      await action("copy", block.id || id);
      toastLearned(learned);
    } catch (e) { toast(e.message); }
  }
  $("#search").addEventListener("input", (e) => { q = e.target.value; loadHistory(); });
  $("#open-history").addEventListener("click", () => api("/api/reveal", {what: "history"}));
  $("#open-history-2").addEventListener("click", () => api("/api/reveal", {what: "history"}));
  $("#open-config").addEventListener("click", () => api("/api/reveal", {what: "config"}));

  // ---- microphone ---------------------------------------------------------------------
  async function loadDevices(refresh) {
    const devs = await api("/api/devices" + (refresh ? "?refresh=1" : ""));
    $("#devices").innerHTML = devs.map((d) => `<div class="dev ${d.selected ? "selected" : ""}" data-name="${esc(d.name)}"><span class="radio"></span><span class="dname">${esc(d.name)}</span><span class="dsub">${d.channels} ch · ${Math.round(d.samplerate)} Hz${d.default ? " · system default" : ""}</span></div>`).join("");
  }
  $("#refresh-devices").addEventListener("click", async () => { try { await loadDevices(true); toast("Device list refreshed"); } catch (e) { toast(e.message); } });
  $("#devices").addEventListener("click", async (e) => {
    const d = e.target.closest(".dev"); if (!d) return;
    try { await api("/api/config", {microphone: d.dataset.name}); toast("Microphone: " + d.dataset.name); loadDevices(); if (micTesting) api("/api/monitor", {on: true}); } catch (err) { toast(err.message); }
  });
  let noneStreak = 0;
  async function startMicTest() { $("#mic-test").textContent = "Stop test"; $("#test-verdict").textContent = "Talk…"; try { await api("/api/monitor", {on: true, seconds: 120}); noneStreak = 0; micTesting = true; } catch (e) { toast(e.message); $("#mic-test").textContent = "Test selected mic"; } }
  async function stopMicTest() { micTesting = false; $("#mic-test").textContent = "Test selected mic"; $("#test-fill").style.width = "0"; $("#test-db").textContent = "—"; try { await api("/api/monitor", {on: false}); } catch (_) {} }
  $("#mic-test").addEventListener("click", () => (micTesting ? stopMicTest() : startMicTest()));

  // ---- vocabulary ---------------------------------------------------------------------
  let vocabTerms = [], vocabRevision = null;   // revision: the server refuses a save based on a stale list
  async function loadVocab() {
    try { const {terms, count, revision} = await api("/api/vocab"); vocabTerms = terms || []; vocabRevision = revision; renderVocab(count); }
    catch (e) { toast(e.message); }
  }
  function renderVocab(count) {
    const n = count ?? vocabTerms.length;
    $("#vocab-count").textContent = n + (n === 1 ? " term" : " terms");
    $("#vocab-list").innerHTML = vocabTerms.map((t, i) => `
      <div class="vocab-item" data-idx="${i}">
        <div class="vocab-row"><span class="term">${esc(t.term)}${t.fuzzy ? ' <span class="chip" title="sound-alike spellings are caught too">≈</span>' : ""}</span><button class="ghost" data-act="remove">Remove</button></div>
        ${t.spoken && t.spoken.length ? `<div class="chips">${t.spoken.map((s) => `<span class="chip">${esc(s)}</span>`).join("")}</div>` : ""}
        ${t.note ? `<div class="note">${esc(t.note)}</div>` : ""}
      </div>`).join("") || `<div class="muted">No terms yet — add one above.</div>`;
  }
  async function saveVocab() {
    try {
      const {terms, count, revision} = await api("/api/vocab", {terms: vocabTerms, revision: vocabRevision});
      vocabTerms = terms || vocabTerms; vocabRevision = revision;
      renderVocab(count);
    } catch (e) {
      await loadVocab();          // the dictionary changed elsewhere (CLI / file): show the real list
      throw e;
    }
  }
  function splitSpoken(s) { return s.split(",").map((x) => x.trim()).filter(Boolean); }
  async function addVocabTerm() {
    const termEl = $("#vocab-term"), spokenEl = $("#vocab-spoken"), noteEl = $("#vocab-note");
    const term = termEl.value.trim();
    if (!term) { termEl.focus(); return; }
    const spoken = splitSpoken(spokenEl.value), note = noteEl.value.trim();
    const fuzzy = !!($("#vocab-fuzzy") && $("#vocab-fuzzy").checked);
    const existing = vocabTerms.find((t) => t.term.toLowerCase() === term.toLowerCase());
    if (existing) { existing.spoken = Array.from(new Set([...(existing.spoken || []), ...spoken])); if (note) existing.note = note; }   // the flag of an existing term is not changed here
    else vocabTerms.push({term, spoken, note, fuzzy});
    try { await saveVocab(); termEl.value = ""; spokenEl.value = ""; noteEl.value = ""; toast("Added " + term); termEl.focus(); }
    catch (e) { toast(e.message); loadVocab(); }
  }
  $("#vocab-add-btn").addEventListener("click", addVocabTerm);
  ["vocab-term", "vocab-spoken", "vocab-note"].forEach((id) => $("#" + id).addEventListener("keydown", (e) => { if (e.key === "Enter") addVocabTerm(); }));
  $("#vocab-list").addEventListener("click", async (e) => {
    const act = e.target.dataset.act; if (!act) return;
    const item = e.target.closest(".vocab-item"); const idx = Number(item.dataset.idx);
    if (act === "remove") {
      const removed = vocabTerms[idx]; vocabTerms.splice(idx, 1);
      try { await saveVocab(); toast("Removed " + (removed ? removed.term : "")); } catch (e2) { toast(e2.message); loadVocab(); }
    }
  });
  $("#open-vocab").addEventListener("click", () => api("/api/reveal", {what: "vocab"}));

  // ---- vocabulary suggestions ("suggested from your dictations") ----------------------
  let vocabSuggestions = [];
  async function loadSuggestions() {
    try {
      const {suggestions, takes} = await api("/api/vocab/suggest?limit=30");
      vocabSuggestions = suggestions || [];
      renderSuggestions(takes);
    } catch (e) { toast(e.message); }
  }
  function renderSuggestions(takes) {
    $("#vocab-suggest-meta").textContent = "from " + (takes ?? 0) + " takes";
    $("#vocab-suggest-list").innerHTML = vocabSuggestions.map((s, i) => {
      const entries = Object.entries(s.spellings || {}).sort((a, b) => b[1] - a[1]);
      const chips = entries.map(([sp, c]) => `<span class="chip">${esc(sp)} ×${c}</span>`).join("");
      return `<div class="suggest-item" data-idx="${i}">
        <div class="suggest-row">
          <span class="term">${esc(s.term)}</span>
          <div class="suggest-actions"><button class="ghost" data-act="use">Use…</button><button data-act="addasis">Add as is</button></div>
        </div>
        <div class="chips">${chips}</div>
      </div>`;
    }).join("") || `<div class="muted">Nothing to suggest yet.</div>`;
  }
  $("#vocab-suggest-list").addEventListener("click", async (e) => {
    const act = e.target.dataset.act; if (!act) return;
    const item = e.target.closest(".suggest-item"); if (!item) return;
    const s = vocabSuggestions[Number(item.dataset.idx)]; if (!s) return;
    $("#vocab-term").value = s.term;
    $("#vocab-spoken").value = (s.spoken || []).join(", ");
    if (act === "use") { $("#vocab-note").value = ""; $("#vocab-term").focus(); $("#vocab-term").select(); }
    else if (act === "addasis") { await addVocabTerm(); loadSuggestions(); }
  });

  // ---- voice bank -----------------------------------------------------------------------
  const LANG_FLAG = {es: "ES", en: "EN"};
  function fmtMin(m) { return m < 1 ? Math.round(m * 60) + " s" : m.toFixed(1) + " min"; }
  function snrClass(x) { return x === null || x === undefined ? "none" : x >= 28 ? "green" : x >= 20 ? "amber" : "red"; }
  function ring(frac) {
    const r = 56, c = 2 * Math.PI * r, off = c * (1 - Math.max(0, Math.min(1, frac)));
    return `<svg viewBox="0 0 132 132"><circle class="bg" cx="66" cy="66" r="${r}" fill="none" stroke-width="12"/>` +
      `<circle class="fg" cx="66" cy="66" r="${r}" fill="none" stroke-width="12" stroke-linecap="round" stroke-dasharray="${c.toFixed(1)}" stroke-dashoffset="${off.toFixed(1)}"/></svg>`;
  }
  function renderVoiceBank(v) {
    vbOn = !!v.on;
    const badge = $("#vb-badge");
    badge.className = "state-badge " + (v.on ? "rec" : "idle");
    $("#vb-badge-text").textContent = v.on ? "Recording your voice" : "Off — not recording";
    $("#vb-toggle").textContent = v.on ? "Turn off" : "Turn on";
    $("#vb-toggle").className = v.on ? "" : "primary";
    $("#vb-nav-dot").classList.toggle("hidden", !v.on);
    const job = v.export || {};
    $("#vb-langs").innerHTML = ["es", "en"].map((k) => {
      const L = v.langs[k];
      const frac = L.goal_min ? L.usable_min / L.goal_min : 0;
      const exporting = job.running && job.lang === k;
      const reasons = Object.entries(L.reasons || {}).map(([r, n]) => `${esc(r)} ×${n}`).join(", ");
      const pace = L.ready ? `<div class="vb-ready">✓ Ready for ElevenLabs${L.full ? " — full, collection stopped" : " — keep going toward 3 h for the best clone"}</div>`
        : L.eta_days !== null && L.eta_days !== undefined ? `<div class="vb-facts">At your pace (~${L.usable_min_per_day} usable min/day): 30 min in about <b>${L.eta_days < 1 ? "less than a day" : L.eta_days + " days"}</b></div>`
        : L.takes && !L.recent_usable_share ? `<div class="vb-facts">No usable takes lately — see Room quality below</div>` : "";
      const btn = L.usable_min > 0
        ? `<button class="primary" data-vb-export="${k}" ${exporting || job.running ? "disabled" : ""}>${exporting ? "Exporting…" : "Export for ElevenLabs"}</button>`
        : `<button disabled>Export — needs usable minutes</button>`;
      return `<div class="vb-lang ${L.ready ? "ready" : ""}">
        <div class="ring">${ring(frac)}<div class="mid"><b>${L.usable_min.toFixed(1)}</b><span>of ${L.goal_min} min usable</span></div></div>
        <div class="vb-lang-info">
          <div class="vb-lang-name"><span class="flag">${LANG_FLAG[k]}</span>${esc(L.name)}</div>
          <div class="vb-facts"><b>${fmtMin(L.recorded_min)}</b> recorded · <b>${L.usable_takes}</b> of <b>${L.takes}</b> takes usable${reasons ? `<br>Not usable: ${reasons}` : ""}</div>
          ${pace}${btn}
        </div></div>`;
    }).join("");
    const room = v.room;
    if (room) {
      const x = room.median_snr, pct = Math.max(0, Math.min(100, x / 40 * 100));
      $("#vb-marker").style.left = pct + "%";
      $("#vb-marker-label").textContent = x.toFixed(0) + " dB";
      const head = room.verdict === "good" ? `✓ Good — your voice is ${x.toFixed(0)} dB above the room.`
        : room.verdict === "usable" ? `Usable — your voice is ${x.toFixed(0)} dB above the room. 28+ is ideal.`
        : `Too noisy — your voice is only ${x.toFixed(0)} dB above the room (needs 20+).`;
      const tip = room.verdict === "good" ? "Keep this distance and this room."
        : "Speak 10–15 cm from the mic and switch off the AC / fans while you dictate. The gain knob won't help: it raises the noise too.";
      $("#vb-room-text").innerHTML = `${head}<div class="tip">Median of your last 10 takes · last take ${room.last_snr.toFixed(0)} dB. ${tip}</div>`;
    } else {
      $("#vb-marker").style.left = "0%"; $("#vb-marker-label").textContent = "";
      $("#vb-room-text").textContent = "No takes yet — hold Ctrl and talk.";
    }
    $("#vb-takes").innerHTML = v.takes.length ? v.takes.map((t) => {
      const time = (t.ts || "").slice(11, 16);
      const snr = t.snr_db === null || t.snr_db === undefined ? "—" : t.snr_db.toFixed(0) + " dB";
      return `<div class="vb-take ${t.usable ? "ok" : ""}">
        <span class="t">${esc(time)}</span><span class="flag">${esc(LANG_FLAG[t.lang] || t.lang || "?")}</span>
        <span class="s">${Math.round(t.seconds || 0)} s</span><span class="snr ${snrClass(t.snr_db)}">${snr}</span>
        <span class="words">${t.text ? "“" + esc(t.text) + "”" : `<span class="muted">${esc(t.status)}</span>`}</span>
        ${t.text ? `<span class="why">${t.usable ? "✓ usable" : esc(t.status)}</span>` : ""}
      </div>`;
    }).join("") : `<div class="muted">No takes yet.</div>`;
    // sidebar mini progress
    $("#side-vb-state").innerHTML = v.on ? `<span class="rec">●</span> Voice bank recording` : "Voice bank off";
    for (const k of ["es", "en"]) {
      const L = v.langs[k];
      $("#side-vb-" + k).style.width = Math.min(100, L.usable_min / L.goal_min * 100) + "%";
      $("#side-vb-" + k + "-n").textContent = `${L.usable_min.toFixed(1)}/${L.goal_min}`;
    }
    // one toast per export job (each run has its own id, so a repeated failure is announced again)
    if (!job.running && job.id && job.id !== vbExportSeen && (job.result || job.error)) {
      vbExportSeen = job.id;
      toast(job.result ? `Exported ${job.result.minutes} min — folder opened` : "Export failed: " + job.error);
    }
  }
  async function loadVoiceBank() { try { renderVoiceBank(await api("/api/voicebank")); } catch (e) { /* app restarting */ } }
  $("#vb-langs").addEventListener("click", async (e) => {
    const b = e.target.closest("[data-vb-export]"); if (!b) return;
    try { await api("/api/voicebank/export", {lang: b.dataset.vbExport}); toast("Exporting… the folder opens when it is ready"); loadVoiceBank(); }
    catch (err) { toast(err.message); }
  });
  $("#vb-toggle").addEventListener("click", async () => {
    const want = !vbOn;          // captured: a poll may land while the request is in flight
    try { await api("/api/config", {voice_bank: want}); toast(want ? "Voice bank on" : "Voice bank off"); loadVoiceBank(); } catch (e) { toast(e.message); }
  });
  $("#vb-folder").addEventListener("click", () => api("/api/reveal", {what: "voicebank"}));
  $("#side-vb").addEventListener("click", () => showTab("voicebank"));
  setInterval(() => { vbTick++; if (tab === "voicebank" || vbTick % 5 === 0) loadVoiceBank(); }, 3000);
  loadVoiceBank();

  // ---- settings -------------------------------------------------------------------------
  async function loadConfig() {
    const c = await api("/api/config");
    $$("[data-key]").forEach((el) => { const v = c[el.dataset.key]; if (el.type === "checkbox") el.checked = !!v; else el.value = v === null || v === undefined ? "" : v; });
  }
  $$("[data-key]").forEach((el) => el.addEventListener("change", async () => {
    const key = el.dataset.key; let v = el.type === "checkbox" ? el.checked : el.value;
    if (key === "language" && v === "") v = null;
    if (key === "max_gain_db") v = Number(v);
    try { await api("/api/config", {[key]: v}); toast("Saved"); } catch (e) { toast(e.message); loadConfig(); }
  }));
  $("#autostart").addEventListener("change", async (e) => { try { const r = await api("/api/autostart", {on: e.target.checked}); toast(r.ok ? "Saved" : "Could not change login item"); } catch (err) { toast(err.message); } });
  $$("[data-open]").forEach((b) => b.addEventListener("click", () => api("/api/open-settings", {pane: b.dataset.open})));
  document.addEventListener("visibilitychange", () => { if (document.visibilityState === "hidden" && micTesting) stopMicTest(); });
})();
