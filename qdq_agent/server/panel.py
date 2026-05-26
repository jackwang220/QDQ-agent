"""Single-page HITL panel — vanilla HTML + JS, no framework, no build step."""

PANEL_HTML = r"""<!doctype html>
<html lang="zh-TW">
<head>
<meta charset="utf-8">
<title>QDQ Agent Panel</title>
<style>
  :root {
    --fg: #1a1a1a; --bg: #f1f5f9; --muted: #64748b; --border: #e2e8f0;
    --accent: #2563eb; --danger: #dc2626; --ok: #16a34a; --warn: #d97706;
    --surface: #ffffff;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: system-ui, -apple-system, sans-serif; background: var(--bg); color: var(--fg); }

  /* header */
  header { background: var(--surface); border-bottom: 1px solid var(--border);
           padding: 0.65em 1.2em; display: flex; align-items: center; gap: 0.8em; }
  header h1 { font-size: 1em; font-weight: 700; }
  header .sub { font-size: 0.82em; color: var(--muted); }
  header .links { margin-left: auto; }
  header .links a { margin-left: 1em; color: var(--accent); text-decoration: none; font-size: 0.85em; }

  /* layout */
  .layout { display: grid; grid-template-columns: 280px 1fr; gap: 0.9em;
             padding: 0.9em; max-width: 1380px; margin: 0 auto; }

  /* cards */
  .card { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 0.9em; }
  .card h2 { font-size: 0.75em; font-weight: 700; text-transform: uppercase;
             letter-spacing: 0.06em; color: var(--muted); margin-bottom: 0.7em; }

  /* forms */
  label { display: block; font-size: 0.76em; color: var(--muted); margin-top: 0.55em; margin-bottom: 0.18em; }
  input, select { font: inherit; font-size: 0.88em; padding: 0.38em 0.55em;
                  border: 1px solid var(--border); border-radius: 5px; width: 100%;
                  background: var(--surface); }
  input[type=number] { width: auto; }
  input[type=radio]  { width: auto; }
  button { font: inherit; font-size: 0.88em; padding: 0.4em 0.9em;
           border: 1px solid var(--border); background: var(--surface);
           border-radius: 5px; cursor: pointer; }
  button.primary { background: var(--accent); color: #fff; border-color: var(--accent); }
  button.ok      { background: var(--ok);     color: #fff; border-color: var(--ok); }
  button.warn-btn{ background: var(--warn);   color: #fff; border-color: var(--warn); }
  button.muted   { background: #f8fafc; color: var(--muted); }
  button:disabled { opacity: 0.4; cursor: not-allowed; }
  .btn-row { display: flex; gap: 0.5em; margin-top: 0.8em; flex-wrap: wrap; }
  .full { width: 100%; }
  .mt { margin-top: 0.55em; }

  /* stage steps */
  .steps { list-style: none; }
  .steps li { display: flex; align-items: center; gap: 0.5em;
              padding: 0.28em 0; font-size: 0.82em; border-left: 2px solid var(--border);
              padding-left: 0.7em; margin-left: 0.25em; }
  .steps li:last-child { border-left-color: transparent; }
  .step-dot { width: 10px; height: 10px; border-radius: 50%; background: #cbd5e1;
              flex-shrink: 0; margin-left: -1.05em; border: 2px solid var(--surface);
              transition: background 0.2s; }
  .step-dot.done   { background: var(--ok); }
  .step-dot.active { background: var(--accent); box-shadow: 0 0 0 3px #bfdbfe; }
  .step-dot.paused { background: var(--warn); box-shadow: 0 0 0 3px #fde68a; }
  .step-label { flex: 1; color: var(--muted); }
  .step-label.active { color: var(--accent); font-weight: 600; }
  .step-label.paused { color: var(--warn);   font-weight: 600; }
  .step-label.done   { color: var(--fg); }
  .hitl-tag { font-size: 0.68em; background: #fef3c7; color: var(--warn);
              border: 1px solid #fde68a; border-radius: 3px; padding: 0 0.35em; }

  /* badges */
  .badge { display: inline-block; padding: 0.08em 0.45em; border-radius: 3px;
           font-size: 0.72em; font-weight: 600; }
  .badge.ok   { background: #dcfce7; color: var(--ok); }
  .badge.warn { background: #fef3c7; color: var(--warn); }
  .badge.info { background: #dbeafe; color: var(--accent); }

  /* HITL banner */
  .hitl-banner { border: 2px solid var(--warn); border-radius: 7px;
                 background: #fffbeb; padding: 0.85em 1em; margin-bottom: 0.9em; }
  .hitl-banner h3 { font-size: 0.95em; color: #92400e; margin-bottom: 0.5em; }
  .info-grid { display: grid; grid-template-columns: 130px 1fr; gap: 0.25em 0.5em;
               font-size: 0.86em; align-items: baseline; }
  .info-grid .k { color: var(--muted); }

  /* tables */
  table { width: 100%; border-collapse: collapse; font-size: 0.85em; margin-top: 0.5em; }
  th, td { padding: 0.35em 0.5em; text-align: left; border-bottom: 1px solid var(--border); }
  th { color: var(--muted); font-size: 0.8em; font-weight: 600; }
  .fl-input { width: 66px !important; text-align: right; }

  /* code */
  code { font-family: ui-monospace, monospace; font-size: 0.83em;
         background: #f1f5f9; padding: 0.1em 0.3em; border-radius: 3px; }
  pre  { font-family: ui-monospace, monospace; font-size: 0.82em; background: #f1f5f9;
         padding: 0.7em; border-radius: 5px; overflow: auto; white-space: pre-wrap; }

  /* big status */
  .big { text-align: center; padding: 4em 1em; color: var(--muted); }
  .big .icon { font-size: 2.6em; display: block; margin-bottom: 0.3em; }
  .big h2 { font-size: 1.1em; color: var(--fg); margin-bottom: 0.4em; }
  .spinner { display: inline-block; width: 2.4em; height: 2.4em;
             border: 3px solid var(--border); border-top-color: var(--accent);
             border-radius: 50%; animation: spin 0.75s linear infinite; }
  @keyframes spin { to { transform: rotate(360deg); } }

  /* radio group */
  .rg label { display: flex; align-items: center; gap: 0.45em; font-size: 0.88em;
              color: var(--fg); cursor: pointer; margin: 0.32em 0; }

  /* event log */
  .elog { background: #0f172a; color: #cbd5e1; padding: 0.7em 0.85em; border-radius: 6px;
          font-family: ui-monospace, monospace; font-size: 0.78em;
          min-height: 80px; max-height: 170px; overflow-y: auto; }
  .es { color: #93c5fd; } .ei { color: #fbbf24; font-weight:600; }
  .ed { color: #86efac; } .ee { color: #fca5a5; } .em { color: #94a3b8; }

  /* toast */
  .toast { position: fixed; right: 1em; bottom: 1em; padding: 0.7em 1em;
           border-radius: 7px; box-shadow: 0 4px 12px rgba(0,0,0,.12);
           background: var(--surface); max-width: 360px; z-index: 999; font-size: 0.88em; }
  .toast.ok    { background: #dcfce7; border: 1px solid var(--ok); color: #166534; }
  .toast.error { background: #fee2e2; border: 1px solid var(--danger); color: #991b1b; }

  fieldset { border: 1px solid var(--border); border-radius: 5px; padding: 0.7em; margin: 0.5em 0; }
  legend { font-size: 0.76em; color: var(--muted); padding: 0 0.35em; }
  .bottom { max-width: 1380px; margin: 0 auto; padding: 0 0.9em 1.2em; }
</style>
</head>
<body>

<header>
  <h1>QDQ Agent Panel</h1>
  <span class="sub">Human-in-the-Loop 控制介面</span>
  <div class="links">
    <a href="/docs">Swagger</a>
    <a href="/status">JSON</a>
  </div>
</header>

<div class="layout">

  <!-- ── LEFT: config form + stage list ──────────────────────────────────── -->
  <div>

    <!-- Run config form (top, primary action) -->
    <div class="card" style="margin-bottom:0.9em">
      <h2>Run Configuration</h2>

      <label>Config YAML</label>
      <input id="cfg-path" value="pipeline_config.yaml" placeholder="pipeline_config.yaml">

      <label>model.txt 路徑 <span style="color:var(--muted);font-weight:400">（留空 = 由 YAML 決定）</span></label>
      <input id="model-txt" placeholder="auto（由 YAML 決定）">

      <label>Detect layer 覆蓋 <span style="color:var(--muted);font-weight:400">（留空 = 自動推斷）</span></label>
      <input type="number" id="detect-layer" placeholder="e.g. 105" style="width:130px">

      <label>Thread ID</label>
      <input id="thread-id" value="qdq_run_01">

      <div class="mt">
        <button id="btn-start" class="primary full">▶ Start Pipeline</button>
      </div>
    </div>

    <!-- Pipeline progress (compact) -->
    <div class="card">
      <h2>Pipeline Progress</h2>
      <ul class="steps" id="steps"></ul>
    </div>

  </div>

  <!-- ── RIGHT: status / HITL forms ──────────────────────────────────────── -->
  <div class="card" id="right-panel">

    <div id="view-idle" class="big">
      <span class="icon">⚙️</span>
      <h2>Ready</h2>
      <p>Fill in the config on the left and click <strong>Start Pipeline</strong>.</p>
    </div>

    <div id="view-running" style="display:none" class="big">
      <span class="icon"><span class="spinner"></span></span>
      <h2>Running…</h2>
      <p>Stage: <code id="run-stage">—</code></p>
      <p style="font-size:0.82em;margin-top:0.4em" id="run-elapsed"></p>
    </div>

    <div id="view-hitl" style="display:none"></div>

    <div id="view-done" style="display:none" class="big">
      <span class="icon" id="done-icon"></span>
      <h2 id="done-title"></h2>
      <div id="done-body" style="max-width:480px;margin:0 auto;text-align:left"></div>
    </div>

  </div>
</div>

<!-- ── Event log ─────────────────────────────────────────────────────────── -->
<div class="bottom">
  <div class="card">
    <h2 style="margin-bottom:0.5em">
      Event Log
      <button id="btn-clear" style="float:right;font-size:0.8em">clear</button>
    </h2>
    <div class="elog" id="elog"><span class="em">Waiting for events…</span></div>
  </div>
</div>

<script>
// ── Stage groups — ids are current_stage VALUES emitted by pipeline nodes ────
const GROUPS = [
  { label: "Startup",              ids: ["config_loaded","step0_done","step0_skipped","model_txt_loaded","layer_constants_inferred","layer_constants_skipped"] },
  { label: "HITL — Quantizer mode", ids: ["quantizer_mode_confirmed"], hitl: true },
  { label: "Step 1 — Export Excel", ids: ["step1_done","unknown_fix_suggested"] },
  { label: "HITL — Unknown fix",    ids: ["unknown_fix_confirmed","unknown_fix_skipped"], hitl: true },
  { label: "Detect layer",          ids: ["detect_layer_suggested"] },
  { label: "HITL — Detect layer",   ids: ["detect_layer_confirmed"], hitl: true },
  { label: "Steps 2–4 + Q/DQ scan", ids: ["step2_done","step2_skipped","step3_done","step4_done","qdq_coverage_scanned","qdq_coverage_skipped"] },
  { label: "HITL — Q/DQ coverage",  ids: ["qdq_coverage_confirmed"], hitl: true },
  { label: "Step 5 + postprocess",  ids: ["step5_done","postprocess_inferred"] },
  { label: "HITL — Input bias",     ids: ["input_bias_confirmed"], hitl: true },
  { label: "Step 6 — Final ONNX",   ids: ["step6_done"] },
];

// Map interrupt type → the stage id that HITL node will emit after human responds
const HITL_TO_STAGE = {
  "quantizer_mode_review": "quantizer_mode_confirmed",
  "unknown_fix_review":    "unknown_fix_confirmed",
  "detect_layer_review":   "detect_layer_confirmed",
  "qdq_coverage_review":   "qdq_coverage_confirmed",
  "input_bias_review":     "input_bias_confirmed",
};

const MODE_INFO = {
  "DFPQuantizer_TwoScale_negFL":  ["雙 scale，非對稱（正負各一組 FL）","✅ 完整支援"],
  "DFPQuantizer_TwoScale_Symm":   ["雙 scale，對稱（只有正 scale）",   "✅ 完整支援"],
  "DFPQuantizer_negFL":           ["單 scale，非對稱",                 "✅ 完整支援"],
  "DFPQuantizer_negFL_stdFL":     ["單 scale，非對稱，用 std 算 FL",   "✅ 完整支援"],
  "DFPQuantizer":                 ["單 scale，用 max 算 FL",            "✅ 完整支援"],
  "DFPQuantizer_stdFL":           ["單 scale，用 std 算 FL",            "✅ 完整支援"],
  "DFPQuantizer_ThreeScale_Symm": ["三 scale，對稱",                   "⚠️ Step 1 解析可能不完整"],
  "DFPQuantizer_ThreeScale_negFL":["三 scale，非對稱",                 "⚠️ Step 1 解析可能不完整"],
  "FPQuantizer":                  ["固定 FL（bias 用）",                "✅ 完整支援"],
};

// ── State ────────────────────────────────────────────────────────────────────
let status   = "idle";
let doneIds  = new Set();
let activeId = null;
let startTs  = null;
let ticker   = null;

// ── Render stage steps ───────────────────────────────────────────────────────
function renderSteps() {
  const ul = document.getElementById("steps");
  ul.innerHTML = GROUPS.map((g, i) => {
    const isActive  = g.ids.includes(activeId);
    const allDone   = !isActive && g.ids.some(id => doneIds.has(id));
    const isPaused  = isActive && status === "paused";

    let dotCls = "step-dot", lblCls = "step-label";
    if (allDone)      { dotCls += " done";   lblCls += " done"; }
    else if (isPaused){ dotCls += " paused"; lblCls += " paused"; }
    else if (isActive){ dotCls += " active"; lblCls += " active"; }

    const tag = g.hitl ? `<span class="hitl-tag">HITL</span>` : "";
    return `<li><span class="${dotCls}"></span>
              <span class="${lblCls}">${g.label} ${tag}</span></li>`;
  }).join("");
}

// ── Views ────────────────────────────────────────────────────────────────────
function show(name) {
  ["idle","running","hitl","done"].forEach(v =>
    document.getElementById("view-" + v).style.display = v === name ? "" : "none");
}

function setRunning(stage) {
  status = "running";
  if (stage) activeId = stage;
  document.getElementById("run-stage").textContent = stage || "—";
  show("running");
  renderSteps();
  if (!ticker && startTs) {
    ticker = setInterval(() => {
      const s = Math.round((Date.now() - startTs) / 1000);
      document.getElementById("run-elapsed").textContent =
        `Elapsed: ${Math.floor(s/60)}m ${s%60}s`;
    }, 1000);
  }
}

function setDone(success, errors) {
  status = "done"; activeId = null;
  clearInterval(ticker); ticker = null;
  document.getElementById("done-icon").textContent  = success ? "✅" : "❌";
  document.getElementById("done-title").textContent = success ? "Pipeline completed!" : "Pipeline failed";
  document.getElementById("done-body").innerHTML = success
    ? `<p style="color:var(--ok);margin-top:.5em">All steps completed successfully.</p>`
    : `<p style="color:var(--danger);margin-top:.5em">Errors:</p><ul style="margin:.4em 0 0 1.2em">`
      + (errors||[]).map(e=>`<li style="font-size:.88em">${e}</li>`).join("") + "</ul>";
  show("done"); renderSteps();
}

// ── HITL renderers ───────────────────────────────────────────────────────────
function renderHITL(data) {
  status = "paused";
  const c = document.getElementById("view-hitl");
  const t = data.type || "";

  // quantizer_mode_review
  if (t === "quantizer_mode_review") {
    const det = data.detected || "";
    const [desc, sup] = MODE_INFO[det] || ["未知", "❓"];
    const supBadge = sup.startsWith("✅")
      ? `<span class="badge ok">${sup}</span>`
      : `<span class="badge warn">${sup}</span>`;
    const opts = Object.keys(MODE_INFO)
      .map(k => `<option value="${k}"${k===det?" selected":""}>${k}</option>`).join("");
    c.innerHTML = `
      <div class="hitl-banner">
        <h3>⚠️ HITL — Quantizer Mode 確認</h3>
        <div class="info-grid">
          <span class="k">偵測到的 mode</span><code>${det}</code>
          <span class="k">說明</span><span>${desc}</span>
          <span class="k">Pipeline 支援</span><span>${supBadge}</span>
        </div>
      </div>
      <label>選擇 mode（可覆蓋偵測結果）</label>
      <select id="qmode">${opts}</select>
      <div class="btn-row">
        <button class="ok" onclick="respond('')">✅ Accept — 保留偵測值</button>
        <button class="primary" onclick="respond(document.getElementById('qmode').value)">Override — 用選取值</button>
      </div>`;
  }

  // detect_layer_review
  else if (t === "detect_layer_review") {
    const sug = data.suggestion ?? "";
    c.innerHTML = `
      <div class="hitl-banner">
        <h3>⚠️ HITL — Detect Layer 確認</h3>
        <div class="info-grid">
          <span class="k">Agent 建議值</span>
          <code style="font-size:1.25em">${sug}</code>
        </div>
        <p style="font-size:.82em;color:var(--muted);margin-top:.5em">
          IDetect / Detect 層的 layer index。regex 唯一匹配直接採用，否則由 LLM 推測。
        </p>
      </div>
      <label>Detect layer index（修改後點 Override）</label>
      <input type="number" id="dlayer" value="${sug}" style="width:130px">
      <div class="btn-row">
        <button class="ok" onclick="respond('')">✅ Accept (${sug})</button>
        <button class="primary" onclick="respond(document.getElementById('dlayer').value)">Override</button>
      </div>`;
  }

  // unknown_fix_review
  else if (t === "unknown_fix_review") {
    const ss = data.suggestions || [];
    const rows = ss.map((s,i) => `<tr>
      <td>${i+1}</td><td>${s.layer??""}</td>
      <td><code>${s.component??""}</code></td>
      <td><code>${s.role_type??""}</code></td>
      <td><code>${s.suggested_pattern??""}</code></td>
      <td style="color:var(--muted);font-size:.8em">${s.reason??""}</td>
    </tr>`).join("");
    c.innerHTML = `
      <div class="hitl-banner">
        <h3>⚠️ HITL — Unknown Node Fix</h3>
        <p style="font-size:.87em">
          Agent 找到 <strong>${ss.length}</strong> 個 unknown <code>role_type</code>。<br>
          請手動修改 <code>export_model_excel.py</code> 套用建議 pattern，完成後點擊確認。
        </p>
      </div>
      <table>
        <thead><tr><th>#</th><th>Layer</th><th>Component</th><th>Role type</th><th>建議 pattern</th><th>原因</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>
      <div class="btn-row">
        <button class="ok" onclick="respond('')">✅ 已修改 — 重新執行 Step 1</button>
        <button class="muted" onclick="respond('skip')">Skip — 跳過</button>
      </div>`;
  }

  // qdq_coverage_review
  else if (t === "qdq_coverage_review") {
    const ss = data.suggestions || [];
    const rows = ss.map((s,i) => `<tr>
      <td>${i+1}</td>
      <td style="font-size:.79em"><code>${s.name??""}</code></td>
      <td>${s.op_type??""}</td><td>${s.layer??""}</td>
      <td><input type="number" step="0.5" class="fl-input" id="fl-${i}"
           value="${s.fl!==null&&s.fl!==undefined?s.fl:""}" placeholder="?"></td>
      <td style="color:var(--muted);font-size:.79em">${s.reason??""}</td>
    </tr>`).join("");
    c.innerHTML = `
      <div class="hitl-banner">
        <h3>⚠️ HITL — Q/DQ Coverage Review</h3>
        <p style="font-size:.87em">
          <strong>${ss.length}</strong> 個節點缺少 Q/DQ 包覆。<br>
          可編輯 FL 欄位後接受，或全部跳過。FL 留空的節點將被忽略。
        </p>
      </div>
      <table>
        <thead><tr><th>#</th><th>Node name</th><th>Op</th><th>Layer</th><th>FL</th><th>原因</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>
      <div class="btn-row">
        <button class="ok" onclick="submitQDQ(${ss.length})">✅ Accept all</button>
        <button class="muted" onclick="respond('skip')">Skip all</button>
      </div>`;
  }

  // input_bias_review
  else if (t === "input_bias_review") {
    const cur = data.current_value ?? -0.5;
    const isCustom = (cur !== -0.5 && cur !== 0.0);
    c.innerHTML = `
      <div class="hitl-banner">
        <h3>⚠️ HITL — Input Bias 確認</h3>
        <div class="info-grid">
          <span class="k">目前設定值</span>
          <code style="font-size:1.2em">${cur}</code>
        </div>
        <p style="font-size:.82em;color:var(--muted);margin-top:.5em">
          input_bias 在推論前加到 model input（pixel/255 正規化後）。<br>
          訓練時有做 −0.5 平移 → <code>−0.5</code><br>
          訓練時只做 /255，無平移 → <code>0.0</code>
        </p>
      </div>
      <fieldset>
        <legend>選擇 input_bias</legend>
        <div class="rg">
          <label><input type="radio" name="bias" value="-0.5"   ${!isCustom&&cur==-0.5?"checked":""}> <code>−0.5</code> — 訓練時做 /255 再 −0.5 平移</label>
          <label><input type="radio" name="bias" value="0.0"    ${!isCustom&&cur==0.0?"checked":""}> <code>0.0</code> — 訓練時只做 /255，無平移</label>
          <label><input type="radio" name="bias" value="custom" ${isCustom?"checked":""}> 自訂數值</label>
        </div>
        <div id="custom-wrap" style="${isCustom?"":"display:none"};margin-top:.4em">
          <input type="number" step="0.001" id="custom-bias"
                 value="${isCustom?cur:""}" placeholder="e.g. −0.485" style="width:150px">
        </div>
      </fieldset>
      <div class="btn-row">
        <button class="ok" onclick="submitBias()">✅ Confirm</button>
        <button class="muted" onclick="respond('')">Accept (keep ${cur})</button>
      </div>`;
    setTimeout(() => {
      document.querySelectorAll("input[name=bias]").forEach(r =>
        r.addEventListener("change", () =>
          document.getElementById("custom-wrap").style.display =
            r.value === "custom" ? "" : "none"));
    }, 0);
  }

  // generic fallback
  else {
    c.innerHTML = `
      <div class="hitl-banner">
        <h3>⚠️ HITL — ${t}</h3>
        <pre>${data.message || JSON.stringify(data, null, 2)}</pre>
      </div>
      <label>Response（空字串 = Enter）</label>
      <input id="generic-resp" placeholder="留空直接 Submit = 按 Enter">
      <div class="btn-row">
        <button class="ok" onclick="respond(document.getElementById('generic-resp').value)">Submit</button>
        <button class="muted" onclick="respond('')">Accept (Enter)</button>
      </div>`;
  }

  show("hitl"); renderSteps();
}

// ── HITL helpers ─────────────────────────────────────────────────────────────
function submitQDQ(n) {
  const parts = [];
  for (let i = 0; i < n; i++) {
    const el = document.getElementById("fl-" + i);
    parts.push(el ? (el.value.trim() || "-") : "-");
  }
  respond(parts.join(","));
}
function submitBias() {
  const r = document.querySelector("input[name=bias]:checked");
  if (!r) return;
  respond(r.value === "custom"
    ? document.getElementById("custom-bias").value.trim()
    : r.value);
}

// ── API ───────────────────────────────────────────────────────────────────────
function toast(msg, kind = "ok") {
  const el = document.createElement("div");
  el.className = "toast " + kind; el.textContent = msg;
  document.body.appendChild(el);
  setTimeout(() => el.remove(), 4000);
}
async function api(method, path, body) {
  const opts = { method, headers: {} };
  if (body !== undefined) { opts.headers["Content-Type"] = "application/json"; opts.body = JSON.stringify(body); }
  const r = await fetch(path, opts);
  let p; try { p = await r.json(); } catch { p = {}; }
  if (!r.ok) throw new Error(p.detail || JSON.stringify(p));
  return p;
}
async function respond(value) {
  try {
    await api("POST", "/respond", { response: value });
    show("running"); status = "running";
  } catch (e) { toast("Error: " + e.message, "error"); }
}

// ── Event log ─────────────────────────────────────────────────────────────────
let logReady = false;
function logEv(cls, text) {
  const el = document.getElementById("elog");
  if (!logReady) { el.innerHTML = ""; logReady = true; }
  const d = document.createElement("div"); d.className = cls;
  d.textContent = `[${new Date().toLocaleTimeString()}] ${text}`;
  el.appendChild(d); el.scrollTop = el.scrollHeight;
}

// ── WebSocket ─────────────────────────────────────────────────────────────────
function handleMsg(msg) {
  if (msg.type === "ping") return;

  if (msg.type === "status") {
    const s = msg.status;
    if      (s === "running")               setRunning(msg.final_stage);
    else if (s === "paused" && msg.interrupt) renderHITL(msg.interrupt);
    else if (s === "done")  setDone(msg.final_success, msg.final_errors);
    else if (s === "error") setDone(false, [msg.error || "Unknown error"]);
    return;
  }
  if (msg.type === "stage") {
    if (activeId && activeId !== msg.stage) doneIds.add(activeId);
    activeId = msg.stage;
    if (status !== "paused") setRunning(msg.stage);
    logEv("es", `stage: ${msg.stage}  (${msg.elapsed}s)`);
    renderSteps();
  }
  else if (msg.type === "interrupt") {
    logEv("ei", `⏸ HITL: ${msg.data?.type || "?"}`);
    // Advance activeId to the HITL group's stage so progress dot turns amber
    const hitlStage = HITL_TO_STAGE[msg.data?.type];
    if (hitlStage) {
      if (activeId && activeId !== hitlStage) doneIds.add(activeId);
      activeId = hitlStage;
    }
    renderHITL(msg.data);
  }
  else if (msg.type === "resumed") {
    logEv("em", `▶ resumed — "${(msg.response||"").slice(0,40)||"⏎"}"`);
    setRunning(activeId);
  }
  else if (msg.type === "done") {
    logEv(msg.success ? "ed" : "ee",
          msg.success ? "✅ done!" : `❌ failed: ${(msg.errors||[]).join("; ")}`);
    setDone(msg.success, msg.errors);
  }
  else if (msg.type === "error") {
    logEv("ee", `❌ ${msg.error}`);
    setDone(false, [msg.error]);
  }
}

function connect() {
  const p = location.protocol === "https:" ? "wss:" : "ws:";
  const ws = new WebSocket(`${p}//${location.host}/events`);
  ws.onmessage = e => handleMsg(JSON.parse(e.data));
  ws.onclose   = () => { logEv("em","WS closed — reconnecting…"); setTimeout(connect, 3000); };
  ws.onerror   = () => logEv("ee", "WS error");
}

// ── Wiring ────────────────────────────────────────────────────────────────────
document.getElementById("btn-start").addEventListener("click", async () => {
  const body = {
    config_path:   document.getElementById("cfg-path").value.trim(),
    model_txt:     document.getElementById("model-txt").value.trim(),
    thread_id:     document.getElementById("thread-id").value.trim(),
  };
  const dl = document.getElementById("detect-layer").value.trim();
  if (dl) body.detect_layer = parseInt(dl, 10);

  try {
    await api("POST", "/run", body);
    doneIds.clear(); activeId = null; logReady = false;
    document.getElementById("elog").innerHTML = "";
    startTs = Date.now();
    setRunning(null);
    toast("Pipeline started!", "ok");
  } catch(e) { toast(e.message, "error"); }
});

document.getElementById("btn-clear").addEventListener("click", () => {
  document.getElementById("elog").innerHTML = ""; logReady = false;
});

renderSteps();
connect();
</script>
</body>
</html>
"""
