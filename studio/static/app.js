/* Presentation and request state only. All scientific data comes from Python. */
"use strict";
const $ = (id) => document.getElementById(id);
const state = {catalogue: null, runs: [], selected: new Set(), viewing: null, historyKey: ""};
function el(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}
function notice(message) { $("notice").textContent = message; }
async function api(path, body) {
  const response = await fetch(path, body === undefined ? {} : {
    method: "POST", headers: {"Content-Type": "application/json", "X-QSVT-Studio": "1"}, body: JSON.stringify(body)
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
  return data;
}
function guard(fn) { return async (...args) => { try { await fn(...args); } catch (error) { notice(error.message); } }; }
function option(select, value, label) { const node = el("option", label); node.value = value; select.append(node); }
function button(text, action) { const node = el("button", text); node.type = "button"; node.addEventListener("click", guard(action)); return node; }
function format(value) {
  if (typeof value === "number") return Number.isInteger(value) ? String(value) : value.toPrecision(5);
  return typeof value === "object" ? JSON.stringify(value) : String(value);
}
function workflow(id) { return state.catalogue.workflows[id]; }
function renderComposer(request) {
  const entry = workflow(request ? request.workflow : $("workflow").value);
  $("workflow").value = entry.id;
  $("description").textContent = entry.description;
  $("controls").replaceChildren();
  const groups = new Map();
  const advanced = el("details"); advanced.id = "advanced-settings";
  advanced.append(el("summary", "Advanced settings"));
  for (const [key, spec] of Object.entries(entry.settings)) {
    const groupKey = `${spec.advanced ? "advanced" : "basic"}:${spec.group}`;
    if (!groups.has(groupKey)) {
      const fieldset = el("fieldset"); fieldset.append(el("legend", spec.group));
      groups.set(groupKey, fieldset); (spec.advanced ? advanced : $("controls")).append(fieldset);
    }
    const value = request && Object.hasOwn(request.settings, key) ? request.settings[key] : spec.default;
    const label = el("label", spec.label);
    let input;
    if (spec.fixed) {
      input = el("input"); input.type = "hidden"; input.value = JSON.stringify(value);
      label.append(el("output", String(value)));
    } else if (spec.control === "select") {
      input = el("select"); spec.values.forEach((v,i) => option(input, JSON.stringify(v), spec.value_labels?.[i] || String(v))); input.value = JSON.stringify(value);
    } else {
      input = el("input"); input.type = spec.control === "boolean" ? "checkbox" : "number";
      if (input.type === "checkbox") input.checked = value;
      else { input.value = value; input.min = spec.min; input.max = spec.max; input.step = spec.step || (spec.integer ? "1" : "any"); input.required = true; }
    }
    input.id = `setting-${key}`; input.dataset.key = key;
    input.title = `API argument: ${key}. Default from ${spec.default_source}.`;
    label.append(input); groups.get(groupKey).append(label);
    if (spec.help) {
      const help = el("p", spec.help, "field-help"); help.id = `help-${key}`;
      input.setAttribute("aria-describedby", help.id); groups.get(groupKey).append(help);
    }
  }
  $("controls").append(advanced);
}
function loadPreset(id) {
  const preset = state.catalogue.presets.find(p => p.id === id);
  renderComposer({schema_version: state.catalogue.schema_version, workflow: preset.workflow, settings: preset.settings});
  $("preset").value = preset.id;
  $("preset-source").textContent = `${preset.purpose}${preset.recommended ? " · Recommended starting point" : ""}. ${preset.description} Source: ${preset.source} · revision ${preset.revision}`;
}
function executionRequested(request) {
  return Boolean(request.settings[workflow(request.workflow).execution_setting]);
}
function executionLabel(request, entry) {
  if (!entry.circuit_execution) return "Polynomial design · no QNode";
  if (!executionRequested(request)) return entry.no_execution_label;
  return request.settings.shots == null ? "Analytic QNode requested" : `Finite QNode requested · ${request.settings.shots} shots`;
}
function draft() {
  const entry = workflow($("workflow").value), settings = {};
  for (const [key, spec] of Object.entries(entry.settings)) {
    const input = $(`setting-${key}`);
    settings[key] = spec.control === "boolean" ? input.checked : spec.control === "select" ? JSON.parse(input.value) : Number(input.value);
  }
  return {schema_version: state.catalogue.schema_version, workflow: entry.id, settings};
}
function download(name, value) {
  const url = URL.createObjectURL(new Blob([JSON.stringify(value, null, 2)], {type: "application/json"}));
  const link = el("a"); link.href = url; link.download = name; link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
}
async function reuse(id) {
  const saved = await api(`/api/runs/${id}/reuse`);
  const request = await api("/api/validate", saved);
  renderComposer(request); $("preset").value = ""; $("preset-source").textContent = `Restored exact saved configuration · ${id.slice(0, 8)}`;
  $("viewer").close(); $("workflow").focus(); window.scrollTo({top: 0, behavior: "smooth"});
  notice(`Saved configuration restored${saved.schema_version !== request.schema_version ? "; schema 1.0 upgraded to 1.1 with its original solver defaults" : ""}. Change a setting and run a new experiment.`);
}
function selection() {
  $("selection-count").textContent = state.selected.size ? `${state.selected.size} selected · match target/problem settings` : "Select compatible runs to compare";
  $("compare").disabled = state.selected.size < 2;
}
function toggleSelection(id) {
  if (state.selected.has(id)) state.selected.delete(id); else state.selected.add(id);
  renderGallery(); selection();
}
function renderGallery() {
  const search = $("search").value.toLowerCase(), filter = $("status-filter").value;
  let runs = state.runs.filter(r => {
    const s = r.request.settings;
    return (!search || JSON.stringify(r).toLowerCase().includes(search)) &&
      (!$("workflow-filter").value || r.request.workflow === $("workflow-filter").value) &&
      (!filter || (filter === "active" ? !["completed", "failed", "cancelled"].includes(r.status) : r.status === filter)) &&
      (!$("encoding-filter").value || (s.access_model || s.block_encoding) === $("encoding-filter").value) &&
      (!$("execution-filter").value || String(executionRequested(r.request)) === $("execution-filter").value) &&
      (!$("favorites").checked || r.favorite);
  });
  if ($("sort").value === "oldest") runs = runs.toReversed ? runs.toReversed() : [...runs].reverse();
  $("count").textContent = `${runs.length} visible / ${state.runs.length} saved`;
  $("gallery").replaceChildren();
  if (!runs.length) {
    const empty = el("div", undefined, "empty"); empty.append(el("h2", state.runs.length ? "No matching experiments" : "Your next experiment starts here"), el("p", "Choose a workflow or a published example, then run it to build your experiment gallery."));
    $("gallery").append(empty);
  }
  for (const run of runs) {
    const entry = workflow(run.request.workflow), settings = run.request.settings;
    const card = el("article", undefined, `card${state.selected.has(run.id) ? " selected" : ""}`);
    const top = el("div", undefined, "card-top"); top.append(el("h3", entry.name), el("span", run.status.replaceAll("_", " "), `status ${run.status}`)); card.append(top);
    if (run.artifacts.includes("preview.png")) {
      const image = el("img", undefined, "preview"); image.src = `/api/runs/${run.id}/preview.png`; image.alt = `${entry.name}: package diagnostics`; image.loading = "lazy"; card.append(image);
    } else card.append(el("div", run.error ? `${run.error.type}: ${run.error.message}` : "Scientific preview appears after execution", `pending${run.error ? " error" : ""}`));
    const body = el("div", undefined, "card-body");
    const config = Object.entries(settings).filter(([k]) => ["degree", "tolerance", "access_model", "block_encoding", "min_degree", "max_degree", "time", "acceptance_tolerance"].includes(k)).map(([k,v]) => `${k} = ${format(v)}`).join(" · ");
    body.append(el("div", config, "meta"), el("div", executionLabel(run.request, entry), "meta"));
    if (run.progress?.message && !["completed", "failed", "cancelled"].includes(run.status)) body.append(el("div", run.progress.message, "progress meta"));
    const keys = ["synthesis_quality.status", "synthesis.angle_solver", "component_synthesis_quality.cosine.status", "component_synthesis_quality.sine.status", "diagnostics.max_error", "degree_search.achieved_error", "synthesis.reconstruction_max_error", "state_relative_error", "operator_relative_error", "acceptance.status"];
    keys.filter(k => run.metrics?.[k] !== undefined).forEach(k => body.append(el("div", `${k}: ${format(run.metrics[k])}`, "meta")));
    if (run.package_call_seconds !== undefined) body.append(el("div", `Package wall time: ${format(run.package_call_seconds)} s`, "meta"));
    body.append(el("div", `${new Date(run.created_at).toLocaleString()} · ${run.id.slice(0,8)}`, "meta"));
    const actions = el("div", undefined, "actions");
    actions.append(button("View", () => showRun(run.id)), button("Reuse", () => reuse(run.id)));
    const favorite = button(run.favorite ? "★ Saved" : "☆ Save", async () => { await api(`/api/runs/${run.id}/favorite`, {favorite: !run.favorite}); await refresh(); }); favorite.setAttribute("aria-pressed", run.favorite); actions.append(favorite);
    if (run.status === "completed") { const compare = button(state.selected.has(run.id) ? "✓ Selected" : "Compare", () => toggleSelection(run.id)); compare.setAttribute("aria-pressed", state.selected.has(run.id)); actions.append(compare); }
    if (["configured", "validating", "executing"].includes(run.status)) actions.append(button("Cancel", async () => { await api(`/api/runs/${run.id}/cancel`, {}); notice(`Cancellation requested for ${run.id.slice(0,8)}.`); await refresh(); }));
    body.append(actions); card.append(body); $("gallery").append(card);
  }
}
function detail(title, value, open=false) {
  const node = el("details"); node.open = open;
  node.append(el("summary", title), el("pre", JSON.stringify(value, null, 2))); return node;
}
function qualityPanel(title, quality) {
  const labels = {passed: "Reconstruction passed", solver_failed: "Solver failed", reconstruction_failed: "Reconstruction failed", reconstruction_unavailable: "Reconstruction unavailable"};
  const node = el("div", undefined, quality.reconstruction_passed ? "acceptance" : "acceptance error");
  node.append(el("strong", `${title}: ${labels[quality.status] || quality.status}`));
  node.append(el("p", `Solver: ${quality.angle_solver}. Returned finite phases: ${quality.solver_returned_phases}. Maximum sampled error: ${format(quality.reconstruction_max_error)}; tolerance: ${format(quality.tolerance)}.`));
  if (quality.error) node.append(el("p", `${quality.error_type}: ${quality.error}`));
  node.append(el("p", quality.interpretation, "muted")); return node;
}
function metricTable(values) {
  const table = el("table");
  Object.entries(values).forEach(([key,value]) => { const row = el("tr"); row.append(el("th", key), el("td", format(value))); table.append(row); });
  return table;
}
async function showRun(id) {
  const run = await api(`/api/runs/${id}`); state.viewing = id;
  $("viewer-title").textContent = `${workflow(run.request.workflow).name} · ${id.slice(0,8)}`;
  const body = $("viewer-body"); body.replaceChildren();
  const actions = el("div", undefined, "actions");
  actions.append(button("Reuse configuration", () => reuse(id)), button("Export configuration", () => download(`${id}-request.json`, run.request)));
  if (run.report) actions.append(button("Export results", () => download(`${id}-report.json`, run.report)));
  if (run.status === "completed") actions.append(button("Select for comparison", () => { toggleSelection(id); notice("Run selection updated."); }));
  body.append(actions, el("p", `Lifecycle: ${run.status}. Completion and acceptance are separate.`, "muted"));
  if (run.progress?.message) body.append(el("p", run.progress.message, "progress"));
  if (run.error) body.append(el("p", `${run.error.type}: ${run.error.message}`, "error"));
  (run.warnings || []).forEach(w => body.append(el("p", w, "error")));
  if (run.report?.acceptance) {
    const a = run.report.acceptance;
    body.append(el("div", `${a.status} · scope: ${a.scope} · full_qsvt_acceptance: ${a.full_qsvt_acceptance}. See all required checks and omitted components below.`, "acceptance"));
  }
  if (run.report?.synthesis_quality) body.append(qualityPanel("Phase synthesis", run.report.synthesis_quality));
  for (const [name, quality] of Object.entries(run.report?.component_synthesis_quality || {})) body.append(qualityPanel(`${name} phase synthesis`, quality));
  if (run.report?.synthesis && !run.report.synthesis_quality) body.append(el("p", "Historical synthesis report: no reconstruction-quality assessment was saved. Inspect its error and original acceptance checks.", "muted"));
  if (run.artifacts.includes("preview.png")) {
    const plot = el("img", undefined, "large-plot"); plot.src = `/api/runs/${id}/preview.png`; plot.alt = "Scientific diagnostics from the saved package report"; body.append(plot);
  }
  for (const [filename, title] of [["phases.png", "Synthesized phases"], ["spectrum.png", "Spectral response"], ["resources.png", "Resource report"]]) {
    if (!run.artifacts.includes(filename)) continue;
    body.append(el("h3", title));
    const plot = el("img", undefined, "large-plot"); plot.src = `/api/runs/${id}/${filename}`; plot.alt = `${title} from the saved package report`; body.append(plot);
  }
  if (run.metrics) body.append(metricTable(run.metrics));
  body.append(detail("Problem / resolved configuration", run.request, true));
  const report = run.report || {};
  for (const key of ["diagnostics", "coeffs", "compatibility", "degree_search", "synthesis", "execution", "resources", "resource_report", "acceptance", "truth_contract", "error_budget", "block_encoding_spec", "observable_values", "physical_observables", "cos_coeffs", "sin_coeffs", "scaled_operator", "qsvt_execution", "component_error_ledger", "circuit_resource_ledger"]) {
    if (report[key] != null) body.append(detail(key.replaceAll("_", " "), report[key], key === "acceptance" || key === "truth_contract"));
  }
  body.append(detail("Complete scientific report", report), detail("Lifecycle and reproducibility", {events: run.events, environment: run.environment, package_call_seconds: run.package_call_seconds, studio_elapsed_seconds: run.studio_elapsed_seconds}));
  const links = el("div", undefined, "actions");
  run.artifacts.forEach(name => { const link = el("a", `Open ${name}`); link.href = `/api/runs/${id}/${name}`; link.target = "_blank"; link.rel = "noopener"; links.append(link); }); body.append(links);
  if (!$("viewer").open) $("viewer").showModal();
}
async function showComparison() {
  const ids = [...state.selected], result = await api("/api/compare", {ids});
  const body = $("comparison-body"); body.replaceChildren(el("p", "Same target/problem. Errors retain their package definitions. Logical resource estimates are distinct from executed circuit evidence. Wall time includes Python simulation and diagnostics.", "muted"));
  const plot = el("img", undefined, "large-plot"); plot.src = `/api/compare.png?${ids.map(id => `id=${id}`).join("&")}`; plot.alt = "Stored scientific curves overlaid for compatible experiments"; body.append(plot);
  const rows = result.runs.map(r => ({...Object.fromEntries(Object.entries(r.request.settings).map(([k,v]) => [`request.${k}`,v])), ...r.metrics, "package_call_seconds (wall time)": r.package_call_seconds}));
  const keys = [...new Set(rows.flatMap(r => Object.keys(r)))];
  const table = el("table"), head = el("tr"); head.append(el("th", "Stored field")); result.runs.forEach(r => head.append(el("th", r.id.slice(0,8)))); table.append(head);
  keys.forEach(k => { const tr = el("tr"); tr.append(el("th", k)); rows.forEach(r => tr.append(el("td", r[k] === undefined ? "—" : format(r[k])))); table.append(tr); });
  const wrap = el("div", undefined, "table-wrap"); wrap.append(table); body.append(wrap, button("Export comparison table (JSON)", () => download("qsvt-comparison.json", result)));
  $("comparison").showModal();
}
async function refresh() {
  const data = await api("/api/runs"), key = JSON.stringify(data.runs);
  if (state.historyKey !== key) {
    state.historyKey = key; state.runs = data.runs; renderGallery();
    if ($("viewer").open && state.viewing) await showRun(state.viewing);
  }
}
async function init() {
  state.catalogue = await api("/api/catalogue");
  for (const entry of Object.values(state.catalogue.workflows)) { option($("workflow"), entry.id, entry.name); option($("workflow-filter"), entry.id, entry.name); }
  const encodings = new Set();
  for (const entry of Object.values(state.catalogue.workflows)) for (const key of ["access_model", "block_encoding"]) (entry.settings[key]?.values || []).forEach(v => encodings.add(v));
  encodings.forEach(v => option($("encoding-filter"), v, v));
  state.catalogue.presets.forEach(p => option($("preset"), p.id, p.name));
  // JSON object key order is not a presentation default (the API sorts keys).
  loadPreset(workflow("sign").recommended_preset); selection();
  $("workflow").addEventListener("change", () => loadPreset(workflow($("workflow").value).recommended_preset));
  $("package-defaults").addEventListener("click", () => {
    renderComposer(); $("preset").value = "";
    $("preset-source").textContent = "Package defaults, with catalogue choices for required arguments. Solver completion does not guarantee reconstruction accuracy.";
  });
  $("controls").addEventListener("input", () => {
    $("preset").value = ""; $("preset-source").textContent = "Custom configuration · changed from the loaded settings.";
  });
  $("preset").addEventListener("change", guard(async () => {
    if ($("preset").value === "") return;
    loadPreset($("preset").value);
  }));
  $("composer").addEventListener("submit", guard(async event => {
    event.preventDefault(); $("run").disabled = true;
    try { const run = await api("/api/runs", draft()); notice(`Experiment ${run.id.slice(0,8)} queued. You can continue composing.`); await refresh(); }
    finally { $("run").disabled = false; }
  }));
  $("export-draft").addEventListener("click", guard(async () => download("qsvt-request.json", await api("/api/validate", draft()))));
  $("import").addEventListener("change", guard(async event => {
    const file = event.target.files[0]; if (!file) return;
    if (file.size > 65536) throw new Error("Configuration file exceeds 64 KiB.");
    const original = JSON.parse(await file.text()), resolved = await api("/api/validate", original);
    renderComposer(resolved); $("preset").value=""; $("preset-source").textContent=`Imported ${file.name}`;
    notice(original.schema_version === resolved.schema_version ? "Configuration imported." : `Configuration imported; schema ${original.schema_version} upgraded to ${resolved.schema_version} with compatible defaults.`); event.target.value="";
  }));
  for (const id of ["search", "workflow-filter", "status-filter", "encoding-filter", "execution-filter", "sort", "favorites"]) $(id).addEventListener("input", renderGallery);
  $("compare").addEventListener("click", guard(showComparison));
  $("clear-selection").addEventListener("click", () => { state.selected.clear(); selection(); renderGallery(); });
  $("close-viewer").addEventListener("click", () => $("viewer").close());
  $("close-comparison").addEventListener("click", () => $("comparison").close());
  await refresh();
  async function poll() { try { await refresh(); } catch (error) { notice(`Connection: ${error.message}`); } finally { setTimeout(poll, 2000); } }
  setTimeout(poll, 2000);
}
guard(init)();
