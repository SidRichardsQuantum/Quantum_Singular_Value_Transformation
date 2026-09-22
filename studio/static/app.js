/* Presentation and request state only. All scientific data comes from Python. */
"use strict";
const $ = (id) => document.getElementById(id);
const state = {catalogue: null, runs: [], selected: new Set(), viewing: null, historyKey: "", offset: 0, total: 0, matching: 0, active: 0, refreshSequence: 0, composing: null, viewerKey: "", drafts: {}};
const filterIds = ["search", "workflow-filter", "status-filter", "encoding-filter", "execution-filter", "sort", "favorites"];
const storageKey = "qsvt-studio-ui-v1";
function saveUI() {
  if (!state.composing) return;
  const inputs = {};
  document.querySelectorAll("#controls [data-key]").forEach(input => {
    inputs[input.dataset.key] = input.type === "checkbox" ? input.checked : input.value;
  });
  state.drafts[state.composing] = {inputs, preset: $("preset").value, source: $("preset-source").textContent, advanced: $("advanced-settings").open};
  const filters = Object.fromEntries(filterIds.map(id => [id, id === "favorites" ? $(id).checked : $(id).value]));
  try { localStorage.setItem(storageKey, JSON.stringify({workflow: state.composing, drafts: state.drafts, filters})); }
  catch { /* Storage may be unavailable; the in-memory drafts still work. */ }
}
function restoreDraft(id) {
  const saved = state.drafts[id];
  if (!saved || !saved.inputs || typeof saved.inputs !== "object") return false;
  renderComposer({workflow: id, settings: {}});
  document.querySelectorAll("#controls [data-key]").forEach(input => {
    const value = saved.inputs[input.dataset.key];
    if (input.type === "hidden") return;
    if (input.type === "checkbox" && typeof value === "boolean") input.checked = value;
    else if (typeof value === "string") {
      if (input.tagName !== "SELECT" || [...input.options].some(option => option.value === value)) input.value = value;
    }
  });
  $("preset").value = typeof saved.preset === "string" ? saved.preset : "";
  $("preset-source").textContent = typeof saved.source === "string" ? saved.source : "Restored browser draft";
  $("advanced-settings").open = saved.advanced === true;
  return true;
}
function el(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}
function notice(message) { $("notice").textContent = message; }
function connectionFailure(error) {
  state.connectionMessage = `Connection: ${error.message}`;
  notice(state.connectionMessage);
}
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
  state.composing = entry.id;
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
  saveUI();
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
  saveUI();
  renderComposer(request); $("preset").value = ""; $("preset-source").textContent = `Restored exact saved configuration · ${id.slice(0, 8)}`;
  $("viewer").close(); $("workflow").focus(); window.scrollTo({top: 0, behavior: "smooth"});
  saveUI();
  notice(`Saved configuration restored${saved.schema_version !== request.schema_version ? `; schema ${saved.schema_version} upgraded to ${request.schema_version} with compatible defaults` : ""}. Change a setting and run a new experiment.`);
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
  const runs = state.runs;
  $("count").textContent = `${runs.length} visible / ${state.matching} matching / ${state.total} saved`;
  $("page-status").textContent = state.matching ? `${state.offset + 1}–${state.offset + runs.length} of ${state.matching}` : "No results";
  $("previous-page").disabled = state.offset === 0;
  $("next-page").disabled = state.offset + runs.length >= state.matching;
  $("gallery").replaceChildren();
  if (!runs.length) {
    const empty = el("div", undefined, "empty"); empty.append(el("h2", state.total ? "No matching experiments" : "Your next experiment starts here"), el("p", "Choose a workflow or a published example, then run it to build your experiment gallery."));
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
    body.append(outcomes(run));
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
function outcomes(run) {
  const panel = el("div", undefined, "outcomes"), m = run.metrics || {};
  const add = (label, value, tone="") => {
    const row = el("p", undefined, tone); row.append(el("strong", `${label}: `), document.createTextNode(value)); panel.append(row);
  };
  const executed = m["execution.succeeded"] ?? m["qsvt_execution.succeeded"];
  add("Execution", executed === true ? "Finite QNode succeeded" : executed === false ? "Finite QNode did not succeed" : executionRequested(run.request) ? "Requested; no execution verdict recorded" : "Not requested", executed === false ? "error" : "");
  const labels = {passed: "Reconstruction passed", solver_failed: "Solver failed", reconstruction_failed: "Reconstruction failed", reconstruction_unavailable: "Reconstruction unavailable"};
  const qualities = [["", m["synthesis_quality.status"]], ["Cosine: ", m["component_synthesis_quality.cosine.status"]], ["Sine: ", m["component_synthesis_quality.sine.status"]]].filter(([, value]) => value !== undefined);
  add("Phase reconstruction", qualities.length ? qualities.map(([name, value]) => name + (labels[value] || value)).join(" · ") : "No assessment recorded", qualities.some(([, value]) => value !== "passed") ? "error" : "");
  add("Scientific acceptance", m["acceptance.status"] === undefined ? "No acceptance verdict recorded" : String(m["acceptance.status"]).replaceAll("_", " "), m["acceptance.status"] === "acceptance_criteria_not_met" ? "error" : "");
  panel.append(el("p", "Completion means a report was saved. Reconstruction and acceptance apply only to their stated checks.", "muted"));
  return panel;
}
function sectionSignature(section) {
  return [...section.childNodes].map(node => new XMLSerializer().serializeToString(node)).join("");
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
async function showRun(id, updating=false) {
  const run = await api(`/api/runs/${id}`);
  if (updating && (!$("viewer").open || state.viewing !== id)) return;
  const key = JSON.stringify(run);
  if (updating && state.viewerKey === key) return;
  const sameRun = state.viewing === id;
  state.viewing = id; state.viewerKey = key;
  state.viewerActive = !["completed", "failed", "cancelled"].includes(run.status);
  const {report: ignoredReport, ...summary} = run;
  state.viewerSummaryKey = JSON.stringify(summary);
  const scroll = sameRun ? $("viewer").scrollTop : 0;
  $("viewer-title").textContent = `${workflow(run.request.workflow).name} · ${id.slice(0,8)}`;
  const body = el("div");
  const navigation = el("nav", undefined, "report-nav"); navigation.setAttribute("aria-label", "Report sections");
  const sections = {};
  for (const [key, title] of [["overview", "Overview"], ["accuracy", "Accuracy"], ["phases", "Phases"], ["resources", "Resources"], ["reproducibility", "Reproducibility"]]) {
    const link = el("a", title); link.href = `#report-${key}`; navigation.append(link);
    const section = el("section"); section.id = `report-${key}`; section.append(el("h3", title)); sections[key] = section;
  }
  body.append(navigation, ...Object.values(sections));
  const overview = sections.overview;
  const actions = el("div", undefined, "actions");
  actions.append(button("Reuse configuration", () => reuse(id)), button("Export configuration", () => download(`${id}-request.json`, run.request)));
  if (run.report) actions.append(button("Export results", () => download(`${id}-report.json`, run.report)));
  if (run.status === "completed") actions.append(button("Select for comparison", () => { toggleSelection(id); notice("Run selection updated."); }));
  overview.append(actions, outcomes(run), el("p", `Lifecycle: ${run.status}. Completion and acceptance are separate.`, "muted"));
  if (run.progress?.message) overview.append(el("p", run.progress.message, "progress"));
  if (run.error) overview.append(el("p", `${run.error.type}: ${run.error.message}`, "error"));
  (run.warnings || []).forEach(w => overview.append(el("p", w, "error")));
  if (run.report?.acceptance) {
    const a = run.report.acceptance;
    sections.accuracy.append(el("div", `${a.status} · scope: ${a.scope} · full_qsvt_acceptance: ${a.full_qsvt_acceptance}. See all required checks and omitted components below.`, "acceptance"));
  }
  if (run.report?.synthesis_quality) sections.phases.append(qualityPanel("Phase synthesis", run.report.synthesis_quality));
  for (const [name, quality] of Object.entries(run.report?.component_synthesis_quality || {})) sections.phases.append(qualityPanel(`${name} phase synthesis`, quality));
  if (run.report?.synthesis && !run.report.synthesis_quality) sections.phases.append(el("p", "Historical synthesis report: no reconstruction-quality assessment was saved. Inspect its error and original acceptance checks.", "muted"));
  if (run.artifacts.includes("preview.png")) {
    const plot = el("img", undefined, "large-plot"); plot.src = `/api/runs/${id}/preview.png`; plot.alt = "Scientific diagnostics from the saved package report"; overview.append(plot);
  }
  for (const [filename, title] of [["phases.png", "Synthesized phases"], ["spectrum.png", "Spectral response"], ["resources.png", "Resource report"]]) {
    if (!run.artifacts.includes(filename)) continue;
    const section = filename === "phases.png" ? sections.phases : filename === "resources.png" ? sections.resources : sections.accuracy;
    section.append(el("h4", title));
    const plot = el("img", undefined, "large-plot"); plot.src = `/api/runs/${id}/${filename}`; plot.alt = `${title} from the saved package report`; section.append(plot);
  }
  if (run.metrics) sections.accuracy.append(metricTable(run.metrics));
  sections.reproducibility.append(detail("Problem / resolved configuration", run.request, true));
  const report = run.report || {};
  for (const key of ["diagnostics", "coeffs", "compatibility", "degree_search", "synthesis", "execution", "resources", "resource_report", "acceptance", "truth_contract", "error_budget", "block_encoding_spec", "observable_values", "physical_observables", "cos_coeffs", "sin_coeffs", "scaled_operator", "qsvt_execution", "component_error_ledger", "circuit_resource_ledger"]) {
    const group = ["synthesis", "coeffs", "cos_coeffs", "sin_coeffs", "compatibility"].includes(key) ? "phases" : ["resources", "resource_report", "circuit_resource_ledger"].includes(key) ? "resources" : "accuracy";
    if (report[key] != null) sections[group].append(detail(key.replaceAll("_", " "), report[key], key === "acceptance" || key === "truth_contract"));
  }
  sections.reproducibility.append(detail("Complete scientific report", report), detail("Lifecycle and reproducibility", {events: run.events, environment: run.environment, package_call_seconds: run.package_call_seconds, studio_elapsed_seconds: run.studio_elapsed_seconds}));
  const links = el("div", undefined, "actions");
  run.artifacts.forEach(name => { const link = el("a", `Open ${name}`); link.href = `/api/runs/${id}/${name}`; link.target = "_blank"; link.rel = "noopener"; links.append(link); }); sections.reproducibility.append(links);
  for (const section of Object.values(sections)) if (section.children.length === 1) section.append(el("p", "No evidence recorded for this section.", "muted"));
  const current = $("viewer-body");
  if (!sameRun || !current.children.length) current.replaceChildren(...body.children);
  else {
    // Keep unchanged sections (and their focused controls) in the DOM.
    for (const section of Object.values(sections)) {
      const previous = $(section.id);
      const signature = sectionSignature(section);
      if (previous.dataset.signature === signature) continue;
      const expanded = new Map([...previous.querySelectorAll("details")].map(node => [node.querySelector("summary").textContent, node.open]));
      const focused = previous.contains(document.activeElement) ? document.activeElement : null;
      const focusText = focused?.textContent;
      section.querySelectorAll("details").forEach(node => { const title = node.querySelector("summary").textContent; if (expanded.has(title)) node.open = expanded.get(title); });
      section.dataset.signature = signature;
      previous.replaceWith(section);
      if (focused) [...section.querySelectorAll("button, a, summary")].find(node => node.textContent === focusText)?.focus({preventScroll: true});
    }
  }
  current.querySelectorAll(":scope > section").forEach(section => { if (!section.dataset.signature) section.dataset.signature = sectionSignature(section); });
  if (!$("viewer").open) $("viewer").showModal();
  $("viewer").scrollTop = scroll;
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
  const sequence = ++state.refreshSequence;
  const params = new URLSearchParams({limit: "24", offset: String(state.offset)});
  for (const id of filterIds) params.set(id.replace("-filter", ""), id === "favorites" ? String($(id).checked) : $(id).value);
  const data = await api(`/api/runs?${params}`);
  if (sequence !== state.refreshSequence) return;
  if (state.connectionMessage && $("notice").textContent === state.connectionMessage) notice("");
  state.connectionMessage = "";
  state.active = data.active; state.offset = data.offset; state.total = data.total; state.matching = data.matching;
  const key = JSON.stringify(data);
  if (state.historyKey !== key) {
    state.historyKey = key; state.runs = data.runs; renderGallery();
  }
  const viewed = state.runs.find(run => run.id === state.viewing);
  if ($("viewer").open && state.viewing && (state.viewerActive || (viewed && JSON.stringify(viewed) !== state.viewerSummaryKey))) await showRun(state.viewing, true);
}
let pollTimer;
function schedulePoll(failed=false) {
  clearTimeout(pollTimer);
  if (document.hidden) return;
  pollTimer = setTimeout(async () => {
    try { await refresh(); schedulePoll(); }
    catch (error) { connectionFailure(error); schedulePoll(true); }
  }, failed ? 30000 : state.active ? 2000 : 15000);
}
async function init() {
  state.catalogue = await api("/api/catalogue");
  for (const entry of Object.values(state.catalogue.workflows)) { option($("workflow"), entry.id, entry.name); option($("workflow-filter"), entry.id, entry.name); }
  const encodings = new Set();
  for (const entry of Object.values(state.catalogue.workflows)) for (const key of ["access_model", "block_encoding"]) (entry.settings[key]?.values || []).forEach(v => encodings.add(v));
  encodings.forEach(v => option($("encoding-filter"), v, v));
  state.catalogue.presets.forEach(p => option($("preset"), p.id, p.name));
  // JSON object key order is not a presentation default (the API sorts keys).
  let saved;
  try { saved = JSON.parse(localStorage.getItem(storageKey)); } catch { /* Ignore unavailable or damaged browser storage. */ }
  if (saved?.drafts && typeof saved.drafts === "object" && !Array.isArray(saved.drafts)) state.drafts = saved.drafts;
  if (!saved || !Object.hasOwn(state.catalogue.workflows, saved.workflow) || !restoreDraft(saved.workflow)) loadPreset(workflow("sign").recommended_preset);
  for (const id of filterIds) {
    const value = saved?.filters?.[id];
    if (id === "favorites") $(id).checked = value === true;
    else if (typeof value === "string" && ($(id).tagName !== "SELECT" || [...$(id).options].some(option => option.value === value))) $(id).value = value;
  }
  selection();
  $("workflow").addEventListener("change", () => { const id = $("workflow").value; saveUI(); if (!restoreDraft(id)) loadPreset(workflow(id).recommended_preset); saveUI(); });
  $("package-defaults").addEventListener("click", () => {
    renderComposer(); $("preset").value = "";
    $("preset-source").textContent = "Package defaults, with catalogue choices for required arguments. Solver completion does not guarantee reconstruction accuracy.";
    saveUI();
  });
  $("controls").addEventListener("input", () => {
    $("preset").value = ""; $("preset-source").textContent = "Custom configuration · changed from the loaded settings.";
    saveUI();
  });
  $("controls").addEventListener("toggle", saveUI, true);
  $("preset").addEventListener("change", guard(async () => {
    if ($("preset").value === "") return;
    loadPreset($("preset").value);
  }));
  $("composer").addEventListener("submit", guard(async event => {
    event.preventDefault(); saveUI(); $("run").disabled = true;
    try { const run = await api("/api/runs", draft()); notice(`Experiment ${run.id.slice(0,8)} queued. You can continue composing.`); await refresh(); schedulePoll(); }
    finally { $("run").disabled = false; }
  }));
  $("export-draft").addEventListener("click", guard(async () => download("qsvt-request.json", await api("/api/validate", draft()))));
  $("import").addEventListener("change", guard(async event => {
    const file = event.target.files[0]; if (!file) return;
    if (file.size > 65536) throw new Error("Configuration file exceeds 64 KiB.");
    const original = JSON.parse(await file.text()), resolved = await api("/api/validate", original);
    saveUI();
    renderComposer(resolved); $("preset").value=""; $("preset-source").textContent=`Imported ${file.name}`;
    notice(original.schema_version === resolved.schema_version ? "Configuration imported." : `Configuration imported; schema ${original.schema_version} upgraded to ${resolved.schema_version} with compatible defaults.`); event.target.value=""; saveUI();
  }));
  let filterTimer;
  for (const id of filterIds) $(id).addEventListener("input", () => {
    saveUI(); state.offset = 0; ++state.refreshSequence; clearTimeout(filterTimer);
    filterTimer = setTimeout(guard(refresh), id === "search" ? 200 : 0);
  });
  $("previous-page").addEventListener("click", guard(async () => { state.offset = Math.max(0, state.offset - 24); await refresh(); }));
  $("next-page").addEventListener("click", guard(async () => { state.offset += 24; await refresh(); }));
  document.addEventListener("visibilitychange", async () => {
    clearTimeout(pollTimer);
    if (document.hidden) return;
    try { await refresh(); schedulePoll(); }
    catch (error) { connectionFailure(error); schedulePoll(true); }
  });
  window.addEventListener("pagehide", saveUI);
  $("compare").addEventListener("click", guard(showComparison));
  $("clear-selection").addEventListener("click", () => { state.selected.clear(); selection(); renderGallery(); });
  $("close-viewer").addEventListener("click", () => $("viewer").close());
  $("close-comparison").addEventListener("click", () => $("comparison").close());
  await refresh();
  schedulePoll();
}
guard(init)();
