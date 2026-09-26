/* Preparation workspace; all state is refreshed from persisted API records. */
const $ = id => document.getElementById(id);
let taskId = localStorage.getItem("iforecast-task-id");
let prep = null;
let file = null;
let assistantAvailable = false;
let opt = null;
let setupInfo = null;
let viewStage = "training";
let selectedTrialId = null;
let lastTaskStage = null;

function node(tag, text, className) {
  const item = document.createElement(tag);
  if (text !== undefined) item.textContent = String(text);
  if (className) item.className = className;
  return item;
}
function clear(id) { const target = $(id); target.replaceChildren(); return target; }
function addButton(id, label, callback, kind = "primary") {
  const button = node("button", label, kind);
  button.type = "button";
  button.onclick = () => work(callback);
  $(id).append(button);
}
function field(parent, label, type, value, options) {
  const wrap = node("label", label, "field");
  const input = node(options ? "select" : "input");
  if (options) {
    for (const [text, val] of options) {
      const option = node("option", text);
      option.value = val;
      input.append(option);
    }
  } else input.type = type;
  input.value = value ?? "";
  wrap.append(input);
  parent.append(wrap);
  return input;
}
async function request(path, method = "GET", body, multipart = false) {
  const response = await fetch(path, {
    method,
    headers: multipart ? {} : body ? {"Content-Type": "application/json"} : {},
    body: multipart ? body : body ? JSON.stringify(body) : undefined
  });
  const result = await response.json();
  if (!response.ok) throw new Error(typeof result.detail === "string" ? result.detail : JSON.stringify(result.detail));
  return result;
}
function action(name, extra = {}) {
  return request(`/tasks/${taskId}/preparation/actions/${name}`, "POST", {
    expected_version: prep.version, ...extra
  });
}
async function refresh() {
  const token = refresh.generation = (refresh.generation || 0) + 1;
  const requestedTask = taskId;
  let requestedStage = viewStage;
  const current = () => refresh.generation === token && taskId === requestedTask &&
    viewStage === requestedStage;
  const nextPrep = await request(`/tasks/${requestedTask}/preparation`);
  if (!current()) return;
  const task = await request(`/tasks/${requestedTask}`);
  if (!current()) return;
  const stageTraining = task.stage === "OPTIMIZATION";
  const stageDeployment = task.stage === "DEPLOYMENT";
  let nextStage = requestedStage;
  if (lastTaskStage !== task.stage && (stageTraining || stageDeployment))
    nextStage = stageDeployment ? "deployment" : "training";
  if (!stageTraining && !stageDeployment) nextStage = "preparation";
  let nextOpt = null, nextSetupInfo = null;
  if (stageTraining) {
    try { nextOpt = await request(`/tasks/${requestedTask}/optimization`); }
    catch (error) {
      if (!current()) return;
      if (!error.message.includes("optimization session not found")) throw error;
      nextSetupInfo = await request(`/tasks/${requestedTask}/optimization/setup`);
    }
  }
  if (!current()) return;
  const canDeploy = stageDeployment || (stageTraining && nextOpt?.session.phase === "completed");
  if (nextStage === "deployment" && !canDeploy) nextStage = "training";
  const training = stageTraining && nextStage === "training";
  const deployment = canDeploy && nextStage === "deployment";
  if (deployment && await loadDeployment() === false) return;
  if (!current()) return;
  prep = nextPrep;
  opt = nextOpt;
  setupInfo = nextSetupInfo;
  viewStage = nextStage;
  requestedStage = nextStage;
  lastTaskStage = task.stage;
  $("task-id").textContent = deployment ? "Forecast workspace" :
    `Task ${requestedTask.slice(0, 8)}`;
  $("training-nav").disabled = !stageTraining;
  $("deployment-nav").disabled = !canDeploy;
  const name = deployment ? "Deployment" : training ? "Training & Evaluation" : "Preparation";
  $("stage-badge").textContent = name;
  document.querySelectorAll(".stage").forEach((button, index) =>
    button.classList.toggle("active", index === (deployment ? 2 : training ? 1 : 0)));
  for (const id of ["upload-section", "schema-section", "mapping-section", "quality-section", "plan-section", "overview-section", "task-section", "review-section"])
    $(id).hidden = training || deployment;
  $("training-workspace").hidden = !training;
  $("deployment-workspace").hidden = !deployment;
  document.querySelector(".page-heading h1").textContent =
    deployment ? "Load forecast and similar days" : training ? "Training & Evaluation" : "Dataset preparation";
  $("workflow-status").textContent = deployment ?
    (deploymentState?.forecast ? "Forecast ready" : "Ready to forecast") : task.workflow_status;
  $("step-pill").textContent = deployment
    ? (deploymentState?.forecast ?
      new Intl.DateTimeFormat(undefined, {dateStyle: "medium",
        timeZone: deploymentState.forecast.origin.timezone_name || "UTC"})
        .format(new Date(deploymentState.forecast.target_timestamps[0]))
      : "Set up forecast")
    : training ? (opt?.session.phase || "run setup").replaceAll("_", " ")
    : prep.step.replaceAll("_", " ").toLowerCase();
  $("deployment-status").textContent = deployment && deploymentState?.forecast
    ? (deploymentState.inspected?.version_number ? "Adjusted forecast" : "Original forecast")
    : "";
  const qualityStatus = prep.quality
    ? ` · ${prep.quality.missing_timestamps} missing timestamps, ${prep.quality.duplicate_timestamps} duplicates`
    : "";
  $("manager-status").textContent = deployment
    ? "Ask Task Manager about this forecast"
    : training
      ? opt ? `${opt.session.mode} · round ${opt.summary.current_round} · ${opt.session.phase}` : "Preparation frozen · configure a search run"
      : `Current step: ${prep.step.replaceAll("_", " ")}${qualityStatus}`;
  $("manager-subtitle").textContent = deployment ? "Forecast assistance" : training ? "Optimization guidance" : "Preparation guidance";
  $("chat-label").textContent = deployment ? "Message Task Manager" : training ? "Ask about progress or propose search guidance" : "Ask Task Manager about this preparation";
  $("chat-input").placeholder = deployment ? "Compare with last week, or reduce after 15:00." : training ? "Prefer LSTM, or describe a search preference." : "Ask why, or request a change…";
  render();
  if (training) renderTraining();
  if (deployment) renderDeployment();
  const messages = await request(`/tasks/${requestedTask}/messages`);
  if (!current()) return;
  const conversation = clear("conversation");
  for (const message of messages.filter(m => m.topic === "chat" &&
      ["user", "task_manager"].includes(m.source_role))) {
    const content = message.payload?.text;
    if (!content) continue;
    const user = message.source_role === "user";
    const bubble = node("div", undefined, `bubble ${user ? "user" : "task-manager"}`);
    bubble.append(node("strong", user ? "You" : "Task Manager"), node("p", content));
    conversation.append(bubble);
  }
}
async function work(callback) {
  try {
    $("notice").textContent = "";
    await callback();
    await refresh();
  } catch (error) {
    // A rejected mutation may still persist guidance history and advance versions.
    try { await refresh(); } catch { /* Keep the original rejection visible. */ }
    $("notice").textContent = error.message;
  }
}
function stats(parent, entries) {
  const grid = node("div", undefined, "stat-grid");
  for (const [label, value] of entries) {
    const item = node("div", undefined, "stat");
    item.append(node("small", label), node("strong", value));
    grid.append(item);
  }
  parent.append(grid);
}
function warnings(parent, items) {
  for (const text of items || []) parent.append(node("div", text, "warning"));
}
function chart(parent, title, points, color) {
  if (!points || !points.length) return;
  parent.append(node("div", title, "chart-title"));
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", "0 0 600 190");
  svg.setAttribute("class", "chart");
  svg.setAttribute("role", "img");
  svg.setAttribute("aria-label", title);
  const values = points.map(p => Number(p[1]));
  const minimum = Math.min(...values), maximum = Math.max(...values);
  const span = maximum - minimum || 1;
  const path = document.createElementNS("http://www.w3.org/2000/svg", "polyline");
  path.setAttribute("fill", "none");
  path.setAttribute("stroke", color);
  path.setAttribute("stroke-width", "2");
  path.setAttribute("points", values.map((v, i) => `${20 + i * 560 / Math.max(1, values.length - 1)},${168 - (v - minimum) * 145 / span}`).join(" "));
  svg.append(path);
  parent.append(svg);
}
function render() {
  const step = prep.step;
  $("upload-button").disabled = step !== "UPLOAD_DATASET";
  $("upload-state").textContent = prep.source ? `${prep.original_filename} · ID ${prep.source.dataset_id} · SHA-256 ${prep.source.sha256.slice(0, 12)}…` : "No dataset uploaded";
  const schema = clear("schema-content"), schemaAction = clear("schema-action");
  if (prep.schema_inspection) {
    stats(schema, [["Rows", prep.schema_inspection.row_count], ["Columns", prep.schema_inspection.columns.length], ["Likely interval", prep.schema_inspection.interval_candidates[0] || "Unknown"]]);
    const table = node("table");
    const head = node("tr");
    for (const label of ["Column", "Type", "Missing", "Date", "Numeric", "Hints"]) head.append(node("th", label));
    table.append(head);
    for (const col of prep.schema_inspection.columns) {
      const row = node("tr");
      for (const value of [col.name, col.physical_dtype, col.missing_count, Math.round(col.datetime_parse_ratio * 100) + "%", Math.round(col.numeric_parse_ratio * 100) + "%", col.semantic_hints.join(", ")]) row.append(node("td", value));
      table.append(row);
    }
    schema.append(table);
  } else schema.append(node("div", "Upload a dataset to inspect its columns.", "empty"));
  if (step === "INSPECT_SCHEMA") addButton("schema-action", "Inspect schema", () => action("inspect"));
  if (step === "PROPOSE_COLUMN_MAPPING") addButton("schema-action", "Propose column mapping", () => action("propose-mapping"));

  const mapping = clear("mapping-content"); clear("mapping-action");
  if (prep.mapping_draft && prep.schema_inspection) {
    const names = prep.schema_inspection.columns.map(c => c.name);
    const options = [["Select column", ""], ...names.map(n => [n, n])];
    const grid = node("div", undefined, "field-grid");
    const time = field(grid, "Time column", "text", prep.mapping_draft.timestamp_column, options);
    const load = field(grid, "Load target", "text", prep.mapping_draft.target_column, options);
    const tempOptions = [["None", ""], ...names.map(n => [n, n])];
    const temperature = field(grid, "Temperature", "text", prep.mapping_draft.temperature_column, tempOptions);
    mapping.append(grid);
    const other = node("div", undefined, "check-list");
    const chosen = new Set([prep.mapping_draft.timestamp_column, prep.mapping_draft.target_column, prep.mapping_draft.temperature_column]);
    const checks = [];
    for (const col of prep.schema_inspection.columns) {
      if (chosen.has(col.name)) continue;
      const label = node("label"), check = node("input");
      check.type = "checkbox";
      check.checked = prep.mapping_draft.other_features.includes(col.name);
      checks.push([col.name, check]);
      label.append(check, document.createTextNode(" " + col.name));
      other.append(label);
    }
    mapping.append(node("p", "Optional other features (numeric/binary columns only):"), other);
    warnings(mapping, prep.mapping_draft.ambiguous_roles.map(x => `Ambiguous ${x}; choose a column explicitly.`));
    if (step === "CONFIRM_COLUMN_MAPPING") {
      if (assistantAvailable) addButton("mapping-action", "Get recommendation", () => request(`/tasks/${taskId}/preparation/assistant/propose-mapping`, "POST", {expected_version: prep.version}), "secondary");
      addButton("mapping-action", "Save mapping", async () => {
        const draft = {
          ...prep.mapping_draft,
          timestamp_column: time.value || null,
          target_column: load.value || null,
          temperature_column: temperature.value || null,
          temperature_available: Boolean(temperature.value),
          other_features: checks.filter(x => x[1].checked && ![time.value, load.value, temperature.value].includes(x[0])).map(x => x[0]),
          excluded_features: checks.filter(x => !x[1].checked).map(x => x[0])
        };
        await request(`/tasks/${taskId}/preparation/mapping`, "PUT", {expected_version: prep.version, draft});
      });
      addButton("mapping-action", "Confirm mapping", () => action("confirm-mapping"), "secondary");
    }
  } else mapping.append(node("div", "A mapping proposal will appear after inspection.", "empty"));

  const quality = clear("quality-content"); clear("quality-action");
  if (prep.quality) {
    stats(quality, [["Rows", prep.quality.row_count], ["Missing timestamps", prep.quality.missing_timestamps], ["Duplicates", prep.quality.duplicate_timestamps], ["Irregular intervals", prep.quality.irregular_intervals]]);
    warnings(quality, prep.quality.warnings);
    const list = node("ul", undefined, "compact");
    for (const [name, count] of Object.entries(prep.quality.invalid_numeric)) list.append(node("li", `${name}: ${count} invalid, ${prep.quality.missing_values[name]} missing, ${prep.quality.thousands_separator_counts[name] || 0} thousands separators`));
    quality.append(list);
  } else quality.append(node("div", "Quality diagnostics require a confirmed mapping.", "empty"));
  if (step === "ANALYZE_DATA_QUALITY") addButton("quality-action", "Analyze quality", () => action("analyze-quality"));
  if (step === "PROPOSE_PREPARATION") addButton("quality-action", "Propose preparation", () => action("propose-plan"));

  const plan = clear("plan-content"); clear("plan-action");
  if (prep.plan_draft) {
    warnings(plan, prep.plan_draft.rationale);
    const grid = node("div", undefined, "field-grid");
    const opts = choices => choices.map(x => [x.replaceAll("_", " "), x]);
    const timezone = field(grid, "Dataset timezone", "text", prep.plan_draft.timezone_name);
    const duplicate = field(grid, "Duplicate timestamps", "text", prep.plan_draft.duplicate_policy, opts(["reject", "first", "last", "mean"]));
    const missingTime = field(grid, "Missing timestamps", "text", prep.plan_draft.missing_timestamp_policy, opts(["reject", "interpolate", "forward_fill"]));
    const missingValue = field(grid, "Missing/invalid values", "text", prep.plan_draft.missing_value_policy, opts(["reject", "interpolate", "forward_fill", "drop"]));
    const invalid = field(grid, "Invalid time rows", "text", prep.plan_draft.invalid_row_policy, opts(["reject", "drop"]));
    plan.append(grid);
    if (step === "CONFIRM_PREPARATION") {
      if (assistantAvailable) addButton("plan-action", "Get recommendation", () => request(`/tasks/${taskId}/preparation/assistant/propose-plan`, "POST", {expected_version: prep.version}), "secondary");
      addButton("plan-action", "Save decisions", async () => {
        await request(`/tasks/${taskId}/preparation/plan`, "PUT", {expected_version: prep.version, draft: {...prep.plan_draft, timezone_name: timezone.value, duplicate_policy: duplicate.value, missing_timestamp_policy: missingTime.value, missing_value_policy: missingValue.value, invalid_row_policy: invalid.value}});
      });
      addButton("plan-action", "Approve preparation", () => action("confirm-plan"), "secondary");
    }
  } else plan.append(node("div", "Cleaning decisions will appear after quality analysis.", "empty"));
  if (step === "APPLY_PREPARATION") addButton("plan-action", "Apply approved plan", () => action("apply", {action_id: crypto.randomUUID()}));

  const overview = clear("overview-content"); clear("overview-action");
  if (prep.overview) {
    stats(overview, [["Prepared rows", prep.overview.row_count], ["Sampling interval", prep.overview.frequency], ["Mean load", prep.overview.load_mean.toFixed(2)]]);
    overview.append(node("p", `${prep.overview.time_start} — ${prep.overview.time_end}`));
    chart(overview, "Load overview", prep.overview.primary.points, "#2d648f");
    if (prep.overview.secondary) chart(overview, prep.overview.secondary.name + " overview", prep.overview.secondary.points, "#7a9c70");
    overview.append(node("p", `Other features: ${Object.keys(prep.overview.other_feature_stats).join(", ") || "none"}`));
  } else overview.append(node("div", "Prepared dataset summaries and charts will appear here.", "empty"));
  if (step === "GENERATE_DATASET_OVERVIEW") addButton("overview-action", "Generate overview", () => action("generate-overview"));

  const task = clear("task-content"); clear("task-action");
  if (prep.task_draft) {
    const grid = node("div", undefined, "field-grid");
    const delta = field(grid, "Lead time Δ", "number", prep.task_draft.delta);
    const horizon = field(grid, "Horizon H (current adapters: 1)", "number", prep.task_draft.horizon);
    const unit = field(grid, "Lead unit", "text", prep.task_draft.time_unit, [["Samples", "samples"], ["Hours", "hours"]]);
    const output = field(grid, "Forecast output", "text", prep.task_draft.output.representation, [["Point", "point"], ["Quantile", "quantile"]]);
    const objective = field(grid, "Evaluation objective", "text", prep.task_draft.objective_id, [["MAE", "mae"], ["MAPE", "mape"], ["CRPS", "crps"], ["Weighted MAE", "weighted_mae"], ["Asymmetric MAE", "asymmetric_mae"]]);
    const quantiles = field(grid, "Quantile levels (comma-separated)", "text", prep.task_draft.output.quantile_levels.join(", ") || "0.1, 0.5, 0.9");
    const toggleQuantiles = () => { quantiles.parentElement.style.display = output.value === "quantile" ? "flex" : "none"; };
    const rule = prep.task_draft.metric_spec?.time_range;
    const rangeStart = field(grid, "Weighted interval start (target local time)", "time", rule?.start_local?.slice(0, 5) || "");
    const rangeEnd = field(grid, "Weighted interval end, exclusive", "time", rule?.end_local?.slice(0, 5) || "");
    const rangeWeight = field(grid, "Weight inside interval (> 0)", "number", rule?.weight ?? "");
    rangeWeight.step = "any";
    const asymOver = field(grid, "Overestimation weight (> 0)", "number", prep.task_draft.metric_spec?.over_weight ?? "");
    const asymUnder = field(grid, "Underestimation weight (> 0)", "number", prep.task_draft.metric_spec?.under_weight ?? "");
    asymOver.step = asymUnder.step = "any";
    const toggleMetric = () => {
      for (const item of [rangeStart, rangeEnd, rangeWeight])
        item.parentElement.style.display = objective.value === "weighted_mae" ? "flex" : "none";
      for (const item of [asymOver, asymUnder])
        item.parentElement.style.display = objective.value === "asymmetric_mae" ? "flex" : "none";
    };
    objective.onchange = toggleMetric;
    output.onchange = () => {
      objective.value = output.value === "quantile" ? "crps" : "mae";
      toggleQuantiles();
      toggleMetric();
    };
    toggleQuantiles();
    toggleMetric();
    const train = field(grid, "Training fraction", "number", prep.task_draft.train_fraction);
    const validation = field(grid, "Validation fraction", "number", prep.task_draft.validation_fraction);
    const test = field(grid, "Test fraction", "number", prep.task_draft.test_fraction);
    const seed = field(grid, "Protocol seed", "number", prep.task_draft.seed);
    let temperature = null, provenance = null;
    const otherControls = {};
    if (prep.capabilities?.has_temperature) {
      temperature = field(grid, "Temperature availability", "text", prep.task_draft.temperature_policy?.kind, [["Observed/history only", "observed"], ["Known ahead", "known_ahead"], ["Forecast product", "forecast"], ["Perfect forecast (experimental)", "perfect_forecast"]]);
      provenance = field(grid, "Forecast source / oracle protocol ID", "text", prep.task_draft.temperature_policy?.source_ref || prep.task_draft.temperature_policy?.protocol_id || "");
    }
    for (const name of prep.capabilities?.available_other_features || []) {
      const current = prep.task_draft.other_policies[name] || {kind: "observed", role: "other"};
      const policy = field(grid, `${name} availability`, "text", current.kind, [["Observed/history only", "observed"], ["Known ahead", "known_ahead"], ["Forecast product", "forecast"]]);
      const source = field(grid, `${name} forecast source`, "text", current.source_ref || "");
      otherControls[name] = {policy, source};
    }
    task.append(grid);
    task.append(node("p", "Split fractions and temperature availability are explicit experimental assumptions. Forecast-product issues require availability metadata at training time."));
    if (step === "CONFIGURE_FORECAST_TASK") {
      if (assistantAvailable) addButton("task-action", "Get recommendation", () => request(`/tasks/${taskId}/preparation/assistant/propose-task`, "POST", {expected_version: prep.version}), "secondary");
      addButton("task-action", "Save task configuration", async () => {
        const representation = output.value;
        const policy = temperature ? {kind: temperature.value, role: "temperature", source_ref: temperature.value === "forecast" ? provenance.value : null, protocol_id: temperature.value === "perfect_forecast" ? provenance.value : null} : null;
        const other_policies = Object.fromEntries(Object.entries(otherControls).map(([name, control]) => [name, {kind: control.policy.value, role: "other", source_ref: control.policy.value === "forecast" ? control.source.value : null, protocol_id: null}]));
        const metricSpec = objective.value === "weighted_mae"
          ? {kind: "weighted", base_metric: "mae", timezone_name: prep.prepared.dataset.timezone_name,
             time_range: {start_local: rangeStart.value, end_local: rangeEnd.value, weight: Number(rangeWeight.value)}}
          : objective.value === "asymmetric_mae"
            ? {kind: "asymmetric", base_metric: "mae", over_weight: Number(asymOver.value), under_weight: Number(asymUnder.value)}
            : null;
        if (objective.value === "weighted_mae" && (!rangeStart.value || !rangeEnd.value || !rangeWeight.value))
          throw new Error("Weighted MAE needs a start, end and positive weight.");
        if (objective.value === "asymmetric_mae" && (!asymOver.value || !asymUnder.value || Number(asymOver.value) <= 0 || Number(asymUnder.value) <= 0))
          throw new Error("Asymmetric MAE needs positive overestimation and underestimation weights.");
        const draft = {...prep.task_draft, delta: Number(delta.value), horizon: Number(horizon.value), time_unit: unit.value, output: {representation, quantile_levels: representation === "quantile" ? quantiles.value.split(",").map(Number) : []}, objective_id: objective.value, metric_spec: metricSpec, temperature_policy: policy, other_policies, train_fraction: Number(train.value), validation_fraction: Number(validation.value), test_fraction: Number(test.value), seed: Number(seed.value)};
        await request(`/tasks/${taskId}/preparation/task`, "PUT", {expected_version: prep.version, draft});
      });
      addButton("task-action", "Review preparation", () => action("review"), "secondary");
    }
  } else task.append(node("div", "Configure the forecast after preparing the dataset.", "empty"));

  const review = clear("review-content"); clear("review-action");
  if (["REVIEW_PREPARATION", "CONFIRM_TASK", "PREPARATION_READY"].includes(step)) {
    stats(review, [["Dataset", prep.original_filename], ["Snapshot", prep.prepared.dataset.snapshot_id.slice(0, 8)], ["Rows", prep.prepared.row_count]]);
    const list = node("ul", undefined, "compact");
    for (const line of [
      `Time: ${prep.confirmed_mapping.timestamp_column}; load: ${prep.confirmed_mapping.target_column}`,
      `Temperature: ${prep.capabilities.has_temperature ? "present" : "absent"}; other features: ${prep.capabilities.available_other_features.join(", ") || "none"}`,
      `Range: ${prep.prepared.time_start} — ${prep.prepared.time_end}; frequency: ${prep.prepared.dataset.frequency}`,
      `Applied: ${prep.prepared.applied_transformations.join("; ")}`,
      `Forecast: Δ=${prep.task_draft.delta} ${prep.task_draft.time_unit}, H=${prep.task_draft.horizon}, ${prep.task_draft.output.representation}, objective ${prep.task_draft.objective_id}`,
      prep.task_draft.metric_spec?.kind === "weighted"
        ? `Weighted MAE: [${prep.task_draft.metric_spec.time_range.start_local}, ${prep.task_draft.metric_spec.time_range.end_local}) target local time in ${prep.task_draft.metric_spec.timezone_name}; weight ${prep.task_draft.metric_spec.time_range.weight}, otherwise 1`
        : prep.task_draft.metric_spec?.kind === "asymmetric"
          ? `Asymmetric MAE: overestimation weight ${prep.task_draft.metric_spec.over_weight}; underestimation weight ${prep.task_draft.metric_spec.under_weight}; exact matches contribute zero`
          : "Evaluation metric uses standard, unweighted scoring",
      `Split: ${prep.task_draft.train_fraction}/${prep.task_draft.validation_fraction}/${prep.task_draft.test_fraction}; seed ${prep.task_draft.seed}`,
      `Temperature policy: ${prep.task_draft.temperature_policy?.kind || "none"}`
    ]) list.append(node("li", line));
    review.append(list);
    warnings(review, prep.quality?.warnings);
  } else review.append(node("div", "The complete preparation review will appear here.", "empty"));
  if (step === "REVIEW_PREPARATION") addButton("review-action", "Validate final review", () => action("validate-review"));
  if (step === "CONFIRM_TASK") addButton("review-action", "Confirm and freeze task", () => action("confirm-task"));
  if (step === "PREPARATION_READY") addButton("review-action", "Continue to Training & Evaluation", () => action("continue"));
}
$("dataset-file").onchange = event => { file = event.target.files[0]; $("upload-state").textContent = file?.name || ""; };
const drop = $("drop-zone");
drop.ondragover = event => { event.preventDefault(); drop.classList.add("drag"); };
drop.ondragleave = () => drop.classList.remove("drag");
drop.ondrop = event => { event.preventDefault(); drop.classList.remove("drag"); file = event.dataTransfer.files[0]; $("upload-state").textContent = file?.name || ""; };
$("upload-button").onclick = () => work(async () => {
  if (!file) throw new Error("Choose a CSV or Parquet file first.");
  const form = new FormData();
  form.append("expected_version", prep.version);
  form.append("action_id", crypto.randomUUID());
  form.append("file", file);
  await request(`/tasks/${taskId}/preparation/upload`, "POST", form, true);
});
$("new-task").onclick = () => work(async () => {
  const task = await request("/tasks", "POST");
  taskId = task.task_id;
  localStorage.setItem("iforecast-task-id", taskId);
  assistantAvailable = (await request(`/tasks/${taskId}/preparation/assistant/status`)).available;
});
$("open-task").onclick = () => work(async () => {
  const entered = prompt("Enter a task ID to open:");
  if (!entered) return;
  await request(`/tasks/${encodeURIComponent(entered)}`);
  taskId = entered;
  localStorage.setItem("iforecast-task-id", taskId);
  assistantAvailable = (await request(`/tasks/${taskId}/preparation/assistant/status`)).available;
});
$("chat-form").onsubmit = event => {
  event.preventDefault();
  const text = $("chat-input").value.trim();
  if (!text) return;
  work(async () => {
    if (viewStage === "deployment") {
      if (!deploymentState?.latest || deploymentState.selectedSession?.session_id !== deploymentState.latest.session_id)
        throw new Error("Select the latest deployment session to chat with Task Manager.");
      const requestedTask = taskId;
      const session = deploymentState.latest;
      const answer = await request(`/tasks/${requestedTask}/deployment/chat`, "POST", {
        session_id: session.session_id, expected_version: session.version, text
      });
      if (taskId !== requestedTask || viewStage !== "deployment" ||
          selectedDeploymentSessionId !== session.session_id) return;
      if (answer.sensitivity) sensitivityResult = answer.sensitivity;
    } else if (viewStage === "training" && opt) {
      await request(`/tasks/${taskId}/optimization/chat`, "POST", {expected_version: opt.session.version, text});
    } else {
      await request(`/tasks/${taskId}/preparation/chat`, "POST", {expected_version: prep.version, text});
    }
    $("chat-input").value = "";
  });
};
if (!window.IFORECAST_PAPER_DEMO) work(async () => {
  if (taskId) {
    try { await request(`/tasks/${taskId}`); }
    catch { taskId = null; }
  }
  if (!taskId) {
    const task = await request("/tasks", "POST");
    taskId = task.task_id;
    localStorage.setItem("iforecast-task-id", taskId);
  }
  assistantAvailable = (await request(`/tasks/${taskId}/preparation/assistant/status`)).available;
});
