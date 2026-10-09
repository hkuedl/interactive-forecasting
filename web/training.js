/* Deterministic Training & Evaluation panel over persisted run/guidance APIs. */
function trainingAction(name) {
  return request(`/tasks/${taskId}/optimization/${name}`, "POST", {
    expected_version: opt.session.version
  });
}
function guidanceLabel(command) {
  const families = command.families?.join(", ") || "";
  if (command.operation === "allocate_family_trials")
    return `Next batch: ${Object.entries(command.allocation || {}).map(([family, count]) => `${count} ${family}`).join(", ")}`;
  if (command.operation === "prefer_family") return `Prioritize ${families} in the next batch`;
  if (command.operation === "exclude_family") return `Exclude ${families} from later search`;
  if (command.operation === "restrict_families") return `Explore only ${families} in later search`;
  if (command.operation === "narrow_parameter")
    return `Narrow ${command.parameter} to ${command.low}–${command.high}`;
  if (command.operation === "restrict_choices")
    return `Limit ${command.parameter} to ${(command.choices || []).join(", ")}`;
  if (command.operation === "fix_parameter")
    return `Fix ${command.parameter} at ${command.value}`;
  if (command.operation === "force_feature" || command.operation === "disable_feature")
    return `${command.operation === "force_feature" ? "Require" : "Disable"} ${command.parameter}`;
  if (command.operation === "enqueue_candidate")
    return `Try a ${command.candidate?.family || "specified"} candidate in the next batch`;
  if (command.operation === "local_refinement") return "Refine around a completed candidate";
  return command.operation.replaceAll("_", " ");
}
function tableView(parent, headers, rows) {
  const table = node("table");
  const header = node("tr");
  for (const label of headers) header.append(node("th", label));
  table.append(header);
  for (const row of rows) {
    const item = node("tr");
    for (const value of row) item.append(node("td", value ?? "—"));
    table.append(item);
  }
  parent.append(table);
}
function comparisonChart(parent, title, points) {
  if (!points.length) return;
  parent.append(node("div", title, "chart-title"));
  const values = points.flatMap(p => [p.truth, p.prediction]);
  const low = Math.min(...values), high = Math.max(...values), span = high - low || 1;
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", "0 0 600 190");
  svg.setAttribute("class", "chart");
  svg.setAttribute("role", "img");
  svg.setAttribute("aria-label", title);
  for (const [fieldName, color] of [["truth", "#2d648f"], ["prediction", "#c77447"]]) {
    const path = document.createElementNS("http://www.w3.org/2000/svg", "polyline");
    path.setAttribute("fill", "none");
    path.setAttribute("stroke", color);
    path.setAttribute("stroke-width", "2");
    path.setAttribute("points", points.map((p, i) =>
      `${20 + i * 560 / Math.max(1, points.length - 1)},${168 - (p[fieldName] - low) * 145 / span}`
    ).join(" "));
    svg.append(path);
  }
  parent.append(svg, node("small", "Blue: actual load · Orange: predicted load", "meta"));
}
function renderTraining() {
  const setup = clear("run-setup-content");
  const summary = clear("optimization-summary");
  const controls = clear("optimization-controls");
  const progress = clear("optimization-progress");
  const family = clear("optimization-family");
  const guidance = clear("optimization-guidance");
  const diagnostics = clear("optimization-diagnostics");
  const trials = clear("optimization-trials");
  clear("optimization-trial-detail");
  if (!opt) {
    if (setupInfo) {
      stats(setup, [
        ["Frozen definition", setupInfo.definition?.definition_id?.slice(0, 8) || "—"],
        ["Protocol", setupInfo.protocol?.protocol_version || "—"],
        ["Eligible families", setupInfo.effective_search_space?.families?.length || 0]
      ]);
      setup.append(node("p", "The source dataset, chronological split, and effective search space are frozen. Supply the remaining run protocol explicitly."));
    }
    const grid = node("div", undefined, "field-grid");
    const mode = field(grid, "Optimization mode", "text", "vanilla_bo", [
      ["Vanilla BO", "vanilla_bo"], ["LLM-guided", "llm_guided"],
      ["Human + LLM-guided", "human_llm_guided"]
    ]);
    setup.append(grid);
    const label = node("label", "Explicit run protocol JSON", "field");
    const input = node("textarea");
    input.rows = 12;
    input.className = "config-editor";
    input.placeholder = '{"spec_id":"...","backend":{...},"schedule":{...},"templates":{...},"metric":{...},"experiment_seed":0}';
    input.value = localStorage.getItem(`iforecast-run-setup-${taskId}`) || "";
    input.oninput = () => localStorage.setItem(`iforecast-run-setup-${taskId}`, input.value);
    label.append(input);
    setup.append(label, node("p", "No manuscript settings are supplied by the app. A complete candidate template and origin schedule are required for the frozen task."));
    addButton("run-setup-content", "Create search run", async () => {
      const payload = JSON.parse(input.value);
      payload.mode = mode.value;
      payload.pause_at_boundary = mode.value === "human_llm_guided";
      await request(`/tasks/${taskId}/optimization/runs`, "POST", payload);
    });
    return;
  }
  const state = opt.session, view = opt.visualizations;
  stats(setup, [
    ["Mode", state.mode], ["Run ID", state.run_id.slice(0, 8)],
    ["Phase", state.phase.replaceAll("_", " ")]
  ]);
  if (state.last_error) warnings(setup, [state.last_error]);
  stats(summary, [
    ["Current round", view.current_round],
    ["Completed trials", view.completed_trials],
    ["Best validation objective", view.best_objective ?? "—"],
    ["Best family", view.best_family || "—"],
    ["Remaining budget", view.remaining_budget],
    ["Stopping", view.stopping_status]
  ]);
  if (["initial_exploration", "boundary", "executing"].includes(state.phase)) {
    addButton("optimization-controls", state.phase === "executing" ? "Recover / resume round" : "Run next round", () => trainingAction("advance"));
  }
  if (state.phase === "waiting_for_user") {
    controls.append(node("p", "Review results, discuss the next batch, or continue when ready."));
    addButton("optimization-controls", "Continue next round", () => trainingAction("resume"));
  }
  if (!["completed", "failed", "cancelled"].includes(state.phase)) {
    addButton("optimization-controls", "Cancel search", () => trainingAction("cancel"), "secondary");
  }
  const scored = view.progress.filter(p => p.objective !== null);
  if (scored.length) {
    chart(progress, "Validation objective by trial", scored.map(p => [p.trial_number, p.objective]), "#2d648f");
    chart(progress, "Best-so-far validation objective", scored.map(p => [p.trial_number, p.best_so_far]), "#569977");
  } else progress.append(node("p", "No completed validation trials yet."));
  if (view.families.length) tableView(family,
    ["Family", "Trials", "Completed", "Failed", "Best validation objective"],
    view.families.map(p => [p.family, p.trials, p.completed, p.failed, p.best_objective])
  );
  guidance.append(node("p", `Effective search space: ${view.effective_space_version} · ${view.effective_space_id}`));
  if (state.proposed_plan) {
    const plan = state.proposed_plan;
    guidance.append(node("h3", "Proposed next-round plan"));
    guidance.append(node("p", plan.rationale));
    if (plan.commands.length) {
      const list = node("ul");
      for (const command of plan.commands) list.append(node("li", guidanceLabel(command)));
      guidance.append(list);
    } else guidance.append(node("p", "No search intervention; continue with the current search space."));
    if (plan.persistent_approval === "pending") {
      guidance.append(node("p", "A proposed ongoing search-space restriction needs your decision before the next batch."));
      addButton("optimization-guidance", "Approve ongoing restriction", () => trainingAction("plan/approve"), "secondary");
      addButton("optimization-guidance", "Discard ongoing restriction", () => trainingAction("plan/discard"), "secondary");
    } else if (plan.persistent_approval !== "none") {
      guidance.append(node("p", `Ongoing restriction: ${plan.persistent_approval}.`));
    }
  }
  if (view.guidance.length) tableView(guidance,
    ["Round", "Source", "Action", "Status", "Space after"],
    view.guidance.map(g => [g.round_number, g.source, g.commands.map(c => c.operation).join(", "), g.status, g.space_after?.slice(0, 8) || "—"])
  );
  for (const item of view.guidance) {
    if (item.original_text || item.rationale || item.error) guidance.append(
      node("p", `Round ${item.round_number} ${item.source}: ${item.original_text || item.rationale || ""} ${item.error || ""}`, "meta")
    );
  }
  if (state.mode === "human_llm_guided" && ["boundary", "waiting_for_user"].includes(state.phase)) {
    const grid = node("div", undefined, "field-grid");
    const operation = field(grid, "Structured action", "text", "prefer_family", [
      ["Prefer family", "prefer_family"], ["Exclude family", "exclude_family"],
      ["Restrict to family", "restrict_families"], ["Allocate family trials", "allocate_family_trials"]
    ]);
    const familyInput = field(grid, "Family", "text", "Linear", [
      "Linear", "SVR", "MLP", "XGBoost", "LSTM", "GRU", "CNN"
    ].map(name => [name, name]));
    const count = field(grid, "Allocation count", "number", 1);
    guidance.append(grid);
    const label = node("label", "Advanced typed guidance JSON (optional array)", "field");
    const input = node("textarea");
    input.rows = 5;
    input.className = "config-editor";
    input.placeholder = '[{"operation":"narrow_parameter","parameter":"MLP.hidden_size","low":4,"high":16}]';
    input.value = state.guidance_draft ? JSON.stringify(state.guidance_draft.commands, null, 2) : "";
    label.append(input);
    guidance.append(label);
    addButton("optimization-guidance", "Save guidance draft", async () => {
      const selected = operation.value, name = familyInput.value;
      const command = selected === "allocate_family_trials"
        ? {operation: selected, allocation: {[name]: Number(count.value)}}
        : {operation: selected, families: [name]};
      const commands = input.value.trim() ? JSON.parse(input.value) : [command];
      await request(`/tasks/${taskId}/optimization/guidance`, "PUT", {
        expected_version: state.version, commands
      });
    });
    if (state.guidance_draft) {
      guidance.append(node("p", "Draft is unconfirmed. Confirm it before continuing."));
      if (state.proposed_plan) guidance.append(node("p", "Confirming this workspace draft replaces the proposed next-round plan."));
      addButton("optimization-guidance", "Confirm structured guidance", () => trainingAction("guidance/confirm"), "secondary");
      addButton("optimization-guidance", "Clear draft", () => trainingAction("guidance/clear"), "secondary");
    }
  } else if (state.mode === "vanilla_bo") {
    guidance.append(node("p", "Vanilla BO does not accept Model Manager or human search guidance."));
  } else {
    guidance.append(node("p", "Model Manager guidance is validated between rounds. Human search guidance is disabled."));
  }
  if (view.feature_status === "available") tableView(diagnostics,
    ["Feature choice", "Value", "Trials", "Mean validation objective"],
    view.feature_performance.map(p => [p.dimension, p.choice, p.trials, p.mean_objective])
  );
  else diagnostics.append(node("p", "Feature/performance comparison: insufficient data."));
  if (view.influence_status === "available") tableView(diagnostics,
    ["Parameter", "Method", "Trials", "Score"],
    view.parameter_influence.map(p => [p.parameter, p.method, p.sample_size, p.score.toFixed(3)])
  );
  else diagnostics.append(node("p", "Hyperparameter influence: insufficient data."));
  if (view.trial_catalog.length) {
    const table = node("table");
    const head = node("tr");
    for (const label of ["Trial", "Family", "Status", "Objective", "Detail"]) head.append(node("th", label));
    table.append(head);
    for (const trial of view.trial_catalog) {
      const row = node("tr");
      for (const value of [trial.number, trial.family, trial.status, trial.objective ?? "—"])
        row.append(node("td", value));
      const cell = node("td"), button = node("button", "Inspect", "secondary");
      button.type = "button";
      button.onclick = () => work(async () => { selectedTrialId = trial.trial_id; });
      cell.append(button);
      row.append(cell);
      table.append(row);
    }
    trials.append(table);
    if (!selectedTrialId) selectedTrialId = view.trial_catalog[0].trial_id;
    void loadTrialDetail();
  } else trials.append(node("p", "No trials have run yet."));
}
async function loadTrialDetail() {
  if (!selectedTrialId || !opt) return;
  try {
    const detail = await request(`/tasks/${taskId}/optimization/trials/${selectedTrialId}`);
    const target = clear("optimization-trial-detail");
    const trial = detail.trial;
    stats(target, [
      ["Family", trial.request.family], ["Status", trial.status],
      ["Validation objective", trial.objective ?? "—"],
      ["Duration (s)", detail.duration_seconds.toFixed(2)]
    ]);
    target.append(node("p", `Trial ${trial.number} · round ${trial.round_number} · source ${trial.request.source} · space ${trial.space_id}`, "meta"));
    if (trial.failure_message) warnings(target, [trial.failure_message]);
    const config = node("pre");
    config.className = "config-view";
    config.textContent = JSON.stringify({
      feature_recipe: trial.candidate?.candidate?.features,
      hyperparameters: trial.candidate?.candidate?.hyperparameters,
      validation_metrics: trial.metrics,
      artifact_references: detail.artifact_references
    }, null, 2);
    target.append(config);
    if (detail.training_loss_curve?.length)
      chart(target, "Training loss by epoch", detail.training_loss_curve.map((v, i) => [i + 1, v]), "#2d648f");
    else target.append(node("p", "Epoch-level training loss unavailable for this model."));
    if (detail.validation_loss_curve?.length)
      chart(target, "Validation loss by epoch", detail.validation_loss_curve.map((v, i) => [i + 1, v]), "#569977");
    else target.append(node("p", "Epoch-level validation loss unavailable for this model."));
    if (detail.prediction_status === "available")
      comparisonChart(target, `Validation actual versus predicted load${detail.representative_quantile ? ` (q=${detail.representative_quantile})` : ""}`, detail.prediction_points);
    else target.append(node("p", "Validation prediction plot unavailable for this trial."));
  } catch (error) {
    $("notice").textContent = error.message;
  }
}
$("preparation-nav").onclick = () => { viewStage = "preparation"; work(async () => {}); };
$("training-nav").onclick = () => { viewStage = "training"; work(async () => {}); };
