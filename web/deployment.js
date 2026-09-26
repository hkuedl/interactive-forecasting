/* Deployment UI: backend records are authoritative; local state is only view selection. */
let deploymentState = null;
let selectedDeploymentSessionId = null;
let selectedDeploymentVersionId = null;
let sensitivityResult = null;
let deploymentLoadToken = 0;

async function optionalDeployment(path, absentText) {
  try { return await request(path); }
  catch (error) {
    if (error.message.includes(absentText)) return null;
    throw error;
  }
}
function deploymentAdjustmentDescription(item, zone = "UTC") {
  const labels = {
    time_scaling: "Time-based scaling", load_scaling: "Load-based scaling",
    external_scaling: "External-variable scaling", manual_override: "Manual override"
  };
  const name = labels[item.adjustment_type] || "Adjustment";
  if (item.adjustment_type === "manual_override")
    return name + " · " + (item.manual_replacements?.length || 0) + " target value(s)";
  const percent = item.lambda_value == null ? ""
    : " " + (item.lambda_value >= 0 ? "+" : "") +
      deploymentValue(item.lambda_value * 100, 0) + "%";
  const time = item.start_at ? " from " + deploymentDate(item.start_at,
    {hour: "2-digit", minute: "2-digit", hourCycle: "h23", timeZone: zone}) : "";
  const threshold = item.threshold == null ? "" :
    " when " + (item.external_variable || "load") +
    (item.comparison === "lt" ? " < " : " > ") + deploymentValue(item.threshold);
  return name + percent + time + threshold;
}
async function loadDeployment() {
  const token = ++deploymentLoadToken;
  const requestedTask = taskId;
  const requestedSession = selectedDeploymentSessionId;
  const requestedVersion = selectedDeploymentVersionId;
  const requestedStage = typeof viewStage === "undefined" ? null : viewStage;
  const current = () => token === deploymentLoadToken && taskId === requestedTask &&
    selectedDeploymentSessionId === requestedSession &&
    selectedDeploymentVersionId === requestedVersion &&
    (typeof viewStage === "undefined" ? null : viewStage) === requestedStage;
  try {
    const base = "/tasks/" + requestedTask + "/deployment";
    const sessions = await request(base + "/sessions");
    const latest = sessions.at(-1) || null;
    const selectedSession = sessions.find(x => x.session_id === requestedSession) || latest;
    const state = {taskId: requestedTask, sessions, latest, selectedSession, forecast: null, original: null,
      versions: [], adjustments: [], reference: null, variables: [], inspected: null};
    if (selectedSession?.forecast_id) {
      const prefix = base + "/forecasts/" + selectedSession.forecast_id;
      const record = await request(prefix);
      state.forecast = record.forecast;
      state.original = record.original;
      state.versions = await request(prefix + "/versions");
      state.adjustments = await request(prefix + "/adjustments");
      state.reference = await optionalDeployment(prefix + "/references", "not been generated");
      state.variables = await request(prefix + "/sensitivity/variables");
      state.inspected = state.versions.find(x => x.version_id === requestedVersion)
        || state.versions.find(x => x.version_id === selectedSession.current_version_id)
        || state.original;
    }
    if (!current()) return false;
    selectedDeploymentSessionId = selectedSession?.session_id || null;
    selectedDeploymentVersionId = state.inspected?.version_id || null;
    if (!selectedSession?.forecast_id || sensitivityResult?.forecast_id !== selectedSession.forecast_id)
      sensitivityResult = null;
    deploymentState = state;
    return true;
  } catch (error) {
    if (current()) throw error;
    return false;
  }
}
async function startDeploymentSession(session = null) {
  const requestedTask = taskId;
  const requestedSession = selectedDeploymentSessionId;
  const requestedStage = typeof viewStage === "undefined" ? null : viewStage;
  const created = await request("/tasks/" + requestedTask + "/deployment/start" +
    (session ? "?new_session=true" : ""), "POST",
    session ? {session_id: session.session_id, expected_version: session.version} : undefined);
  if (taskId !== requestedTask || selectedDeploymentSessionId !== requestedSession ||
      (typeof viewStage === "undefined" ? null : viewStage) !== requestedStage) return;
  selectedDeploymentSessionId = created.session_id;
  selectedDeploymentVersionId = null;
  sensitivityResult = null;
}
function deploymentButton(parent, label, callback, kind = "secondary") {
  const button = node("button", label, kind);
  button.type = "button";
  button.onclick = () => work(callback);
  parent.append(button);
}
function deploymentEmpty(parent, message) { parent.append(node("p", message, "empty")); }
function deploymentValue(value, digits = 1) {
  return Number(value).toLocaleString("en-US", {maximumFractionDigits: digits,
    minimumFractionDigits: digits});
}
function deploymentDate(value, options = {}) {
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return String(value || "—");
  return new Intl.DateTimeFormat("en-US", options).format(parsed);
}
function deploymentPlot(parent, title, lines, band = null, options = {}) {
  const available = lines.filter(line => line.values?.length);
  if (!available.length) return;
  parent.append(node("div", title, "chart-title"));
  const values = available.flatMap(line => line.values);
  if (band) values.push(...band.lower, ...band.upper);
  const min = options.yDomain?.[0] ?? Math.min(...values);
  const max = options.yDomain?.[1] ?? Math.max(...values);
  const padding = (max - min || Math.max(Math.abs(max) * 0.05, 1)) * 0.08;
  const low = min - padding, high = max + padding;
  const frame = {left: 76, right: 630, top: 18, bottom: 186};
  const x = (index, count) => count === 1
    ? (frame.left + frame.right) / 2
    : frame.left + index * (frame.right - frame.left) / (count - 1);
  const y = value => frame.bottom - (value - low) * (frame.bottom - frame.top) / (high - low);
  const ns = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(ns, "svg");
  svg.setAttribute("viewBox", "0 0 660 250");
  svg.setAttribute("class", "chart deployment-chart");
  svg.setAttribute("role", "img");
  svg.setAttribute("aria-label", title + ". " + available.map(line => line.label).join(", "));
  const add = (tag, attributes, label) => {
    const item = document.createElementNS(ns, tag);
    for (const [key, value] of Object.entries(attributes)) item.setAttribute(key, value);
    if (label !== undefined) item.textContent = String(label);
    svg.append(item);
    return item;
  };
  for (const fraction of [0, 0.5, 1]) {
    const position = frame.bottom - fraction * (frame.bottom - frame.top);
    add("line", {x1: frame.left, y1: position, x2: frame.right, y2: position,
      stroke: "#e4ebf1", "stroke-width": 1});
    add("text", {x: frame.left - 9, y: position + 4, "text-anchor": "end",
      fill: "#63788a", "font-size": 12}, deploymentValue(low + fraction * (high - low), 1));
  }
  add("line", {x1: frame.left, y1: frame.bottom, x2: frame.right, y2: frame.bottom,
    stroke: "#91a5b4", "stroke-width": 1.2});
  const count = options.times?.length || available[0].values.length;
  const tickIndices = count === 1 ? [0] :
    [...new Set([0, 0.25, 0.5, 0.75, 1].map(fraction => Math.round(fraction * (count - 1))))];
  for (const index of tickIndices) {
    const position = x(index, count);
    add("line", {x1: position, y1: frame.bottom, x2: position, y2: frame.bottom + 5,
      stroke: "#91a5b4", "stroke-width": 1});
    let label = String(index);
    if (options.times?.[index]) {
      label = deploymentDate(options.times[index], {hour: "2-digit", minute: "2-digit",
        hourCycle: "h23", timeZone: options.timeZone || "UTC"});
    }
    add("text", {x: position, y: frame.bottom + 21, "text-anchor": "middle",
      fill: "#63788a", "font-size": 12}, label);
  }
  add("text", {x: (frame.left + frame.right) / 2, y: 240, "text-anchor": "middle",
    fill: "#526d81", "font-size": 12}, options.xLabel || "Time of day");
  add("text", {x: 15, y: (frame.top + frame.bottom) / 2,
    transform: "rotate(-90 15 " + (frame.top + frame.bottom) / 2 + ")",
    "text-anchor": "middle", fill: "#526d81", "font-size": 12},
    options.yLabel || "Load");
  if (band?.lower.length && band.lower.length === band.upper.length) {
    const upper = band.upper.map((value, index) => x(index, band.upper.length) + "," + y(value));
    const lower = band.lower.map((value, index) => x(index, band.lower.length) + "," + y(value)).reverse();
    add("polygon", {points: [...upper, ...lower].join(" "), fill: "#dfeef7"});
  }
  for (const line of available) {
    const points = line.values.map((value, index) => x(index, line.values.length) + "," + y(value));
    if (line.values.length > 1)
      add("polyline", {fill: "none", stroke: line.color, "stroke-width": 3,
        "stroke-linecap": "round", "stroke-linejoin": "round",
        "stroke-dasharray": line.dash || "none", points: points.join(" ")});
    if (line.values.length === 1 || line.highlightIndex !== undefined) {
      const index = line.highlightIndex ?? 0;
      if (index >= 0 && index < line.values.length)
        add("circle", {cx: x(index, line.values.length), cy: y(line.values[index]), r: 5,
          fill: line.color, stroke: "#fff", "stroke-width": 1.5});
    }
  }
  parent.append(svg);
  const legend = node("div", undefined, "chart-legend");
  for (const line of available) {
    const item = node("span", undefined, "chart-legend-item");
    const swatch = node("span", undefined, "chart-swatch");
    swatch.style.backgroundColor = line.color;
    item.append(swatch, node("span", line.label));
    legend.append(item);
  }
  if (band) legend.append(node("span", "Pale blue: outer quantiles", "chart-legend-item"));
  parent.append(legend);
}
function predictionDisplay(prediction) {
  if (!prediction.levels) return {
    central: prediction.values, lower: null, upper: null, label: "Point forecast",
    cells: prediction.values.map(value => deploymentValue(value))
  };
  const levels = prediction.levels;
  const center = levels.reduce((best, value, i) =>
    Math.abs(value - 0.5) < Math.abs(levels[best] - 0.5) ? i : best, 0);
  return {
    central: prediction.values.map(row => row[center]),
    lower: prediction.values.map(row => row[0]),
    upper: prediction.values.map(row => row.at(-1)),
    label: "Central q=" + levels[center],
    cells: prediction.values.map(row =>
      row.map((value, i) => "q" + levels[i] + "=" +
        Number(Number(value).toFixed(1)).toLocaleString("en-US")).join(" · "))
  };
}
function renderDeploymentSummary(state) {
  const summary = clear("deployment-summary"), setup = clear("deployment-setup");
  if (!state.latest) {
    deploymentEmpty(summary, "No deployment session. Start one after search completes.");
    deploymentButton(setup, "Start deployment session", () => startDeploymentSession(), "primary");
    return;
  }
  const session = state.selectedSession;
  const latest = session.session_id === state.latest.session_id;
  const zone = state.forecast?.origin?.timezone_name || "UTC";
  if (state.sessions.length > 1) {
    const selector = field(summary, "Forecast session", "text", session.session_id,
      [...state.sessions].reverse().map((item, index) => [
        deploymentDate(item.created_at, {dateStyle: "medium", timeStyle: "short"}) +
          " · " + (index === 0 ? "Latest" : "Earlier"), item.session_id
      ]));
    selector.onchange = () => work(async () => {
      selectedDeploymentSessionId = selector.value;
      selectedDeploymentVersionId = null;
      sensitivityResult = null;
    });
  }
  if (!latest) summary.append(node("p",
    "Prior session · read-only. Switch to the latest session to take action.", "meta"));
  if (state.forecast) {
    const current = state.versions.find(item => item.version_id === session.current_version_id)
      || state.original;
    const first = state.forecast.target_timestamps[0];
    const last = state.forecast.target_timestamps.at(-1);
    const targetDate = deploymentDate(first, {dateStyle: "long", timeZone: zone});
    const time = value => deploymentDate(value, {hour: "2-digit", minute: "2-digit",
      hourCycle: "h23", timeZone: zone});
    const period = first === last ? time(first) + " " + zone
      : time(first) + "–" + time(last) + " " + zone;
    const applied = state.adjustments.find(item => item.applied_version_id === current.version_id);
    const adjustment = applied ? deploymentAdjustmentDescription(applied, zone) : "None";
    stats(summary, [
      ["Dataset", state.presentation?.dataset_label || "Prepared dataset"],
      ["Target date", targetDate],
      ["Forecast period", period],
      ["Model", state.presentation?.model_label || "Selected saved model"],
      ["Forecast type", state.forecast.prediction_representation === "quantile" ?
        "Quantile" : "Point"],
      ["Current version", current.version_number ? "Adjusted v" + current.version_number
        : "Original v0"],
      ["Adjustment", adjustment]
    ]);
  } else {
    summary.append(node("p", "Ready to prepare the next forecast.", "meta"));
  }
  const technical = node("details", undefined, "deployment-technical");
  technical.append(node("summary", "Technical details"));
  const details = node("div", undefined, "deployment-technical-grid");
  details.append(node("p", "Session " + session.session_id));
  details.append(node("p", "Selected trial " + session.selected_trial_id));
  details.append(node("p", "Model artifact " + session.model_artifact.uri));
  details.append(node("p", "Status " + session.state.replaceAll("_", " ")));
  technical.append(details);
  summary.append(technical);
  if (latest && session.forecast_id && !session.pending_adjustment_id) {
    deploymentButton(setup, "Start another session", () => startDeploymentSession(session));
    if (session.state !== "COMPLETED")
      deploymentButton(setup, "Complete session", () =>
        request("/tasks/" + taskId + "/deployment/complete", "POST",
          {session_id: session.session_id, expected_version: session.version}));
  }
  if (!latest || session.forecast_id) return;
  const upload = node("input");
  upload.type = "file";
  upload.accept = ".csv,.parquet";
  const uploadLabel = node("label", "New observations (frozen source columns)", "field");
  uploadLabel.append(upload);
  setup.append(uploadLabel);
  deploymentButton(setup, "Upload observations", async () => {
    const file = upload.files?.[0];
    if (!file) throw new Error("Choose a CSV or Parquet file.");
    const form = new FormData();
    form.append("session_id", session.session_id);
    form.append("expected_version", session.version);
    form.append("file", file);
    await request("/tasks/" + taskId + "/deployment/upload", "POST", form, true);
  }, "primary");
  if (session.raw_upload) {
    const label = node("label", "Future auxiliaries JSON (include availability metadata)", "field");
    const input = node("textarea");
    input.rows = 5;
    input.className = "config-editor";
    input.value = JSON.stringify(session.future_auxiliaries || [], null, 2);
    label.append(input);
    setup.append(label);
    deploymentButton(setup, "Validate deployment input", () =>
      request("/tasks/" + taskId + "/deployment/validate", "POST", {
        session_id: session.session_id, expected_version: session.version, future_auxiliaries: JSON.parse(input.value)
      }), "primary");
  }
  if (session.validation) {
    setup.append(node("p", session.validation.ready ? "Backend validation passed." :
      "Input unavailable; correct data and validate again.",
      session.validation.ready ? "meta" : "warning"));
    for (const item of session.validation.history_deficits || [])
      setup.append(node("p", item.column + ": " + item.required_steps +
        " history steps required, " + item.available_steps + " available.", "warning"));
    for (const item of session.validation.missing_auxiliaries || [])
      setup.append(node("p", "Missing future auxiliary: " + item, "warning"));
  }
  if (session.state === "READY_TO_FORECAST")
    deploymentButton(setup, "Generate original forecast", () =>
      request("/tasks/" + taskId + "/deployment/generate", "POST",
        {session_id: session.session_id, expected_version: session.version}), "primary");
}
function renderDeploymentForecast(state) {
  const parent = clear("deployment-forecast");
  if (!state.forecast)
    return deploymentEmpty(parent, "Generate a forecast to see saved-model output.");
  const current = state.inspected;
  const shown = predictionDisplay(current.prediction);
  const baseline = predictionDisplay(state.original.prediction);
  const adjusted = current.version_number !== 0;
  const zone = state.forecast.origin.timezone_name || "UTC";
  const target = current.prediction.keys[0]?.target;
  const targetTime = target ? deploymentDate(target, {hour: "2-digit", minute: "2-digit",
    hourCycle: "h23", timeZone: zone}) : "target time";
  const headline = node("div", undefined, "forecast-highlight");
  headline.append(node("span", (adjusted ? "Adjusted forecast · " : "Saved-model forecast · ") +
    targetTime + " " + zone, "meta"));
  headline.append(node("strong", deploymentValue(shown.central[0]), "forecast-value"));
  if (adjusted) headline.append(node("span",
    "Original v0: " + deploymentValue(baseline.central[0]) +
    " · current v" + current.version_number + ": " + deploymentValue(shown.central[0]),
    "forecast-comparison"));
  parent.append(headline);
  parent.append(node("p", "Inspecting v" + current.version_number +
    (adjusted ? " · orange adjusted; blue original v0." : " · original model forecast."),
    "forecast-context"));
  const demo = state.presentation?.day_profile;
  if (demo) {
    parent.append(node("p", demo.disclosure, "forecast-disclosure"));
    const series = [{label: "Original v0 · illustrative day profile",
      values: demo.original_values, color: "#2d648f", highlightIndex: demo.target_index}];
    if (adjusted) series.push({label: "Adjusted v" + current.version_number +
      " · target hour only", values: demo.adjusted_values,
      color: "#c77447", dash: "7 5", highlightIndex: demo.target_index});
    deploymentPlot(parent, "Target-day load profile · illustrative context",
      series, null, {times: demo.timestamps, timeZone: zone,
        yLabel: "Load", xLabel: "Hour of day"});
  } else if (shown.central.length === 1 && state.reference?.d_minus_1?.available) {
    const references = [state.reference.d_minus_1, state.reference.d_minus_7,
      state.reference.d_minus_365].filter(item => item.available);
    parent.append(node("p",
      "The selected model predicts one target hour. Full-day curves below are historical " +
      "context, not additional model forecasts.", "forecast-disclosure"));
    deploymentPlot(parent, "Historical load context around the target day",
      references.map((item, index) => ({
        label: item.label + " · " + item.date,
        values: item.load_profile.map(point => point.value),
        color: ["#2d648f", "#569977", "#8275aa"][index]
      })), null, {times: references[0].load_profile.map(point => point.timestamp),
        timeZone: zone, yLabel: "Load", xLabel: "Hour of day"});
  } else {
    if (shown.central.length === 1)
      parent.append(node("p", "This saved model predicts one target hour, not a full-day curve.",
        "forecast-disclosure"));
    deploymentPlot(parent, "Saved-model forecast by target time", [
      {label: "Inspected v" + current.version_number + ": " + shown.label,
        values: shown.central, color: adjusted ? "#c77447" : "#2d648f"},
      ...(adjusted ? [{label: "Original v0: " + baseline.label,
        values: baseline.central, color: "#2d648f"}] : [])
    ], shown.lower ? {lower: shown.lower, upper: shown.upper} : null,
    {times: current.prediction.keys.map(key => key.target), timeZone: zone,
      yLabel: "Load", xLabel: "Target time"});
  }
  tableView(parent, ["Target time", "Original v0", "Inspected v" + current.version_number],
    current.prediction.keys.map((key, index) =>
      [deploymentDate(key.target, {dateStyle: "medium", timeStyle: "short", timeZone: zone}),
        baseline.cells[index], shown.cells[index]]));
  if (current.prediction.levels)
    parent.append(node("p", "All quantile levels are retained: " +
      current.prediction.levels.join(", ") + ". The line is " + shown.label + ".", "meta"));
}
function renderDeploymentCalendar(state) {
  const parent = clear("deployment-calendar");
  if (!state.forecast) return deploymentEmpty(parent, "Calendar references require a forecast.");
  if (!state.reference) {
    deploymentEmpty(parent, "Reference analysis has not been generated.");
    if (state.selectedSession.session_id === state.latest.session_id)
      deploymentButton(parent, "Analyze reference days", () =>
        request("/tasks/" + taskId + "/deployment/forecasts/" +
          state.forecast.forecast_id + "/references/analyze", "POST", {
            session_id: state.selectedSession.session_id,
            expected_version: state.selectedSession.version
          }), "primary");
    return;
  }
  const zone = state.forecast.origin.timezone_name || "UTC";
  parent.append(node("p", "Reference days for " + state.reference.target_date +
    ". Dates are exact; missing or incomplete days are shown as unavailable."));
  const days = [state.reference.d_minus_1, state.reference.d_minus_7,
    state.reference.d_minus_365];
  const allLoads = days.filter(item => item.available)
    .flatMap(item => item.load_profile.map(point => point.value));
  const scale = allLoads.length ? [Math.min(...allLoads), Math.max(...allLoads)] : undefined;
  const meanings = {"D-1": "Previous day", "D-7": "One week earlier",
    "D-365": "One year earlier"};
  for (const item of days) {
    const section = node("div", undefined, "reference-row");
    section.append(node("strong", item.label + " · " + meanings[item.label] +
      " · " + item.date));
    if (item.available) {
      deploymentPlot(section, item.label + " historical load",
        [{label: item.label + " · " + item.date,
          values: item.load_profile.map(point => point.value), color: "#2d648f"}],
        null, {times: item.load_profile.map(point => point.timestamp),
          timeZone: zone, yDomain: scale, yLabel: "Load", xLabel: "Hour of day"});
      const detail = node("details", undefined, "profile-details");
      detail.append(node("summary", "View hourly values"));
      tableView(detail, ["Time", "Historical load"],
        item.load_profile.map(point => [
          deploymentDate(point.timestamp, {hour: "2-digit", minute: "2-digit",
            hourCycle: "h23", timeZone: zone}), deploymentValue(point.value)
        ]));
      section.append(detail);
    } else section.append(node("p", item.reason || "Reference unavailable.", "warning"));
    parent.append(section);
  }
}
function renderDeploymentWeather(state) {
  const parent = clear("deployment-weather");
  if (!state.forecast) return deploymentEmpty(parent, "Weather analogs require a forecast.");
  if (!state.reference)
    return deploymentEmpty(parent, "Run reference analysis to inspect weather analogs.");
  const analysis = state.reference;
  if (!analysis.weather_available)
    return deploymentEmpty(parent, "Weather analogs unavailable: " +
      (analysis.weather_unavailable_reason || "required weather data absent") + ".");
  const zone = state.forecast.origin.timezone_name || "UTC";
  parent.append(node("p", "Weather-similar historical days, ranked by normalized temperature " +
    "profile distance. Lower is more similar.", "meta"));
  tableView(parent, ["Rank", "Historical day", "Similarity distance"],
    analysis.weather_analogs.map((item, index) =>
      [index + 1, item.date, deploymentValue(item.distance, 4)]));
  deploymentPlot(parent, "Target-day and analog temperature profiles", [
    {label: "Target-day temperature",
      values: analysis.target_weather_profile.map(point => point.value),
      color: "#2d648f"},
    ...analysis.weather_analogs.map((item, index) => ({
      label: item.date + " temperature",
      values: item.weather_profile.map(point => point.value),
      color: ["#c77447", "#569977", "#8275aa"][index % 3]
    }))
  ], null, {times: analysis.target_weather_profile.map(point => point.timestamp),
    timeZone: zone, yLabel: "Temperature", xLabel: "Hour of day"});
  deploymentPlot(parent, "Historical load on weather-analog days",
    analysis.weather_analogs.map((item, index) => ({
      label: item.date + " load", values: item.load_profile.map(point => point.value),
      color: ["#c77447", "#569977", "#8275aa"][index % 3]
    })), null, {times: analysis.weather_analogs[0]?.load_profile.map(point => point.timestamp),
      timeZone: zone, yLabel: "Load", xLabel: "Hour of day"});
  for (const item of analysis.weather_analogs) {
    const detail = node("details", undefined, "profile-details");
    detail.append(node("summary", item.date + " · distance " +
      deploymentValue(item.distance, 4) + " · view hourly values"));
    tableView(detail, ["Time", "Load", "Temperature"],
      item.load_profile.map((point, index) => [
        deploymentDate(point.timestamp, {hour: "2-digit", minute: "2-digit",
          hourCycle: "h23", timeZone: zone}),
        deploymentValue(point.value), deploymentValue(item.weather_profile[index]?.value)
      ]));
    parent.append(detail);
  }

}
function adjustmentForm(parent, state) {
  const session = state.selectedSession, forecast = state.forecast;
  const grid = node("div", undefined, "field-grid");
  const type = field(grid, "Adjustment type", "text", "time_scaling", [
    ["Manual Override", "manual_override"], ["Time-Based Scaling", "time_scaling"],
    ["Load-Based Scaling", "load_scaling"],
    ["External-Variable-Based Scaling", "external_scaling"]
  ]);
  if (forecast.prediction_representation === "quantile")
    [...type.options].find(x => x.value === "load_scaling").disabled = true;
  const params = node("div", undefined, "field-grid");
  const timestamps = forecast.target_timestamps.map(x => [x, x]);
  let controls = {};
  const draw = () => {
    params.replaceChildren();
    controls = {};
    if (type.value === "manual_override") {
      controls.timestamp = field(params, "Target timestamp to replace", "text",
        forecast.target_timestamps[0], timestamps);
      controls.values = field(params, forecast.prediction_representation === "quantile"
        ? "Replacement values (all " + forecast.prediction.levels.length + " quantiles, ordered)"
        : "Replacement load value", "text", "");
    } else {
      controls.lambda = field(params, "Lambda (-0.10 means -10%)", "number", "-0.10");
      controls.lambda.step = "any";
      if (type.value === "time_scaling") {
        controls.start = field(params, "Start time (included)", "text",
          forecast.target_timestamps[0], timestamps);
        controls.end = field(params, "End time (included; optional)", "text", "",
          [["No end limit", ""], ...timestamps]);
      } else {
        controls.comparison = field(params, "Strict comparison", "text", "gt",
          [["> threshold", "gt"], ["< threshold", "lt"]]);
        controls.threshold = field(params, "Threshold", "number", "");
        controls.threshold.step = "any";
        if (type.value === "external_scaling") {
          const names = [...new Set((session.future_auxiliaries || []).map(x => x.column))];
          controls.variable = field(params, "Supplied external variable", "text",
            names[0] || "", names.length ? names.map(x => [x, x]) : [["None supplied", ""]]);
        }
      }
    }
  };
  type.onchange = draw;
  draw();
  parent.append(grid, params);
  if (forecast.prediction_representation === "quantile")
    parent.append(node("p", "Quantile levels stay separate. Load-threshold scaling is unsupported; manual override requires every quantile.", "warning"));
  if (!(session.future_auxiliaries || []).length)
    parent.append(node("p", "External scaling requires supplied origin-available values.", "meta"));
  deploymentButton(parent, "Save typed adjustment draft", async () => {
    const proposal = {adjustment_type: type.value};
    if (type.value === "manual_override") {
      const values = controls.values.value.split(",").map(x => Number(x.trim()));
      const count = forecast.prediction_representation === "quantile"
        ? forecast.prediction.levels.length : 1;
      if (controls.values.value.trim() === "" || values.length !== count ||
          values.some(x => !Number.isFinite(x)))
        throw new Error("Provide exactly " + count + " finite replacement value(s).");
      proposal.manual_replacements = [{timestamp: controls.timestamp.value, values}];
    } else {
      if (controls.lambda.value.trim() === "" ||
          !Number.isFinite(Number(controls.lambda.value)))
        throw new Error("Provide a finite lambda.");
      proposal.lambda_value = Number(controls.lambda.value);
      if (type.value === "time_scaling") {
        proposal.start_at = controls.start.value;
        if (controls.end.value) proposal.end_at = controls.end.value;
      } else {
        if (controls.threshold.value.trim() === "" ||
            !Number.isFinite(Number(controls.threshold.value)))
          throw new Error("Provide a finite threshold.");
        proposal.threshold = Number(controls.threshold.value);
        proposal.comparison = controls.comparison.value;
        if (type.value === "external_scaling")
          proposal.external_variable = controls.variable.value;
      }
    }
    await request("/tasks/" + taskId + "/deployment/forecasts/" +
      forecast.forecast_id + "/adjustments", "POST",
      {session_id: session.session_id, expected_version: session.version, proposal});
  }, "primary");
}
function renderDeploymentAdjustment(state) {
  const parent = clear("deployment-adjustment");
  if (!state.forecast)
    return deploymentEmpty(parent, "Generate a forecast before proposing an adjustment.");
  const session = state.selectedSession;
  const latest = session.session_id === state.latest.session_id;
  const pending = state.adjustments.find(x => x.adjustment_id === session.pending_adjustment_id);
  if (pending) {
    parent.append(node("p", "Draft · " +
      deploymentAdjustmentDescription(pending, state.forecast.origin.timezone_name || "UTC") +
      ". No forecast value has changed.", "warning"));
    const prefix = "/tasks/" + taskId + "/deployment/adjustments/" + pending.adjustment_id;
    if (latest && session.state === "ADJUSTMENT_DRAFT_PENDING")
      deploymentButton(parent, "Validate draft deterministically", () =>
        request(prefix + "/validate", "POST", {session_id: session.session_id, expected_version: session.version}), "primary");
    if (latest && session.state === "WAITING_FOR_USER_CONFIRMATION") {
      parent.append(node("p", "Backend validation passed. Confirm to create a new immutable version, or reject without changing values.", "meta"));
      deploymentButton(parent, "Confirm adjustment", async () => {
        const key = "iforecast-confirm-" + pending.adjustment_id;
        let confirmation = sessionStorage.getItem(key);
        if (!confirmation) {
          confirmation = crypto.randomUUID();
          sessionStorage.setItem(key, confirmation);
        }
        await request(prefix + "/confirm", "POST",
          {session_id: session.session_id, expected_version: session.version, confirmation_id: confirmation});
        sessionStorage.removeItem(key);
        if (taskId === state.taskId && selectedDeploymentSessionId === session.session_id &&
            selectedDeploymentVersionId === state.inspected.version_id)
          selectedDeploymentVersionId = null;
      }, "primary");
    }
    if (latest) deploymentButton(parent, "Reject draft", () =>
      request(prefix + "/reject", "POST", {session_id: session.session_id, expected_version: session.version}));
  } else if (latest && session.state !== "COMPLETED") adjustmentForm(parent, state);
  else deploymentEmpty(parent, latest ? "Session complete; start another session." :
    "Prior session is read-only.");
  if (state.adjustments.length) {
    const versions = Object.fromEntries(state.versions.map(item =>
      [item.version_id, "v" + item.version_number]));
    tableView(parent, ["When", "Adjustment", "Status", "Version"],
      state.adjustments.map(item => [
        deploymentDate(item.created_at, {dateStyle: "medium", timeStyle: "short"}),
        deploymentAdjustmentDescription(item, state.forecast.origin.timezone_name || "UTC"),
        item.status === "applied" ? "Applied" : item.status === "rejected" ?
          "Rejected" : "Draft",
        versions[item.applied_version_id] || "—"
      ]));
  }
  else parent.append(node("p", "No adjustments recorded.", "meta"));
}
function renderDeploymentVersions(state) {
  const parent = clear("deployment-versions");
  if (!state.forecast)
    return deploymentEmpty(parent, "Version history begins with original v0.");
  const byId = Object.fromEntries(state.versions.map(item =>
    [item.version_id, item.version_number]));
  const adjustments = Object.fromEntries(state.adjustments.map(item =>
    [item.adjustment_id, item]));
  const selector = field(parent, "Compare with the original forecast", "text",
    state.inspected.version_id, state.versions.map(item => [
      item.version_number === 0 ? "Original v0" : "Adjusted v" + item.version_number,
      item.version_id
    ]));
  selector.onchange = () => work(async () => { selectedDeploymentVersionId = selector.value; });
  tableView(parent, ["Version", "Change", "When", "Status"],
    state.versions.map(item => [
      item.version_number === 0 ? "Original v0" : "Adjusted v" + item.version_number,
      item.adjustment_id ?
        deploymentAdjustmentDescription(adjustments[item.adjustment_id] ||
          {adjustment_type: "adjustment"}, state.forecast.origin.timezone_name || "UTC")
        : "Saved-model forecast",
      deploymentDate(item.created_at, {dateStyle: "medium", timeStyle: "short"}),
      item.version_id === state.selectedSession.current_version_id ? "Current" : "Earlier"
    ]));
  if (state.inspected.version_number > 0) {
    parent.append(node("p", "Adjusted v" + state.inspected.version_number +
      " follows v" + byId[state.inspected.parent_version_id] +
      "; the original forecast remains unchanged.", "meta"));
    tableView(parent, ["Affected time", "Before", "After"],
      state.inspected.value_changes.map(change => [
        deploymentDate(change.timestamp, {dateStyle: "medium", timeStyle: "short",
          timeZone: state.forecast.origin.timezone_name || "UTC"}),
        change.before.map(value => deploymentValue(value)).join(", "),
        change.after.map(value => deploymentValue(value)).join(", ")
      ]));
  }
}
function renderDeploymentSensitivity(state) {
  const parent = clear("deployment-sensitivity");
  if (!state.forecast)
    return deploymentEmpty(parent, "Sensitivity requires an original forecast.");
  parent.append(node("p", "Explore how a change in an external input affects the saved-model forecast. This analysis does not change any forecast version."));
  if (!state.variables.length)
    return deploymentEmpty(parent, "No numeric external input is used by the frozen recipe.");
  const grid = node("div", undefined, "field-grid");
  const variable = field(grid, "External input", "text", state.variables[0],
    state.variables.map(name => [name.replaceAll("_", " ").replace(/\b\w/g,
      letter => letter.toUpperCase()), name]));
  const kind = field(grid, "Perturbation", "text", "absolute",
    [["Add/subtract units", "absolute"], ["Percentage of input", "percent"]]);
  const amount = field(grid, "Signed amount", "number", "1");
  amount.step = "any";
  parent.append(grid);
  deploymentButton(parent, "Run read-only sensitivity", async () => {
    if (amount.value.trim() === "" || !Number.isFinite(Number(amount.value)))
      throw new Error("Provide a finite signed perturbation.");
    const requestedStage = typeof viewStage === "undefined" ? null : viewStage;
    const result = await request("/tasks/" + state.taskId + "/deployment/forecasts/" +
      state.forecast.forecast_id + "/sensitivity", "POST", {
        base_version_id: state.original.version_id, variable: variable.value,
        perturbation_type: kind.value, value: Number(amount.value)
      });
    if (taskId === state.taskId && deploymentState === state &&
        selectedDeploymentSessionId === state.selectedSession.session_id &&
        (typeof viewStage === "undefined" ? null : viewStage) === requestedStage)
      sensitivityResult = result;
  }, "primary");
  if (!sensitivityResult || sensitivityResult.forecast_id !== state.forecast.forecast_id) {
    parent.append(node("p", "No sensitivity comparison yet. Run the analysis to compare with the original forecast.", "meta"));
    return;
  }
  const base = predictionDisplay(sensitivityResult.baseline_prediction);
  const changed = predictionDisplay(sensitivityResult.perturbed_prediction);
  const label = sensitivityResult.variable.replaceAll("_", " ").replace(/\b\w/g,
    letter => letter.toUpperCase());
  const signed = (sensitivityResult.value >= 0 ? "+" : "") +
    deploymentValue(sensitivityResult.value);
  const unit = sensitivityResult.perturbation_type === "percent" ? "%" : " input units";
  parent.append(node("p", label + " " + signed + unit +
    " · compared with original v0. The saved model is rerun; no forecast version changes.",
    "meta"));
  deploymentPlot(parent, "Original versus sensitivity scenario", [
    {label: "Original " + base.label, values: base.central, color: "#2d648f"},
    {label: "Input changed · " + changed.label, values: changed.central, color: "#c77447"}
  ], changed.lower ? {lower: changed.lower, upper: changed.upper} : null,
  {times: sensitivityResult.baseline_prediction.keys.map(key => key.target),
    timeZone: state.forecast.origin.timezone_name || "UTC",
    yLabel: "Load", xLabel: "Target time"});
  const levels = sensitivityResult.baseline_prediction.levels;
  tableView(parent, ["Target", "Baseline", "Perturbed", "Delta (perturbed - baseline)"],
    sensitivityResult.baseline_prediction.keys.map((key, i) => [
      key.target, base.cells[i], changed.cells[i],
      sensitivityResult.deltas[i].map((value, j) =>
        levels ? "q" + levels[j] + "=" + value : String(value)).join(" · ")
    ]));
}
function renderDeployment() {
  if (!deploymentState) return;
  renderDeploymentSummary(deploymentState);
  renderDeploymentForecast(deploymentState);
  renderDeploymentCalendar(deploymentState);
  renderDeploymentWeather(deploymentState);
  renderDeploymentAdjustment(deploymentState);
  renderDeploymentVersions(deploymentState);
  renderDeploymentSensitivity(deploymentState);
}
$("deployment-nav").onclick = () => { viewStage = "deployment"; work(async () => {}); };
