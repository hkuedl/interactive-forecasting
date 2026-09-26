/* Demo-only presentation of recorded Deployment data. */
function polishPaperDemo(fixture) {
  const time = fixture.metadata.target_timestamp;
  const hour = deploymentDate(time, {hour: "2-digit", minute: "2-digit", hourCycle: "h23"});
  const date = deploymentDate(time, {dateStyle: "long"});
  const adjustment = fixture.adjustments[0].lambda_value;
  const original = fixture.original.prediction.values[0];
  const adjusted = fixture.versions.at(-1).prediction.values[0];
  const summary = clear("deployment-summary");
  clear("deployment-setup");
  stats(summary, [
    ["Dataset", "GEFCom2014"], ["Target date", date], ["Target hour", hour],
    ["Forecast type", "Point forecast"], ["Current version", "Adjusted"],
    ["Adjustment", "+" + (adjustment * 100).toFixed(0) + "% at " + hour]
  ]);

  const forecast = clear("deployment-forecast");
  const metrics = node("div", undefined, "paper-metrics");
  for (const [label, value] of [
    ["Adjusted forecast", deploymentValue(adjusted, 2)],
    ["Original forecast", deploymentValue(original, 2)],
    ["Change", "+" + (adjustment * 100).toFixed(1) + "%"]
  ]) {
    const metric = node("div", undefined, "paper-metric");
    metric.append(node("small", label), node("strong", value));
    metrics.append(metric);
  }
  forecast.append(metrics);
  const profile = fixture.presentation.day_profile;
  deploymentPlot(forecast, "Target-day load profile", [
    {label: "Original", values: profile.original_values, color: "#51799c",
      highlightIndex: profile.target_index},
    {label: "Adjusted", values: profile.adjusted_values, color: "#bd7354",
      highlightIndex: profile.target_index}
  ], null, {times: profile.timestamps, yLabel: "Load", xLabel: "Hour of day"});
  tableView(forecast, ["Target time", "Original", "Adjusted"], [
    [deploymentDate(time, {dateStyle: "medium"}) + ", " + hour,
      deploymentValue(original, 2), deploymentValue(adjusted, 2)]
  ]);

  const calendar = $("deployment-calendar");
  calendar.querySelector("p").textContent = "Selected reference days for comparison.";
  const weather = $("deployment-weather");
  weather.querySelector("p").textContent = "Weather-based reference days ranked by similarity.";
  weather.querySelector("table th:last-child").textContent = "Similarity score";
  const titles = weather.querySelectorAll(".chart-title");
  titles[0].textContent = "Temperature profiles";
  titles[1].textContent = "Historical load profiles";
  const charts = weather.querySelectorAll("svg.deployment-chart");
  const legends = weather.querySelectorAll(".chart-legend");
  const grid = node("div", undefined, "paper-weather-grid");
  for (let index = 0; index < titles.length; index++) {
    const card = node("div", undefined, "paper-weather-chart");
    card.append(titles[index], charts[index], legends[index]);
    grid.append(card);
  }
  weather.append(grid);
}
if (window.IFORECAST_PAPER_DEMO) {
  document.body.classList.add("paper-demo");
  fetch("/data/examples/deployment_paper_demo.json").then(async response => {
    if (!response.ok) throw new Error("Demo fixture unavailable.");
    const fixture = await response.json();
    const formatDate = deploymentDate;
    deploymentDate = (value, options = {}) =>
      formatDate(value, {timeZone: "UTC", ...options});
    const formatValue = deploymentValue;
    deploymentValue = (value, digits = 1) =>
      formatValue(value, digits === 1 ? 2 : digits);
    const base = "/tasks/" + fixture.metadata.task_id + "/deployment";
    request = async (path, method = "GET") => {
      if (method !== "GET") throw new Error("This paper preview is read-only.");
      const prefix = base + "/forecasts/" + fixture.forecast.forecast_id;
      const responses = new Map([
        [base + "/sessions", [fixture.session]],
        [prefix, {forecast: fixture.forecast, original: fixture.original}],
        [prefix + "/versions", fixture.versions],
        [prefix + "/adjustments", fixture.adjustments],
        [prefix + "/references", fixture.reference],
        [prefix + "/sensitivity/variables", fixture.variables]
      ]);
      if (!responses.has(path)) throw new Error("No demo response for " + path);
      return responses.get(path);
    };
    taskId = fixture.metadata.task_id;
    viewStage = "deployment";
    lastTaskStage = "DEPLOYMENT";
    $("task-id").textContent = "GEFCom2014 load forecast";
    $("stage-badge").textContent = "Deployment";
    $("workflow-status").textContent = "Forecast ready";
    $("deployment-status").textContent = "Adjusted";
    $("preparation-nav").classList.remove("active");
    $("training-nav").classList.remove("active");
    $("deployment-nav").classList.add("active");
    $("deployment-nav").disabled = false;
    $("training-workspace").hidden = true;
    $("deployment-workspace").hidden = false;
    for (const section of document.querySelectorAll("main > section.card")) section.hidden = true;
    $("new-task").hidden = true;
    $("open-task").hidden = true;
    const heading = document.querySelector(".page-heading");
    heading.querySelector(".eyebrow").textContent = "STAGE 03 · DEPLOYMENT";
    heading.querySelector("h1").textContent = "Load forecast and similar days";
    heading.querySelector("p:last-child").textContent =
      "Review the forecast together with calendar and weather-based reference days.";
    $("step-pill").textContent = deploymentDate(fixture.metadata.target_timestamp, {dateStyle: "long"});
    document.querySelector(".nav-note").textContent =
      "Preparation → Training & Evaluation → Deployment";
    $("manager-subtitle").textContent = "Forecast support";
    $("manager-status").textContent = "Discuss this forecast with Task Manager";
    const conversation = clear("conversation");
    for (const message of fixture.messages) {
      if (!["user", "task_manager"].includes(message.role)) continue;
      const bubble = node("div", undefined, "bubble " +
        (message.role === "user" ? "user" : "task-manager"));
      bubble.append(node("strong", message.role === "user" ? "You" : "Task Manager"));
      bubble.append(node("p", message.text));
      conversation.append(bubble);
    }
    $("chat-form").hidden = false;
    $("chat-label").textContent = "Message Task Manager";
    $("chat-input").placeholder = "Ask about this forecast…";
    $("chat-form").onsubmit = event => event.preventDefault();
    if (!(await loadDeployment())) throw new Error("Demo session did not load.");
    deploymentState.presentation = fixture.presentation;
    renderDeployment();
    polishPaperDemo(fixture);
    document.body.dataset.paperBottom =
      $("deployment-weather-section").getBoundingClientRect().bottom.toFixed(2);
    for (const id of ["deployment-adjustment-section", "deployment-versions-section",
      "deployment-sensitivity-section"]) $(id).hidden = true;
    $("notice").textContent = "";
    $("notice").hidden = true;
    for (const [name, id] of [
      ["references", "deployment-calendar-section"],
      ["weather", "deployment-weather-section"]
    ]) {
      const box = $(id).getBoundingClientRect();
      document.body.dataset["paperCrop" + name[0].toUpperCase() + name.slice(1)] =
        [box.left, box.top, box.width, box.height].map(value => value.toFixed(2)).join(",");
    }
    const visibleText = document.body.innerText.toLowerCase();
    for (const phrase of ["paper preview", "gef14.csv", "cnn", "perfect temperature",
      "timezone", "illustrative", "model-predicted", "saved model", "protocol",
      "artifact", "forecast version history", "sensitivity analysis"]) {
      if (visibleText.includes(phrase))
        throw new Error("Internal copy is visible in the figure: " + phrase);
    }
    document.body.dataset.paperReady = "true";
    document.title = "Interactive Forecasting · Deployment";
    window.paperDemoReady = true;
  }).catch(error => {
    $("notice").textContent = "Paper demo failed: " + error.message;
    window.paperDemoError = error.message;
  });
}
