"""Entry point for all non-human public forecasting experiments."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import platform
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import architecture_specific
import fixed_default
import llm_guided
import optuna
import pandas
import probabilistic
import sklearn
import torch
import vanilla_bo
import xgboost
from common import (
    FAMILIES,
    ROOT,
    build_context,
    discover,
    fingerprint,
    manifest,
)

from interactive_forecasting.domain.types import ModelFamily

EXPERIMENTS = (
    "fixed_default",
    "vanilla_bo",
    "llm_guided",
    "probabilistic",
    "architecture_specific",
)
OUTPUT = ROOT / "experiments" / "output"


def plan(args: argparse.Namespace) -> list[tuple[str, str, str, Path, ModelFamily | None]]:
    """Expand dataset filters into the individual runs requested by the CLI."""
    jobs = []
    for dataset, series, path in discover(args.dataset):
        if args.series and series != args.series:
            continue
        for experiment in [args.experiment] if args.experiment else EXPERIMENTS:
            families = (
                ((ModelFamily(args.model),) if args.model else FAMILIES)
                if experiment in {"fixed_default", "architecture_specific"}
                else (None,)
            )
            if args.model and experiment not in {"fixed_default", "architecture_specific"}:
                raise ValueError("--model applies only to fixed_default or architecture_specific")
            for family in families:
                jobs.append((experiment, dataset, series, path, family))
    return jobs


def _aggregate(records: list[dict]) -> list[dict]:
    """Predeclared unweighted mean over successful series; Zone 9 is excluded."""
    buckets: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for item in records:
        if item.get("status") != "completed":
            continue
        if item["dataset"] == "gefcom2012" and item["series"] == "zone_9":
            continue
        buckets[(item["experiment"], item["dataset"], item.get("family") or "joint")].append(item)
    rows = []
    for (experiment, dataset, family), items in sorted(buckets.items()):
        metrics = {}
        for key in ("mae", "mape", "crps"):
            values = [
                float(item["final_metrics"][key])
                for item in items
                if key in item.get("final_metrics", {})
            ]
            if values:
                metrics[key] = sum(values) / len(values)
        rows.append(
            {
                "experiment": experiment,
                "dataset": dataset,
                "family": family,
                "series_count": len(items),
                "aggregation": "arithmetic_mean_of_series",
                "metrics": metrics,
            }
        )
    return rows


def report(document: dict) -> str:
    """Render per-series and aggregate results as a readable HTML report."""

    def esc(value: object) -> str:
        return html.escape(str(value), quote=True)

    rows = []
    for item in document["results"]:
        metric = item.get("final_metrics", {})
        rows.append(
            "<tr>"
            + "".join(
                f"<td>{esc(value)}</td>"
                for value in (
                    item.get("experiment"),
                    item.get("dataset"),
                    item.get("series"),
                    item.get("family") or "joint",
                    item.get("status"),
                    metric.get("mae", "—"),
                    metric.get("mape", "—"),
                    metric.get("crps", "—"),
                    item.get("selected_trial_number", "—"),
                    item.get("reason", ""),
                )
            )
            + "</tr>"
        )
    aggregates = []
    for row in document["aggregates"]:
        aggregates.append(
            "<tr>"
            + "".join(
                f"<td>{esc(value)}</td>"
                for value in (
                    row["experiment"],
                    row["dataset"],
                    row["family"],
                    row["series_count"],
                    row["metrics"].get("mae", "—"),
                    row["metrics"].get("mape", "—"),
                    row["metrics"].get("crps", "—"),
                )
            )
            + "</tr>"
        )
    settings = esc(json.dumps(document.get("settings", {}), indent=2, sort_keys=True))
    environment = esc(json.dumps(document["environment"], indent=2, sort_keys=True))
    registry = esc(json.dumps(document["registry"], indent=2, sort_keys=True))
    return f"""<!doctype html><html lang="en"><meta charset="utf-8">
<title>Interactive Forecasting — public automatic experiments</title>
<style>body{{font:15px/1.5 system-ui,sans-serif;color:#203145;
max-width:1200px;margin:32px auto;padding:0 20px}}
h1,h2{{color:#15385a}}table{{border-collapse:collapse;width:100%;margin:18px 0 30px}}
th,td{{padding:8px 10px;border-bottom:1px solid #d9e3ed;text-align:left;vertical-align:top}}
th{{background:#eef3f8}}pre{{white-space:pre-wrap;background:#f3f6f9;padding:16px}}
.note{{background:#eef3f8;padding:12px 16px;border-left:4px solid #5d83a4}}</style>
<h1>Interactive Forecasting — public automatic experiments</h1>
<p class="note">Status: {esc(document["status"])}. Results use only the supplied frozen settings.
Independent successful series contribute equally to dataset means.
GEFCom2012 Zone 9 is reported individually but excluded from its family mean.
Missing or failed series are never imputed.</p>
<h2>Per-series results</h2><table><thead><tr><th>Experiment</th><th>Dataset</th><th>Series</th>
<th>Family</th><th>Status</th><th>MAE</th><th>MAPE (%)</th><th>CRPS</th>
<th>Selected trial</th><th>Reason</th></tr></thead><tbody>{"".join(rows)}</tbody></table>
<h2>Dataset-family summaries</h2><table><thead><tr><th>Experiment</th><th>Dataset</th>
<th>Family</th><th>Series</th><th>Mean MAE</th><th>Mean MAPE</th><th>Mean CRPS</th>
</tr></thead><tbody>{"".join(aggregates)}</tbody></table>
<h2>Public task registry</h2><pre>{registry}</pre>
<h2>Frozen settings and budgets</h2><pre>{settings}</pre>
<h2>Software environment</h2><pre>{environment}</pre>
<h2>Manuscript correspondence</h2><p>Tables III–IV: production canonical space.
Table XII and Fig. 11 Default: fixed_default. Tables V, VII (Vanilla), XIV–XV,
Fig. 6(a), Fig. 11 Automated: vanilla_bo. Table VI automatic: probabilistic.
Table VII LLM-Guided: llm_guided. Table XIII: architecture_specific.
Human+LLM/Proposed results are outside this package.</p></html>"""


def main(argv: list[str] | None = None) -> int:
    """Plan runs, execute each one independently, and write the combined reports."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=EXPERIMENTS)
    parser.add_argument("--dataset", choices=("gefcom2014", "gefcom2012", "gefcom2017"))
    parser.add_argument("--series")
    parser.add_argument("--model", choices=[item.value for item in FAMILIES])
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(os.environ["IF_EXPERIMENT_PROTOCOL"])
        if os.getenv("IF_EXPERIMENT_PROTOCOL")
        else None,
        help="Explicit frozen protocol JSON; required for execution",
    )
    parser.add_argument(
        "--budget", type=int, help="Explicit budget override, recorded as distinct run"
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    jobs = plan(args)
    if not jobs:
        raise ValueError("no public tasks matched filters")
    config = json.loads(args.protocol.read_text()) if args.protocol else {}
    registry = {
        name: {
            "manifest": manifest(name),
            "series": [series for ds, series, _ in discover(name)],
        }
        for name in ("gefcom2014", "gefcom2012", "gefcom2017")
    }
    document: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "planned" if args.dry_run else "running",
        "registry": registry,
        "settings": config,
        "protocol_file": str(args.protocol.resolve()) if args.protocol else None,
        "protocol_sha256": hashlib.sha256(args.protocol.read_bytes()).hexdigest()
        if args.protocol
        else None,
        "budget_override": args.budget,
        "environment": {
            "python": platform.python_version(),
            "pandas": pandas.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
            "xgboost": xgboost.__version__,
            "optuna": optuna.__version__,
        },
        "results": [],
        "aggregates": [],
    }
    if not args.dry_run and not config:
        raise ValueError(
            "Pass the frozen experiment protocol using --protocol or IF_EXPERIMENT_PROTOCOL"
        )
    prior: dict[str, dict[str, Any]] = {}
    previous = OUTPUT / "results.json"
    # Resume accepts only completed runs whose inputs and settings still match.
    if args.resume and previous.is_file():
        old = json.loads(previous.read_text())
        prior = {item["key"]: item for item in old.get("results", []) if "key" in item}
    for experiment, dataset, series, path, family in jobs:
        key = ":".join((experiment, dataset, series, family.value if family else "joint"))
        digest = fingerprint(experiment, dataset, series, path, config, family, args.budget)
        item: dict[str, Any]
        if args.dry_run:
            item = {
                "key": key,
                "fingerprint": digest,
                "experiment": experiment,
                "dataset": dataset,
                "series": series,
                "family": family.value if family else None,
                "status": "planned",
            }
        elif (
            args.resume
            and key in prior
            and prior[key].get("fingerprint") == digest
            and prior[key].get("status") == "completed"
        ):
            item = {**prior[key], "resumed": True}
        else:
            try:
                # Keep failures visible without stopping the remaining jobs.
                if experiment == "llm_guided":
                    llm_settings = config.get("llm", {})
                    if llm_settings != {
                        "provider": "openai",
                        "model": "gpt-4o",
                        "temperature": 0,
                    }:
                        raise ValueError("LLM settings must be OpenAI gpt-4o at temperature 0")
                    if not os.getenv("OPENAI_API_KEY"):
                        raise RuntimeError(
                            "OpenAI provider unavailable: OPENAI_API_KEY is not configured"
                        )
                workspace = ROOT / "artifacts" / "experiments" / digest[:16] / str(uuid4())
                context = build_context(
                    dataset,
                    series,
                    path,
                    config,
                    workspace,
                    quantile=experiment == "probabilistic",
                    family=family,
                    budget=args.budget,
                )
                if experiment == "fixed_default":
                    assert family is not None
                    item = fixed_default.run(context, family)
                elif experiment == "vanilla_bo":
                    item = vanilla_bo.run(context)
                elif experiment == "llm_guided":
                    llm_settings = config["llm"]
                    item = llm_guided.run(
                        context, llm_settings["model"], temperature=llm_settings["temperature"]
                    )
                elif experiment == "probabilistic":
                    item = probabilistic.run(context)
                else:
                    assert family is not None
                    item = architecture_specific.run(context, family)
                item.update(
                    key=key,
                    fingerprint=digest,
                    family=family.value if family else None,
                    resumed=False,
                )
            except Exception as exc:
                item = {
                    "key": key,
                    "fingerprint": digest,
                    "experiment": experiment,
                    "dataset": dataset,
                    "series": series,
                    "family": family.value if family else None,
                    "status": "unavailable"
                    if experiment == "llm_guided"
                    and isinstance(exc, (RuntimeError, ValueError))
                    and ("provider" in str(exc).lower() or "model" in str(exc).lower())
                    else "failed",
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            print(f"{key}: {item['status']}", flush=True)
        # Save after each job so long-running batches retain their progress.
        document["results"].append(item)
        document["aggregates"] = _aggregate(document["results"])
        document["status"] = (
            "planned"
            if args.dry_run
            else "completed"
            if all(row["status"] == "completed" for row in document["results"])
            else "partial"
        )
        OUTPUT.mkdir(parents=True, exist_ok=True)
        # One compact public JSON and one HTML report; internal artifacts stay in ArtifactStore.
        json_temp = OUTPUT / "results.json.tmp"
        html_temp = OUTPUT / "report.html.tmp"
        json_temp.write_text(json.dumps(document, indent=2, default=str))
        html_temp.write_text(report(document))
        json_temp.replace(OUTPUT / "results.json")
        html_temp.replace(OUTPUT / "report.html")
    return 0 if document["status"] in {"planned", "completed"} else 1


if __name__ == "__main__":
    sys.exit(main())
