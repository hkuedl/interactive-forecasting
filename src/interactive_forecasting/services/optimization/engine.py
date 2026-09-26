"""Deterministic, synchronous search rounds above the audited Forecasting Core."""

from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone
from uuid import UUID

from interactive_forecasting.domain.forecasting import OriginSchedule, ResolvedCandidate
from interactive_forecasting.domain.models import (
    DatasetSnapshot,
    EvaluationProtocol,
    ExperimentRun,
    TaskDefinition,
)
from interactive_forecasting.domain.search import (
    CandidateRequest,
    GuidanceCommand,
    SearchSpace,
    SpaceChange,
    TrialResult,
    ValidationMetricConfig,
)
from interactive_forecasting.domain.types import JobStatus, ModelFamily
from interactive_forecasting.services.models.core import ModelRegistry
from interactive_forecasting.services.optimization.backend import SearchBackend
from interactive_forecasting.services.optimization.execution import TrialExecutor
from interactive_forecasting.services.optimization.space import CandidateResolver
from interactive_forecasting.storage.interfaces import ExperimentRepositoryPort


class SearchEngine:
    """Owns run state. The backend sees only proposals and validation outcomes."""

    def __init__(
        self,
        repository: ExperimentRepositoryPort,
        backend: SearchBackend,
        executor: TrialExecutor,
    ):
        self._repository = repository
        self._backend = backend
        self._executor = executor
        self._restored_run_id: UUID | None = None

    def create_run(
        self,
        *,
        spec_id: str,
        task: TaskDefinition,
        snapshot: DatasetSnapshot,
        protocol: EvaluationProtocol,
        schedule: OriginSchedule,
        space: SearchSpace,
        templates: dict[ModelFamily, ResolvedCandidate],
        metric: ValidationMetricConfig,
        experiment_seed: int,
    ) -> ExperimentRun:
        config = self._backend.config
        if config.direction != "minimize":
            raise ValueError("built-in MAE/MAPE/CRPS objectives must be minimized")
        if task.dataset_id != snapshot.dataset_id:
            raise ValueError("task and dataset snapshot disagree")
        if self._executor.snapshot_sha256 != snapshot.artifact.sha256:
            raise ValueError("validation evaluator must load the declared snapshot artifact")
        if task.objective_id != metric.objective or metric.objective not in protocol.metric_ids:
            raise ValueError("task/protocol/validation objective disagree")
        if task.metric_spec != protocol.metric_spec:
            raise ValueError("task and protocol metric specifications disagree")
        if metric.objective == "weighted_mae" and (
            task.metric_spec is None or task.metric_spec.objective_id != metric.objective
        ):
            raise ValueError("weighted MAE requires the frozen task MetricSpec")
        if metric.objective == "asymmetric_mae" and (
            task.metric_spec is None or task.metric_spec.objective_id != metric.objective
        ):
            raise ValueError("asymmetric MAE requires the frozen task MetricSpec")
        if (task.forecast_output.representation == "quantile") != (metric.objective == "crps"):
            raise ValueError("quantile tasks require CRPS; point tasks require point objective")
        if not 0 <= experiment_seed <= 2**32 - 1:
            raise ValueError("experiment seed out of range")
        eligible = self._eligible(space, task, templates)
        if not eligible:
            raise ValueError("no capability-eligible family with a complete template")
        for family in eligible:
            base = templates[family]
            if (
                base.index.frequency != snapshot.frequency
                or snapshot.timezone_name != task.timezone_name
            ):
                raise ValueError("candidate index or timezone differs from dataset snapshot")
        payload = {
            "task": task.model_dump(mode="json"),
            "snapshot": snapshot.model_dump(mode="json"),
            "protocol": protocol.model_dump(mode="json"),
            "schedule": schedule.model_dump(mode="json"),
            "space": space.model_dump(mode="json"),
            "backend": config.model_dump(mode="json"),
            "metric": metric.model_dump(mode="json"),
            "templates": {key.value: val.model_dump(mode="json") for key, val in templates.items()},
            "experiment_seed": experiment_seed,
        }
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        run = ExperimentRun(
            spec_id=spec_id,
            config_hash=digest,
            task_definition=task,
            dataset_snapshot=snapshot,
            protocol=protocol,
            origin_schedule=schedule,
            canonical_space=space,
            effective_space=space,
            space_history=(space,),
            templates=templates,
            backend_config=config,
            metric_config=metric,
            experiment_seed=experiment_seed,
            runtime_metadata={"python": platform.python_version(), **self._backend.metadata()},
        )
        return self._repository.create(run)

    @staticmethod
    def _eligible(
        space: SearchSpace,
        task: TaskDefinition,
        templates: dict[ModelFamily, ResolvedCandidate],
    ) -> tuple[ModelFamily, ...]:
        registry = ModelRegistry()
        return tuple(
            item.family
            for item in space.families
            if item.family in templates
            and (
                registry.capabilities(item.family).quantile
                if task.forecast_output.representation == "quantile"
                else registry.capabilities(item.family).point
            )
        )

    def _load(self, run_id: UUID) -> ExperimentRun:
        run = self._repository.get(run_id)
        if run is None or run.task_definition is None or run.effective_space is None:
            raise ValueError("unknown or incomplete experiment run")
        if run.backend_config != self._backend.config:
            raise ValueError("backend configuration differs from saved run")
        if self._restored_run_id is None:
            self._backend.restore_completed(
                run.trials, {space.space_id: space for space in run.space_history}
            )
            self._restored_run_id = run_id
        elif self._restored_run_id != run_id:
            raise ValueError("one engine instance handles one experiment run")
        return run

    def _save(self, run: ExperimentRun, **updates: object) -> ExperimentRun:
        revised = run.model_copy(update={"version": run.version + 1, **updates})
        return self._repository.save(revised, expected_version=run.version)

    def apply_guidance(self, run_id: UUID, commands: tuple[GuidanceCommand, ...]) -> ExperimentRun:
        run = self._load(run_id)
        if run.status not in {JobStatus.QUEUED, JobStatus.RUNNING}:
            raise ValueError("guidance is accepted only before a search round completes")
        assert run.effective_space is not None and run.task_definition is not None
        if any(trial.round_number == run.completed_rounds for trial in run.trials):
            raise ValueError(
                "guidance requires a round boundary; resume the interrupted batch first"
            )
        space = run.effective_space
        pending = list(run.pending_requests)
        allocation = dict(run.next_allocation)
        for command in commands:
            if command.operation in {"exclude_family", "restrict_families"}:
                space = space.derive(
                    SpaceChange(operation=command.operation, families=command.families)
                )
            elif command.operation == "fix_parameter":
                space = space.derive(
                    SpaceChange(operation="fix", parameter=command.parameter, value=command.value)
                )
            elif command.operation == "narrow_parameter":
                space = space.derive(
                    SpaceChange(
                        operation="narrow",
                        parameter=command.parameter,
                        low=command.low,
                        high=command.high,
                    )
                )
            elif command.operation == "restrict_choices":
                space = space.derive(
                    SpaceChange(
                        operation="restrict_choices",
                        parameter=command.parameter,
                        choices=command.choices,
                    )
                )
            elif command.operation in {"force_feature", "disable_feature"}:
                space = space.derive(
                    SpaceChange(
                        operation=command.operation,
                        parameter=command.parameter,
                        value=command.value,
                    )
                )
            elif command.operation == "enqueue_candidate":
                if command.candidate is None:
                    raise ValueError("enqueue needs a candidate")
                space.validate_partial_values(command.candidate.family, command.candidate.values)
                pending.append(command.candidate.model_copy(update={"source": "enqueued"}))
            elif command.operation == "local_refinement":
                if command.candidate is None or command.radius is None:
                    raise ValueError("local refinement needs an existing candidate and radius")
                if not any(
                    result.request.family == command.candidate.family
                    and result.request.values == command.candidate.values
                    and result.status == "completed"
                    for result in run.trials
                ):
                    raise ValueError("local refinement requires a completed candidate")
                space = space.derive(
                    SpaceChange(operation="fix_family", families=(command.candidate.family,))
                )
                for domain in space.domains_for(command.candidate.family):
                    if domain.kind not in {"integer", "float"}:
                        continue
                    center = command.candidate.values.get(domain.name)
                    if (
                        not isinstance(center, (int, float))
                        or domain.low is None
                        or domain.high is None
                    ):
                        continue
                    width = (domain.high - domain.low) * command.radius
                    low = max(domain.low, center - width)
                    high = min(domain.high, center + width)
                    if domain.kind == "integer":
                        step = int(domain.step or 1)
                        low = domain.low + int((low - domain.low) // step) * step
                        high = domain.low + int((high - domain.low) // step) * step
                        high = max(low, high)
                    space = space.derive(
                        SpaceChange(
                            operation="narrow",
                            parameter=(
                                f"feature.{domain.name}"
                                if domain in space.features
                                else f"{command.candidate.family.value}.{domain.name}"
                            ),
                            low=low,
                            high=high,
                        )
                    )
            elif command.operation in {"allocate_family_trials", "prefer_family"}:
                if command.operation == "prefer_family":
                    if len(command.families) != 1:
                        raise ValueError("prefer_family needs exactly one family")
                    allocation[command.families[0]] = 1
                else:
                    if not command.allocation or any(
                        count <= 0 for count in command.allocation.values()
                    ):
                        raise ValueError("family allocation requires positive counts")
                    allocation = dict(command.allocation)
            else:
                raise ValueError("unsupported structured guidance")
        eligible = self._eligible(space, run.task_definition, run.templates)
        if not eligible or any(family not in eligible for family in allocation):
            raise ValueError("guidance leaves no eligible family or invalid allocation")
        if pending and any(request.family not in eligible for request in pending):
            raise ValueError("pending candidate is excluded by guidance")
        for request in pending:
            space.validate_partial_values(request.family, request.values)
        proposed = run.model_copy(
            update={"next_allocation": allocation, "pending_requests": tuple(pending)}
        )
        self._remaining_batch(proposed, eligible)
        return self._save(
            run,
            effective_space=space,
            space_history=run.space_history
            + ((space,) if space.space_id != run.effective_space.space_id else ()),
            guidance_history=run.guidance_history + commands,
            pending_requests=tuple(pending),
            next_allocation=allocation,
        )

    @staticmethod
    def _remaining_batch(
        run: ExperimentRun, eligible: tuple[ModelFamily, ...], *, limit: int | None = None
    ) -> tuple[int, list[ModelFamily]]:
        """Preflight the original slot order, offset by durable trial records (H4/H5)."""
        assert run.backend_config is not None
        config = run.backend_config
        consumed = sum(trial.round_number == run.completed_rounds for trial in run.trials)
        size = min(limit or config.batch_size, config.trial_budget - len(run.trials) + consumed)
        slots = max(0, size - consumed)
        allocation = [family for family, count in run.next_allocation.items() for _ in range(count)]
        if len(allocation) > size:
            raise ValueError("family allocation exceeds the upcoming round size")
        remaining = allocation[consumed:]
        if len(run.pending_requests) > config.trial_budget - len(run.trials):
            raise ValueError("pending candidates exceed the remaining trial budget")
        if (remaining or run.pending_requests) and run.completed_rounds >= config.max_rounds:
            raise ValueError("no search rounds remain for guidance")
        for slot, request in enumerate(run.pending_requests[:slots]):
            allowed = (remaining[slot],) if slot < len(remaining) else eligible
            if request.family not in allowed:
                raise ValueError("pending candidate conflicts with family allocation")
        return slots, remaining

    def cancel(self, run_id: UUID) -> ExperimentRun:
        run = self._load(run_id)
        if run.status not in {JobStatus.QUEUED, JobStatus.RUNNING}:
            raise ValueError("run is already terminal")
        return self._save(run, status=JobStatus.CANCELLED, ended_at=datetime.now(timezone.utc))

    def request_stop(self, run_id: UUID) -> ExperimentRun:
        run = self._load(run_id)
        if run.status not in {JobStatus.QUEUED, JobStatus.RUNNING}:
            raise ValueError("run is already terminal")
        return self._save(run, stop_requested=True)

    def run_fixed(self, run_id: UUID, request: CandidateRequest) -> ExperimentRun:
        if request.source != "fixed_reference":
            raise ValueError("fixed execution requires fixed_reference provenance")
        run = self._load(run_id)
        if run.trials:
            raise ValueError("fixed execution requires a fresh run")
        assert run.effective_space is not None
        run.effective_space.validate_values(request.family, request.values)
        run = self._save(run, pending_requests=(request,))
        return self._run_round(run.run_id, limit=1, force_complete=True)

    def run_round(
        self, run_id: UUID, *, guidance: tuple[GuidanceCommand, ...] = ()
    ) -> ExperimentRun:
        if guidance:
            self.apply_guidance(run_id, guidance)
        return self._run_round(run_id)

    def _run_round(
        self, run_id: UUID, *, limit: int | None = None, force_complete: bool = False
    ) -> ExperimentRun:
        run = self._load(run_id)
        if run.status not in {JobStatus.QUEUED, JobStatus.RUNNING}:
            raise ValueError("run is terminal")
        if run.stop_requested:
            return self._complete(run)
        assert run.backend_config is not None and run.task_definition is not None
        assert run.effective_space is not None and run.protocol is not None
        assert run.metric_config is not None and run.experiment_seed is not None
        config = run.backend_config
        in_progress = any(trial.round_number == run.completed_rounds for trial in run.trials)
        if (
            len(run.trials) >= config.trial_budget and not in_progress
        ) or run.completed_rounds >= config.max_rounds:
            return self._complete(run)
        eligible = self._eligible(run.effective_space, run.task_definition, run.templates)
        slots, allocation = self._remaining_batch(run, eligible, limit=limit)
        if run.status == JobStatus.QUEUED:
            run = self._save(run, status=JobStatus.RUNNING, started_at=datetime.now(timezone.utc))
        assert run.task_definition is not None and run.protocol is not None
        assert run.effective_space is not None and run.metric_config is not None
        assert run.experiment_seed is not None
        space = run.effective_space
        resolver = CandidateResolver(run.task_definition, run.protocol, run.templates)
        previous_scores = [
            trial.objective
            for trial in run.trials
            if trial.round_number < run.completed_rounds
            and trial.status == "completed"
            and trial.objective is not None
        ]
        before_best = min(previous_scores) if previous_scores else None
        for slot in range(slots):
            number = len(run.trials)
            seed = (run.experiment_seed + number) % (2**32)
            allowed = (allocation[slot],) if slot < len(allocation) else eligible
            if run.pending_requests:
                request = run.pending_requests[0]
                if request.family not in allowed:
                    raise ValueError("pending candidate conflicts with family allocation")
                self._backend.enqueue(request, space)
            proposal = self._backend.propose(space, allowed, seed=seed)
            started = datetime.now(timezone.utc)
            try:
                trial = resolver.resolve(proposal.request, space)
            except Exception as exc:
                result = TrialResult(
                    number=number,
                    round_number=run.completed_rounds,
                    request=proposal.request,
                    space_id=space.space_id,
                    status="failed",
                    failure_type=type(exc).__name__,
                    failure_message=str(exc),
                    started_at=started,
                    ended_at=datetime.now(timezone.utc),
                )
                artifact = None
            else:
                result, artifact = self._executor.execute(
                    run.run_id,
                    run.spec_id,
                    trial,
                    run.metric_config,
                    number=number,
                    round_number=run.completed_rounds,
                )
            refs = run.artifact_refs + ((artifact,) if artifact is not None else ())
            run = self._save(
                run,
                trials=(*run.trials, result),
                artifact_refs=refs,
                pending_requests=run.pending_requests[1:]
                if run.pending_requests
                else run.pending_requests,
            )
            self._backend.observe(proposal.token, result)
        after_best = self._best_objective(run)
        improved = after_best is not None and (
            before_best is None
            or (
                after_best < before_best
                if config.direction == "minimize"
                else after_best > before_best
            )
        )
        stale = 0 if improved else run.stale_rounds + 1
        run = self._save(
            run,
            completed_rounds=run.completed_rounds + 1,
            stale_rounds=stale,
            next_allocation={},
        )
        if (
            force_complete
            or len(run.trials) >= config.trial_budget
            or run.completed_rounds >= config.max_rounds
            or (config.no_improvement_rounds is not None and stale >= config.no_improvement_rounds)
        ):
            return self._complete(run)
        return run

    @staticmethod
    def _best_objective(run: ExperimentRun) -> float | None:
        values = [
            trial.objective
            for trial in run.trials
            if trial.status == "completed" and trial.objective is not None
        ]
        if not values:
            return None
        assert run.backend_config is not None
        return min(values) if run.backend_config.direction == "minimize" else max(values)

    def _complete(self, run: ExperimentRun) -> ExperimentRun:
        assert run.backend_config is not None
        completed = [
            trial
            for trial in run.trials
            if trial.status == "completed" and trial.objective is not None
        ]
        best = None
        if completed:
            best = (
                min(
                    completed,
                    key=lambda item: item.objective if item.objective is not None else float("inf"),
                )
                if run.backend_config.direction == "minimize"
                else max(
                    completed,
                    key=lambda item: (
                        item.objective if item.objective is not None else float("-inf")
                    ),
                )
            )
        return self._save(
            run,
            status=JobStatus.COMPLETED if best else JobStatus.FAILED,
            selected_trial_id=best.trial_id if best else None,
            ended_at=datetime.now(timezone.utc),
        )
