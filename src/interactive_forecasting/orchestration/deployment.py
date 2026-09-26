"""Application-owned deployment transitions over the selected persisted validation artifact."""

from __future__ import annotations

import json
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

from interactive_forecasting.domain.models import (
    Adjustment,
    AdjustmentProposal,
    ArtifactRef,
    DeploymentReadiness,
    DeploymentSession,
    Event,
    ExperimentRun,
    Forecast,
    ForecastVersion,
    FutureAuxiliaryValue,
    ReferenceAnalysis,
    SensitivityRequest,
    SensitivityResult,
    utc_now,
)
from interactive_forecasting.domain.preparation import PreparationRecord
from interactive_forecasting.domain.search import TrialResult
from interactive_forecasting.domain.types import Actor, Stage, Topic
from interactive_forecasting.orchestration.state_machine import WorkflowStateMachine
from interactive_forecasting.services.deployment.core import forecast_examples, prepare_context
from interactive_forecasting.services.deployment.postprocessing import apply_adjustment
from interactive_forecasting.services.deployment.references import analyze_references
from interactive_forecasting.services.deployment.sensitivity import (
    available_variables,
    run_sensitivity,
)
from interactive_forecasting.services.models.core import FittedCoreModel, ModelRegistry
from interactive_forecasting.storage.artifacts import ArtifactStore
from interactive_forecasting.storage.preparation import PreparationRepository
from interactive_forecasting.storage.sql import (
    DeploymentRepository,
    ExperimentRepository,
    MessageRepository,
    OptimizationSessionRepository,
    TaskRepository,
)


class DeploymentWorkflow:
    def __init__(
        self,
        tasks: TaskRepository,
        preparations: PreparationRepository,
        optimization_sessions: OptimizationSessionRepository,
        runs: ExperimentRepository,
        deployments: DeploymentRepository,
        store: ArtifactStore,
        machine: WorkflowStateMachine,
        messages: MessageRepository | None = None,
    ) -> None:
        self.tasks = tasks
        self.preparations = preparations
        self.optimization_sessions = optimization_sessions
        self.runs = runs
        self.deployments = deployments
        self.store = store
        self.machine = machine
        self.messages = messages

    def _selection(
        self, task_id: UUID
    ) -> tuple[PreparationRecord, ExperimentRun, TrialResult, ArtifactRef]:
        task = self.tasks.get(task_id)
        session = self.optimization_sessions.get(task_id)
        prepared = self.preparations.get(task_id)
        if task is None or session is None or prepared is None:
            raise ValueError("task has no frozen preparation and optimization session")
        run = self.runs.get(session.run_id)
        if (
            run is None
            or run.status.value != "completed"
            or run.selected_trial_id is None
            or run.task_definition is None
            or run.dataset_snapshot is None
            or run.protocol is None
            or not prepared.frozen
            or prepared.prepared is None
            or prepared.definition != run.task_definition
            or prepared.protocol != run.protocol
            or prepared.prepared.dataset != run.dataset_snapshot
            or task.definition_id != prepared.definition.definition_id
        ):
            raise ValueError("deployment requires completed selection and frozen task artifacts")
        if task.stage not in {Stage.OPTIMIZATION, Stage.DEPLOYMENT}:
            raise ValueError("task is not ready for deployment")
        trial = next((item for item in run.trials if item.trial_id == run.selected_trial_id), None)
        if (
            trial is None
            or trial.status != "completed"
            or trial.candidate is None
            or trial.artifact_uri is None
        ):
            raise ValueError("selected trial has no completed model artifact")
        reference = next(
            (item for item in run.artifact_refs if item.uri == trial.artifact_uri), None
        )
        if reference is None or reference.sha256 != trial.artifact_sha256:
            raise ValueError("selected model artifact reference mismatch")
        self.store.read_bytes(reference)
        return prepared, run, trial, reference

    def readiness(self, task_id: UUID) -> DeploymentReadiness:
        _prepared, run, trial, reference = self._selection(task_id)
        return DeploymentReadiness(
            ready=True,
            task_id=task_id,
            optimization_run_id=run.run_id,
            selected_trial_id=trial.trial_id,
            model_artifact=reference,
            session=self.deployments.get(task_id),
        )

    def start(
        self,
        task_id: UUID,
        *,
        new_session: bool = False,
        session_id: UUID | None = None,
        expected_version: int | None = None,
    ) -> DeploymentSession:
        prepared, run, trial, reference = self._selection(task_id)
        existing = self.deployments.get(task_id)
        if existing is not None and not new_session:
            return existing
        if new_session:
            if session_id is None or expected_version is None:
                raise ValueError(
                    "starting another session requires current session identity and version"
                )
            self._session(task_id, expected_version, session_id=session_id)
        if existing is not None and (
            existing.forecast_id is None or existing.pending_adjustment_id is not None
        ):
            raise ValueError("finish or reject the current session draft before starting another")
        task = self.tasks.get(task_id)
        assert task is not None
        if task.stage == Stage.OPTIMIZATION:
            changed = self.machine.transition_stage(
                task, Stage.DEPLOYMENT, expected_version=task.version
            )
            self.tasks.save_transition(
                changed,
                Event(
                    task_id=task_id,
                    source_role=Actor.SYSTEM,
                    topic=Topic.SYSTEM,
                    message_type="deployment_started",
                    payload={"run_id": str(run.run_id), "selected_trial_id": str(trial.trial_id)},
                    outcome="advanced",
                ),
                expected_version=task.version,
            )
        elif task.stage != Stage.DEPLOYMENT:
            raise ValueError("task is not in the deployment stage")
        del prepared
        return self.deployments.create(
            DeploymentSession(
                task_id=task_id,
                run_id=run.run_id,
                selected_trial_id=trial.trial_id,
                model_artifact=reference,
            )
        )

    def _session(
        self, task_id: UUID, expected_version: int, *, session_id: UUID
    ) -> DeploymentSession:
        session = self.deployments.get(task_id)
        if session is None:
            raise ValueError("deployment has not started")
        if session.session_id != session_id or session.version != expected_version:
            from interactive_forecasting.orchestration.state_machine import VersionConflict

            raise VersionConflict("deployment session identity or version mismatch")
        prepared, run, trial, reference = self._selection(task_id)
        del prepared
        if (
            session.run_id != run.run_id
            or session.selected_trial_id != trial.trial_id
            or session.model_artifact != reference
        ):
            raise ValueError("deployment selection differs from frozen validation winner")
        return session

    def upload(
        self,
        task_id: UUID,
        expected_version: int,
        content: bytes,
        extension: str,
        *,
        session_id: UUID,
    ) -> DeploymentSession:
        session = self._session(task_id, expected_version, session_id=session_id)
        if session.forecast_id is not None:
            raise ValueError("original forecast is immutable")
        from interactive_forecasting.services.preparation import read_tabular

        read_tabular(content, extension)
        reference = self.store.put_bytes(
            f"deployment/{task_id}/uploads/{uuid4()}{extension}", content
        )
        changed = session.model_copy(
            update={
                "state": "VALIDATING_DEPLOYMENT_INPUT",
                "version": session.version + 1,
                "raw_upload": reference,
                "upload_extension": extension,
                "future_auxiliaries": (),
                "context_artifact": None,
                "validation": None,
            }
        )
        return self.deployments.save(changed, expected_version=session.version)

    def validate(
        self,
        task_id: UUID,
        expected_version: int,
        future_values: tuple[FutureAuxiliaryValue, ...] = (),
        *,
        session_id: UUID,
    ) -> DeploymentSession:
        session = self._session(task_id, expected_version, session_id=session_id)
        if session.forecast_id is not None:
            raise ValueError("original forecast is immutable")
        if session.raw_upload is None or session.upload_extension is None:
            raise ValueError("upload deployment observations first")
        prepared, _run, trial, reference = self._selection(task_id)
        assert (
            prepared.prepared is not None
            and prepared.definition is not None
            and trial.candidate is not None
        )
        fitted = FittedCoreModel.load(self.store, reference)
        selected = trial.candidate.candidate
        adapter = ModelRegistry().resolve(selected)
        canonical = selected.model_copy(
            update={"hyperparameters": adapter.validate(selected).model_dump(exclude_none=True)}
        )
        if fitted.candidate != canonical:
            raise ValueError("saved model differs from selected resolved candidate")
        if (
            fitted.candidate.output != prepared.definition.forecast_output
            or fitted.candidate.auxiliary_policies != prepared.definition.auxiliary_policies
            or fitted.candidate.index.delta != prepared.definition.delta
            or fitted.candidate.index.horizon != prepared.definition.horizon
            or fitted.candidate.index.offset_unit != prepared.definition.time_unit
        ):
            raise ValueError("selected model differs from frozen task definition")
        context = prepare_context(
            self.store,
            prepared.prepared,
            fitted,
            task_id,
            trial.trial_id,
            reference,
            self.store.read_bytes(session.raw_upload),
            session.upload_extension,
            future_values,
        )
        context_ref = None
        if context.validation.ready:
            context_ref = self.store.put_bytes(
                f"deployment/{task_id}/contexts/{uuid4()}.csv",
                context.historical.to_csv(index=False).encode(),
            )
        changed = session.model_copy(
            update={
                "state": (
                    "READY_TO_FORECAST"
                    if context.validation.ready
                    else "WAITING_FOR_DEPLOYMENT_DATA"
                ),
                "version": session.version + 1,
                "future_auxiliaries": future_values,
                "context_artifact": context_ref,
                "validation": context.validation,
            }
        )
        return self.deployments.save(changed, expected_version=session.version)

    def generate(
        self, task_id: UUID, expected_version: int, *, session_id: UUID
    ) -> tuple[Forecast, ForecastVersion]:
        session = self._session(task_id, expected_version, session_id=session_id)
        if session.forecast_id is not None:
            raise ValueError("original forecast is immutable")
        if (
            session.state != "READY_TO_FORECAST"
            or session.validation is None
            or session.raw_upload is None
            or session.upload_extension is None
            or session.context_artifact is None
        ):
            raise ValueError("deployment input must validate before forecasting")
        prepared, run, trial, reference = self._selection(task_id)
        assert prepared.prepared is not None
        fitted = FittedCoreModel.load(self.store, reference)
        context = prepare_context(
            self.store,
            prepared.prepared,
            fitted,
            task_id,
            trial.trial_id,
            reference,
            self.store.read_bytes(session.raw_upload),
            session.upload_extension,
            session.future_auxiliaries,
        )
        if (
            not context.validation.ready
            or context.validation != session.validation
            or context.data is None
        ):
            raise ValueError("deployment context changed since validation")
        if (
            self.store.read_bytes(session.context_artifact)
            != context.historical.to_csv(index=False).encode()
        ):
            raise ValueError("prepared context artifact differs from validated input")
        prediction = fitted.predict(context.data, forecast_examples(context.origin))
        future_artifact = self.store.put_bytes(
            f"deployment/{task_id}/inputs/{uuid4()}.json",
            json.dumps(
                [item.model_dump(mode="json") for item in session.future_auxiliaries],
                sort_keys=True,
            ).encode(),
        )
        forecast = Forecast(
            task_id=task_id,
            optimization_run_id=run.run_id,
            selected_trial_id=trial.trial_id,
            deployment_session_id=session.session_id,
            model_artifact=reference,
            origin=context.origin,
            target_timestamps=context.origin.targets,
            prediction_representation=fitted.candidate.output.representation,
            prediction=prediction,
            raw_upload=session.raw_upload,
            context_artifact=session.context_artifact,
            future_auxiliary_artifact=future_artifact,
            prepared_snapshot=prepared.prepared.dataset.artifact,
        )
        original = ForecastVersion(
            forecast_id=forecast.forecast_id,
            version_number=0,
            prediction_representation=fitted.candidate.output.representation,
            prediction=prediction,
            provenance={
                "model_artifact_sha256": reference.sha256,
                "context_sha256": session.context_artifact.sha256,
            },
        )
        changed = session.model_copy(
            update={
                "state": "FORECAST_GENERATED",
                "version": session.version + 1,
                "forecast_id": forecast.forecast_id,
                "current_version_id": original.version_id,
            }
        )
        self.deployments.commit_forecast(
            changed, forecast, original, expected_version=session.version
        )
        return forecast, original

    def get_forecast(self, task_id: UUID, forecast_id: UUID) -> tuple[Forecast, ForecastVersion]:
        forecast = self.deployments.forecast(forecast_id)
        original = self.deployments.original(forecast_id)
        if forecast is None or original is None or forecast.task_id != task_id:
            raise ValueError("forecast not found for task")
        return forecast, original

    def list_sessions(self, task_id: UUID) -> list[DeploymentSession]:
        self._selection(task_id)
        return self.deployments.list_sessions(task_id)

    def list_forecasts(self, task_id: UUID) -> list[Forecast]:
        self._selection(task_id)
        return self.deployments.list_forecasts(task_id)

    def analyze(
        self,
        task_id: UUID,
        forecast_id: UUID,
        *,
        session_id: UUID,
        expected_version: int,
        top_k: int = 3,
    ) -> ReferenceAnalysis:
        session = self._session(task_id, expected_version, session_id=session_id)
        if session.forecast_id != forecast_id:
            raise ValueError("reference analysis requires the current session forecast")
        forecast, _ = self.get_forecast(task_id, forecast_id)
        existing = self.deployments.reference(forecast_id)
        if existing is not None:
            if existing.top_k != top_k:
                raise ValueError("reference analysis is frozen with a different top_k")
            return existing
        prepared, _, _, _ = self._selection(task_id)
        assert prepared.prepared is not None and prepared.definition is not None
        temperature = next(
            (
                name
                for name, role in prepared.prepared.mapping.auxiliary_roles.items()
                if role == "temperature"
            ),
            None,
        )
        analysis = analyze_references(
            forecast,
            self.store,
            frequency=prepared.prepared.dataset.frequency,
            temperature_column=temperature,
            auxiliary_policies=prepared.definition.auxiliary_policies,
            top_k=top_k,
        )
        changed = session.model_copy(
            update={
                "version": session.version + 1,
                "reference_analysis_id": analysis.analysis_id,
            }
        )
        return self.deployments.save_reference(changed, analysis, expected_version=session.version)

    def reference(self, task_id: UUID, forecast_id: UUID) -> ReferenceAnalysis:
        self.get_forecast(task_id, forecast_id)
        analysis = self.deployments.reference(forecast_id)
        if analysis is None:
            raise ValueError("reference analysis has not been generated")
        return analysis

    def versions(self, task_id: UUID, forecast_id: UUID) -> list[ForecastVersion]:
        self.get_forecast(task_id, forecast_id)
        return self.deployments.versions(forecast_id)

    def version(self, task_id: UUID, forecast_id: UUID, version_id: UUID) -> ForecastVersion:
        self.get_forecast(task_id, forecast_id)
        version = self.deployments.version(version_id)
        if version is None or version.forecast_id != forecast_id:
            raise ValueError("forecast version not found")
        return version

    def adjustments(self, task_id: UUID, forecast_id: UUID) -> list[Adjustment]:
        self.get_forecast(task_id, forecast_id)
        return self.deployments.adjustments(forecast_id)

    def _adjustment_context(
        self, task_id: UUID, forecast_id: UUID
    ) -> tuple[
        Forecast, DeploymentSession, ForecastVersion, tuple[FutureAuxiliaryValue, ...], dict
    ]:
        session = self.deployments.get(task_id)
        if session is None or session.forecast_id != forecast_id:
            raise ValueError("adjustments require the current deployment session")
        forecast, original = self.get_forecast(task_id, forecast_id)
        parent = (
            self.deployments.version(session.current_version_id)
            if session.current_version_id is not None
            else original
        )
        if parent is None:
            raise ValueError("current forecast version is missing")
        prepared, _, _, _ = self._selection(task_id)
        assert prepared.definition is not None
        future = tuple(
            FutureAuxiliaryValue.model_validate(item)
            for item in json.loads(self.store.read_bytes(forecast.future_auxiliary_artifact))
        )
        return forecast, session, parent, future, prepared.definition.auxiliary_policies

    def create_draft(
        self,
        task_id: UUID,
        forecast_id: UUID,
        expected_version: int,
        proposal: AdjustmentProposal,
        *,
        session_id: UUID,
        source: str,
        user_text: str | None = None,
    ) -> Adjustment:
        self._session(task_id, expected_version, session_id=session_id)
        forecast, session, parent, future, policies = self._adjustment_context(task_id, forecast_id)
        if session.version != expected_version:
            from interactive_forecasting.orchestration.state_machine import VersionConflict

            raise VersionConflict("deployment session version mismatch")
        if session.pending_adjustment_id is not None or session.state == "COMPLETED":
            raise ValueError("finish the pending adjustment before creating another")
        if source not in {"user", "deployment_operator_proposal"}:
            raise ValueError("invalid adjustment source")
        draft = Adjustment.model_validate(
            {
                **proposal.model_dump(mode="json"),
                "task_id": task_id,
                "forecast_id": forecast_id,
                "parent_version_id": parent.version_id,
                "source": source,
                "user_request_text": user_text,
            }
        )
        apply_adjustment(forecast, parent, draft, future_values=future, auxiliary_policies=policies)
        changed = session.model_copy(
            update={
                "version": session.version + 1,
                "state": "ADJUSTMENT_DRAFT_PENDING",
                "pending_adjustment_id": draft.adjustment_id,
            }
        )
        self.deployments.save_draft(changed, draft, expected_version=session.version)
        return draft

    def validate_draft(
        self, task_id: UUID, adjustment_id: UUID, expected_version: int, *, session_id: UUID
    ) -> Adjustment:
        self._session(task_id, expected_version, session_id=session_id)
        draft = self.deployments.adjustment(adjustment_id)
        if draft is None or draft.task_id != task_id or draft.status != "draft":
            raise ValueError("pending draft not found")
        forecast, session, parent, future, policies = self._adjustment_context(
            task_id, draft.forecast_id
        )
        if session.version != expected_version or session.pending_adjustment_id != adjustment_id:
            from interactive_forecasting.orchestration.state_machine import VersionConflict

            raise VersionConflict("draft does not match current session")
        if draft.parent_version_id != parent.version_id:
            raise ValueError("draft parent version is stale")
        apply_adjustment(forecast, parent, draft, future_values=future, auxiliary_policies=policies)
        changed = session.model_copy(
            update={"version": session.version + 1, "state": "WAITING_FOR_USER_CONFIRMATION"}
        )
        self.deployments.save(changed, expected_version=session.version)
        return draft

    def reject_draft(
        self, task_id: UUID, adjustment_id: UUID, expected_version: int, *, session_id: UUID
    ) -> Adjustment:
        self._session(task_id, expected_version, session_id=session_id)
        draft = self.deployments.adjustment(adjustment_id)
        if draft is None or draft.task_id != task_id or draft.status != "draft":
            raise ValueError("pending draft not found")
        session = self.deployments.get(task_id)
        if (
            session is None
            or session.version != expected_version
            or session.pending_adjustment_id != adjustment_id
        ):
            from interactive_forecasting.orchestration.state_machine import VersionConflict

            raise VersionConflict("draft does not match current session")
        rejected = draft.model_copy(update={"status": "rejected"})
        changed = session.model_copy(
            update={
                "version": session.version + 1,
                "state": (
                    "FORECAST_VERSION_UPDATED"
                    if self.deployments.versions(draft.forecast_id)[-1].version_number > 0
                    else "REFERENCE_ANALYSIS_AVAILABLE"
                    if session.reference_analysis_id is not None
                    else "FORECAST_GENERATED"
                ),
                "pending_adjustment_id": None,
            }
        )
        self.deployments.update_draft(changed, rejected, expected_version=session.version)
        return rejected

    def _confirmation_event(self, applied: Adjustment) -> None:
        if self.messages is None or applied.confirmation_id is None:
            return
        self.messages.append(
            Event(
                message_id=uuid5(NAMESPACE_URL, f"deployment-confirmation:{applied.adjustment_id}"),
                task_id=applied.task_id,
                source_role=Actor.USER,
                topic=Topic.DEPLOY,
                message_type="deployment.adjustment.confirmed",
                payload={
                    "adjustment_id": str(applied.adjustment_id),
                    "confirmation_id": str(applied.confirmation_id),
                    "version_id": str(applied.applied_version_id),
                },
                outcome="applied",
            )
        )

    def confirm_draft(
        self,
        task_id: UUID,
        adjustment_id: UUID,
        expected_version: int,
        confirmation_id: UUID,
        *,
        session_id: UUID,
    ) -> ForecastVersion:
        draft = self.deployments.adjustment(adjustment_id)
        if draft is None or draft.task_id != task_id:
            raise ValueError("adjustment not found")
        forecast, _ = self.get_forecast(task_id, draft.forecast_id)
        if forecast.deployment_session_id != session_id:
            raise ValueError("confirmation belongs to a different deployment session")
        if draft.status == "applied":
            if draft.confirmation_id != confirmation_id:
                raise ValueError("confirmation ID differs from previously applied adjustment")
            self._confirmation_event(draft)
            assert draft.applied_version_id is not None
            version = self.deployments.version(draft.applied_version_id)
            if version is None:
                raise ValueError("applied version is missing")
            return version
        if draft.status != "draft":
            raise ValueError("rejected adjustment cannot be confirmed")
        self._session(task_id, expected_version, session_id=session_id)
        forecast, session, parent, future, policies = self._adjustment_context(
            task_id, draft.forecast_id
        )
        if (
            session.version != expected_version
            or session.pending_adjustment_id != adjustment_id
            or session.state != "WAITING_FOR_USER_CONFIRMATION"
            or parent.version_id != draft.parent_version_id
        ):
            from interactive_forecasting.orchestration.state_machine import VersionConflict

            raise VersionConflict("adjustment confirmation is stale or not approved")
        version = apply_adjustment(
            forecast, parent, draft, future_values=future, auxiliary_policies=policies
        )
        applied = Adjustment.model_validate(
            draft.model_copy(
                update={
                    "status": "applied",
                    "confirmed_at": utc_now(),
                    "confirmation_id": confirmation_id,
                    "applied_version_id": version.version_id,
                }
            ).model_dump(mode="json")
        )
        changed = session.model_copy(
            update={
                "version": session.version + 1,
                "state": "FORECAST_VERSION_UPDATED",
                "pending_adjustment_id": None,
                "current_version_id": version.version_id,
            }
        )
        self.deployments.commit_adjustment(
            changed, applied, version, expected_version=session.version
        )
        self._confirmation_event(applied)
        return version

    def sensitivity_variables(self, task_id: UUID, forecast_id: UUID) -> tuple[str, ...]:
        forecast, _ = self.get_forecast(task_id, forecast_id)
        prepared, _, _, reference = self._selection(task_id)
        if prepared.prepared is None or forecast.model_artifact != reference:
            raise ValueError("sensitivity forecast differs from frozen selected model")
        fitted = FittedCoreModel.load(self.store, reference)
        return available_variables(fitted)

    def sensitivity(
        self, task_id: UUID, forecast_id: UUID, request: SensitivityRequest
    ) -> SensitivityResult:
        forecast, original = self.get_forecast(task_id, forecast_id)
        session = (
            self.deployments.get_session(forecast.deployment_session_id)
            if forecast.deployment_session_id is not None
            else None
        )
        if (
            session is None
            or session.task_id != task_id
            or session.forecast_id != forecast_id
            or session.upload_extension is None
            or request.base_version_id != original.version_id
        ):
            raise ValueError("sensitivity requires a stored original forecast session")
        prepared, _, trial, reference = self._selection(task_id)
        if (
            prepared.prepared is None
            or forecast.model_artifact != reference
            or forecast.prepared_snapshot != prepared.prepared.dataset.artifact
        ):
            raise ValueError("sensitivity inputs differ from frozen selected artifact")
        fitted = FittedCoreModel.load(self.store, reference)
        future = tuple(
            FutureAuxiliaryValue.model_validate(item)
            for item in json.loads(self.store.read_bytes(forecast.future_auxiliary_artifact))
        )
        context = prepare_context(
            self.store,
            prepared.prepared,
            fitted,
            task_id,
            trial.trial_id,
            reference,
            self.store.read_bytes(forecast.raw_upload),
            session.upload_extension,
            future,
        )
        if not context.validation.ready or context.historical.to_csv(
            index=False
        ).encode() != self.store.read_bytes(forecast.context_artifact):
            raise ValueError("stored forecast context no longer reproduces validated input")
        return run_sensitivity(forecast, original, request, context, fitted)

    def complete(
        self, task_id: UUID, expected_version: int, *, session_id: UUID
    ) -> DeploymentSession:
        session = self._session(task_id, expected_version, session_id=session_id)
        if session.forecast_id is None or session.pending_adjustment_id is not None:
            raise ValueError("cannot complete a session without forecast or with pending draft")
        if session.state == "COMPLETED":
            return session
        changed = session.model_copy(update={"version": session.version + 1, "state": "COMPLETED"})
        return self.deployments.save(changed, expected_version=session.version)
