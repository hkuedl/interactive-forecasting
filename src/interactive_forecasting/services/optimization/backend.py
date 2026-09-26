"""Optuna is confined behind a typed search backend; callers never see its trials."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
from uuid import UUID, uuid4

import optuna
from optuna.distributions import CategoricalDistribution, FloatDistribution, IntDistribution
from optuna.trial import TrialState

from interactive_forecasting.domain.search import (
    BackendConfig,
    CandidateRequest,
    ParameterDomain,
    Scalar,
    SearchSpace,
    TrialResult,
)
from interactive_forecasting.domain.types import ModelFamily


@dataclass(frozen=True)
class BackendProposal:
    token: UUID
    request: CandidateRequest


class SearchBackend(Protocol):
    config: BackendConfig

    def propose(
        self, space: SearchSpace, eligible: tuple[ModelFamily, ...], *, seed: int
    ) -> BackendProposal: ...
    def enqueue(self, request: CandidateRequest, space: SearchSpace) -> None: ...
    def observe(self, token: UUID, result: TrialResult) -> None: ...
    def metadata(self) -> dict[str, str | int | float | bool]: ...
    def restore_completed(
        self, results: tuple[TrialResult, ...], spaces: dict[UUID, SearchSpace]
    ) -> None: ...


def _distribution(domain: ParameterDomain) -> optuna.distributions.BaseDistribution:
    if domain.kind == "categorical":
        return CategoricalDistribution(list(domain.choices))
    if domain.kind == "fixed":
        return CategoricalDistribution([domain.value])
    assert domain.low is not None and domain.high is not None
    if domain.kind == "integer":
        return IntDistribution(
            int(domain.low), int(domain.high), step=int(domain.step or 1), log=domain.log
        )
    return FloatDistribution(domain.low, domain.high, step=domain.step, log=domain.log)


def _key(family: ModelFamily, name: str) -> str:
    # SearchSpace UUIDs identify provenance, not independent sampler dimensions.
    return f"search:{family.value}:{name}"


def _family_key(eligible: tuple[ModelFamily, ...]) -> str:
    return f"search:family:{','.join(family.value for family in eligible)}"


class OptunaBackend:
    """In-memory ask/tell TPE. Completed history can be replayed, not sampler RNG state."""

    def __init__(self, config: BackendConfig):
        self.config = config
        self._study = self._new_study()
        self._pending: dict[
            UUID, tuple[optuna.trial.Trial | None, SearchSpace, CandidateRequest]
        ] = {}
        self._queue: list[tuple[CandidateRequest, SearchSpace]] = []
        self._history: list[tuple[TrialResult, SearchSpace]] = []
        self._active_space: SearchSpace | None = None
        self._eligible: tuple[ModelFamily, ...] = ()

    def _new_study(self) -> optuna.Study:
        config = self.config
        sampler = optuna.samplers.TPESampler(
            seed=config.seed,
            n_startup_trials=config.n_startup_trials,
            n_ei_candidates=config.n_ei_candidates,
            consider_prior=None,
            prior_weight=config.prior_weight,
            consider_magic_clip=config.consider_magic_clip,
            consider_endpoints=config.consider_endpoints,
            gamma=None,
            weights=None,
            multivariate=config.multivariate,
            group=config.group,
            warn_independent_sampling=config.warn_independent_sampling,
            constant_liar=config.constant_liar,
            constraints_func=None,
        )
        return optuna.create_study(direction=config.direction, sampler=sampler)

    def metadata(self) -> dict[str, str | int | float | bool]:
        return {
            "backend": "optuna",
            "backend_version": optuna.__version__,
            "sampler": self.config.sampler,
            "seed": self.config.seed,
            "direction": self.config.direction,
            "n_startup_trials": self.config.n_startup_trials,
            "n_ei_candidates": self.config.n_ei_candidates,
            "prior_weight": self.config.prior_weight,
            "consider_magic_clip": self.config.consider_magic_clip,
            "consider_endpoints": self.config.consider_endpoints,
            "gamma_rule": self.config.gamma_rule,
            "weights_rule": self.config.weights_rule,
            "group": self.config.group,
            "warn_independent_sampling": self.config.warn_independent_sampling,
            "constraints": self.config.constraints,
            "multivariate": self.config.multivariate,
            "constant_liar": self.config.constant_liar,
            "history_policy": "compatible_dimensions_v1",
            "sampler_parameter_keys": "run_local_v1",
            "rng_replay": "seeded_reconstruction",
        }

    def enqueue(self, request: CandidateRequest, space: SearchSpace) -> None:
        space.validate_partial_values(request.family, request.values)
        self._queue.append((request, space))

    def propose(
        self, space: SearchSpace, eligible: tuple[ModelFamily, ...], *, seed: int
    ) -> BackendProposal:
        if not eligible:
            raise ValueError("no capability-eligible model family")
        eligible = tuple(sorted(eligible, key=lambda family: family.value))
        self._activate(space, eligible)
        queued: CandidateRequest | None = None
        if self._queue:
            queued, queued_space = self._queue.pop(0)
            if queued_space.space_id != space.space_id or queued.family not in eligible:
                raise ValueError("enqueued candidate is incompatible with active space")
            try:
                space.validate_values(queued.family, queued.values)
            except ValueError:
                fixed = {
                    _family_key(eligible): (queued.family.value),
                    **{_key(queued.family, name): value for name, value in queued.values.items()},
                }
                self._study.enqueue_trial(fixed)
            else:
                token = uuid4()
                self._pending[token] = (None, space, queued)
                return BackendProposal(token, queued)
        trial = self._study.ask()
        family_names = [family.value for family in eligible]
        family = ModelFamily(trial.suggest_categorical(_family_key(eligible), family_names))
        values: dict[str, Scalar] = {}
        for domain in space.domains_for(family):
            if (
                domain.condition_on is not None
                and values.get(domain.condition_on) != domain.condition_value
            ):
                continue
            name = _key(family, domain.name)
            if domain.kind == "fixed":
                values[domain.name] = domain.value
            elif domain.kind == "categorical":
                values[domain.name] = trial.suggest_categorical(name, list(domain.choices))
            elif domain.kind == "integer":
                assert domain.low is not None and domain.high is not None
                values[domain.name] = trial.suggest_int(
                    name,
                    int(domain.low),
                    int(domain.high),
                    step=int(domain.step or 1),
                    log=domain.log,
                )
            else:
                assert domain.low is not None and domain.high is not None
                values[domain.name] = trial.suggest_float(
                    name,
                    domain.low,
                    domain.high,
                    step=domain.step,
                    log=domain.log,
                )
        request = CandidateRequest(
            family=family,
            values=values,
            source=queued.source if queued is not None else "sampled",
            seed=queued.seed if queued is not None else seed,
        )
        space.validate_values(family, values)
        token = uuid4()
        self._pending[token] = (trial, space, request)
        return BackendProposal(token, request)

    def observe(self, token: UUID, result: TrialResult) -> None:
        if token not in self._pending:
            raise ValueError("unknown or already observed proposal")
        trial, space, request = self._pending[token]
        if result.request != request or result.space_id != space.space_id:
            raise ValueError("observed result does not match pending proposal")
        del self._pending[token]
        if trial is not None:
            if result.status == "completed":
                assert result.objective is not None
                self._study.tell(trial, result.objective)
            else:
                self._study.tell(trial, state=TrialState.FAIL)
        elif result.status == "completed":
            self._add_completed(result, space)
        if result.status == "completed" and result.candidate is not None:
            self._history.append((result, space))

    def restore_completed(
        self, results: tuple[TrialResult, ...], spaces: dict[UUID, SearchSpace]
    ) -> None:
        self._history = [
            (result, spaces[result.candidate.space_id])
            for result in results
            if result.status == "completed" and result.candidate is not None
        ]
        if spaces:
            active = next(reversed(spaces.values()))
            self._activate(
                active,
                tuple(
                    sorted(
                        (item.family for item in active.families), key=lambda family: family.value
                    )
                ),
            )

    def _activate(self, space: SearchSpace, eligible: tuple[ModelFamily, ...]) -> None:
        previous = self._active_space
        changed = (
            previous is None
            or previous.families != space.families
            or previous.features != space.features
            or self._eligible != eligible
        )
        self._active_space, self._eligible = space, eligible
        if changed and self._history:
            if self._pending:
                raise ValueError("cannot change sampler domains while proposals are pending")
            # Replay every completed outcome, projecting compatible observations into
            # the active distributions (including restricted categorical/family choices).
            self._study = self._new_study()
            for result, source_space in self._history:
                self._add_completed(result, source_space)

    def _add_completed(self, result: TrialResult, space: SearchSpace) -> None:
        assert result.candidate is not None
        request = result.candidate.request
        params: dict[str, Scalar] = {}
        distributions: dict[str, optuna.distributions.BaseDistribution] = {}
        active = self._active_space or space
        eligible = self._eligible or tuple(item.family for item in active.families)
        if request.family in eligible:
            family_key = _family_key(eligible)
            params[family_key] = request.family.value
            distributions[family_key] = CategoricalDistribution([item.value for item in eligible])
            source_domains = {domain.name: domain for domain in space.domains_for(request.family)}
            for domain in active.domains_for(request.family):
                original = source_domains.get(domain.name)
                if (
                    original is None
                    or domain.kind == "fixed"
                    or domain.name not in request.values
                    or not domain.contains(request.values[domain.name])
                    or (
                        original.kind != "fixed"
                        and (original.kind, original.log) != (domain.kind, domain.log)
                    )
                    or (original.condition_on, original.condition_value)
                    != (domain.condition_on, domain.condition_value)
                ):
                    continue
                name = _key(request.family, domain.name)
                params[name] = request.values[domain.name]
                distributions[name] = _distribution(domain)
        assert result.objective is not None
        self._study.add_trial(
            optuna.trial.create_trial(
                params=params,
                distributions=distributions,
                value=result.objective,
                state=TrialState.COMPLETE,
            )
        )
