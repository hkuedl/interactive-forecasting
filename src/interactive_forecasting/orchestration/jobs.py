"""Persistent job lifecycle policy, separate from future worker execution."""

from interactive_forecasting.domain.models import Job, utc_now
from interactive_forecasting.domain.types import JobStatus
from interactive_forecasting.orchestration.state_machine import InvalidTransition

_ALLOWED: dict[JobStatus, set[JobStatus]] = {
    JobStatus.QUEUED: {JobStatus.RUNNING, JobStatus.CANCELLED},
    JobStatus.RUNNING: {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED},
    JobStatus.COMPLETED: set(),
    JobStatus.FAILED: set(),
    JobStatus.CANCELLED: set(),
}


def advance_job(
    job: Job,
    target: JobStatus,
    *,
    completed: int | None = None,
    total: int | None = None,
    error: str | None = None,
) -> Job:
    if target not in _ALLOWED[job.status]:
        raise InvalidTransition(f"job {job.status} cannot transition to {target}")
    new_completed = completed if completed is not None else job.progress_completed
    new_total = total if total is not None else job.progress_total
    if new_total is not None and new_completed > new_total:
        raise ValueError("completed progress exceeds total")
    if target == JobStatus.COMPLETED and new_total is not None and new_completed != new_total:
        raise ValueError("completed job must reach its progress total")
    if target == JobStatus.FAILED and not error:
        raise ValueError("failed job requires an error")
    return job.model_copy(
        update={
            "status": target,
            "progress_completed": new_completed,
            "progress_total": new_total,
            "error": error,
            "updated_at": utc_now(),
        }
    )
