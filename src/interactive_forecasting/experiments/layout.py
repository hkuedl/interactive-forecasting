"""Artifact directory convention for future experiment runs."""

from pathlib import Path
from uuid import UUID

from interactive_forecasting.storage.artifacts import ArtifactStore


def prepare_run_directory(store: ArtifactStore, spec_id: str, run_id: UUID) -> Path:
    """Create only raw/ and metrics/ directories; no result values or files."""
    return store.reproduction_dir(spec_id, run_id)
