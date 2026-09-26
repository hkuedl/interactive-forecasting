"""Immutable local artifact storage with relative URIs and SHA-256 verification."""

import hashlib
import os
import re
from pathlib import Path
from uuid import UUID

from interactive_forecasting.domain.models import ArtifactManifest, ArtifactRef


class ArtifactConflict(ValueError):
    pass


class ArtifactStore:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _resolve(self, relative_uri: str) -> Path:
        relative = Path(relative_uri)
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise ValueError("artifact URI must stay under the artifact root")
        target = (self.root / relative).resolve()
        if not target.is_relative_to(self.root):
            raise ValueError("artifact URI escapes artifact root")
        return target

    def put_bytes(self, relative_uri: str, content: bytes) -> ArtifactRef:
        target = self._resolve(relative_uri)
        target.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(content).hexdigest()
        try:
            descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            if target.read_bytes() != content:
                raise ArtifactConflict(
                    f"immutable artifact already exists: {relative_uri}"
                ) from None
        else:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
        return ArtifactRef(uri=relative_uri, sha256=digest, size_bytes=len(content))

    def read_bytes(self, reference: ArtifactRef) -> bytes:
        data = self._resolve(reference.uri).read_bytes()
        if (
            len(data) != reference.size_bytes
            or hashlib.sha256(data).hexdigest() != reference.sha256
        ):
            raise ArtifactConflict("artifact checksum mismatch")
        return data

    def reproduction_dir(self, spec_id: str, run_id: UUID) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", spec_id) or spec_id in {".", ".."}:
            raise ValueError("invalid experiment spec id")
        relative = f"reproduction/{spec_id}/{run_id}"
        target = self._resolve(relative)
        for child in ("raw", "metrics"):
            (target / child).mkdir(parents=True, exist_ok=True)
        return target

    def write_manifest(self, manifest: ArtifactManifest) -> ArtifactRef:
        self.reproduction_dir(manifest.spec_id, manifest.run_id)
        relative = f"reproduction/{manifest.spec_id}/{manifest.run_id}/manifest.json"
        data = manifest.model_dump_json(indent=2).encode("utf-8")
        return self.put_bytes(relative, data)
