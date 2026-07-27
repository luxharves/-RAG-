"""Document manifest: track file hashes, versions, chunk counts for incremental updates."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from dataclasses import dataclass, field


def file_hash(path: str | Path) -> str:
    """SHA256 hex digest of file contents."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class DocManifest:
    """Manifest for one document."""

    source_file: str
    document_id: str
    file_hash: str
    file_size: int
    total_pages: int
    text_pages: int
    num_chunks: int
    version: str = ""  # first 16 chars of file_hash
    last_updated: str = ""

    def __post_init__(self) -> None:
        if not self.version:
            self.version = self.file_hash[:16]
        if not self.last_updated:
            self.last_updated = time.strftime("%Y-%m-%dT%H:%M:%S")

    def to_dict(self) -> dict:
        return {
            "source_file": self.source_file,
            "document_id": self.document_id,
            "file_hash": self.file_hash,
            "file_size": self.file_size,
            "total_pages": self.total_pages,
            "text_pages": self.text_pages,
            "num_chunks": self.num_chunks,
            "version": self.version,
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DocManifest":
        return cls(**d)


class ManifestStore:
    """Persistent store of document manifests."""

    def __init__(self, store_dir: str | Path):
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self._manifests: dict[str, DocManifest] = {}
        self._load()

    def _path(self) -> Path:
        return self.store_dir / "manifests.json"

    def _load(self) -> None:
        p = self._path()
        if p.exists():
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
            self._manifests = {
                k: DocManifest.from_dict(v) for k, v in data.items()
            }

    def save(self) -> None:
        with open(self._path(), "w", encoding="utf-8") as f:
            json.dump(
                {k: v.to_dict() for k, v in self._manifests.items()},
                f, ensure_ascii=False, indent=2,
            )

    def get(self, source_file: str) -> DocManifest | None:
        return self._manifests.get(source_file)

    def set(self, manifest: DocManifest) -> None:
        self._manifests[manifest.source_file] = manifest

    def remove(self, source_file: str) -> None:
        self._manifests.pop(source_file, None)

    def all_files(self) -> set[str]:
        return set(self._manifests.keys())

    def classify(
        self, current_files: dict[str, str]  # path → sha256
    ) -> dict[str, list[str]]:
        """Classify files: added, unchanged, modified, deleted.

        Args:
            current_files: {file_path: sha256_hash} for files on disk.

        Returns:
            {"added": [...], "unchanged": [...], "modified": [...], "deleted": [...]}
        """
        stored = self.all_files()
        current = set(current_files.keys())
        result = {"added": [], "unchanged": [], "modified": [], "deleted": []}

        for path in current - stored:
            result["added"].append(path)
        for path in current & stored:
            m = self._manifests[path]
            if m.file_hash == current_files[path]:
                result["unchanged"].append(path)
            else:
                result["modified"].append(path)
        for path in stored - current:
            result["deleted"].append(path)

        return result
