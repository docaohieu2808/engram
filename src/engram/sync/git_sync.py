"""Git-friendly incremental memory sync via compressed JSONL chunks.

Exports memories to .engram/ directory for git-based sharing across machines.
Each sync creates a new chunk file (UUID-named, gzipped JSONL) — no merge conflicts.
Manifest tracks synced/imported state.
"""

from __future__ import annotations

import gzip
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MANIFEST_NAME = "manifest.json"
CHUNKS_DIR = "chunks"


def _find_git_root(start: Path) -> Path | None:
    """Walk up from start until .git directory found."""
    p = start.resolve()
    for _ in range(20):
        if (p / ".git").exists():
            return p
        if p.parent == p:
            break
        p = p.parent
    return None


def _get_sync_dir(sync_dir: str | None = None) -> Path:
    """Resolve .engram/ sync directory, defaulting to git repo root."""
    if sync_dir:
        d = Path(sync_dir).expanduser()
    else:
        root = _find_git_root(Path.cwd())
        if not root:
            raise RuntimeError("Not in a git repository. Use --dir to specify sync directory.")
        d = root / ".engram"
    d.mkdir(parents=True, exist_ok=True)
    (d / CHUNKS_DIR).mkdir(exist_ok=True)
    return d


def _load_manifest(sync_dir: Path) -> dict[str, Any]:
    manifest_path = sync_dir / MANIFEST_NAME
    if manifest_path.exists():
        return json.loads(manifest_path.read_text())
    return {"version": "1", "synced_ids": [], "chunks": [], "imported_chunks": [], "last_sync": None}


def _save_manifest(sync_dir: Path, manifest: dict[str, Any]) -> None:
    manifest["last_sync"] = datetime.now(timezone.utc).isoformat()
    (sync_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2))


async def export_memories(episodic_store: Any, sync_dir: str | None = None) -> dict[str, Any]:
    """Export new memories to .engram/chunks/ as gzipped JSONL.

    Only exports memories not already tracked in manifest.
    Returns dict with new_count and chunk_file.
    """
    d = _get_sync_dir(sync_dir)
    manifest = _load_manifest(d)
    synced_ids = set(manifest["synced_ids"])

    collection = episodic_store._ensure_collection()
    count = collection.count()
    if count == 0:
        return {"new_count": 0, "chunk_file": None}

    data = collection.get(include=["documents", "metadatas"])
    new_records = []
    for i, mem_id in enumerate(data["ids"]):
        if mem_id in synced_ids:
            continue
        new_records.append({
            "id": mem_id,
            "content": data["documents"][i],
            "metadata": data["metadatas"][i],
        })

    if not new_records:
        return {"new_count": 0, "chunk_file": None}

    # Write gzipped JSONL chunk
    chunk_name = f"{uuid.uuid4().hex[:8]}.jsonl.gz"
    chunk_path = d / CHUNKS_DIR / chunk_name
    with gzip.open(chunk_path, "wt", encoding="utf-8") as f:
        for record in new_records:
            f.write(json.dumps(record, default=str) + "\n")

    # Update manifest
    manifest["synced_ids"].extend(r["id"] for r in new_records)
    manifest["chunks"].append({
        "file": chunk_name,
        "count": len(new_records),
        "exported_at": datetime.now(timezone.utc).isoformat(),
    })
    _save_manifest(d, manifest)

    return {"new_count": len(new_records), "chunk_file": chunk_name}


async def import_memories(episodic_store: Any, sync_dir: str | None = None) -> dict[str, Any]:
    """Import memories from .engram/chunks/ not yet in this store.

    Reads manifest to determine which chunks have been imported.
    Uses ChromaDB upsert to handle duplicates gracefully.
    """
    d = _get_sync_dir(sync_dir)
    manifest = _load_manifest(d)
    imported_set = {c["file"] for c in manifest.get("imported_chunks", [])}

    chunk_dir = d / CHUNKS_DIR
    total_imported = 0
    new_chunks: list[dict[str, Any]] = []

    for chunk_path in sorted(chunk_dir.glob("*.jsonl.gz")):
        if chunk_path.name in imported_set:
            continue
        records: list[dict[str, Any]] = []
        with gzip.open(chunk_path, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

        if records:
            collection = episodic_store._ensure_collection()
            ids = [r["id"] for r in records]
            docs = [r["content"] for r in records]
            metas = [r["metadata"] for r in records]
            collection.upsert(ids=ids, documents=docs, metadatas=metas)
            total_imported += len(records)
            new_chunks.append({"file": chunk_path.name, "count": len(records)})

    if new_chunks:
        manifest.setdefault("imported_chunks", []).extend(new_chunks)
        _save_manifest(d, manifest)

    return {"imported": total_imported, "chunks_processed": len(new_chunks)}


def sync_status(sync_dir: str | None = None) -> dict[str, Any]:
    """Return manifest summary without touching the store."""
    try:
        d = _get_sync_dir(sync_dir)
        manifest = _load_manifest(d)
        return {
            "synced_ids_count": len(manifest.get("synced_ids", [])),
            "chunks": len(manifest.get("chunks", [])),
            "imported_chunks": len(manifest.get("imported_chunks", [])),
            "last_sync": manifest.get("last_sync"),
            "sync_dir": str(d),
        }
    except RuntimeError as e:
        return {"error": str(e)}
