"""Backup and restore utilities for engram data."""

from __future__ import annotations

import json
import logging
import sys
import tarfile
import tempfile
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("engram")

_VERSION = "0.2.0"


async def backup(episodic, graph, output_path: str) -> dict:
    """Export all episodic and semantic data to a tar.gz archive.

    Returns manifest dict with counts and timestamp.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        # --- Export episodic memories via ChromaDB ---
        collection = episodic._ensure_collection()
        count = collection.count()
        if count > 0:
            data = collection.get(include=["documents", "metadatas"])
            (tmp_path / "episodic.json").write_text(json.dumps(data, default=str))

        # --- Export semantic graph ---
        graph_data = await graph.dump()
        (tmp_path / "semantic.json").write_text(json.dumps(graph_data, default=str))

        # --- Manifest ---
        manifest = {
            "version": _VERSION,
            "timestamp": datetime.now().isoformat(),
            "episodic_count": count,
            "semantic_nodes": len(graph_data.get("nodes", [])),
        }
        (tmp_path / "manifest.json").write_text(json.dumps(manifest))

        # --- Package all files into tar.gz ---
        with tarfile.open(output_path, "w:gz") as tar:
            for f in tmp_path.iterdir():
                tar.add(f, arcname=f.name)

    return manifest


async def restore(episodic, graph, archive_path: str) -> dict:
    """Import episodic and semantic data from a tar.gz backup archive.

    Episodic memories are added directly via ChromaDB collection upsert.
    Semantic nodes and edges are imported via graph methods.
    Returns counts of restored items.
    """
    from engram.models import SemanticEdge, SemanticNode

    path = Path(archive_path)
    if not path.exists():
        raise FileNotFoundError(f"Archive not found: {archive_path}")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        with tarfile.open(archive_path, "r:gz") as tar:
            # Safe extraction: Python 3.12+ supports filter="data" to block
            # path traversal (e.g. ../../etc/passwd). Fall back to manual
            # validation on older Python versions.
            if sys.version_info >= (3, 12):
                tar.extractall(tmp_path, filter="data")
            else:
                for member in tar.getmembers():
                    # Resolve member path relative to destination; reject if it
                    # would escape the destination directory.
                    member_path = (tmp_path / member.name).resolve()
                    try:
                        member_path.relative_to(tmp_path.resolve())
                    except ValueError:
                        raise ValueError(
                            f"Archive member '{member.name}' would extract outside "
                            "destination directory — possible path traversal attack."
                        )
                tar.extractall(tmp_path)

        manifest_file = tmp_path / "manifest.json"
        manifest = json.loads(manifest_file.read_text()) if manifest_file.exists() else {}

        # --- Restore episodic memories ---
        episodic_count = 0
        episodic_file = tmp_path / "episodic.json"
        if episodic_file.exists():
            data = json.loads(episodic_file.read_text())
            ids = data.get("ids", [])
            documents = data.get("documents", [])
            metadatas = data.get("metadatas", [])
            if ids:
                collection = episodic._ensure_collection()
                collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
                episodic_count = len(ids)

        # --- Restore semantic graph ---
        node_count = 0
        edge_count = 0
        semantic_file = tmp_path / "semantic.json"
        if semantic_file.exists():
            graph_data = json.loads(semantic_file.read_text())
            node_failures = 0
            for node_dict in graph_data.get("nodes", []):
                try:
                    node = SemanticNode(**node_dict)
                    await graph.add_node(node)
                    node_count += 1
                except Exception as exc:
                    node_failures += 1
                    logger.warning("Failed to restore semantic node %r: %s", node_dict, exc)
            edge_failures = 0
            for edge_dict in graph_data.get("edges", []):
                try:
                    edge = SemanticEdge(**edge_dict)
                    await graph.add_edge(edge)
                    edge_count += 1
                except Exception as exc:
                    edge_failures += 1
                    logger.warning("Failed to restore semantic edge %r: %s", edge_dict, exc)

    return {
        "source_version": manifest.get("version"),
        "source_timestamp": manifest.get("timestamp"),
        "episodic_restored": episodic_count,
        "semantic_nodes_restored": node_count,
        "semantic_edges_restored": edge_count,
        "node_failures": node_failures,
        "edge_failures": edge_failures,
    }
