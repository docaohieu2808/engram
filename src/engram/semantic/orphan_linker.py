from __future__ import annotations

import collections
from dataclasses import dataclass

from engram.models import SemanticEdge


@dataclass
class LinkSuggestion:
    orphan_key: str
    target_key: str
    relation: str
    co_mentions: int


def _infer_relation(orphan_type: str, target_type: str) -> tuple[str | None, bool]:
    """Return (relation, direction_from_target_to_orphan?).

    None relation means "do not auto-link" (too ambiguous).
    """
    if orphan_type in {"Technology", "Service"} and target_type == "Project":
        return "uses", True
    if orphan_type == "Server" and target_type == "Project":
        return "deployed_on", True
    if orphan_type == "Environment" and target_type == "Project":
        return "runs_on", True
    if orphan_type == "Person" and target_type == "Project":
        return "works_on", False
    return None, False


async def suggest_orphan_links(graph, episodic_store, recent: int = 1000, min_co_mentions: int = 3) -> list[LinkSuggestion]:
    """Suggest links for orphan nodes using episodic co-mentions.

    Strategy:
    - orphan node = node with degree 0 in semantic graph
    - if orphan entity and connected entity co-occur in many recent memories,
      suggest a typed edge
    """
    nodes = await graph.get_nodes()
    edges = await graph.get_edges()

    out_deg = collections.Counter(e.from_node for e in edges)
    in_deg = collections.Counter(e.to_node for e in edges)

    node_type: dict[str, str] = {n.key: n.type for n in nodes}
    name_to_keys: dict[str, list[str]] = collections.defaultdict(list)
    for n in nodes:
        name_to_keys[n.name.casefold()].append(n.key)

    orphan_keys = {k for k in node_type if out_deg[k] == 0 and in_deg[k] == 0}
    connected_keys = {k for k in node_type if (out_deg[k] + in_deg[k]) > 0}

    recents = await episodic_store.get_recent(recent)
    pair_counts: collections.Counter[tuple[str, str]] = collections.Counter()

    for m in recents:
        ents = [str(e).strip() for e in (m.entities or []) if str(e).strip()]
        if len(ents) < 2:
            continue

        present: set[str] = set()
        for en in ents:
            present.update(name_to_keys.get(en.casefold(), []))

        if len(present) < 2:
            continue

        orphan_present = [k for k in present if k in orphan_keys]
        connected_present = [k for k in present if k in connected_keys]

        for ok in orphan_present:
            for tk in connected_present:
                if ok != tk:
                    pair_counts[(ok, tk)] += 1

    existing = {(e.from_node, e.to_node, e.relation) for e in edges}
    suggestions: list[LinkSuggestion] = []

    for (orphan_key, target_key), cnt in pair_counts.items():
        if cnt < min_co_mentions:
            continue

        orphan_type = node_type.get(orphan_key, "")
        target_type = node_type.get(target_key, "")
        relation, from_target = _infer_relation(orphan_type, target_type)
        if not relation:
            continue

        from_key, to_key = (target_key, orphan_key) if from_target else (orphan_key, target_key)
        if (from_key, to_key, relation) in existing:
            continue

        suggestions.append(
            LinkSuggestion(
                orphan_key=orphan_key,
                target_key=target_key,
                relation=relation,
                co_mentions=cnt,
            )
        )

    suggestions.sort(key=lambda s: s.co_mentions, reverse=True)
    return suggestions


async def apply_suggestions(graph, suggestions: list[LinkSuggestion], limit: int = 50) -> int:
    applied = 0
    nodes = await graph.get_nodes()
    by_key = {n.key: n for n in nodes}
    for s in suggestions[:limit]:
        ot = by_key.get(s.orphan_key).type if s.orphan_key in by_key else ""
        tt = by_key.get(s.target_key).type if s.target_key in by_key else ""
        relation, from_target = _infer_relation(ot, tt)
        if not relation:
            continue
        from_key, to_key = (s.target_key, s.orphan_key) if from_target else (s.orphan_key, s.target_key)

        edge = SemanticEdge(
            from_node=from_key,
            to_node=to_key,
            relation=relation,
            weight=min(1.0, 0.4 + s.co_mentions * 0.08),
            attributes={"source": "orphan_autolink", "co_mentions": s.co_mentions},
        )
        await graph.add_edge(edge)
        applied += 1
    return applied
