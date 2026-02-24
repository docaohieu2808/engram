"""HTTP webhook server for external agent integration."""

from __future__ import annotations

from typing import Any, Optional

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from engram.config import Config
from engram.episodic.store import EpisodicStore
from engram.models import MemoryType
from engram.reasoning.engine import ReasoningEngine
from engram.semantic.graph import SemanticGraph


# --- Request/Response Models ---


class IngestRequest(BaseModel):
    messages: list[dict[str, Any]]


class RememberRequest(BaseModel):
    content: str
    memory_type: MemoryType = MemoryType.FACT
    priority: int = 5
    entities: list[str] = []
    tags: list[str] = []


class ThinkRequest(BaseModel):
    question: str


class QueryRequest(BaseModel):
    keyword: str
    node_type: Optional[str] = None  # renamed from `type` to avoid builtin shadow


class SummarizeRequest(BaseModel):
    count: int = 20
    save: bool = False


# --- App Factory ---


def create_app(
    episodic: EpisodicStore,
    graph: SemanticGraph,
    engine: ReasoningEngine,
    ingest_fn: Any = None,
) -> FastAPI:
    """Create FastAPI app wired to memory stores."""
    app = FastAPI(title="engram", description="Memory traces for AI agents")

    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.post("/ingest")
    async def ingest(req: IngestRequest):
        if ingest_fn:
            result = await ingest_fn(req.messages)
            return {"status": "ok", "result": result.model_dump()}
        return {"status": "error", "message": "Ingest function not configured"}

    @app.post("/remember")
    async def remember(req: RememberRequest):
        mem_id = await episodic.remember(
            req.content, memory_type=req.memory_type, priority=req.priority,
            entities=req.entities, tags=req.tags,
        )
        return {"status": "ok", "id": mem_id}

    @app.post("/think")
    async def think(req: ThinkRequest):
        answer = await engine.think(req.question)
        return {"status": "ok", "answer": answer}

    @app.get("/recall")
    async def recall(
        query: str,
        limit: int = 5,
        offset: int = 0,
        memory_type: Optional[str] = None,
        tags: Optional[str] = None,  # comma-separated tag list
    ):
        """Search episodic memories with optional filters and pagination."""
        filters = {"memory_type": memory_type} if memory_type else None
        tag_list = [t.strip() for t in tags.split(",")] if tags else None
        results = await episodic.search(query, limit=limit + offset, filters=filters, tags=tag_list)
        paginated = results[offset:offset + limit]

        # Also search semantic graph for matching entities
        graph_nodes = await graph.query(query)
        graph_results = []
        for node in graph_nodes[:3]:
            related = await graph.get_related([node.name])
            edges = related.get(node.name, {}).get("edges", [])
            graph_results.append({
                "node": node.model_dump(),
                "edges": [e.model_dump() for e in edges[:5]],
            })

        return {
            "status": "ok",
            "results": [r.model_dump() for r in paginated],
            "graph_results": graph_results,
            "total": len(results),
            "offset": offset,
            "limit": limit,
        }

    @app.get("/query")
    async def query(
        keyword: str,
        node_type: Optional[str] = None,
        related_to: Optional[str] = None,
        offset: int = 0,
        limit: int = 50,
    ):
        """Query semantic graph by keyword, type, or relatedness."""
        if related_to:
            related = await graph.get_related([related_to], depth=2)
            nodes = []
            for data in related.values():
                if isinstance(data, dict):
                    nodes.extend(data.get("nodes", []))
        else:
            nodes = await graph.query(keyword, type=node_type)

        paginated = nodes[offset:offset + limit]
        return {
            "status": "ok",
            "results": [n.model_dump() for n in paginated],
            "total": len(nodes),
            "offset": offset,
            "limit": limit,
        }

    @app.post("/cleanup")
    async def cleanup():
        """Delete all expired memories from episodic store."""
        deleted = await episodic.cleanup_expired()
        return {"status": "ok", "deleted": deleted}

    @app.post("/summarize")
    async def summarize(req: SummarizeRequest):
        """Summarize recent N memories into key insights using LLM."""
        summary = await engine.summarize(n=req.count, save=req.save)
        return {"status": "ok", "summary": summary}

    @app.get("/status")
    async def status():
        ep_stats = await episodic.stats()
        sem_stats = await graph.stats()
        return {"episodic": ep_stats, "semantic": sem_stats}

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    return app


def run_server(
    episodic: EpisodicStore,
    graph: SemanticGraph,
    engine: ReasoningEngine,
    config: Config,
    ingest_fn: Any = None,
) -> None:
    """Run HTTP server."""
    app = create_app(episodic, graph, engine, ingest_fn)
    uvicorn.run(app, host=config.serve.host, port=config.serve.port)
