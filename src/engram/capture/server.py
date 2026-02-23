"""HTTP webhook server for external agent integration."""

from __future__ import annotations

from typing import Any

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
    memory_type: str = "fact"
    priority: int = 5
    entities: list[str] = []


class ThinkRequest(BaseModel):
    question: str


class QueryRequest(BaseModel):
    keyword: str
    type: str | None = None


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
        allow_origins=["http://localhost:*", "http://127.0.0.1:*"],
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
        mem_type = MemoryType(req.memory_type)
        mem_id = await episodic.remember(
            req.content, memory_type=mem_type, priority=req.priority,
            entities=req.entities,
        )
        return {"status": "ok", "id": mem_id}

    @app.post("/think")
    async def think(req: ThinkRequest):
        answer = await engine.think(req.question)
        return {"status": "ok", "answer": answer}

    @app.get("/recall")
    async def recall(query: str, limit: int = 5):
        results = await episodic.search(query, limit=limit)
        return {"status": "ok", "results": [r.model_dump() for r in results]}

    @app.get("/query")
    async def query(keyword: str, type: str | None = None):
        results = await graph.query(keyword, type=type)
        return {"status": "ok", "results": [n.model_dump() for n in results]}

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
