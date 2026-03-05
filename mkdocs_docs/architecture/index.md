# Architecture

Engram is a dual-memory AI system modeled after how humans store and retrieve information.

## Core Concept

Humans use two complementary memory systems:

- **Episodic memory** — "What happened?" — specific events, timestamped experiences
- **Semantic memory** — "What do I know?" — facts, concepts, relationships

Engram mirrors this with a vector store (episodic) and a knowledge graph (semantic), unified by an LLM reasoning engine.

## System Diagram

```mermaid
flowchart TD
    subgraph Interfaces
        CLI["CLI (Typer)"]
        MCP["MCP Server (stdio)"]
        HTTP["HTTP API (FastAPI)"]
        WS["WebSocket /ws"]
    end

    subgraph Agents
        CC["Claude Code"]
        OC["OpenClaw"]
        CU["Cursor"]
        ANY["Any MCP Client"]
    end

    CC & OC & CU & ANY --> MCP
    Interfaces --> Auth

    Auth["Auth Middleware\n(JWT + RBAC, optional)"]
    Auth --> Tenant["TenantContext\n(ContextVar)"]

    Tenant --> RP["Recall Pipeline"]

    RP --> EP["EpisodicStore\n(Qdrant)"]
    RP --> SG["SemanticGraph\n(NetworkX + SQLite/PG)"]
    RP --> Fed["Federation Layer"]

    EP & SG --> RE["Reasoning Engine\n(Gemini via litellm)"]
    EP --> Cache["Redis Cache\n(optional)"]
    WS --> EB["Event Bus\n(push events)"]

    subgraph Fed["Federated Knowledge"]
        M0["mem0"]
        LR["LightRAG"]
        GR["Graphiti"]
        Custom["REST / File / PG / MCP"]
    end
```

## Layers

| Layer | Component | Technology |
|-------|-----------|------------|
| Interface | CLI, MCP, HTTP API, WebSocket | Typer, FastMCP, FastAPI |
| Auth | JWT + API keys, RBAC | python-jose |
| Tenancy | ContextVar propagation | Python contextvars |
| Recall | Pipeline: decide > resolve > search > fuse | Custom |
| Episodic | Vector store | Qdrant (embedded or server) |
| Semantic | Knowledge graph | NetworkX + SQLite/PostgreSQL |
| Reasoning | LLM synthesis | Gemini via litellm |
| Capture | Session watchers | inotify/watchdog |
| Federation | External providers | REST, File, PG, MCP adapters |
| Cache | Result caching | Redis (optional) |
| Observability | Tracing + audit | OpenTelemetry, JSONL |

## Data Flow: Recall

```mermaid
sequenceDiagram
    participant Agent
    participant Pipeline as Recall Pipeline
    participant Episodic as EpisodicStore
    participant Graph as SemanticGraph
    participant LLM as Reasoning Engine

    Agent->>Pipeline: recall("production incidents last week")
    Pipeline->>Pipeline: Query Decision (trivial check)
    Pipeline->>Pipeline: Temporal Resolution ("last week" → date range)
    Pipeline->>Pipeline: Entity Resolution (pronouns → named entities)
    par Parallel search
        Pipeline->>Episodic: vector similarity search
        Pipeline->>Graph: entity + keyword search
    end
    Episodic-->>Pipeline: episodic results
    Graph-->>Pipeline: graph results
    Pipeline->>Pipeline: Dedup + composite scoring
    Pipeline->>LLM: fuse context (think mode)
    LLM-->>Agent: synthesized answer
```

## Data Flow: Ingestion

```mermaid
sequenceDiagram
    participant Source as Agent / Watcher
    participant Gate as Entity Gate
    participant Extractor as Entity Extractor
    participant Classifier as Memory Classifier
    participant Episodic as EpisodicStore
    participant Graph as SemanticGraph

    Source->>Gate: ingest(messages)
    Gate->>Extractor: extract entities (LLM)
    Extractor-->>Gate: entities found?
    alt No entities
        Gate-->>Source: skip (noise filtered)
    else Entities found
        Gate->>Classifier: classify memory type
        Classifier-->>Gate: fact / decision / preference / ...
        Gate->>Episodic: store with embedding
        Gate->>Graph: upsert entities + relations
        Gate-->>Source: stored
    end
```

## Component Deep Dives

- [Episodic Memory](episodic-memory.md) — Qdrant vector store, decay, scoring
- [Semantic Graph](semantic-graph.md) — NetworkX graph, SQLite/PG backend, query DSL
- [Recall Pipeline](recall-pipeline.md) — Full pipeline walkthrough
- [Entity-Gated Ingestion](entity-gated-ingestion.md) — Why and how entities gate storage
