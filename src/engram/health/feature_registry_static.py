"""Static hardcoded lists for the feature registry: CLI commands, HTTP endpoints, MCP tools, etc."""

from __future__ import annotations

# --- CLI Commands (49) ---
CLI_COMMANDS: list[tuple[str, str]] = [
    # system.py
    ("engram health", "System"),
    ("engram status", "System"),
    ("engram dump", "System"),
    ("engram cleanup", "System"),
    ("engram summarize", "System"),
    ("engram ingest", "System"),
    ("engram watch", "System"),
    ("engram decay", "System"),
    ("engram consolidate", "System"),
    ("engram session-start", "System"),
    ("engram session-end", "System"),
    ("engram tui", "System"),
    ("engram resource-status", "System"),
    ("engram constitution-status", "System"),
    ("engram scheduler-status", "System"),
    ("engram graph", "System"),
    ("engram serve", "System"),
    # episodic.py
    ("engram remember", "Memory"),
    ("engram recall", "Memory"),
    # reasoning.py
    ("engram think", "Reasoning"),
    ("engram ask", "Reasoning"),
    # semantic.py
    ("engram add node", "Graph"),
    ("engram add edge", "Graph"),
    ("engram remove node", "Graph"),
    ("engram remove edge", "Graph"),
    ("engram autolink-orphans", "Graph"),
    ("engram search-graph", "Graph"),
    # feedback_cmd.py
    ("engram feedback", "Feedback"),
    # resolve_cmd.py
    ("engram resolve", "Pipeline"),
    # audit_cmd.py
    ("engram audit", "Enterprise"),
    # backup_cmd.py
    ("engram backup", "Enterprise"),
    ("engram restore", "Enterprise"),
    # migrate_cmd.py
    ("engram migrate", "Enterprise"),
    # sync_cmd.py
    ("engram sync", "Enterprise"),
    # benchmark_cmd.py
    ("engram benchmark", "Enterprise"),
    # providers_cmd.py
    ("engram providers", "Integration"),
    ("engram providers list", "Integration"),
    ("engram providers test", "Integration"),
    ("engram providers stats", "Integration"),
    ("engram providers add", "Integration"),
    # auth_cmd.py
    ("engram auth create-key", "Auth"),
    ("engram auth list-keys", "Auth"),
    ("engram auth revoke-key", "Auth"),
    # config_cmd.py
    ("engram config show", "Config"),
    ("engram config set", "Config"),
    ("engram config get", "Config"),
    ("engram schema show", "Config"),
    ("engram schema init", "Config"),
    ("engram schema validate", "Config"),
]

# --- HTTP API Endpoints (35) ---
HTTP_ENDPOINTS: list[tuple[str, str, str]] = [
    # memory_routes.py  → /api/v1/
    ("POST", "/api/v1/remember", "Memory"),
    ("POST", "/api/v1/think", "Memory"),
    ("POST", "/api/v1/ingest", "Memory"),
    ("GET",  "/api/v1/recall", "Memory"),
    ("GET",  "/api/v1/query", "Memory"),
    ("POST", "/api/v1/feedback", "Memory"),
    ("POST", "/api/v1/cleanup", "Memory"),
    ("POST", "/api/v1/cleanup/dedup", "Memory"),
    ("POST", "/api/v1/summarize", "Memory"),
    ("GET",  "/api/v1/status", "Memory"),
    # memories_crud.py → /api/v1/
    ("GET",    "/api/v1/memories", "CRUD"),
    ("GET",    "/api/v1/memories/export", "CRUD"),
    ("POST",   "/api/v1/memories/bulk-delete", "CRUD"),
    ("GET",    "/api/v1/memories/{memory_id}", "CRUD"),
    ("PUT",    "/api/v1/memories/{memory_id}", "CRUD"),
    ("DELETE", "/api/v1/memories/{memory_id}", "CRUD"),
    # graph_routes.py → /api/v1/
    ("GET",    "/api/v1/graph/data", "Graph"),
    ("POST",   "/api/v1/graph/nodes", "Graph"),
    ("PUT",    "/api/v1/graph/nodes/{node_key}", "Graph"),
    ("DELETE", "/api/v1/graph/nodes/{node_key}", "Graph"),
    ("POST",   "/api/v1/graph/edges", "Graph"),
    ("DELETE", "/api/v1/graph/edges", "Graph"),
    ("GET",    "/api/v1/feedback/history", "Graph"),
    # admin_routes.py → /api/v1/
    ("POST", "/api/v1/auth/token", "Admin"),
    ("GET",  "/api/v1/providers", "Admin"),
    ("GET",  "/api/v1/audit/log", "Admin"),
    ("GET",  "/api/v1/scheduler/tasks", "Admin"),
    ("POST", "/api/v1/scheduler/tasks/{task_name}/run", "Admin"),
    ("POST", "/api/v1/benchmark/run", "Admin"),
    # built-in FastAPI / server
    ("GET", "/health", "Core"),
    ("GET", "/", "Core"),
    ("GET", "/ws", "Core"),
    ("GET", "/api/v1/config", "Config"),
    ("GET", "/api/v1/schema", "Config"),
    ("GET", "/docs", "Core"),
]

# --- MCP Tools (20) ---
MCP_TOOLS: list[tuple[str, str]] = [
    # episodic_tools.py
    ("engram_remember", "Memory"),
    ("engram_recall", "Memory"),
    ("engram_get_memory", "Memory"),
    ("engram_timeline", "Memory"),
    ("engram_cleanup", "Memory"),
    ("engram_cleanup_dedup", "Memory"),
    ("engram_ingest", "Memory"),
    ("engram_feedback", "Memory"),
    ("engram_auto_feedback", "Memory"),
    # reasoning_tools.py
    ("engram_ask", "Reasoning"),
    ("engram_think", "Reasoning"),
    ("engram_summarize", "Reasoning"),
    ("engram_status", "Reasoning"),
    # semantic_tools.py
    ("engram_add_entity", "Graph"),
    ("engram_add_relation", "Graph"),
    ("engram_query_graph", "Graph"),
    # session_tools.py
    ("engram_session_start", "Session"),
    ("engram_session_end", "Session"),
    ("engram_session_summary", "Session"),
    ("engram_session_context", "Session"),
]

# --- WebSocket Commands (7) ---
WS_COMMANDS: list[tuple[str, str]] = [
    ("remember", "WS Command"),
    ("recall", "WS Command"),
    ("think", "WS Command"),
    ("feedback", "WS Command"),
    ("query", "WS Command"),
    ("ingest", "WS Command"),
    ("status", "WS Command"),
]

# --- WebSocket Events (4) ---
WS_EVENTS: list[tuple[str, str]] = [
    ("memory_created", "WS Event"),
    ("memory_updated", "WS Event"),
    ("memory_deleted", "WS Event"),
    ("feedback_recorded", "WS Event"),
]

# --- Pipeline Stages (13) ---
PIPELINE_STAGES: list[tuple[str, str]] = [
    # Recall pipeline (6)
    ("Decision (skip trivial)", "Recall"),
    ("Entity Resolution", "Recall"),
    ("Parallel Search (semantic+FTS5+lexical)", "Recall"),
    ("Result Fusion + Dedup", "Recall"),
    ("Re-rank (similarity×importance)", "Recall"),
    ("Format by memory type", "Recall"),
    # Ingest pipeline (4)
    ("Poisoning Guard", "Ingest"),
    ("Auto Memory Detection", "Ingest"),
    ("Entity Extraction (LLM)", "Ingest"),
    ("Semantic Graph Upsert", "Ingest"),
    # Feedback pipeline (3)
    ("Confidence Scoring", "Feedback"),
    ("Importance Adjustment", "Feedback"),
    ("Auto-Delete Check (3× negative)", "Feedback"),
]

# --- Algorithms (19) ---
ALGORITHMS: list[tuple[str, str]] = [
    ("Ebbinghaus Exponential Decay", "Memory"),
    ("Composite Scoring (similarity×retention×recency×frequency)", "Ranking"),
    ("Cosine Similarity Dedup (threshold)", "Storage"),
    ("FTS5 BM25 Full-Text Search", "Search"),
    ("Lexical Keyword Fallback Search", "Search"),
    ("Parallel Async Search Gather", "Search"),
    ("LLM Entity Extraction (JSON)", "Extraction"),
    ("Schema-Guided Relation Extraction", "Extraction"),
    ("Pronoun Coreference Resolution", "Resolution"),
    ("Temporal Expression Resolution", "Resolution"),
    ("Poisoning Guard (LLM classifier)", "Security"),
    ("JWT HS256 Authentication", "Security"),
    ("Sliding Window Rate Limiter", "Rate Limit"),
    ("Redis Cache (recall/think/query)", "Cache"),
    ("HNSW Vector Index (ChromaDB)", "Storage"),
    ("Graph Shortest Path (sqlite)", "Graph"),
    ("Auto-Link Orphan Nodes", "Graph"),
    ("Memory Cluster Consolidation (LLM)", "Consolidation"),
    ("Fail-Open Embedding Queue", "Reliability"),
]

# --- Middleware (6) ---
MIDDLEWARE: list[tuple[str, str]] = [
    ("CorrelationId middleware", "Core"),
    ("Rate Limit middleware", "Security"),
    ("Auth / JWT middleware", "Security"),
    ("CORS middleware", "Core"),
    ("Content Size limit middleware", "Security"),
    ("Input Sanitization middleware", "Security"),
]

# --- Integrations (22) ---
INTEGRATIONS: list[tuple[str, str]] = [
    # Providers / adapters
    ("REST provider adapter", "Provider"),
    ("File provider adapter", "Provider"),
    ("PostgreSQL provider adapter", "Provider"),
    ("MCP provider adapter", "Provider"),
    # Watchers
    ("OpenClaw session watcher", "Watcher"),
    ("Claude Code session watcher", "Watcher"),
    ("Inbox file watcher (inbox/)", "Watcher"),
    # Discovery
    ("Local discovery (localhost probe)", "Discovery"),
    ("Remote host discovery", "Discovery"),
    ("Custom endpoint discovery", "Discovery"),
    # Infrastructure
    ("Redis cache (aioredis)", "Infrastructure"),
    ("Redis event bus (pub/sub)", "Infrastructure"),
    ("Redis rate limiter", "Infrastructure"),
    ("ChromaDB vector store", "Infrastructure"),
    ("SQLite semantic graph", "Infrastructure"),
    ("PostgreSQL semantic graph", "Infrastructure"),
    ("FTS5 SQLite full-text index", "Infrastructure"),
    ("OpenTelemetry (OTLP traces)", "Observability"),
    ("File audit trail (JSONL)", "Observability"),
    ("Retrieval audit trail (JSONL)", "Observability"),
    ("Watcher daemon (fork + PID)", "Observability"),
    ("LiteLLM multi-provider bridge", "LLM"),
]

# --- Data Model Values (37) ---
DATA_MODEL_VALUES: list[tuple[str, str]] = [
    # MemoryType (10)
    ("MemoryType.FACT", "MemoryType"),
    ("MemoryType.DECISION", "MemoryType"),
    ("MemoryType.PREFERENCE", "MemoryType"),
    ("MemoryType.TODO", "MemoryType"),
    ("MemoryType.ERROR", "MemoryType"),
    ("MemoryType.CONTEXT", "MemoryType"),
    ("MemoryType.WORKFLOW", "MemoryType"),
    ("MemoryType.LESSON", "MemoryType"),
    ("MemoryType.AVOIDANCE", "MemoryType"),
    ("MemoryType.PERSONALITY_TRAIT", "MemoryType"),
    # Priority (5)
    ("Priority.LOWEST (1)", "Priority"),
    ("Priority.LOW (3)", "Priority"),
    ("Priority.NORMAL (5)", "Priority"),
    ("Priority.HIGH (7)", "Priority"),
    ("Priority.CRITICAL (10)", "Priority"),
    # FeedbackType (2)
    ("FeedbackType.POSITIVE", "FeedbackType"),
    ("FeedbackType.NEGATIVE", "FeedbackType"),
    # Role (3)
    ("Role.USER", "Role"),
    ("Role.ADMIN", "Role"),
    ("Role.SERVICE", "Role"),
    # ErrorCode (11)
    ("ErrorCode.AUTH_REQUIRED", "ErrorCode"),
    ("ErrorCode.AUTH_INVALID", "ErrorCode"),
    ("ErrorCode.FORBIDDEN", "ErrorCode"),
    ("ErrorCode.NOT_FOUND", "ErrorCode"),
    ("ErrorCode.VALIDATION_ERROR", "ErrorCode"),
    ("ErrorCode.RATE_LIMITED", "ErrorCode"),
    ("ErrorCode.CONTENT_TOO_LARGE", "ErrorCode"),
    ("ErrorCode.EMBEDDING_ERROR", "ErrorCode"),
    ("ErrorCode.LLM_ERROR", "ErrorCode"),
    ("ErrorCode.STORE_ERROR", "ErrorCode"),
    ("ErrorCode.INTERNAL_ERROR", "ErrorCode"),
    # WSEvent types (4)
    ("WSEvent.memory_created", "WSEvent"),
    ("WSEvent.memory_updated", "WSEvent"),
    ("WSEvent.memory_deleted", "WSEvent"),
    ("WSEvent.feedback_recorded", "WSEvent"),
    # ResourceTier (2)
    ("ResourceTier.FULL", "ResourceTier"),
    ("ResourceTier.STANDARD", "ResourceTier"),
]

# --- Runtime Features (22) ---
RUNTIME_FEATURES: list[tuple[str, str]] = [
    ("Constitution law enforcement", "Safety"),
    ("Resource tier monitor (full/standard/basic/readonly)", "Safety"),
    ("Scheduler (cron-style task runner)", "Automation"),
    ("Memory backup to JSON", "Data"),
    ("Memory restore from JSON", "Data"),
    ("FTS5 index rebuild", "Data"),
    ("Semantic dedup cleanup", "Data"),
    ("Memory consolidation (LLM cluster merge)", "Automation"),
    ("Auto session capture (watcher daemon)", "Capture"),
    ("OpenClaw session auto-capture", "Capture"),
    ("Claude Code session auto-capture", "Capture"),
    ("Inbox auto-ingest (file watcher)", "Capture"),
    ("TUI (terminal user interface)", "UI"),
    ("Web UI (static SPA)", "UI"),
    ("WebSocket real-time push events", "API"),
    ("MCP server (stdio transport)", "API"),
    ("HTTP REST API (FastAPI)", "API"),
    ("CLI (Typer)", "API"),
    ("Python SDK (engram package)", "SDK"),
    ("Namespace isolation (multi-tenant)", "Multi-Tenant"),
    ("Event bus (memory or Redis pub/sub)", "Events"),
    ("OTEL distributed tracing", "Observability"),
]
