# Brain Features Architecture (v0.3.2)

Detailed architecture for the four brain-only features added in v0.3.2. See [system-architecture.md](./system-architecture.md) for the full system overview.

---

## Layer 3h: Memory Audit Trail

**Path:** `src/engram/audit.py` (extended), `src/engram/episodic/store.py` (wired)

**Purpose:** Provide a traceable record of every episodic memory mutation for debugging, compliance, and reversibility.

**Components:**
- **log_modification()** — Records action with before/after values, mod_type, reversible flag, timestamp
- **read_recent(n)** — Retrieves last N audit entries from the log
- **MODIFICATION_TYPES:** memory_create, memory_delete, memory_update, metadata_update, config_change, batch_create, cleanup_expired
- **Wired into:** remember(), delete(), update_metadata(), _update_topic(), cleanup_expired()

**Data Flow:**
```
EpisodicStore.remember() → log_modification(type=memory_create, after=new_memory)
EpisodicStore.delete()   → log_modification(type=memory_delete, before=old_memory)
EpisodicStore.update_*() → log_modification(type=memory_update, before=old, after=new)
```

---

## Layer 3i: Resource-Aware Retrieval

**Path:** `src/engram/resource_tier.py`, `src/engram/reasoning/engine.py` (integrated)

**Purpose:** Gracefully degrade LLM-dependent operations when the LLM provider is unreliable, ensuring the system remains functional at reduced capability.

**Components:**
- **ResourceMonitor** — Sliding-window tracker for LLM call success/failure rates
- **4 Tiers:**

  | Tier | LLM Calls | Behavior |
  |------|-----------|----------|
  | FULL | Allowed | Full synthesis + reasoning |
  | STANDARD | Allowed | Reduced context window |
  | BASIC | Skipped | Raw recall results, no synthesis |
  | READONLY | Blocked | No mutations, read-only |

- **Auto-recovery:** Tier promotes back after 60s cooldown without failures
- **Integration:** think() and summarize() check tier before issuing LLM calls; BASIC returns raw results
- **CLI:** `engram resource-status` shows current tier and sliding-window stats

---

## Layer 3j: Data Constitution

**Path:** `src/engram/constitution.py`, `src/engram/reasoning/engine.py` (integrated)

**Purpose:** Enforce ethical and operational constraints on LLM reasoning by injecting a compact law prefix into every prompt.

**Components:**
- **3 Laws:** (1) namespace isolation — only access own tenant data; (2) no fabrication — only synthesize from stored memories; (3) audit rights — every operation is loggable
- **Auto-creation:** ~/.engram/constitution.md created on first load if absent
- **SHA-256 verification:** Hash stored alongside constitution; tampered files detected on load
- **Prompt injection:** Compact prefix prepended to all reasoning engine and summarize prompts
- **CLI:** `engram constitution-status` shows laws, file path, and hash verification result

---

## Layer 3k: Consolidation Scheduler

**Path:** `src/engram/scheduler.py`, `src/engram/cli/system.py` (integrated)

**Purpose:** Run periodic maintenance tasks automatically in the background without blocking the main process.

**Components:**
- **Asyncio recursive setTimeout pattern** — Each task schedules its own next run after completion, preventing overlap
- **3 default tasks:**

  | Task | Interval | LLM Required |
  |------|----------|--------------|
  | cleanup_expired | Daily | No |
  | consolidate_memories | Every 6h | Yes |
  | decay_report | Daily | No |

- **Tier awareness:** LLM-dependent tasks skipped when ResourceMonitor reports BASIC tier or lower
- **State persistence:** ~/.engram/scheduler_state.json tracks last run, next run, task history
- **Auto-start:** Starts when `engram watch` is invoked
- **CLI:** `engram scheduler-status` shows all tasks with last/next run times

**Data Flow:**
```
engram watch →  Scheduler.start()
                  ├─ _schedule(cleanup_expired, interval=24h)
                  ├─ _schedule(consolidate_memories, interval=6h)
                  └─ _schedule(decay_report, interval=24h)
                          ↓ (on each tick)
                  check resource_tier → skip if BASIC
                  run task → log result → reschedule
                  save state to scheduler_state.json
```
