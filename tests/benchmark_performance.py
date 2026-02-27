"""Engram server performance benchmark.

Measures p50/p95/p99 latency for remember, recall, think, and health endpoints
across different concurrency levels and memory dataset sizes.

Usage:
    python tests/benchmark_performance.py --host 127.0.0.1 --port 8765
    python tests/benchmark_performance.py --quick
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

import aiohttp


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class LatencyResult:
    """Collected latencies for a single operation."""
    operation: str
    latencies: list[float] = field(default_factory=list)
    errors: int = 0

    def p50(self) -> float:
        return statistics.median(self.latencies) if self.latencies else 0.0

    def p95(self) -> float:
        if not self.latencies:
            return 0.0
        sorted_l = sorted(self.latencies)
        idx = int(len(sorted_l) * 0.95)
        return sorted_l[min(idx, len(sorted_l) - 1)]

    def p99(self) -> float:
        if not self.latencies:
            return 0.0
        sorted_l = sorted(self.latencies)
        idx = int(len(sorted_l) * 0.99)
        return sorted_l[min(idx, len(sorted_l) - 1)]

    def mean(self) -> float:
        return statistics.mean(self.latencies) if self.latencies else 0.0

    def total(self) -> int:
        return len(self.latencies) + self.errors


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

async def timed_post(
    session: aiohttp.ClientSession,
    url: str,
    payload: dict,
    result: LatencyResult,
) -> None:
    """Execute a POST, record latency in ms, increment error count on failure."""
    t0 = time.perf_counter()
    try:
        async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=30)) as resp:
            await resp.read()
            if resp.status >= 500:
                result.errors += 1
                return
    except Exception:
        result.errors += 1
        return
    result.latencies.append((time.perf_counter() - t0) * 1000)


async def timed_get(
    session: aiohttp.ClientSession,
    url: str,
    result: LatencyResult,
) -> None:
    """Execute a GET, record latency in ms."""
    t0 = time.perf_counter()
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
            await resp.read()
            if resp.status >= 500:
                result.errors += 1
                return
    except Exception:
        result.errors += 1
        return
    result.latencies.append((time.perf_counter() - t0) * 1000)


# ---------------------------------------------------------------------------
# Benchmark runners
# ---------------------------------------------------------------------------

async def bench_health(base_url: str, iterations: int, concurrency: int) -> LatencyResult:
    result = LatencyResult("health")
    url = f"{base_url}/health"
    async with aiohttp.ClientSession() as session:
        for batch_start in range(0, iterations, concurrency):
            batch_size = min(concurrency, iterations - batch_start)
            tasks = [timed_get(session, url, result) for _ in range(batch_size)]
            await asyncio.gather(*tasks)
    return result


async def bench_remember(base_url: str, iterations: int, concurrency: int) -> LatencyResult:
    result = LatencyResult("remember")
    url = f"{base_url}/api/v1/remember"
    async with aiohttp.ClientSession() as session:
        for batch_start in range(0, iterations, concurrency):
            batch_size = min(concurrency, iterations - batch_start)
            payloads = [
                {"content": f"Benchmark memory #{batch_start + i}: the sky is blue on a clear day", "memory_type": "fact"}
                for i in range(batch_size)
            ]
            tasks = [timed_post(session, url, p, result) for p in payloads]
            await asyncio.gather(*tasks)
    return result


async def bench_recall(base_url: str, iterations: int, concurrency: int) -> LatencyResult:
    result = LatencyResult("recall")
    url = f"{base_url}/api/v1/recall"
    queries = [
        "sky color on clear day",
        "benchmark memory test",
        "blue sky weather",
        "clear day facts",
        "memory performance test",
    ]
    async with aiohttp.ClientSession() as session:
        for batch_start in range(0, iterations, concurrency):
            batch_size = min(concurrency, iterations - batch_start)
            tasks = [
                timed_post(session, url, {"query": queries[(batch_start + i) % len(queries)]}, result)
                for i in range(batch_size)
            ]
            await asyncio.gather(*tasks)
    return result


async def bench_think(base_url: str, iterations: int, concurrency: int) -> LatencyResult:
    result = LatencyResult("think")
    url = f"{base_url}/api/v1/think"
    questions = [
        "What do I know about the sky?",
        "Summarize benchmark results",
        "What facts have been stored?",
    ]
    async with aiohttp.ClientSession() as session:
        for batch_start in range(0, iterations, concurrency):
            batch_size = min(concurrency, iterations - batch_start)
            tasks = [
                timed_post(session, url, {"question": questions[(batch_start + i) % len(questions)]}, result)
                for i in range(batch_size)
            ]
            await asyncio.gather(*tasks)
    return result


# ---------------------------------------------------------------------------
# Memory scale test: sequential insert then query
# ---------------------------------------------------------------------------

async def bench_memory_scale(
    base_url: str,
    memory_count: int,
    query_iterations: int,
) -> dict[str, LatencyResult]:
    """Insert `memory_count` memories sequentially, then run recall queries."""
    remember_url = f"{base_url}/api/v1/remember"
    recall_url = f"{base_url}/api/v1/recall"

    insert_result = LatencyResult(f"remember@{memory_count}")
    recall_result = LatencyResult(f"recall@{memory_count}")

    print(f"  Inserting {memory_count} memories...", end="", flush=True)
    async with aiohttp.ClientSession() as session:
        # Insert in batches of 10 to avoid overwhelming the server
        batch_size = 10
        for i in range(0, memory_count, batch_size):
            end = min(i + batch_size, memory_count)
            tasks = [
                timed_post(
                    session,
                    remember_url,
                    {"content": f"Scale test memory #{j}: topic_{j % 50} detail data point {j}", "memory_type": "fact"},
                    insert_result,
                )
                for j in range(i, end)
            ]
            await asyncio.gather(*tasks)
            if (i // batch_size) % 10 == 0:
                print(".", end="", flush=True)

        print(f" done ({insert_result.errors} errors)")

        print(f"  Querying {query_iterations} times...", end="", flush=True)
        for i in range(0, query_iterations, batch_size):
            end = min(i + batch_size, query_iterations)
            tasks = [
                timed_post(
                    session,
                    recall_url,
                    {"query": f"topic_{j % 50} detail"},
                    recall_result,
                )
                for j in range(i, end)
            ]
            await asyncio.gather(*tasks)
        print(f" done ({recall_result.errors} errors)")

    return {"insert": insert_result, "recall": recall_result}


# ---------------------------------------------------------------------------
# Memory usage via /health
# ---------------------------------------------------------------------------

async def get_server_memory_mb(base_url: str) -> Optional[float]:
    """Attempt to read memory_mb from /health endpoint JSON."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{base_url}/health",
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    # Try common field names
                    for key in ("memory_mb", "memory", "rss_mb", "mem_mb"):
                        if key in data:
                            return float(data[key])
                    # Nested structure e.g. {"system": {"memory_mb": ...}}
                    if isinstance(data.get("system"), dict):
                        for key in ("memory_mb", "memory", "rss_mb"):
                            if key in data["system"]:
                                return float(data["system"][key])
    except Exception:
        pass

    # Fallback: try psutil for local process memory approximation
    try:
        import psutil
        import os
        proc = psutil.Process(os.getpid())
        return proc.memory_info().rss / 1024 / 1024
    except ImportError:
        pass

    return None


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def _col(value: str, width: int, align: str = "<") -> str:
    return f"{value:{align}{width}}"


def print_latency_table(results: list[LatencyResult], title: str) -> None:
    cols = [
        ("Operation", 22),
        ("Count", 7),
        ("Errors", 7),
        ("Mean ms", 9),
        ("p50 ms", 9),
        ("p95 ms", 9),
        ("p99 ms", 9),
    ]
    total_width = sum(w for _, w in cols) + len(cols) * 3 + 1
    sep = "-" * total_width

    print(f"\n{title}")
    print(sep)
    header = " | ".join(_col(name, w) for name, w in cols)
    print(f"| {header} |")
    print(sep)
    for r in results:
        row = " | ".join([
            _col(r.operation, cols[0][1]),
            _col(str(r.total()), cols[1][1], ">"),
            _col(str(r.errors), cols[2][1], ">"),
            _col(f"{r.mean():.1f}", cols[3][1], ">"),
            _col(f"{r.p50():.1f}", cols[4][1], ">"),
            _col(f"{r.p95():.1f}", cols[5][1], ">"),
            _col(f"{r.p99():.1f}", cols[6][1], ">"),
        ])
        print(f"| {row} |")
    print(sep)


def print_scale_table(scale_results: dict[int, dict[str, LatencyResult]]) -> None:
    cols = [
        ("Memory Count", 14),
        ("Insert p50 ms", 14),
        ("Insert p95 ms", 14),
        ("Recall p50 ms", 14),
        ("Recall p95 ms", 14),
        ("Insert Errors", 14),
        ("Recall Errors", 14),
    ]
    total_width = sum(w for _, w in cols) + len(cols) * 3 + 1
    sep = "-" * total_width

    print(f"\nMemory Scale Test Results")
    print(sep)
    header = " | ".join(_col(name, w) for name, w in cols)
    print(f"| {header} |")
    print(sep)
    for count, res in sorted(scale_results.items()):
        ins = res["insert"]
        rec = res["recall"]
        row = " | ".join([
            _col(str(count), cols[0][1], ">"),
            _col(f"{ins.p50():.1f}", cols[1][1], ">"),
            _col(f"{ins.p95():.1f}", cols[2][1], ">"),
            _col(f"{rec.p50():.1f}", cols[3][1], ">"),
            _col(f"{rec.p95():.1f}", cols[4][1], ">"),
            _col(str(ins.errors), cols[5][1], ">"),
            _col(str(rec.errors), cols[6][1], ">"),
        ])
        print(f"| {row} |")
    print(sep)


# ---------------------------------------------------------------------------
# Server connectivity check
# ---------------------------------------------------------------------------

async def check_server(base_url: str) -> bool:
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{base_url}/health",
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                return resp.status < 500
    except Exception as exc:
        print(f"[error] Cannot connect to {base_url}: {exc}", file=sys.stderr)
        return False


# ---------------------------------------------------------------------------
# Main benchmark orchestration
# ---------------------------------------------------------------------------

async def run_concurrency_benchmarks(
    base_url: str,
    concurrency_levels: list[int],
    iterations: int,
    quick: bool,
) -> None:
    """Run remember/recall/think/health at each concurrency level."""
    # think is slow (LLM call) — reduce iterations unless user specified more
    think_iters = min(iterations, 5 if quick else 20)

    for concurrency in concurrency_levels:
        print(f"\n[Concurrency = {concurrency}]")
        results: list[LatencyResult] = []

        print("  Benchmarking health...", end=" ", flush=True)
        results.append(await bench_health(base_url, iterations, concurrency))
        print("done")

        print("  Benchmarking remember...", end=" ", flush=True)
        results.append(await bench_remember(base_url, iterations, concurrency))
        print("done")

        print("  Benchmarking recall...", end=" ", flush=True)
        results.append(await bench_recall(base_url, iterations, concurrency))
        print("done")

        print(f"  Benchmarking think ({think_iters} iters)...", end=" ", flush=True)
        results.append(await bench_think(base_url, think_iters, concurrency))
        print("done")

        print_latency_table(results, f"Concurrency = {concurrency}, Iterations = {iterations}")

        mem_mb = await get_server_memory_mb(base_url)
        if mem_mb is not None:
            print(f"  Server memory usage: {mem_mb:.1f} MB")


async def run_scale_benchmarks(
    base_url: str,
    memory_counts: list[int],
    query_iterations: int,
) -> None:
    """Insert growing sets of memories and measure query latency degradation."""
    scale_results: dict[int, dict[str, LatencyResult]] = {}

    for count in memory_counts:
        print(f"\n[Scale Test: {count} memories]")
        scale_results[count] = await bench_memory_scale(base_url, count, query_iterations)

        mem_mb = await get_server_memory_mb(base_url)
        if mem_mb is not None:
            print(f"  Server memory after {count} inserts: {mem_mb:.1f} MB")

    print_scale_table(scale_results)


async def main() -> int:
    parser = argparse.ArgumentParser(
        description="Engram server performance benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=8765, help="Server port")
    parser.add_argument(
        "--iterations",
        type=int,
        default=50,
        help="Number of requests per operation per concurrency level",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Minimal iterations for CI smoke test (overrides --iterations)",
    )
    parser.add_argument(
        "--skip-scale",
        action="store_true",
        help="Skip the memory scale test (faster run)",
    )
    parser.add_argument(
        "--concurrency",
        nargs="+",
        type=int,
        default=None,
        help="Concurrency levels to test (overrides defaults)",
    )
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"

    # Quick mode overrides
    if args.quick:
        iterations = 10
        concurrency_levels = [1, 5]
        memory_counts = [100]
        query_iterations = 5
    else:
        iterations = args.iterations
        concurrency_levels = args.concurrency or [1, 5, 10, 25]
        memory_counts = [100, 1000, 10000]
        query_iterations = 20

    print(f"Engram Benchmark — {base_url}")
    print(f"Mode: {'quick' if args.quick else 'full'} | Iterations: {iterations}")
    print(f"Concurrency levels: {concurrency_levels}")

    # Verify server is up
    print(f"\nChecking server connectivity...", end=" ", flush=True)
    if not await check_server(base_url):
        print(f"FAILED\nServer not reachable at {base_url}", file=sys.stderr)
        return 1
    print("OK")

    # Report initial memory
    mem_mb = await get_server_memory_mb(base_url)
    if mem_mb is not None:
        print(f"Initial server memory: {mem_mb:.1f} MB")

    # Concurrency benchmarks
    await run_concurrency_benchmarks(base_url, concurrency_levels, iterations, args.quick)

    # Scale benchmarks
    if not args.skip_scale:
        print(f"\n{'='*60}")
        print("Scale Tests: insert N memories then query")
        print(f"Memory counts: {memory_counts} | Query iterations: {query_iterations}")
        await run_scale_benchmarks(base_url, memory_counts, query_iterations)

    print("\nBenchmark complete.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
