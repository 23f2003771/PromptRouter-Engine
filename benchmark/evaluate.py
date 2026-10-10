"""
Benchmark evaluation suite for PromptRouter Engine.
Evaluates thousands of tasks from the 15,011 Databricks Dolly dataset through
my routing pipeline to measure cost savings, token savings, and latency.

Usage:
    python benchmark/evaluate.py --samples 1000 --random     # 1,000 random data points
    python benchmark/evaluate.py --samples 15000             # Full dataset
    python benchmark/evaluate.py --samples 500 --live        # Live TypeSafe API call
"""

import sys
import json
import time
import random
import argparse
from pathlib import Path
from collections import Counter

# ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from router import router_graph
import router.config as config
from router.config import MODEL_COSTS_PER_1K

DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "dolly_15k.jsonl"
BENCHMARK_DIR = Path(__file__).resolve().parent

# baseline model: what it would cost if everything was sent to the flagship model
BASELINE_MODEL = "gpt-5.6-terra"
BASELINE_RATE = MODEL_COSTS_PER_1K.get(BASELINE_MODEL, 0.0045)


def run_benchmark(num_samples: int = 1000, live_api: bool = False, use_random: bool = True, seed: int = 42):
    if not DATA_FILE.exists():
        print(f"Dataset not found at {DATA_FILE}. Run download_dataset.py first!", flush=True)
        return

    # if not in live API mode, temporarily bypass remote network calls for ultra-fast throughput
    saved_key = config.TYPESAFE_API_KEY
    if not live_api:
        config.TYPESAFE_API_KEY = ""

    print(f"Loading tasks from {DATA_FILE.name}...", flush=True)
    all_tasks = []
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                instruction = item.get("instruction", "").strip()
                if instruction:
                    all_tasks.append({
                        "prompt": instruction,
                        "category": item.get("category", "general"),
                    })

    # select sample points: either random or sequential
    if use_random and num_samples < len(all_tasks):
        random.seed(seed)
        tasks = random.sample(all_tasks, num_samples)
        sample_type = f"randomly sampled (seed={seed})"
    else:
        tasks = all_tasks[:num_samples]
        sample_type = "sequential"

    total_tasks = len(tasks)
    mode_label = "Live TypeSafe API" if live_api else "High-Throughput Engine"
    print(f"Running benchmark on {total_tasks:,} tasks [{sample_type}] ({mode_label})...\n", flush=True)

    model_counts = Counter()
    total_tokens = 0
    total_router_cost = 0.0
    total_baseline_cost = 0.0
    total_latency_ms = 0.0
    blocked_count = 0

    start_bench_time = time.perf_counter()

    for idx, item in enumerate(tasks, 1):
        prompt = item["prompt"]
        t0 = time.perf_counter()
        result = router_graph.invoke({"prompt": prompt})
        step_latency = (time.perf_counter() - t0) * 1000.0

        model = result.get("selected_model", "deepseek-v4-flash")
        tokens = result.get("tokens_used", len(prompt.split()))
        router_cost = result.get("estimated_cost", 0.0)

        # calculate baseline cost if sent to expensive flagship model
        baseline_cost = (tokens / 1000.0) * BASELINE_RATE

        model_counts[model] += 1
        total_tokens += tokens
        total_router_cost += router_cost
        total_baseline_cost += baseline_cost
        total_latency_ms += step_latency

        if result.get("is_blocked"):
            blocked_count += 1

        if idx % 250 == 0 or idx == total_tasks:
            print(f"  Processed {idx:>5,} / {total_tasks:,} tasks ({idx/total_tasks*100:5.1f}%)...", flush=True)

    wall_time_sec = time.perf_counter() - start_bench_time
    throughput_qps = round(total_tasks / max(wall_time_sec, 0.001), 1)

    # restore key
    config.TYPESAFE_API_KEY = saved_key

    # compute final aggregated metrics
    avg_latency_ms = round(total_latency_ms / total_tasks, 2)
    total_saved_dollars = round(max(0.0, total_baseline_cost - total_router_cost), 4)
    pct_cost_saved = round((total_saved_dollars / max(total_baseline_cost, 0.0001)) * 100, 2)

    summary = {
        "dataset": "databricks-dolly-15k",
        "tasks_evaluated": total_tasks,
        "sampling_method": "random" if use_random else "sequential",
        "mode": "live_typesafe_api" if live_api else "high_throughput_fastpath",
        "api_calls_made": total_tasks * 2 if live_api else 0,
        "total_wall_time_seconds": round(wall_time_sec, 2),
        "throughput_queries_per_sec": throughput_qps,
        "average_routing_latency_ms": avg_latency_ms,
        "total_tokens_processed": total_tokens,
        "baseline_model": BASELINE_MODEL,
        "baseline_total_cost_usd": round(total_baseline_cost, 4),
        "promptrouter_total_cost_usd": round(total_router_cost, 4),
        "total_cost_saved_usd": total_saved_dollars,
        "percentage_cost_reduction": pct_cost_saved,
        "blocked_injections": blocked_count,
        "model_distribution": dict(model_counts),
    }

    # save results to both specific sample file and general results file
    sample_file = BENCHMARK_DIR / f"benchmark_results_{total_tasks}.json"
    with open(sample_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    default_file = BENCHMARK_DIR / "benchmark_results.json"
    with open(default_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # print benchmark evaluation summary report
    print("\n" + "=" * 65, flush=True)
    print("           PROMPTROUTER BENCHMARK EVALUATION REPORT           ", flush=True)
    print("=" * 65, flush=True)
    print(f" Tasks Evaluated           : {total_tasks:,}", flush=True)
    print(f" Sampling Method           : {sample_type}", flush=True)
    print(f" Total Tokens Processed    : {total_tokens:,}", flush=True)
    print(f" Average Routing Overhead  : {avg_latency_ms:.2f} ms", flush=True)
    print(f" Engine Throughput         : {throughput_qps:,} queries/sec", flush=True)
    print(f" Execution Wall Time       : {wall_time_sec:.2f} seconds", flush=True)
    print("-" * 65, flush=True)
    print(f" Baseline Cost ({BASELINE_MODEL:<10}) : ${total_baseline_cost:.4f}", flush=True)
    print(f" PromptRouter Cost         : ${total_router_cost:.4f}", flush=True)
    print(f" Total Money Saved         : ${total_saved_dollars:.4f}", flush=True)
    print(f" Cost Reduction Achieved   : {pct_cost_saved:.1f}% SAVINGS", flush=True)
    print("-" * 65, flush=True)
    print(" Model Routing Distribution:", flush=True)
    for model, count in model_counts.most_common():
        pct = (count / total_tasks) * 100
        print(f"   - {model:<22}: {count:>5,} tasks ({pct:5.1f}%)", flush=True)
    print("=" * 65, flush=True)
    print(f"Results saved to: {sample_file.name} & benchmark_results.json\n", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PromptRouter benchmark on Dolly 15k dataset.")
    parser.add_argument("--samples", type=int, default=1000, help="Number of tasks to evaluate (default: 1000)")
    parser.add_argument("--live", action="store_true", help="Use live TypeSafe network API calls")
    parser.add_argument("--random", action="store_true", default=True, help="Randomly sample tasks across the dataset")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    args = parser.parse_args()
    run_benchmark(num_samples=args.samples, live_api=args.live, use_random=args.random, seed=args.seed)
