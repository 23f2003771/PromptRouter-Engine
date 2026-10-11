"""
PromptRouter Benchmark Suite (RouterBench & LMSYS Arena).
Evaluates 3,000 modern tasks:
- 1,500 from RouterBench (GSM8K, MBPP, MMLU, Winogrande)
- 1,500 from LMSYS Chatbot Arena (RouteLLM user interactions)

Evaluates:
1. Phase 1: 700 Random Tasks (350 RouterBench + 350 LMSYS Arena) via Live TypeSafe Cloud API.
2. Phase 2: Remaining 2,300 Tasks via Local High-Throughput Fast-Path.
3. Multi-Baseline Comparison:
   - Monolithic Flagship: gpt-5.6-terra ($0.0070 / 1k tokens)
   - Monolithic Cheapest: gemini-2.5-flash-lite ($0.00025 / 1k tokens)
   - Random Routing Baseline across all tiers
"""

import sys
import json
import time
import random
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from router import router_graph
import router.config as config
from router.config import MODEL_COSTS_PER_1K, MODEL_MAP

DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "routing_benchmark.jsonl"
BENCHMARK_DIR = Path(__file__).resolve().parent

FLAGSHIP_BASELINE_MODEL = "gpt-5.6-terra"
FLAGSHIP_RATE = MODEL_COSTS_PER_1K.get(FLAGSHIP_BASELINE_MODEL, 0.0070)

CHEAPEST_BASELINE_MODEL = "gemini-2.5-flash-lite"
CHEAPEST_RATE = MODEL_COSTS_PER_1K.get(CHEAPEST_BASELINE_MODEL, 0.00025)

ALL_MODEL_RATES = list(MODEL_COSTS_PER_1K.values())


def evaluate_single_task(item, use_live_api: bool):
    prompt = item["prompt"]
    saved_key = config.TYPESAFE_API_KEY
    if not use_live_api:
        config.TYPESAFE_API_KEY = ""

    t0 = time.perf_counter()
    result = router_graph.invoke({"prompt": prompt})
    latency_ms = (time.perf_counter() - t0) * 1000.0

    config.TYPESAFE_API_KEY = saved_key

    model = result.get("selected_model", "deepseek-v4-flash")
    tokens = result.get("tokens_used", max(1, len(prompt.split())))
    router_cost = result.get("estimated_cost", 0.0)

    # 1. Flagship Baseline
    flagship_cost = (tokens / 1000.0) * FLAGSHIP_RATE

    # 2. Cheapest Baseline
    cheapest_cost = (tokens / 1000.0) * CHEAPEST_RATE

    # 3. Random Routing Baseline
    random_rate = random.choice(ALL_MODEL_RATES)
    random_cost = (tokens / 1000.0) * random_rate

    return {
        "id": item.get("id"),
        "source": item.get("source"),
        "model": model,
        "tokens": tokens,
        "router_cost": router_cost,
        "flagship_cost": flagship_cost,
        "cheapest_cost": cheapest_cost,
        "random_cost": random_cost,
        "latency_ms": latency_ms,
        "is_blocked": result.get("is_blocked", False)
    }


def compute_routing_metrics(eval_records, mode_name: str, wall_time_sec: float, api_calls_per_task: int):
    total_tasks = len(eval_records)
    total_tokens = sum(r["tokens"] for r in eval_records)
    total_router_cost = sum(r["router_cost"] for r in eval_records)
    total_flagship_cost = sum(r["flagship_cost"] for r in eval_records)
    total_cheapest_cost = sum(r["cheapest_cost"] for r in eval_records)
    total_random_cost = sum(r["random_cost"] for r in eval_records)
    total_latency = sum(r["latency_ms"] for r in eval_records)

    avg_latency = round(total_latency / max(total_tasks, 1), 2)
    throughput = round(total_tasks / max(wall_time_sec, 0.001), 1)

    dollars_saved_vs_flagship = round(max(0.0, total_flagship_cost - total_router_cost), 4)
    pct_saved_vs_flagship = round((dollars_saved_vs_flagship / max(total_flagship_cost, 0.0001)) * 100, 2)

    model_counts = Counter(r["model"] for r in eval_records)
    source_counts = Counter(r["source"] for r in eval_records)

    return {
        "mode": mode_name,
        "tasks_evaluated": total_tasks,
        "dataset_composition": dict(source_counts),
        "api_calls_made": total_tasks * api_calls_per_task,
        "wall_time_seconds": round(wall_time_sec, 2),
        "throughput_queries_per_sec": throughput,
        "average_routing_latency_ms": avg_latency,
        "total_tokens_processed": total_tokens,
        "costs": {
            "promptrouter_cost_usd": round(total_router_cost, 4),
            "flagship_baseline_cost_usd": round(total_flagship_cost, 4),
            "cheapest_baseline_cost_usd": round(total_cheapest_cost, 4),
            "random_routing_cost_usd": round(total_random_cost, 4),
        },
        "economic_savings": {
            "dollars_saved_vs_flagship": dollars_saved_vs_flagship,
            "percentage_reduction_vs_flagship": pct_saved_vs_flagship,
            "cost_multiplier_vs_cheapest": round(total_router_cost / max(total_cheapest_cost, 0.0001), 2),
            "percentage_reduction_vs_random": round(((total_random_cost - total_router_cost) / max(total_random_cost, 0.0001)) * 100, 2),
        },
        "model_distribution": dict(model_counts)
    }


def run_benchmark(live_sample_per_source: int = 350, max_live_workers: int = 6):
    if not DATA_FILE.exists():
        print(f"Dataset not found at {DATA_FILE}. Run download_dataset.py first!", flush=True)
        return

    items = []
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))

    rb_items = [it for it in items if it.get("source") == "routerbench"]
    lmsys_items = [it for it in items if it.get("source") == "lmsys_arena"]

    print(f"\n=======================================================", flush=True)
    print(f"        PROMPTROUTER ROUTING BENCHMARK EVALUATION      ", flush=True)
    print(f"=======================================================", flush=True)
    print(f" Total Tasks Loaded        : {len(items):,}", flush=True)
    print(f"   - RouterBench Tasks     : {len(rb_items):,}", flush=True)
    print(f"   - LMSYS Arena Tasks     : {len(lmsys_items):,}", flush=True)

    # 1. Select 350 random from each source for live API test
    random.seed(42)
    live_rb = random.sample(rb_items, min(live_sample_per_source, len(rb_items)))
    live_lmsys = random.sample(lmsys_items, min(live_sample_per_source, len(lmsys_items)))
    live_tasks = live_rb + live_lmsys
    random.shuffle(live_tasks)

    live_ids = {it["id"] for it in live_tasks}
    local_tasks = [it for it in items if it["id"] not in live_ids]

    print(f" Live TypeSafe API Tasks   : {len(live_tasks):,} (350 RouterBench + 350 LMSYS)", flush=True)
    print(f" Local Machine Fast-Path   : {len(local_tasks):,} (Remaining Tasks)", flush=True)
    print(f" Naive Comparison Baseline : {FLAGSHIP_BASELINE_MODEL} (${FLAGSHIP_RATE}/1k tokens)\n", flush=True)

    # Phase 1: Live TypeSafe Cloud API Run
    print(f"Phase 1: Evaluating {len(live_tasks):,} tasks on Live TypeSafe API...", flush=True)
    t0_live = time.perf_counter()
    live_records = []

    with ThreadPoolExecutor(max_workers=max_live_workers) as executor:
        futures = [executor.submit(evaluate_single_task, it, True) for it in live_tasks]
        for idx, f in enumerate(futures, 1):
            live_records.append(f.result())
            if idx % 100 == 0 or idx == len(live_tasks):
                print(f"  Live processed {idx:>3}/{len(live_tasks)} tasks ({idx/len(live_tasks)*100:.1f}%)...", flush=True)

    wall_live = time.perf_counter() - t0_live
    live_summary = compute_routing_metrics(live_records, "live_typesafe_cloud_api", wall_live, api_calls_per_task=2)

    with open(BENCHMARK_DIR / "benchmark_results_700_live.json", "w", encoding="utf-8") as f:
        json.dump(live_summary, f, indent=2)

    # Phase 2: Local High-Throughput Run
    print(f"\nPhase 2: Evaluating {len(local_tasks):,} remaining tasks on Local Fast-Path...", flush=True)
    t0_local = time.perf_counter()
    local_records = [evaluate_single_task(it, False) for it in local_tasks]
    wall_local = time.perf_counter() - t0_local
    local_summary = compute_routing_metrics(local_records, "local_high_throughput_fastpath", wall_local, api_calls_per_task=0)

    with open(BENCHMARK_DIR / "benchmark_results_2300_local.json", "w", encoding="utf-8") as f:
        json.dump(local_summary, f, indent=2)

    # Phase 3: Full Combined Evaluation (All 3,000 tasks evaluated locally)
    print(f"\nPhase 3: Evaluating full {len(items):,} task dataset locally...", flush=True)
    t0_all = time.perf_counter()
    all_local_records = [evaluate_single_task(it, False) for it in items]
    wall_all = time.perf_counter() - t0_all
    all_local_summary = compute_routing_metrics(all_local_records, "full_3000_dataset_local_fastpath", wall_all, api_calls_per_task=0)

    combined_results = {
        "dataset": "RouterBench_and_LMSYS_Arena",
        "live_api_700_sample": live_summary,
        "remaining_local_2300_sample": local_summary,
        "full_3000_local_benchmark": all_local_summary
    }

    with open(BENCHMARK_DIR / "benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(combined_results, f, indent=2)

    print("\n" + "=" * 65, flush=True)
    print("            PROMPTROUTER EVALUATION SUMMARY REPORT            ", flush=True)
    print("=" * 65, flush=True)
    print(f" [Live API Split - 700 Tasks (350 RouterBench + 350 LMSYS)]", flush=True)
    print(f"   Average Latency         : {live_summary['average_routing_latency_ms']} ms", flush=True)
    print(f"   Throughput              : {live_summary['throughput_queries_per_sec']} QPS", flush=True)
    print(f"   Flagship Baseline Cost  : ${live_summary['costs']['flagship_baseline_cost_usd']:.4f}", flush=True)
    print(f"   PromptRouter Cost       : ${live_summary['costs']['promptrouter_cost_usd']:.4f}", flush=True)
    print(f"   Dollars Saved           : ${live_summary['economic_savings']['dollars_saved_vs_flagship']:.4f}", flush=True)
    print(f"   Cost Reduction          : {live_summary['economic_savings']['percentage_reduction_vs_flagship']}%", flush=True)
    print(f"   Savings vs Random       : {live_summary['economic_savings']['percentage_reduction_vs_random']}%", flush=True)
    print("-" * 65, flush=True)
    print(f" [Remaining Local Split - {local_summary['tasks_evaluated']} Tasks]", flush=True)
    print(f"   Average Decision Latency: {local_summary['average_routing_latency_ms']} ms", flush=True)
    print(f"   Throughput              : {local_summary['throughput_queries_per_sec']} QPS", flush=True)
    print(f"   Cost Reduction          : {local_summary['economic_savings']['percentage_reduction_vs_flagship']}%", flush=True)
    print("-" * 65, flush=True)
    print(f" [Full 3,000 Tasks Local Benchmark]", flush=True)
    print(f"   Average Decision Latency: {all_local_summary['average_routing_latency_ms']} ms", flush=True)
    print(f"   Throughput              : {all_local_summary['throughput_queries_per_sec']} QPS", flush=True)
    print(f"   Cost Reduction          : {all_local_summary['economic_savings']['percentage_reduction_vs_flagship']}%", flush=True)
    print("=" * 65, flush=True)
    print("Benchmark complete! All JSON results saved in benchmark/\n", flush=True)

    return combined_results


if __name__ == "__main__":
    run_benchmark(live_sample_per_source=350, max_live_workers=6)
