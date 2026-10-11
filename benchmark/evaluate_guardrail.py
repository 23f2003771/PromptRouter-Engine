"""
Prompt Injection Guardrail Benchmark Evaluation.
Evaluates PromptRouter's adversarial security layer against real-world labeled attacks:
Dataset: deepset/prompt-injections (662 labeled queries: 263 injection attacks, 399 benign queries)

Evaluates:
- 500 prompts via live TypeSafe SDK 'Score' (1 to 5 scale, threshold > 2.0 blocks).
- Remaining prompts via local regex heuristic guardrail.
- Computes Precision, Recall, F1-Score, False Positive Rate (FPR), and Accuracy.
"""

import sys
import json
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import router.config as config
from router.nodes import check_safety

DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "guardrail_benchmark.jsonl"
OUTPUT_FILE = Path(__file__).resolve().parent / "guardrail_benchmark_results.json"


def evaluate_single_prompt(item, use_live_api: bool):
    prompt = item["prompt"]
    label = item["label"]  # 1 = injection, 0 = benign

    saved_key = config.TYPESAFE_API_KEY
    if not use_live_api:
        config.TYPESAFE_API_KEY = ""

    t0 = time.perf_counter()
    state = {"prompt": prompt}
    res = check_safety(state)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    config.TYPESAFE_API_KEY = saved_key

    is_blocked = res.get("is_blocked", False)
    safety_score = res.get("safety_score", 1.0)

    # Prediction: 1 if blocked, 0 if safe
    pred = 1 if is_blocked else 0

    return {
        "id": item.get("id"),
        "label": label,
        "pred": pred,
        "is_blocked": is_blocked,
        "safety_score": safety_score,
        "latency_ms": latency_ms,
        "is_correct": (pred == label)
    }


def compute_metrics(results):
    total = len(results)
    tp = sum(1 for r in results if r["label"] == 1 and r["pred"] == 1)
    tn = sum(1 for r in results if r["label"] == 0 and r["pred"] == 0)
    fp = sum(1 for r in results if r["label"] == 0 and r["pred"] == 1)
    fn = sum(1 for r in results if r["label"] == 1 and r["pred"] == 0)

    accuracy = round(((tp + tn) / max(total, 1)) * 100, 2)
    precision = round((tp / max(tp + fp, 1)) * 100, 2)
    recall = round((tp / max(tp + fn, 1)) * 100, 2)
    f1 = round((2 * precision * recall / max(precision + recall, 0.001)), 2)
    fpr = round((fp / max(fp + tn, 1)) * 100, 2)
    avg_latency = round(sum(r["latency_ms"] for r in results) / max(total, 1), 2)

    return {
        "total_evaluated": total,
        "true_positives_blocked_attacks": tp,
        "true_negatives_allowed_benign": tn,
        "false_positives_benign_blocked": fp,
        "false_negatives_missed_attacks": fn,
        "accuracy_pct": accuracy,
        "precision_pct": precision,
        "recall_pct": recall,
        "f1_score": f1,
        "false_positive_rate_pct": fpr,
        "avg_latency_ms": avg_latency
    }


def run_guardrail_benchmark(live_sample_count: int = 500, max_workers: int = 6):
    if not DATA_FILE.exists():
        print(f"Dataset not found at {DATA_FILE}. Run download_dataset.py first!", flush=True)
        return

    items = []
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))

    total_items = len(items)
    print(f"\n=======================================================", flush=True)
    print(f"       PROMPTROUTER ADVERSARIAL GUARDRAIL EVALUATION", flush=True)
    print(f"=======================================================", flush=True)
    print(f" Total Labeled Queries in Dataset : {total_items:,}", flush=True)
    print(f" Live TypeSafe API Evaluations    : {live_sample_count:,}", flush=True)
    print(f" Local Regex Guardrail Remainder  : {total_items - live_sample_count:,}\n", flush=True)

    live_items = items[:live_sample_count]
    local_items = items[live_sample_count:]

    # Phase 1: Live TypeSafe API evaluation
    print(f"Phase 1: Running {len(live_items)} prompts on Live TypeSafe API (Score Rubric 1-5)...", flush=True)
    live_results = []
    t_start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(evaluate_single_prompt, item, True) for item in live_items]
        for idx, f in enumerate(futures, 1):
            res = f.result()
            live_results.append(res)
            if idx % 100 == 0 or idx == len(live_items):
                print(f"  Live evaluated {idx}/{len(live_items)} ({idx/len(live_items)*100:.1f}%)...", flush=True)

    live_time = time.perf_counter() - t_start
    live_metrics = compute_metrics(live_results)
    live_metrics["wall_time_sec"] = round(live_time, 2)

    # Phase 2: Local heuristic evaluation on remainder
    print(f"\nPhase 2: Running {len(local_items)} remaining prompts on Local Regex Guardrail...", flush=True)
    local_results = [evaluate_single_prompt(item, False) for item in local_items]
    local_metrics = compute_metrics(local_results)

    # Also evaluate all items locally for full reference
    full_local_results = [evaluate_single_prompt(item, False) for item in items]
    full_local_metrics = compute_metrics(full_local_results)

    summary = {
        "dataset": "deepset/prompt-injections",
        "live_api_500_sample": live_metrics,
        "remaining_local_sample": local_metrics,
        "full_dataset_local_regex": full_local_metrics
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 65, flush=True)
    print("       PROMPTROUTER GUARDRAIL EVALUATION SUMMARY       ", flush=True)
    print("=" * 65, flush=True)
    print(f" Live API Accuracy        : {live_metrics['accuracy_pct']}%", flush=True)
    print(f" Live API Precision       : {live_metrics['precision_pct']}%", flush=True)
    print(f" Live API Recall          : {live_metrics['recall_pct']}%", flush=True)
    print(f" Live API F1 Score        : {live_metrics['f1_score']}", flush=True)
    print(f" Live API False Pos. Rate : {live_metrics['false_positive_rate_pct']}%", flush=True)
    print(f" Live API Avg Latency     : {live_metrics['avg_latency_ms']} ms", flush=True)
    print("-" * 65, flush=True)
    print(f" Local Regex Accuracy     : {full_local_metrics['accuracy_pct']}%", flush=True)
    print(f" Local Regex Recall       : {full_local_metrics['recall_pct']}%", flush=True)
    print(f" Local Regex Precision    : {full_local_metrics['precision_pct']}%", flush=True)
    print(f" Local Regex Latency      : {full_local_metrics['avg_latency_ms']} ms", flush=True)
    print("=" * 65, flush=True)
    print(f"Results saved to: {OUTPUT_FILE.name}\n", flush=True)

    return summary


if __name__ == "__main__":
    run_guardrail_benchmark(live_sample_count=500)
