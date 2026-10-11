"""
Dataset Downloader for PromptRouter Engine Benchmarking.
Downloads modern, gold-standard benchmark datasets used by leading router researchers:
1. RouterBench (withmartian/routerbench): Benchmark prompts from GSM8k, MBPP, MMLU, Winogrande.
2. LMSYS Chatbot Arena (routellm/gpt4_dataset): Real-world user interactions with quality evaluations.
3. Deepset Prompt Injections (deepset/prompt-injections): Real prompt injection attacks vs benign queries.
"""

import io
import json
import pickle
import random
import urllib.request
from pathlib import Path
import pandas as pd
import httpx

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

ROUTERBENCH_URL = (
    "https://huggingface.co/datasets/withmartian/routerbench/resolve/main/routerbench_0shot.pkl"
)
LMSYS_PARQUET_URL = (
    "https://huggingface.co/api/datasets/routellm/gpt4_dataset/parquet/default/validation/0.parquet"
)
DEEPSET_TRAIN_URL = (
    "https://huggingface.co/api/datasets/deepset/prompt-injections/parquet/default/train/0.parquet"
)
DEEPSET_TEST_URL = (
    "https://huggingface.co/api/datasets/deepset/prompt-injections/parquet/default/test/0.parquet"
)

ROUTING_DATASET_FILE = DATA_DIR / "routing_benchmark.jsonl"
GUARDRAIL_DATASET_FILE = DATA_DIR / "guardrail_benchmark.jsonl"


def download_file(url: str, dest_path: Path, label: str):
    if dest_path.exists():
        print(f"[{label}] File already exists at {dest_path.name} ({dest_path.stat().st_size / (1024*1024):.1f} MB)")
        return
    print(f"[{label}] Downloading from {url}...")
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    with urllib.request.urlopen(req) as resp, open(dest_path, "wb") as f:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
    print(f"[{label}] Saved to {dest_path.name} ({dest_path.stat().st_size / (1024*1024):.1f} MB)")


def prepare_routing_benchmark(router_samples: int = 1500, lmsys_samples: int = 1500):
    rb_path = DATA_DIR / "routerbench_0shot.pkl"
    lmsys_path = DATA_DIR / "lmsys_validation.parquet"

    download_file(ROUTERBENCH_URL, rb_path, "RouterBench")
    download_file(LMSYS_PARQUET_URL, lmsys_path, "LMSYS RouteLLM")

    print("\nProcessing RouterBench prompts...")
    with open(rb_path, "rb") as f:
        df_rb = pickle.load(f)
    print(f"Loaded RouterBench DataFrame: {df_rb.shape[0]:,} total rows")

    # sample diverse prompts from RouterBench
    df_rb_clean = df_rb.dropna(subset=["prompt"]).copy()
    if len(df_rb_clean) > router_samples:
        df_rb_sampled = df_rb_clean.sample(n=router_samples, random_state=42)
    else:
        df_rb_sampled = df_rb_clean

    router_tasks = []
    for idx, row in df_rb_sampled.iterrows():
        prompt = str(row["prompt"]).strip()
        if prompt:
            router_tasks.append({
                "id": f"rb_{row.get('sample_id', idx)}",
                "prompt": prompt,
                "source": "routerbench",
                "eval_name": str(row.get("eval_name", "standard")),
            })

    print(f"Extracted {len(router_tasks):,} prompts from RouterBench.")

    print("\nProcessing LMSYS Chatbot Arena prompts...")
    df_lmsys = pd.read_parquet(lmsys_path)
    print(f"Loaded LMSYS DataFrame: {df_lmsys.shape[0]:,} total rows")

    df_lmsys_clean = df_lmsys.dropna(subset=["prompt"]).copy()
    if len(df_lmsys_clean) > lmsys_samples:
        df_lmsys_sampled = df_lmsys_clean.sample(n=lmsys_samples, random_state=42)
    else:
        df_lmsys_sampled = df_lmsys_clean

    lmsys_tasks = []
    for idx, row in df_lmsys_sampled.iterrows():
        prompt = str(row["prompt"]).strip()
        if prompt:
            lmsys_tasks.append({
                "id": f"lmsys_{idx}",
                "prompt": prompt,
                "source": "lmsys_arena",
                "eval_name": "chatbot_arena",
            })

    print(f"Extracted {len(lmsys_tasks):,} prompts from LMSYS Arena.")

    # combine and shuffle
    combined = router_tasks + lmsys_tasks
    random.seed(42)
    random.shuffle(combined)

    print(f"\nWriting {len(combined):,} combined routing tasks to {ROUTING_DATASET_FILE.name}...")
    with open(ROUTING_DATASET_FILE, "w", encoding="utf-8") as f:
        for item in combined:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Routing benchmark dataset ready! Total tasks: {len(combined):,}")


def prepare_guardrail_benchmark():
    train_path = DATA_DIR / "deepset_train.parquet"
    test_path = DATA_DIR / "deepset_test.parquet"

    download_file(DEEPSET_TRAIN_URL, train_path, "Deepset Train")
    download_file(DEEPSET_TEST_URL, test_path, "Deepset Test")

    print("\nProcessing Deepset Prompt Injections dataset...")
    df_train = pd.read_parquet(train_path)
    df_test = pd.read_parquet(test_path)
    df_combined = pd.concat([df_train, df_test], ignore_index=True)

    print(f"Loaded {len(df_combined):,} labeled guardrail items ({sum(df_combined['label'] == 1)} injections, {sum(df_combined['label'] == 0)} benign)")

    guardrail_items = []
    for idx, row in df_combined.iterrows():
        text = str(row["text"]).strip()
        label = int(row["label"])  # 1 = injection/attack, 0 = benign
        if text:
            guardrail_items.append({
                "id": f"guard_{idx}",
                "prompt": text,
                "label": label,
                "is_injection": (label == 1),
            })

    # shuffle with fixed seed
    random.seed(42)
    random.shuffle(guardrail_items)

    with open(GUARDRAIL_DATASET_FILE, "w", encoding="utf-8") as f:
        for item in guardrail_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Saved {len(guardrail_items):,} guardrail evaluation items to {GUARDRAIL_DATASET_FILE.name}.")


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    prepare_routing_benchmark(router_samples=1500, lmsys_samples=1500)
    prepare_guardrail_benchmark()
    print("\n--- All Datasets Downloaded & Prepared Successfully! ---")


if __name__ == "__main__":
    main()
