"""
Script to download the Databricks Dolly 15,000 instruction dataset.
This dataset contains ~15,000 real-world tasks across multiple categories
(creative, factual QA, brainstorming, classification, coding, summarization)
which I use to benchmark my PromptRouter engine.
"""

from pathlib import Path
from collections import Counter
import json
import httpx

DATASET_URL = (
    "https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl"
)
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUT_FILE = DATA_DIR / "dolly_15k.jsonl"


def download_dataset():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if OUTPUT_FILE.exists():
        print(f"Dataset already exists at: {OUTPUT_FILE}")
    else:
        print(f"Downloading 15,000 task dataset from HuggingFace...")
        with httpx.Client(follow_redirects=True, timeout=60.0) as client:
            res = client.get(DATASET_URL)
            res.raise_for_status()
            OUTPUT_FILE.write_bytes(res.content)
        print(f"Downloaded and saved to: {OUTPUT_FILE}")

    # inspect and print summary of categories in the dataset
    categories = Counter()
    total_rows = 0

    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                categories[item.get("category", "unknown")] += 1
                total_rows += 1

    print(f"\n--- Dataset Summary ({total_rows:,} total rows) ---")
    for category, count in categories.most_common():
        pct = (count / total_rows) * 100
        print(f"  {category:<24}: {count:>5,} rows ({pct:.1f}%)")

    print("\nDataset ready for benchmarking!")
    return OUTPUT_FILE


if __name__ == "__main__":
    download_dataset()
