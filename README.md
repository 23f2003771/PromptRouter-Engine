# PromptRouter Engine 🚀

An intelligent, low-latency LLM orchestration and routing engine built with **FastAPI**, **LangGraph**, and **TypeSafe SDK**. 

PromptRouter dynamically evaluates incoming user prompts, guards against prompt injections using a 5-scale security evaluator, and routes tasks to the most cost-effective model across 5 distinct intelligence tiers.

---

## 🏛️ System Architecture & 5-Tier Routing Matrix

PromptRouter implements an automated security guardrail followed by dynamic routing across five specialized model tiers:

```text
               ┌───────────────────────────────┐
               │         Incoming Task         │
               └───────────────┬───────────────┘
                               │
                               ▼
               ┌───────────────────────────────┐
               │    Prompt Injection Guard     │  <-- TypeSafe SDK 'Score' (1 to 5 scale)
               └───────┬───────────────┬───────┘
                       │               │
        Score > 2.0    │               │ Score <= 2.0 (Safe)
     (Jailbreak/Attack)│               │
                       ▼               ▼
          ┌────────────────┐  ┌─────────────────────────┐
          │ Block Request  │  │   LangGraph Classifier  │
          │  & Zero Tokens │  └────────────┬────────────┘
          └────────────────┘               │
                                           ▼
          ┌─────────────────────────────────────────────────────────────────┐
          │                  5-Tier Model Routing Matrix                    │
          └─┬──────────────┬───────────────┬───────────────┬──────────────┬─┘
            │              │               │               │              │
            ▼              ▼               ▼               ▼              ▼
     ┌──────────────┐┌──────────────┐┌──────────────┐┌──────────────┐┌──────────────┐
     │    Tier 1    ││    Tier 2    ││    Tier 3    ││    Tier 4    ││    Tier 5    │
     │ Flagship Eng ││ Flagship Std ││ Budget Mini  ││  Fast Factual││ Micro / Trivial│
     │claude-5-sonnet││gpt-5.6-terra ││ gpt-5.6-luna ││deepseek-flash││gemini-flash-lite│
     │  ($0.0040/1k)││  ($0.0045/1k)││ ($0.00045/1k)││ ($0.00020/1k)││ ($0.00018/1k)│
     └──────────────┘└──────────────┘└──────────────┘└──────────────┘└──────────────┘
```

1. **Security Guardrail Node**: Evaluates prompt injection risk on a 1–5 scale. Prompts scoring > 2.0 are blocked immediately, preventing jailbreaks and saving 100% of downstream tokens.
2. **LangGraph State Engine**: Orchestrates execution using a lightweight `TypedDict` state graph for sub-2ms routing latency.
3. **5-Tier Model Selection**: Routes complex coding to Tier 1, general chat to Tier 2, summarization/JSON extraction to Tier 3, factual queries to Tier 4, and simple formatting to Tier 5.
4. **Economics & Telemetry**: Automatically records latency, token count, cost, and dollars saved vs. always calling flagship models.

---

## 📊 Benchmark Validation Results

Evaluated against human-annotated tasks from the [Databricks Dolly 15k Dataset](https://huggingface.co/datasets/databricks/databricks-dolly-15k):

### Benchmark Summary Comparison

| Metric | 500-Task Live API Validation | 1,000 Random Tasks Benchmark |
| :--- | :--- | :--- |
| **Tasks Evaluated** | 500 tasks | 1,000 tasks (Random Seed 42) |
| **Total Tokens** | 32,424 tokens | 66,055 tokens |
| **Routing Mode** | Live TypeSafe System One API | High-Throughput Engine |
| **Average Latency** | 905.81 ms (Cloud API) | **1.72 ms** (Local Overhead) |
| **Throughput** | 1.1 queries / sec | **578.6 queries / sec** |
| **Baseline Cost (`gpt-5.6-terra`)** | $0.1459 | $0.2972 |
| **PromptRouter Cost** | **$0.0100** | **$0.0259** |
| **Total Dollars Saved** | **$0.1359** | **$0.2714** |
| **Cost Reduction Achieved** | **93.14% SAVINGS** | **91.32% SAVINGS** |

---

### Model Distribution Breakdown

Across both benchmark runs, the vast majority of tasks were handled by ultra-low-cost specialized tiers rather than expensive flagship models:

#### 1,000 Random Tasks Benchmark
* **54.0%** — `gemini-2.5-flash-lite` (540 tasks): Micro formatting, basic binary classification
* **27.6%** — `deepseek-v4-flash` (276 tasks): Factual QA, knowledge retrieval
* **14.5%** — `gpt-5.6-luna` (145 tasks): Everyday summaries, JSON extraction, brainstorming
* **3.9%** — `claude-5-sonnet` (39 tasks): Complex programming, architecture, creative prose

#### 500-Task Live API Validation
* **80.8%** — `deepseek-v4-flash` (404 tasks)
* **14.2%** — `gpt-5.6-luna` (71 tasks)
* **3.4%** — `gemini-2.5-flash-lite` (17 tasks)
* **1.0%** — `gpt-5.6-terra` (5 tasks)
* **0.6%** — `claude-5-sonnet` (3 tasks)

*(Saved in `benchmark/benchmark_results_500.json` and `benchmark/benchmark_results_1000.json`).*

---

## ⚡ Quickstart

### 1. Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/your-username/PromptRouter-Engine.git
cd PromptRouter-Engine
pip install -r requirements.txt
```

### 2. Configure Environment
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
*(Optional: Add your `TYPESAFE_API_KEY` for live cloud classification and scoring).*

### 3. Run the Engine
Start the FastAPI server:
```bash
python -m uvicorn main:app --port 8000
```
API Documentation is live at `http://127.0.0.1:8000/docs`.

---

## 💻 Using the Python SDK

PromptRouter comes with a built-in, lightweight client library:

```python
from sdk import PromptRouterClient

# 1. Connect to engine
client = PromptRouterClient(base_url="http://127.0.0.1:8000")

# 2. Route any task
response = client.route("Write a python function to sort a list.")

print(f"Routed Model : {response.routed_model}")
print(f"Safety Score : {response.safety_score} / 5.0")
print(f"Latency      : {response.latency_ms} ms")
print(f"Cost         : ${response.estimated_cost}")
print(f"Money Saved  : ${response.cost_saved}")
```

Run the included demo:
```bash
python examples/demo.py
```

---

## 🧪 Running the Benchmark Suite

To reproduce the benchmark results:

1. **Download Dataset:**
   ```bash
   python benchmark/download_dataset.py
   ```
2. **Run Benchmark:**
   ```bash
   # 1,000 random samples:
   python benchmark/evaluate.py --samples 1000 --random

   # Full 15,000 tasks:
   python benchmark/evaluate.py --samples 15000
   ```
Results are saved to `benchmark/benchmark_results.json`.

---

## 📂 Project Structure

```text
PromptRouter-Engine/
├── benchmark/
│   ├── download_dataset.py       # Downloads & summarizes 15k Dolly dataset
│   ├── evaluate.py               # Benchmark runner with random sampling
│   ├── benchmark_results_500.json   # Live API 500-sample results
│   ├── benchmark_results_1000.json  # 1,000 random-sample results
│   └── benchmark_results.json
├── data/
│   └── dolly_15k.jsonl           # 15,011 instruction dataset
├── examples/
│   └── demo.py                   # SDK usage demo with security test
├── router/
│   ├── config.py                 # Model pricing, 5-tier mapping, and security rubric
│   ├── graph.py                  # LangGraph workflow with safety conditional edge
│   ├── nodes.py                  # Safety guard, 5-tier classification, and generator
│   └── state.py                  # TypedDict graph state definition
├── sdk/
│   ├── __init__.py
│   └── client.py                 # Lightweight, typed Python client
├── main.py                       # FastAPI application endpoints
├── models.py                     # Pydantic request & response schemas
├── requirements.txt
└── README.md
```
