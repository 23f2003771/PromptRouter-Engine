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
2. **Dual-Path Execution Architecture**:
   - **Deep Semantic Path (TypeSafe SDK Cloud)**: Full semantic classification and probabilistic safety scoring via TypeSafe System One API (~905ms public WAN round-trips over TLS).
   - **High-Throughput Fast-Path (Local Engine)**: In-memory regex guardrails and word-boundary heuristic routing for zero-latency (<2ms, 578+ QPS) execution without burning external API quota.
3. **5-Tier Model Selection**: Routes complex coding to Tier 1/2, general chat to Tier 2, summarization/JSON extraction to Tier 3, factual queries to Tier 4, and simple formatting to Tier 5.
4. **Economics & Telemetry**: Automatically records latency, token count, cost, and dollars saved vs. always calling flagship models.

### 📋 5-Tier Architecture & Complete 10-Task Routing Catalog

PromptRouter configures 10 granular task criteria in `router/config.py` mapped across 5 operational tiers and 9 modern LLMs:

| Tier | Category / Task Type | Target Model | Blended Rate / 1k Tokens | Capabilities & Scope |
| :--- | :--- | :--- | :--- | :--- |
| **Tier 1: Heavy Reasoning** | `reasoning_heavy`<br>`reasoning_premium` | `deepseek-r1`<br>`o3-reasoning` | $0.0012<br>$0.0250 | Complex mathematical proofs, deep algorithmic architecture, autonomous agent planning |
| **Tier 2: Flagship Workhorses** | `code`<br>`creative`<br>`factual_long_context`<br>`general_premium` | `claude-5-sonnet`<br>`gemini-3.1-pro`<br>`gpt-5.6-terra` | $0.0040<br>$0.0040<br>$0.0045<br>$0.0045 | Full-stack software engineering, nuanced editorial prose, 1M+ token document analysis, complex multi-turn dialogue |
| **Tier 3: Performance Mini** | `agentic_fast`<br>`general_budget` | `gemini-3.8-flash`<br>`gpt-5.6-luna` | $0.0015<br>$0.00045 | Low-latency agentic loops, high-velocity JSON schema extraction, structured entity tagging |
| **Tier 4: High-Speed Factual** | `factual` | `deepseek-v4-flash` | $0.00020 | High-throughput factual QA, knowledge retrieval, short summarization |
| **Tier 5: Micro & Floor Fallback** | `fallback_cheapest` | `gemini-2.5-flash-lite` | $0.00018 | Basic string reformatting, binary classification, trivial single-turn text queries |

*(Note: In the Databricks Dolly 15k dataset benchmarks, user tasks concentrated in general QA, summarization, creative text, and programming, which dynamically routed to the 5 representative tier models: `deepseek-v4-flash`, `gpt-5.6-luna`, `gemini-2.5-flash-lite`, `gpt-5.6-terra`, and `claude-5-sonnet`).*

---

## 📊 Benchmark Validation Results

Evaluated against human-annotated tasks from the [Databricks Dolly 15k Dataset](https://huggingface.co/datasets/databricks/databricks-dolly-15k).

Given standard developer quota constraints (e.g. 3,000 total API calls, where each live prompt requires 2 TypeSafe API calls: one for injection scoring and one for classification), the benchmark suite rigorously evaluates both execution modes:

### Benchmark Summary Comparison

| Metric | 500-Task Live API Validation | 1,000 Random Tasks Fast-Path |
| :--- | :--- | :--- |
| **Tasks Evaluated** | 500 tasks | 1,000 tasks (Random Seed 42) |
| **Total Tokens** | 32,424 tokens | 66,055 tokens |
| **Routing Mode** | **Live TypeSafe Cloud API** | **High-Throughput Local Fast-Path** |
| **TypeSafe API Calls Made** | **1,000 calls** (2 per prompt) | **0 calls** (In-Memory Engine) |
| **Average Latency** | **905.81 ms** (Public WAN Network Round-Trips) | **1.72 ms** (Local Gateway Overhead) |
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
git clone https://github.com/23f2003771/PromptRouter-Engine.git
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
