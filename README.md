# PromptRouter Engine 🚀

An exploratory, cost-optimized LLM orchestration gateway and research project built with **FastAPI**, **LangGraph**, and **TypeSafe SDK**.

PromptRouter evaluates incoming user prompts, intercepts prompt injection attempts using a 5-scale security evaluator, and routes queries to the most cost-effective model across 5 intelligence tiers based on task complexity.

---

## 🏛️ System Architecture & 5-Tier Routing Matrix

PromptRouter implements an upstream security guardrail followed by dynamic routing across five specialized model tiers:

```text
               ┌───────────────────────────────┐
               │     Inbound Task Payload      │
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
          ┌─────────────────────────────────────────────────────────────────────────────┐
          │                          5-Tier Model Routing Matrix                        │      
          └─┬────────────────┬───────────────────┬─────────────────┬──────────────────┬─┘
            │                │                   │                 │                  │
            ▼                ▼                   ▼                 ▼                  ▼
    ┌───────────────┐ ┌───────────────┐ ┌─────────────────┐ ┌──────────────┐ ┌──────────────────┐
    │    Tier 1     │ │    Tier 2     │ │    Tier 3       │ │    Tier 4    │ │    Tier 5        │
    │Heavy Reasoning│ │Flagship Work- │ │Performance      │ │Fast Factual  │ │     Default      │
    │& Autonomous   │ │horses         │ │Mini Tier        │ │Knowledge     │ │    Fallback      │
    │deepseek-r1    │ │claude-5-      │ │gemini-3.8-      │ │deepseek-v4-  │ │   gemini-2.5-    │
    │o3-reasoning   │ │sonnet         │ │flash            │ │flash         │ │   flash-lite     │
    │claude-5-5-opus│ │gpt-5.6-terra  │ │gpt-5.6-luna     │ │              │ │                  │
    │($0.0012-$0.03)│ │($0.006-$0.007)│ │($0.0007-$0.0023)│ │($0.00021/1k) │ │($0.00025/1k)     │
    └───────────────┘ └───────────────┘ └─────────────────┘ └──────────────┘ └──────────────────┘
```

1. **Adversarial Guardrail Node**: Evaluates prompt injection risk on a 1–5 scale. Prompts scoring > 2.0 are blocked immediately, preventing model compromise and consuming 0 downstream inference tokens.
2. **Dual-Path Routing Architecture**:
   - **Cloud Semantic Path (TypeSafe SDK)**: Performs semantic classification and probabilistic safety scoring via TypeSafe System One API over public WAN TLS (~835 ms decision latency).
   - **Local Heuristic Fast-Path (In-Memory)**: In-memory regex guardrails and word-boundary keyword routing (~2.5 ms latency, ~400 QPS) for high-throughput testing or when running offline without external API quota.
3. **5-Tier Model Selection**: Routes complex coding and formal reasoning to Tier 1/2, general chat to Tier 2, summarization/JSON extraction to Tier 3, factual queries to Tier 4, and simple formatting to Tier 5.
4. **Economics & Telemetry**: Automatically records latency, token count, cost, and savings against multiple baseline strategies.

---

### 📋 5-Tier Routing Catalog

PromptRouter defines 14 task categories in `router/config.py` mapped across 5 operational tiers and 13 modern LLMs:

| Tier | Category / Task Type | Target Model | Blended Rate / 1k Tokens | Capabilities & Scope |
| :--- | :--- | :--- | :--- | :--- |
| **Tier 1: Heavy Reasoning & Frontiers** | `frontier_autonomous`<br>`agentic_orchestration`<br>`reasoning_heavy`<br>`reasoning_premium`<br>`reasoning_expert` | `gpt-6-astra`<br>`claude-5-1-fable`<br>`deepseek-r1`<br>`o3-reasoning`<br>`claude-5-5-opus` | $0.0300<br>$0.0300<br>$0.0012<br>$0.0050<br>$0.0120 | Long-horizon agentic workflows, formal mathematical proofs, complex algorithmic debugging, and deep regulatory/legal analysis |
| **Tier 2: Flagship Workhorses** | `code`<br>`creative`<br>`factual_long_context`<br>`general_premium`<br>`agentic_professional` | `claude-5-sonnet`<br>`claude-5-sonnet`<br>`gemini-3.1-pro`<br>`gpt-5.6-terra`<br>`gpt-6.1-sol` | $0.0060<br>$0.0060<br>$0.0070<br>$0.0070<br>$0.0060 | Full-stack software engineering, nuanced editorial prose, 1M+ token document analysis, multi-turn consultation, and structured microservice tasks |
| **Tier 3: Performance Mini** | `agentic_fast`<br>`general_budget` | `gemini-3.8-flash`<br>`gpt-5.6-luna` | $0.00225<br>$0.00070 | Low-latency agentic loops, high-velocity JSON schema extraction, structured entity tagging |
| **Tier 4: High-Speed Factual** | `factual` | `deepseek-v4-flash` | $0.00021 | High-throughput factual QA, knowledge retrieval, definitions, short explanations |
| **Tier 5: Default Fallback** | `fallback_cheapest` | `gemini-2.5-flash-lite` | $0.00025 | Basic string reformatting, binary classification, unmatched fallback queries |

---

## 📊 Benchmark Evaluation Results

> **Key Findings:** **Estimated 84.06% cost reduction vs. flagship** on 700 RouterBench/LMSYS queries; **100.0% precision, 56.78% recall** guardrail on 500 deepset prompts.

Evaluated against modern research benchmarks:
- **[RouterBench](https://huggingface.co/datasets/withmartian/routerbench) (1,500 tasks):** MMLU (584), HellaSwag (393), GSM8K arithmetic (302), ARC-Challenge (70), WinoGrande (57), MBPP coding (28), and 66 tasks across smaller domains (Chinese language/poetry, consensus summarization, abstract-to-title, bias detection, accounting audit, MT-Bench). Total = 1,500.
- **[LMSYS Chatbot Arena](https://huggingface.co/datasets/routellm/gpt4_dataset) via RouteLLM (1,500 tasks):** Real-world user interactions reflecting authentic prompt distributions.

### Routing Performance Across Baselines

| Metric | 700-Task Live API Run (350 RB + 350 LMSYS) | Full 3,000-Task Local Fallback |
| :--- | :--- | :--- |
| **Tasks Evaluated** | 700 queries | 3,000 queries |
| **Routing Mode** | **Live TypeSafe Cloud API** (Semantic Router) | **Local Fast-Path Heuristic** (Static Fallback) |
| **External API Calls Made** | **1,400 calls** (2 per prompt) | **0 calls** (In-Memory Engine) |
| **Average Decision Latency** | **835.91 ms** (Public WAN Network) | **2.47 ms** (Local Overhead) |
| **Throughput** | 7.1 QPS (6 worker threads) | **403.1 QPS** |
| **Flagship Baseline (`gpt-5.6-terra` @ $0.0070/1k)** | $0.8487 | $3.6749 |
| **Cheapest Baseline (`deepseek-v4-flash` @ $0.00021/1k)** | $0.0255 | $0.1102 |
| **PromptRouter Cost** | **$0.1353** | **$0.6931** |
| **Dollars Saved vs Flagship** | **$0.7134** | **$2.9818** |
| **Cost Reduction vs Flagship** | **84.06% SAVINGS (Semantic router)** | **81.14% SAVINGS (Heuristic fallback, see analysis)** |
| **Cost Multiplier vs Cheapest** | **5.31x** | **6.29x** |

### 🔍 Critical Analysis: Semantic Router vs. Heuristic Fallback
Cross-tabulating `RouterBench` sub-tasks against selected models reveals the difference between the **live semantic run (84.06%)** and the **3,000-task local fallback (81.14%)**:
- **Fallback Keyword Vulnerability:** In the 3,000-task local run, **91.4% (276 / 302) of GSM8K grade-school math tasks fell through to `gemini-2.5-flash-lite`**. Static regex keywords looked for formal terms (*"proof"*, *"theorem"*, *"calculus"*) and failed to detect arithmetic word problems, defaulting them to the default fallback model.
- **Semantic Routing Behavior:** Of the 75 GSM8K tasks in the live 350 RouterBench sample, the TypeSafe classifier routed 61 (81.3%) to `deepseek-v4-flash` ($0.00021/1k — the cheapest model in the catalog), 11 (14.7%) to `deepseek-r1`, and 3 (4.0%) to `gpt-5.6-luna`. The live router avoids the floor model on GSM8K but still sends most GSM8K tasks to the cheapest tier, so its 84.06% savings carries the same unverified-quality caveat as the local fallback. Whether `deepseek-v4-flash` maintains acceptable accuracy on multi-step arithmetic remains unmeasured without an LLM judge.


---

## 🛡️ Adversarial Guardrail Evaluation

Evaluated against **662 labeled queries** from the Hugging Face research dataset [`deepset/prompt-injections`](https://huggingface.co/datasets/deepset/prompt-injections) (263 attacks, 399 benign):

| Metric | Live TypeSafe API (500 Prompts) | Local Regex Heuristic (662 Prompts) |
| :--- | :--- | :--- |
| **Evaluation Method** | TypeSafe SDK `Score` (1–5 Rubric) | Static Regex Pattern Scanning |
| **Accuracy** | **82.80%** | 60.73% |
| **Precision** | **100.00%** (0 false positives on benign queries) | 100.00% |
| **Recall** | **56.78%** (blocked 113 of 199 attacks in sample) | **1.14%** (blocked 3 of 263 attacks) |
| **F1 Score** | **72.43** | 2.25 |
| **False Positive Rate (FPR)** | **0.00%** | 0.00% |
| **Average Decision Latency** | **405.83 ms** (Public WAN Network) | **0.02 ms** (Local Process) |

**Key Research Finding:** Static keyword regex filters (6 patterns) caught only **1.14%** of real injection attacks, failing on rephrased instructions, roleplays, and obfuscations. Semantic scoring via the TypeSafe API captured **56.78%** of attacks with **0.0% false positives** on benign inputs, substantially outperforming static regex on this dataset.

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
Create a `.env` file from `.env.example`:
```bash
cp .env.example .env
```
*(Optional: Set `TYPESAFE_API_KEY` for live cloud classification and scoring).*

### 3. Run the Engine
Start the FastAPI server:
```bash
python -m uvicorn main:app --port 8000
```
Interactive API documentation is available at `http://127.0.0.1:8000/docs`.

---

## 💻 Using the Python SDK

PromptRouter provides a lightweight client library (`sdk/client.py`):

```python
from sdk import PromptRouterClient

# 1. Connect to engine
client = PromptRouterClient(base_url="http://127.0.0.1:8000")

# 2. Route any task
response = client.route("Write a Python function to sort a list using quicksort.")

print(f"Routed Model : {response.routed_model}")
print(f"Safety Score : {response.safety_score}")
print(f"Cost Saved   : ${response.cost_saved}")
print(f"Tokens Used  : {response.tokens_used}")
```

---

## 🔬 Reproducing the Benchmarks

All evaluation data and test runners are reproducible:

```bash
# 1. Download RouterBench, LMSYS Arena, and deepset/prompt-injections
python benchmark/download_dataset.py

# 2. Run the prompt injection guardrail evaluation (500 live + remainder local)
python benchmark/evaluate_guardrail.py

# 3. Run the routing benchmark (700 live + remainder local + full 3k local)
python benchmark/evaluate.py
```

Results are saved to:
- `benchmark/guardrail_benchmark_results.json`
- `benchmark/benchmark_results_700_live.json`
- `benchmark/benchmark_results_2300_local.json`
- `benchmark/benchmark_results.json`

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
