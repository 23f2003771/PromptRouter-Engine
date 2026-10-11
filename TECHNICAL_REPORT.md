# Technical Report: PromptRouter Engine
**Cost-Optimized LLM Routing and Adversarial Guardrails**

*Author: Shikhar Singh ([qubt.dev](https://qubt.dev))*  
*Codebase: [GitHub: PromptRouter-Engine](https://github.com/23f2003771/PromptRouter-Engine)*  
*Evaluation Datasets: [RouterBench](https://huggingface.co/datasets/withmartian/routerbench), [LMSYS Chatbot Arena](https://huggingface.co/datasets/routellm/gpt4_dataset), [deepset/prompt-injections](https://huggingface.co/datasets/deepset/prompt-injections)*  

---

## Abstract

Monolithic LLM systems route every prompt to high-parameter frontier models, paying flagship rates on queries that need far less compute. **PromptRouter Engine** is an experimental gateway built with FastAPI, LangGraph, and TypeSafe SDK that evaluates prompt injection risk on a 1–5 rubric, classifies task complexity across 14 categories, and dispatches queries across five model tiers.

In live evaluation over **700 queries** sourced equally from **RouterBench** and **LMSYS Chatbot Arena**, PromptRouter achieved an **estimated 84.06% cost reduction vs. flagship** (`gpt-5.6-terra` at $0.0070/1k tokens), spending $0.1353 compared to $0.8487. On an upstream security test of **500 queries** from `deepset/prompt-injections`, the live TypeSafe `Score` guardrail delivered **100.0% precision** (0.0% false positives on benign queries), **56.78% recall** on injection attacks, and **82.8% accuracy** at **405.83 ms** average latency.

This report distinguishes between the **live semantic router** (tested on 700 tasks) and the **local heuristic fallback** (tested on 3,000 tasks). Cross-tabulation shows that the local keyword fallback misroutes 91.4% of GSM8K arithmetic tasks to the floor model (`gemini-2.5-flash-lite`), making its 81.14% paper savings partly an artifact of defaulting to cheap. The live router avoids the floor model on GSM8K but exhibits a similar pattern: of the 75 GSM8K tasks in the live 350 RouterBench sample, 61 (81.3%) went to `deepseek-v4-flash` ($0.00021/1k — the cheapest model in the catalog), 11 (14.7%) to `deepseek-r1`, and 3 (4.0%) to `gpt-5.6-luna`. Both paths send arithmetic tasks to budget models. Because downstream generation quality was not scored with an LLM judge, quality preservation remains an unmeasured question for future work.

---

## 1. Overview & Motivation

In production traffic, query complexity varies widely:
- **Simple queries:** Factual lookups, string reformatting, binary classifications.
- **Moderate queries:** Summaries, JSON extraction, standard explanations.
- **Complex queries:** Code generation, multi-step math, domain-specific reasoning.

Sending all queries to frontier models ($0.0070–$0.0300/1k tokens) inflates spend on tasks where smaller models ($0.00021–$0.00070/1k) might suffice. Conversely, routing everything to the cheapest model ($0.00021/1k) risks poor answers on math and code. Additionally, unguarded gateways expose downstream models to prompt injection payloads.

We evaluate PromptRouter against three baselines:
1. **Monolithic Flagship:** `gpt-5.6-terra` ($0.0070 / 1k tokens).
2. **Always Cheapest:** `deepseek-v4-flash` ($0.00021 / 1k tokens).

---

## 2. System Architecture

PromptRouter is structured as a LangGraph state machine:

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

### 2.1 Model Catalog (13 Models, 14 Categories)

| Tier | Category | Target Model | Rate / 1k Tokens | Intended Scope |
| :--- | :--- | :--- | :--- | :--- |
| **Tier 1: Heavy Reasoning** | `frontier_autonomous`<br>`agentic_orchestration`<br>`reasoning_heavy`<br>`reasoning_premium`<br>`reasoning_expert` | `gpt-6-astra`<br>`claude-5-1-fable`<br>`deepseek-r1`<br>`o3-reasoning`<br>`claude-5-5-opus` | $0.0300<br>$0.0300<br>$0.0012<br>$0.0050<br>$0.0120 | Autonomous loops, formal proofs, complex algorithmic debugging, compliance analysis |
| **Tier 2: Flagship Workhorses** | `code`<br>`creative`<br>`factual_long_context`<br>`general_premium`<br>`agentic_professional` | `claude-5-sonnet`<br>`claude-5-sonnet`<br>`gemini-3.1-pro`<br>`gpt-5.6-terra`<br>`gpt-6.1-sol` | $0.0060<br>$0.0060<br>$0.0070<br>$0.0070<br>$0.0060 | Full-stack engineering, creative prose, long-context analysis, multi-turn dialogue |
| **Tier 3: Performance Mini** | `agentic_fast`<br>`general_budget` | `gemini-3.8-flash`<br>`gpt-5.6-luna` | $0.00225<br>$0.00070 | Low-latency function calling, structured summaries, JSON extraction |
| **Tier 4: Fast Factual** | `factual` | `deepseek-v4-flash` | $0.00021 | Knowledge retrieval, definitions, short explanations |
| **Tier 5: Default Fallback** | `fallback_cheapest` | `gemini-2.5-flash-lite` | $0.00025 | Short string reformatting, binary classification, unmatched queries |

### 2.2 State Machine
Internal state uses standard `typing.TypedDict` (`RouterState`) to avoid schema validation overhead during node hops. External HTTP boundaries in `models.py` use Pydantic for ingress validation.

---

## 3. Adversarial Security Guardrail

Before classification, prompts enter `check_safety`. The node evaluates injection risk on a 1–5 rubric via TypeSafe SDK `Score`:
- **1–2:** Safe / benign inquiry.
- **3:** Suspicious instruction override hints.
- **4–5:** High-risk jailbreaks, prompt extractions, or exploit payloads.

Prompts scoring > 2.0 trigger `block_request`, halting downstream execution and consuming 0 generation tokens.

### 3.1 Evaluation on `deepset/prompt-injections` (662 Prompts)

| Metric | Live TypeSafe API (500 Prompts) | Local Regex Baseline (662 Prompts) |
| :--- | :--- | :--- |
| **Method** | TypeSafe SDK `Score` (1–5 Rubric) | Static Regex Pattern Scanning |
| **True Positives (Blocked Attacks)** | **113** | 3 |
| **True Negatives (Allowed Benign)** | **301** | 399 |
| **False Positives (Benign Blocked)** | **0** | 0 |
| **False Negatives (Missed Attacks)** | **86** | 260 |
| **Accuracy** | **82.80%** | 60.73% |
| **Precision** | **100.00%** | 100.00% |
| **Recall** | **56.78%** | **1.14%** |
| **F1 Score** | **72.43** | 2.25 |
| **False Positive Rate (FPR)** | **0.00%** | 0.00% |
| **Decision Latency** | **405.83 ms** (WAN TLS) | **0.02 ms** (In-Memory) |

**Finding:** Static regex rules caught only 1.14% of attacks (3/263), failing on rephrased instructions and roleplay wrappers. The live semantic guardrail achieved 100% precision (0 benign prompts blocked) and caught 56.78% of attacks.

---

## 4. Routing Evaluation

### 4.1 Datasets
We tested across 3,000 queries from two sources:
1. **[RouterBench](https://huggingface.co/datasets/withmartian/routerbench) (1,500 tasks):** MMLU (584), HellaSwag (393), GSM8K arithmetic (302), ARC-Challenge (70), WinoGrande (57), MBPP coding (28), and 66 tasks across smaller domains (Chinese language/poetry, consensus summarization, abstract-to-title, bias detection, accounting audit, MT-Bench). Total = 1,500.
2. **[LMSYS Chatbot Arena](https://huggingface.co/datasets/routellm/gpt4_dataset) via RouteLLM (1,500 tasks):** Real-world pairwise human-LLM interactions.

### 4.2 Benchmark Results

| Metric | Live API Run (700 Tasks: 350 RB + 350 LMSYS) | Local Fallback Run (3,000 Tasks) |
| :--- | :--- | :--- |
| **Execution Mode** | **Live TypeSafe Cloud API** | **Local Heuristic Fallback** |
| **External API Calls** | **1,400 calls** (2 per query) | **0 calls** |
| **Average Decision Latency** | **835.91 ms** (WAN TLS) | **2.47 ms** (Local Process) |
| **Throughput** | 7.1 QPS (6 worker threads) | **403.1 QPS** |
| **Flagship Baseline (`gpt-5.6-terra`)** | $0.8487 | $3.6749 |
| **PromptRouter Cost** | **$0.1353** | **$0.6931** |
| **Estimated Cost Reduction vs Flagship**| **84.06% ($0.7134 saved)** | **81.14% ($2.9818 saved)** |
| **Cheapest Baseline (`deepseek-v4-flash`)** | $0.0255 | $0.1102 |
| **Cost Multiplier vs Cheapest** | **5.31x** | **6.29x** |

### 4.3 Model Distribution & Cross-Tabulation

#### A. Live Semantic Run (700 Tasks)
- `deepseek-v4-flash`: 352 (50.3%)
- `gemini-2.5-flash-lite`: 126 (18.0%)
- `gpt-5.6-luna`: 104 (14.9%)
- `claude-5-sonnet`: 65 (9.3%)
- `deepseek-r1`: 20 (2.9%)
- `gpt-5.6-terra`: 20 (2.9%)
- `claude-5-5-opus`: 9 (1.3%)
- `blocked`: 4 (0.6%)

Tier 1 and Tier 2 models received **16.3% (114/700)** of queries.

#### B. Local Fallback Run (3,000 Tasks)
- `gemini-2.5-flash-lite`: 2,073 (69.1%)
- `claude-5-sonnet`: 384 (12.8%)
- `deepseek-v4-flash`: 350 (11.7%)
- `gpt-5.6-luna`: 144 (4.8%)
- Other models (`r1`, `opus`, `terra`, `fable`, `pro`, `sol`): 47 (1.6%)
- `blocked`: 2 (<0.1%)

### 4.4 Analysis: The "Default to Cheap" Vulnerability

Cross-tabulating RouterBench sub-tasks against selected models in the local fallback test revealed a clear failure mode:

| RouterBench Sub-Task | Tasks in Sample | Dispatched: `flash-lite` | Dispatched: `sonnet` | Dispatched: `v4-flash` | Dispatched: Other (`r1`, `opus`, etc.) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`mmlu` (Academic Knowledge)** | 584 | **404 (69.2%)** | 27 (4.6%) | 130 (22.3%) | 23 (3.9%) |
| **`hellaswag` (Commonsense)** | 393 | **322 (81.9%)** | 27 (6.9%) | 25 (6.4%) | 19 (4.8%) |
| **`grade-school-math` (GSM8K)** | 302 | **276 (91.4%)** | 10 (3.3%) | 16 (5.3%) | **0 (0.0%)** |
| **`arc-challenge` (Science QA)** | 70 | **52 (74.3%)** | 4 (5.7%) | 14 (20.0%) | 0 (0.0%) |
| **`winogrande` (Coreference)** | 57 | **56 (98.2%)** | 1 (1.8%) | 0 (0.0%) | 0 (0.0%) |
| **`mbpp` (Python Coding)** | 28 | 0 (0.0%) | **28 (100.0%)** | 0 (0.0%) | 0 (0.0%) |

**Observations:**
1. **Keyword Match Failure:** Static regex rules in `TASK_KEYWORDS["reasoning_heavy"]` check for academic vocabulary (*"proof"*, *"theorem"*, *"calculus"*). GSM8K arithmetic word problems rarely use these words. Consequently, **91.4% of GSM8K tasks dropped into the floor fallback (`gemini-2.5-flash-lite`)**. The 81.14% fallback savings is largely an artifact of defaulting most sub-tasks to the floor model.
2. **Live Semantic Behavior:** Of the 75 GSM8K tasks in the live 350 RouterBench sample, the TypeSafe classifier routed 61 (81.3%) to `deepseek-v4-flash` ($0.00021/1k — the cheapest model in the catalog), 11 (14.7%) to `deepseek-r1`, and 3 (4.0%) to `gpt-5.6-luna`, with 0 reaching the floor model. The live router avoids the floor model on GSM8K but still sends most GSM8K tasks to the cheapest tier, so its 84.06% savings carries the same unverified-quality caveat as the local fallback. Whether `deepseek-v4-flash` maintains acceptable accuracy on multi-step arithmetic remains unmeasured without an LLM judge.

---

## 5. Latency Profiles

| Execution Mode | Average Latency | Throughput | Use Case |
| :--- | :--- | :--- | :--- |
| **Local Regex Guardrail** | ~0.02 ms | — | Instant pre-filter for known attack strings |
| **Local Routing Graph** | 2.47 ms | 403.1 QPS | Offline testing, local fallback when API is down |
| **Cloud Guardrail (TypeSafe `Score`)** | 405.83 ms | — | Semantic prompt injection detection |
| **Full Cloud Path (Guardrail + Classifier)** | 835.91 ms | 7.1 QPS (6 threads) | Full semantic evaluation over public WAN TLS |

---

## 6. Implementation Notes

- **Service Framework:** Built with FastAPI (`main.py`) exposing `POST /api/route` and `GET /health`.
- **Failover:** In `router/nodes.py`, external TypeSafe API calls are wrapped in `try...except`. If the network fails, timeouts occur, or quota is exhausted, execution falls back to in-memory regex and keyword heuristics.
- **Client SDK:** A minimal client in `sdk/client.py` wraps the HTTP endpoint with typed return structures.

---

## 7. Limitations & Future Work

1. **Unmeasured Quality:** Cost savings are estimated based on token pricing matrices. Downstream response quality across models was not scored with an LLM judge due to API quota boundaries.
2. **Sequential Cloud Calls:** Safety scoring and classification execute sequentially. Parallelizing them into concurrent async requests could reduce cloud latency by ~40–50%.
3. **Guardrail Coverage:** While achieving 100% precision, the guardrail recall was 56.78%, showing that subtle injection attacks can evade a single-prompt rubric.

---

## 8. Conclusion

PromptRouter Engine demonstrates the mechanics and tradeoffs of multi-tier LLM routing. In live API evaluation across 700 benchmark queries, the semantic router achieved an **estimated 84.06% cost reduction** compared to a monolithic flagship baseline (`gpt-5.6-terra`). Concurrently, the upstream TypeSafe guardrail demonstrated **100.0% precision** (0% false positives) and **56.78% recall** on 500 labeled prompts from `deepset/prompt-injections`. At the same time, analysis of the local fallback highlights the limits of static keyword matching, underscoring that meaningful cost optimization requires semantic evaluation alongside direct quality verification.

*Code and benchmark scripts are available under the MIT License at [GitHub: PromptRouter-Engine](https://github.com/23f2003771/PromptRouter-Engine).*
