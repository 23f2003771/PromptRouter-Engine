# Technical Report: PromptRouter Engine
**An Exploratory Study on Cost-Optimized LLM Routing and Adversarial Guardrails**

*Author: Shikhar Singh ([qubt.dev](https://qubt.dev))*  
*Codebase: [GitHub: PromptRouter-Engine](https://github.com/23f2003771/PromptRouter-Engine)*  
*Evaluation Datasets: [RouterBench](https://huggingface.co/datasets/withmartian/routerbench), [LMSYS Chatbot Arena (RouteLLM)](https://huggingface.co/datasets/routellm/gpt4_dataset), [deepset/prompt-injections](https://huggingface.co/datasets/deepset/prompt-injections)*  

---

## Abstract

Monolithic LLM architectures route all inbound user queries to high-parameter frontier models, incurring substantial inference expenses on tasks that require only modest compute. This technical report investigates **PromptRouter Engine**, an experimental routing gateway built with FastAPI, LangGraph, and TypeSafe SDK designed to evaluate task complexity, enforce pre-execution prompt injection security, and dispatch requests across five capability and economic tiers.

We evaluate PromptRouter across **3,000 queries** sourced equally from **RouterBench** (1,500 tasks spanning GSM8K, MBPP, MMLU, and Winogrande) and **LMSYS Chatbot Arena** (1,500 real-world user interactions). On the full 3,000-task benchmark, PromptRouter achieved an **81.14% cost reduction** compared to a monolithic flagship baseline (`gpt-5.6-terra` at $0.0070/1k tokens), spending $0.6931 versus $3.6749, and an **84.11% reduction** compared to random routing ($4.3616). When evaluated live across 700 queries (350 RouterBench + 350 LMSYS Arena) through WAN TLS calls to the TypeSafe Cloud API, PromptRouter achieved an **84.06% cost reduction** ($0.1353 vs $0.8487).

To verify adversarial resilience, the upstream prompt injection guardrail was tested against **662 labeled queries** from `deepset/prompt-injections`. On a 500-sample live API evaluation using a 1–5 risk rubric, the guardrail achieved **82.8% accuracy**, **100.0% precision** (0.0% false positive rate on benign prompts), and **56.78% recall** with an average latency of **405.83 ms**. In contrast, local regex heuristics caught only **1.14%** of attacks, illustrating the necessity of semantic scoring for adversarial detection. In-memory heuristic routing executes in **~2.47 ms**, providing a functional offline fallback when external network connectivity is unavailable.

---

## 1. Introduction

### 1.1 Context and Motivation
Standard generative AI architectures typically direct all incoming traffic to a single flagship model. In practice, user queries vary widely in computational complexity:
- **Low-complexity queries:** Factual lookups, binary classifications, syntactic reformatting, and brief conversational exchanges.
- **Moderate queries:** Multi-paragraph summarization, structured JSON schema transformation, and general Q&A.
- **High-complexity queries:** Multi-step logical proofs, deep code refactoring, algorithmic synthesis, and domain-specific analysis.

Sending straightforward factual or structural queries to frontier models priced at $0.0070–$0.0300 per 1k tokens consumes budget that yields minimal marginal quality over models costing $0.00021–$0.00070 per 1k tokens. Conversely, always routing to the cheapest model sacrifices quality on complex coding and reasoning tasks.

Furthermore, dispatching raw user prompts directly to downstream LLMs creates vulnerability to prompt injection and jailbreak payloads, consuming inference tokens on adversarial requests.

### 1.2 Project Scope
PromptRouter was built as a personal systems and research exploration to evaluate:
1. **Dynamic Multi-Tier Routing:** Can a structured task matrix route user queries to cost-appropriate models while reserving high-parameter models for demanding tasks?
2. **Pre-Execution Guardrails:** Can an upstream semantic scoring step intercept prompt injection attempts before downstream tokens are spent?
3. **Multi-Baseline Benchmarking:** How does adaptive routing perform when compared against three distinct baselines:
   - A **Monolithic Flagship Baseline** (`gpt-5.6-terra` at $0.0070/1k tokens).
   - An **Always Cheapest Baseline** (`gemini-2.5-flash-lite` at $0.00025/1k tokens).
   - A **Random Routing Baseline** across all available models.

---

## 2. System Architecture

PromptRouter is implemented as a directed acyclic graph using **LangGraph**, coordinating state transitions across safety evaluation, task classification, model selection, and response attribution.

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
          ┌─────────────────────────────────────────────────────────────────────────┐
          │                       5-Tier Model Routing Matrix                       │      
          └─┬────────────────┬─────────────────┬─────────────────┬────────────────┬─┘
            │                │                 │                 │                │
            ▼                ▼                 ▼                 ▼                ▼
    ┌───────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────────┐
    │    Tier 1     │ │    Tier 2    │ │    Tier 3    │ │    Tier 4    │ │    Tier 5        │
    │Heavy Reasoning│ │Flagship Work-│ │Performance   │ │Fast Factual  │ │Micro & Floor   │
    │& Autonomous   │ │horses        │ │Mini Tier     │ │Knowledge     │ │Fallback        │
    │deepseek-r1    │ │claude-5-     │ │gemini-3.8-   │ │deepseek-v4-  │ │gemini-2.5-     │
    │o3-reasoning   │ │sonnet        │ │flash         │ │flash         │ │flash-lite      │
    │claude-5-5-opus│ │gpt-5.6-terra │ │gpt-5.6-luna  │ │              │ │                  │
    │($0.0012-$0.03)│ │($0.006-$0.007│ │($0.0007-$0.00│ │($0.00021/1k) │ │($0.00025/1k)     │
    └───────────────┘ └──────────────┘ └──────────────┘ └──────────────┘ └──────────────────┘
```

### 2.1 The 5-Tier Model Matrix
The routing catalog in `router/config.py` maps 14 specialized task categories across 5 capability tiers:

| Tier | Category / Task Type | Target Model | Blended Rate / 1k Tokens | Scope and Description |
| :--- | :--- | :--- | :--- | :--- |
| **Tier 1: Heavy Reasoning & Frontiers** | `frontier_autonomous`<br>`agentic_orchestration`<br>`reasoning_heavy`<br>`reasoning_premium`<br>`reasoning_expert` | `gpt-6-astra`<br>`claude-5-1-fable`<br>`deepseek-r1`<br>`o3-reasoning`<br>`claude-5-5-opus` | $0.0300<br>$0.0300<br>$0.0012<br>$0.0050<br>$0.0120 | Long-horizon agentic workflows, formal mathematical proofs, complex algorithmic debugging, and deep compliance/legal analysis |
| **Tier 2: Flagship Workhorses** | `code`<br>`creative`<br>`factual_long_context`<br>`general_premium`<br>`agentic_professional` | `claude-5-sonnet`<br>`claude-5-sonnet`<br>`gemini-3.1-pro`<br>`gpt-5.6-terra`<br>`gpt-6.1-sol` | $0.0060<br>$0.0060<br>$0.0070<br>$0.0070<br>$0.0060 | Software engineering, nuanced creative writing, large-context document analysis, multi-turn consultation, and structured microservice tasks |
| **Tier 3: Performance Mini** | `agentic_fast`<br>`general_budget` | `gemini-3.8-flash`<br>`gpt-5.6-luna` | $0.00225<br>$0.00070 | Low-latency voice/function loops, everyday summarization, and JSON schema extraction |
| **Tier 4: Fast Factual** | `factual` | `deepseek-v4-flash` | $0.00021 | Factual question answering, definitions, entity lookups, and historical queries |
| **Tier 5: Micro & Floor Fallback** | `fallback_cheapest` | `gemini-2.5-flash-lite` | $0.00025 | Baseline fallback for unmatched queries and lightweight tasks |

### 2.2 Graph State Design
Internal state transitions in `router/state.py` use Python's standard `typing.TypedDict`:
```python
class RouterState(TypedDict, total=False):
    prompt: str
    safety_score: float
    is_blocked: bool
    task_type: str
    selected_model: str
    response_text: str
    tokens_used: int
    estimated_cost: float
    cost_saved: float
    latency_ms: float
```
External HTTP boundaries in `models.py` use Pydantic (`TaskRequest`, `RoutingResponse`) for request validation, while internal node hops pass dictionary-like state objects to avoid unnecessary validation overhead during graph traversal.

---

## 3. Adversarial Security Guardrail

PromptRouter evaluates input safety prior to task classification. Incoming prompts enter `check_safety` before downstream dispatching.

### 3.1 Rubric-Based Scoring
Rather than a simple binary flag, the guardrail assesses prompt injection risk on a 1-to-5 scale using the TypeSafe SDK `Score` primitive:
- **Level 1 (Safe):** Benign user instruction with zero adversarial intent.
- **Level 2 (Low Risk):** Unusual syntax or academic inquiries without boundary override attempts.
- **Level 3 (Suspicious):** Mild directive alteration or boundary probing (e.g., "disregard prior context").
- **Level 4 (High Risk / Jailbreak):** Explicit instruction overrides, role-playing jailbreaks, or system prompt extraction.
- **Level 5 (Severe / Exploit):** Multi-vector adversarial payloads or obfuscated exploit patterns.

Prompts receiving a risk score greater than 2.0 (`config.INJECTION_THRESHOLD`) trigger the conditional edge `route_safety` to divert directly to `block_request`, returning an alert and consuming 0 downstream inference tokens.

### 3.2 Empirical Guardrail Evaluation
We evaluated the guardrail on **662 labeled prompts** from the Hugging Face research dataset [`deepset/prompt-injections`](https://huggingface.co/datasets/deepset/prompt-injections), consisting of **263 adversarial injection attacks** and **399 benign prompts**.

We tested **500 prompts on the live TypeSafe Cloud API** and compared against a **local regex heuristic baseline** across all 662 prompts:

| Metric | Live TypeSafe API (500 Prompts) | Full Dataset Local Regex (662 Prompts) |
| :--- | :--- | :--- |
| **Total Evaluated** | 500 prompts | 662 prompts |
| **True Positives (Blocked Attacks)** | **113** | 3 |
| **True Negatives (Allowed Benign)** | **301** | 399 |
| **False Positives (Benign Blocked)** | **0** | 0 |
| **False Negatives (Missed Attacks)** | **86** | 260 |
| **Accuracy** | **82.80%** | 60.73% |
| **Precision** | **100.00%** | 100.00% |
| **Recall** | **56.78%** | 1.14% |
| **F1 Score** | **72.43** | 2.25 |
| **False Positive Rate (FPR)** | **0.00%** | 0.00% |
| **Average Decision Latency** | **405.83 ms** | **0.02 ms** |

#### Findings and Research Observations:
1. **Zero False Positives:** The TypeSafe API achieved 100% precision with 0 false positives across 301 benign prompts, meaning normal user requests were never erroneously blocked.
2. **Semantic vs. Keyword Detection:** The static regex baseline exhibited a recall of only **1.14%**, missing 260 of 263 attacks. Simple string matching for phrases like `"ignore previous"` fails against adversarial paraphrasing, roleplay scenarios, and obfuscated prompts. Semantic scoring via the TypeSafe API captured **56.78%** of attacks without false positives.
3. **Latency Profile:** The cloud evaluation requires a WAN round-trip taking **~405.8 ms**. For environments requiring strict sub-50ms ingress, this represents a deliberate tradeoff between security coverage and latency.

---

## 4. Empirical Routing Evaluation

### 4.1 Evaluation Methodology and Datasets
To evaluate routing performance on modern benchmark data, we selected two prominent open research datasets totaling **3,000 tasks**:
1. **[RouterBench](https://huggingface.co/datasets/withmartian/routerbench) (1,500 tasks):** A benchmark created specifically for evaluating LLM routers, spanning reasoning (GSM8K), code (MBPP), knowledge (MMLU), and commonsense (Winogrande).
2. **[LMSYS Chatbot Arena](https://huggingface.co/datasets/routellm/gpt4_dataset) via RouteLLM (1,500 tasks):** Real-world pairwise human-LLM interactions reflecting authentic user prompt distributions.

### 4.2 Benchmark Results Across Modes

We conducted three evaluation runs:
- **700-Task Live API Run:** 350 random RouterBench + 350 random LMSYS Arena tasks executed via live TypeSafe Cloud API calls (1,400 total API calls: safety check + classification).
- **2,300-Task Local Remainder Run:** The remaining 2,300 tasks evaluated locally via compiled regex and word-boundary heuristics.
- **3,000-Task Full Benchmark:** All 3,000 tasks evaluated on the local fast-path engine.

| Metric | 700-Task Live API Run (350 RB + 350 LMSYS) | 2,300-Task Local Remainder Run | Full 3,000-Task Local Benchmark |
| :--- | :--- | :--- | :--- |
| **Tasks Evaluated** | 700 queries | 2,300 queries | 3,000 queries |
| **Execution Mode** | **Live TypeSafe Cloud API** | **Local Fast-Path** | **Local Fast-Path** |
| **External API Calls** | **1,400 calls** (2 per prompt) | **0 calls** | **0 calls** |
| **Total Tokens Processed** | 121,237 tokens | 403,533 tokens | 524,983 tokens |
| **Average Decision Latency** | **835.91 ms** (WAN TLS calls) | **2.63 ms** (In-Memory) | **2.47 ms** (In-Memory) |
| **Throughput** | 7.1 QPS (6 worker threads) | 378.8 QPS | 403.1 QPS |
| **Flagship Baseline Cost (`gpt-5.6-terra`)** | $0.8487 | $2.8247 | $3.6749 |
| **Cheapest Baseline Cost (`gemini-2.5-flash-lite`)** | $0.0303 | $0.1009 | $0.1312 |
| **Random Routing Baseline Cost** | $0.9176 | $3.6120 | $4.3616 |
| **PromptRouter Cumulative Cost** | **$0.1353** | **$0.5446** | **$0.6931** |
| **Dollars Saved vs Flagship** | **$0.7134** | **$2.2801** | **$2.9818** |
| **Cost Reduction vs Flagship** | **84.06%** | **80.72%** | **81.14%** |
| **Cost Reduction vs Random** | **85.26%** | **84.92%** | **84.11%** |
| **Multiplier vs Always Cheapest** | **4.46x** | **5.40x** | **5.28x** |

### 4.3 Multi-Baseline Analysis and Quality Rationale

A common weakness in router evaluations is relying exclusively on a monolithic flagship baseline. We compare PromptRouter against three distinct reference strategies:

1. **Monolithic Flagship (`gpt-5.6-terra` @ $0.0070/1k tokens):**
   - Cost: $3.6749 across 3,000 tasks.
   - PromptRouter achieves **81.14% cost savings** ($0.6931 total cost), demonstrating that the vast majority of everyday queries do not require a $0.0070/1k model.
2. **Random Routing Baseline:**
   - Cost: $4.3616 across 3,000 tasks.
   - PromptRouter achieves **84.11% cost savings**, proving that structured classification outperforms unguided model selection.
3. **Always Cheapest Baseline (`gemini-2.5-flash-lite` @ $0.00025/1k tokens):**
   - Cost: $0.1312 across 3,000 tasks (a theoretical 96.43% cost reduction).
   - **Quality Tradeoff Discussion:** Routing 100% of tasks to the absolute cheapest micro-tier model maximizes paper savings, but inevitably degrades output quality on multi-step reasoning, mathematical proofs (GSM8K), and complex programming tasks (MBPP). PromptRouter incurs **5.28x the cost of the cheapest baseline** ($0.6931 vs. $0.1312) precisely because it selectively routes high-difficulty tasks to Tier 1 and Tier 2 models (`claude-5-sonnet`, `deepseek-r1`, `o3-reasoning`, `gpt-5.6-terra`). This selective allocation preserves response capability on difficult tasks while capturing the bulk of economic savings on straightforward queries.

### 4.4 Model Selection Distribution

The distribution of selected models across the runs reveals how tasks were apportioned:

#### 700-Task Live API Run:
- **50.3% (352 tasks):** `deepseek-v4-flash` ($0.00021/1k) — Factual questions, general knowledge inquiries.
- **18.0% (126 tasks):** `gemini-2.5-flash-lite` ($0.00025/1k) — Short, generic conversational prompts.
- **14.9% (104 tasks):** `gpt-5.6-luna` ($0.00070/1k) — Structured summaries, list extraction, drafting.
- **9.3% (65 tasks):** `claude-5-sonnet` ($0.0060/1k) — Complex coding, algorithm implementation, creative prose.
- **2.9% (20 tasks):** `deepseek-r1` ($0.0012/1k) — Mathematical proofs and logic problems.
- **2.9% (20 tasks):** `gpt-5.6-terra` ($0.0070/1k) — General premium dialogue and multi-turn tasks.
- **1.3% (9 tasks):** `claude-5-5-opus` ($0.0120/1k) — Specialized expert reasoning.
- **0.6% (4 tasks):** `blocked` — Intercepted by prompt injection guardrail.

#### Full 3,000-Task Benchmark (Local Heuristic Engine):
- **69.1% (2,073 tasks):** `gemini-2.5-flash-lite` — Floor fallback for general queries.
- **12.8% (384 tasks):** `claude-5-sonnet` — Code generation and algorithmic prompts.
- **11.7% (350 tasks):** `deepseek-v4-flash` — Explicit factual and explanatory queries.
- **4.8% (144 tasks):** `gpt-5.6-luna` — Summarization and structured formatting tasks.
- **0.4% (13 tasks):** `claude-5-5-opus` — Specialized domain analysis.
- **0.3% (8 tasks):** `deepseek-r1` — Formal mathematical derivation.
- **0.3% (8 tasks):** `o3-reasoning` — Advanced logic tasks.
- **0.3% (8 tasks):** `gpt-5.6-terra` — Premium general assistance.
- **0.2% (6 tasks):** `claude-5-1-fable` — Multi-agent workflow planning.
- **0.1% (3 tasks):** `gemini-3.1-pro` — Long-document context.
- **<0.1% (1 task):** `gpt-6.1-sol` — Professional automation.
- **<0.1% (2 tasks):** `blocked` — Heuristic attack block.

---

## 5. Latency Characterization

A key goal of this report is to present latency transparently without confusing disparate execution modes:

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                           LATENCY PROFILES                              │
├──────────────────────────────────────┬──────────────────────────────────┤
│ Mode                                 │ Average Latency                  │
├──────────────────────────────────────┼──────────────────────────────────┤
│ 1. Local Regex Guardrail (Heuristic) │ ~0.02 ms                         │
│ 2. Local Routing Graph (Fast-Path)   │ 2.47 – 2.63 ms                   │
│ 3. Cloud Guardrail Only (TypeSafe)   │ 405.83 ms                        │
│ 4. Full Cloud Pipeline (Guard+Class) │ 835.91 ms                        │
└──────────────────────────────────────┴──────────────────────────────────┘
```

- **Cloud Semantic Path (835.91 ms):** Involves two sequential WAN HTTPS requests to the TypeSafe API: one for 1–5 risk scoring and one for 14-way task classification. This overhead is typical for cloud-hosted evaluation services and is suitable when classification quality takes precedence over sub-100ms response requirements.
- **Local Heuristic Fast-Path (2.47 ms):** Involves in-memory regex scanning and keyword matching executed within the local Python process. This path processes ~400 QPS on a single core, making it effective for offline evaluation or as a high-throughput fallback.

---

## 6. Implementation Architecture

### 6.1 Service Framework
The engine runs on FastAPI (`main.py`), exposing:
- `POST /api/route`: Accepts `{ "prompt": "..." }`, runs the LangGraph pipeline, and returns the selected model, risk score, estimated cost, and estimated savings.
- `GET /health`: Health-check endpoint confirming graph compilation status.

### 6.2 Offline Failover Handling
In `router/nodes.py`, calls to external services are wrapped in `try...except` blocks. If an external API call raises an exception (due to network timeout, missing credentials, or quota limits), the router automatically executes local keyword matching and regex safety scanning, preventing unhandled exceptions from terminating the request.

### 6.3 Lightweight Client SDK
A Python SDK (`sdk/client.py`) provides typed client access:
```python
from sdk import PromptRouterClient

client = PromptRouterClient(base_url="http://127.0.0.1:8000")
response = client.route("Write a Python function to compute Fibonacci numbers.")
print(f"Model: {response.routed_model} | Saved: ${response.cost_saved}")
```

---

## 7. Limitations and Future Work

As an exploratory project, PromptRouter has several recognized limitations:
1. **Response Quality Scoring:** In this evaluation, cost savings were computed using model pricing matrices and token estimations. While tasks were mapped to models capable of handling them based on benchmark domain (e.g. MBPP to `claude-5-sonnet`, GSM8K to `deepseek-r1`), direct LLM-as-a-judge quality scoring across all generated responses was not conducted due to API quota constraints.
2. **Sequential API Calls in Cloud Mode:** The cloud path executes safety scoring and classification sequentially. Parallelizing these two requests into a single prompt or concurrent asynchronous calls could reduce cloud latency by approximately 40–50%.
3. **Guardrail Recall:** While achieving 100% precision on the evaluated sample, the guardrail recall of 56.78% indicates that sophisticated prompt injections can still evade a single-prompt rubric. Layering additional defense techniques (such as output filtering or perplexity analysis) warrants further investigation.

---

## 8. Conclusion

PromptRouter Engine illustrates the economic feasibility of dynamic LLM task routing. Across 3,000 tasks from RouterBench and LMSYS Chatbot Arena, adaptive routing achieved an **81.14% cost reduction** compared to a monolithic `gpt-5.6-terra` baseline and an **84.11% reduction** compared to random routing. At the same time, testing against `deepset/prompt-injections` demonstrated that semantic guardrails can filter malicious inputs with high precision (100% in our sample), whereas static keyword heuristics miss the vast majority of attacks.

*All code, datasets, evaluation runners, and reproduction results are available under the MIT License at [GitHub: PromptRouter-Engine](https://github.com/23f2003771/PromptRouter-Engine).*
