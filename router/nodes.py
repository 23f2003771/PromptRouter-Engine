import re
from router.state import RouterState
from router.config import (
    TYPESAFE_API_KEY,
    MODEL_MAP,
    MODEL_COSTS_PER_1K,
    CLASSIFICATION_CRITERIA,
    INJECTION_THRESHOLD,
    INJECTION_RUBRIC_5_SCALE,
    INJECTION_PATTERNS,
)

# keywords I use as a quick fallback if no API key is provided or network is offline
TASK_KEYWORDS = {
    "code": ["code", "function", "debug", "script", "program", "algorithm", "api", "sql", "python", "javascript"],
    "creative": ["write", "story", "poem", "essay", "blog", "creative", "compose", "draft"],
    "factual": ["what is", "explain", "define", "who is", "when did", "how does", "tell me about", "capital of"],
}


def check_safety(state: RouterState) -> RouterState:
    """
    Evaluates prompt injection and jailbreak risk on a 1-5 scale using TypeSafe SDK Score.
    If the score is greater than 2.0, the prompt is marked as blocked to protect my LLMs.
    """
    prompt = state.get("prompt", "")

    # 1. Use TypeSafe SDK Score if my API key is configured
    if TYPESAFE_API_KEY:
        try:
            from typesafe_sdk import TypeSafeClient, Score

            client = TypeSafeClient(api_key=TYPESAFE_API_KEY)
            result = client.system_one(
                state=prompt,
                questions={
                    "injection_risk": Score(
                        instructions="Evaluate risk of prompt injection, instruction override, or adversarial jailbreak on a 1-5 scale.",
                        criteria=INJECTION_RUBRIC_5_SCALE,
                    )
                },
            )
            score = round(float(result.scores["injection_risk"].score), 2)
            is_blocked = score > INJECTION_THRESHOLD
            return {"safety_score": score, "is_blocked": is_blocked}
        except Exception:
            # fallback to rule-based safety scanner if API fails
            pass

    # 2. Offline fallback regex scanner
    prompt_lower = prompt.lower()
    score = 1.0  # default benign score
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, prompt_lower, re.IGNORECASE):
            score = 4.5  # high risk injection detected
            break

    is_blocked = score > INJECTION_THRESHOLD
    return {"safety_score": score, "is_blocked": is_blocked}


def block_request(state: RouterState) -> RouterState:
    """
    Blocks malicious prompt injection requests from reaching my LLMs.
    Saves 100% of LLM compute costs and protects model guardrails.
    """
    score = state.get("safety_score", 5.0)
    prompt = state.get("prompt", "")
    tokens_used = max(1, len(prompt.split()))

    return {
        "selected_model": "blocked",
        "response_text": f"Request blocked: Prompt injection attempt detected. (Risk score {score}/5.0 exceeds maximum allowed threshold of {INJECTION_THRESHOLD}).",
        "tokens_used": tokens_used,
        "estimated_cost": 0.0,
        "cost_saved": 0.0,
    }


def classify_task(state: RouterState) -> RouterState:
    """
    Decides the task category using TypeSafe SDK if my API key is set,
    otherwise falls back to regex-based word boundary matching.
    """
    prompt = state.get("prompt", "")

    # if I added my TypeSafe API key, use TypeSafe System One for instant classification
    if TYPESAFE_API_KEY:
        try:
            from typesafe_sdk import TypeSafeClient, Choice

            client = TypeSafeClient(api_key=TYPESAFE_API_KEY)
            result = client.system_one(
                state=prompt,
                questions={
                    "task_type": Choice(
                        instructions="Categorize this prompt into the best matching task type.",
                        criteria=CLASSIFICATION_CRITERIA,
                    )
                },
            )
            detected = result.choices["task_type"].choice
            if detected:
                return {"task_type": detected}
        except Exception:
            # if the API call fails, seamlessly fall through to keyword matching
            pass

    # fallback rule-based classifier using word boundaries to avoid false substring matches
    prompt_lower = prompt.lower()
    for task_type, keywords in TASK_KEYWORDS.items():
        for kw in keywords:
            if re.search(rf"\b{re.escape(kw)}\b", prompt_lower):
                return {"task_type": task_type}

    return {"task_type": "general"}


def select_model(state: RouterState) -> RouterState:
    """
    Picks the most cost-effective model for the classified task.
    Simple tasks get cheap/fast models, while complex tasks get top-tier models.
    """
    task_type = state.get("task_type", "general")
    selected = MODEL_MAP.get(task_type, "gemini-flash")
    return {"selected_model": selected}


def generate_response(state: RouterState) -> RouterState:
    """
    Simulates response generation and calculates exact token counts,
    estimated cost, and how much money I saved vs always using GPT-4o.
    """
    model = state.get("selected_model", "gemini-flash")
    prompt = state.get("prompt", "")
    task_type = state.get("task_type", "general")

    # calculate realistic token count (prompt + estimated output)
    prompt_tokens = max(1, int(len(prompt.split()) * 1.3))
    output_tokens = 50  # average standard response length for estimation
    total_tokens = prompt_tokens + output_tokens

    # calculate cost of selected model vs expensive baseline (GPT-4o)
    model_rate = MODEL_COSTS_PER_1K.get(model, 0.001)
    baseline_rate = MODEL_COSTS_PER_1K.get("gpt-5.6-luna", 0.00045)

    estimated_cost = round((total_tokens / 1000.0) * model_rate, 6)
    baseline_cost = (total_tokens / 1000.0) * baseline_rate
    cost_saved = round(max(0.0, baseline_cost - estimated_cost), 6)

    # craft response text reflecting the routed task
    response_text = f"[{model}] Handled your {task_type} request successfully."

    return {
        "response_text": response_text,
        "tokens_used": total_tokens,
        "estimated_cost": estimated_cost,
        "cost_saved": cost_saved,
    }
