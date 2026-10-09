from router.state import RouterState


# keywords I use to figure out what kind of task this is
TASK_KEYWORDS = {
    "code": ["code", "function", "debug", "script", "program", "algorithm", "api", "sql", "python", "javascript"],
    "creative": ["write", "story", "poem", "essay", "blog", "creative", "compose", "draft"],
    "factual": ["what is", "explain", "define", "who is", "when did", "how does", "tell me about"],
}

# mapping each task type to the model I want to use
MODEL_MAP = {
    "code": "gpt-4o",
    "creative": "claude-sonnet",
    "factual": "gemini-pro",
    "general": "gemini-flash",   # default fallback, cheapest option
}


def classify_task(state: RouterState) -> RouterState:
    """
    Looks at the prompt and figures out what kind of task it is
    based on keyword matching. I'll swap this out for a smarter
    classifier later.
    """
    prompt_lower = state["prompt"].lower()

    for task_type, keywords in TASK_KEYWORDS.items():
        if any(kw in prompt_lower for kw in keywords):
            return {"task_type": task_type}

    # nothing matched, just call it general
    return {"task_type": "general"}


def select_model(state: RouterState) -> RouterState:
    """
    Picks the right model based on the classified task type.
    This is where the cost savings happen — no point sending a
    simple factual question to an expensive model.
    """
    task_type = state.get("task_type", "general")
    selected = MODEL_MAP.get(task_type, "gemini-flash")
    return {"selected_model": selected}


def generate_response(state: RouterState) -> RouterState:
    """
    Placeholder for actual LLM calls. Right now it just returns
    a mock response so I can test the full graph flow end to end.
    I'll plug in LangChain + TypeSafe SDK here in the next step.
    """
    model = state.get("selected_model")
    prompt = state.get("prompt", "")
    task_type = state.get("task_type", "general")

    return {
        "response_text": f"[{model}] would respond to this {task_type} task: '{prompt[:80]}...'",
        "tokens_used": len(prompt.split()) * 2,     # rough estimate for now
        "estimated_cost": 0.0,
    }
