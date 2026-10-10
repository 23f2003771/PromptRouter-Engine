from typing import TypedDict


class RouterState(TypedDict, total=False):
    """
    State flowing through my LangGraph pipeline.
    TypedDict keeps it lightweight and fast inside the graph nodes.
    """
    prompt: str
    safety_score: float     # prompt injection score on 1-5 scale (<= 2 is safe, > 2 is blocked)
    is_blocked: bool        # whether prompt injection was detected
    task_type: str          # classified category: "code", "creative", "factual", etc.
    selected_model: str     # which model got picked by the router
    response_text: str      # LLM response
    tokens_used: int
    estimated_cost: float
    cost_saved: float       # savings compared to default expensive model
    latency_ms: float       # execution time in ms
