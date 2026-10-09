from typing import TypedDict, Optional


class RouterState(TypedDict, total=False):
    """
    This is the state that flows through my LangGraph.
    Using TypedDict instead of Pydantic here because LangGraph
    works with dicts internally, so TypedDict gives me type safety
    without the overhead of Pydantic validation at every node.
    """
    prompt: str
    task_type: str          # classified category: "code", "creative", "factual", etc.
    selected_model: str     # which model got picked by the router
    response_text: str      # final LLM response
    tokens_used: int
    estimated_cost: float
