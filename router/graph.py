from langgraph.graph import StateGraph, START, END
from router.state import RouterState
from router.nodes import (
    check_safety,
    block_request,
    classify_task,
    select_model,
    generate_response,
)


def route_safety(state: RouterState) -> str:
    """
    Conditional edge:
    If prompt injection is detected (score > 2.0), route to block_request.
    Otherwise, continue to the classify node.
    """
    if state.get("is_blocked", False):
        return "block_request"
    return "classify"


def build_router_graph():
    """
    Assembles my routing pipeline with security guardrails:
    START -> check_safety
       |-> (score > 2) -> block_request -> END
       |-> (score <= 2) -> classify -> select_model -> generate -> END
    """
    graph = StateGraph(RouterState)

    # register each step as a node
    graph.add_node("check_safety", check_safety)
    graph.add_node("block_request", block_request)
    graph.add_node("classify", classify_task)
    graph.add_node("select_model", select_model)
    graph.add_node("generate", generate_response)

    # entry point goes to security guard first
    graph.add_edge(START, "check_safety")

    # conditional branch based on injection score
    graph.add_conditional_edges(
        "check_safety",
        route_safety,
        {
            "classify": "classify",
            "block_request": "block_request",
        },
    )

    # standard routing pipeline
    graph.add_edge("classify", "select_model")
    graph.add_edge("select_model", "generate")
    graph.add_edge("generate", END)

    # blocked branch terminates immediately
    graph.add_edge("block_request", END)

    return graph.compile()
