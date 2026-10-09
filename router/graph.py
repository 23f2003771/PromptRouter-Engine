from langgraph.graph import StateGraph, START, END
from router.state import RouterState
from router.nodes import classify_task, select_model, generate_response


def build_router_graph():
    """
    Assembles my routing pipeline as a LangGraph:
    classify → select_model → generate_response → END
    """
    graph = StateGraph(RouterState)

    # register each step as a node
    graph.add_node("classify", classify_task)
    graph.add_node("select_model", select_model)
    graph.add_node("generate", generate_response)

    # define the flow: one node into the next
    graph.add_edge(START, "classify")
    graph.add_edge("classify", "select_model")
    graph.add_edge("select_model", "generate")
    graph.add_edge("generate", END)

    return graph.compile()
