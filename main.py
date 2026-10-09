import time
from fastapi import FastAPI
from models import TaskRequest, RoutingResponse
from router import router_graph

app = FastAPI(
    title="PromptRouter Engine",
    description="An intelligent orchestration engine for routing LLM tasks to optimize cost and latency.",
    version="0.1.0"
)

@app.get("/health")
async def health_check():
    return {"status": "healthy", "engine": "PromptRouter"}

@app.post("/api/route", response_model=RoutingResponse)
async def route_task(request: TaskRequest):
    """
    Runs the incoming prompt through my LangGraph pipeline:
    classify (TypeSafe SDK / heuristics) → select model → generate & calculate cost/savings
    """
    start_time = time.perf_counter()
    result = router_graph.invoke({"prompt": request.prompt})
    duration_ms = round((time.perf_counter() - start_time) * 1000, 2)

    return RoutingResponse(
        original_prompt=request.prompt,
        routed_model=result.get("selected_model", "gemini-flash"),
        response_text=result.get("response_text", ""),
        tokens_used=result.get("tokens_used", 0),
        estimated_cost=result.get("estimated_cost", 0.0),
        cost_saved=result.get("cost_saved", 0.0),
        latency_ms=duration_ms,
    )
