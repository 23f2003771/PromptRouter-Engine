from fastapi import FastAPI
from models import TaskRequest, RoutingResponse

app = FastAPI(
    title="PromptRouter Engine",
    description="An intelligent orchestration engine for routing LLM tasks to optimize cost and latency.",
    version="0.1.0"
)

@app.get("/health")
async def health_check():
    """Simple health check to ensure the engine is running."""
    return {"status": "healthy", "engine": "PromptRouter"}

@app.post("/api/route", response_model=RoutingResponse)
async def route_task(request: TaskRequest):
    """
    Main endpoint to route tasks. 
    I'll integrate my LangGraph workflow here soon to handle the actual LLM routing logic.
    """
    # Just returning a static response for now to test the API structure
    return RoutingResponse(
        original_prompt=request.prompt,
        routed_model="test-model",
        response_text="Routing logic is under construction!",
        tokens_used=0,
        estimated_cost=0.0
    )
