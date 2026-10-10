from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

class TaskRequest(BaseModel):
    prompt: str = Field(..., description="The user's input prompt or task.")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Optional context or previous conversation.")
    
class RoutingResponse(BaseModel):
    original_prompt: str
    routed_model: str = Field(..., description="The name of the LLM chosen by my engine.")
    response_text: str = Field(..., description="The response from the selected LLM or safety guard.")
    tokens_used: Optional[int] = None
    estimated_cost: Optional[float] = None
    cost_saved: Optional[float] = Field(default=None, description="Money saved compared to always calling GPT-4o.")
    latency_ms: Optional[float] = Field(default=None, description="Routing and response latency in milliseconds.")
    safety_score: Optional[float] = Field(default=None, description="Prompt injection risk score on 1-5 scale (<=2 is safe, >2 blocked).")
