from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

class TaskRequest(BaseModel):
    prompt: str = Field(..., description="The user's input prompt or task.")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Optional context or previous conversation.")
    
class RoutingResponse(BaseModel):
    original_prompt: str
    routed_model: str = Field(..., description="The name of the LLM chosen by the engine.")
    response_text: str = Field(..., description="The actual response from the chosen LLM.")
    tokens_used: Optional[int] = None
    estimated_cost: Optional[float] = None
