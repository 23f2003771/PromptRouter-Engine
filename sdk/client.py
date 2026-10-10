import httpx
from models import RoutingResponse


class PromptRouterClient:
    """
    My simple Python client to talk to my PromptRouter API.
    Instead of writing raw HTTP requests manually, I can just call client.route(prompt).
    """

    def __init__(self, base_url: str = "http://127.0.0.1:8000"):
        self.base_url = base_url.rstrip("/")

    def health(self) -> dict:
        """Check if my server is running."""
        response = httpx.get(f"{self.base_url}/health")
        return response.json()

    def route(self, prompt: str) -> RoutingResponse:
        """Send a prompt to my router and get back the model choice, response, and metrics."""
        response = httpx.post(
            f"{self.base_url}/api/route",
            json={"prompt": prompt}
        )
        return RoutingResponse(**response.json())
