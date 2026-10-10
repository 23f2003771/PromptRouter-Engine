"""
Simple script to test my PromptRouter SDK.
Make sure the server is running first:
    python -m uvicorn main:app --port 8000
"""

import sys
from pathlib import Path

# lets Python find my sdk folder
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sdk import PromptRouterClient


def main():
    # 1. Connect to my local engine
    client = PromptRouterClient(base_url="http://127.0.0.1:8000")

    # 2. Check if engine is running
    print("Checking server health...")
    status = client.health()
    print("Status:", status)

    # 3. Test prompts for different tasks
    test_prompts = [
        "Write a python function to sort a list.",
        "What is the capital of France?",
        "Write a poem about the ocean.",
    ]

    print("\n--- Routing Results ---")
    for prompt in test_prompts:
        result = client.route(prompt)
        print(f"\nTask:       {prompt}")
        print(f"Model:      {result.routed_model}")
        print(f"Latency:    {result.latency_ms} ms")
        print(f"Cost:       ${result.estimated_cost}")
        print(f"Cost Saved: ${result.cost_saved}")


if __name__ == "__main__":
    main()
