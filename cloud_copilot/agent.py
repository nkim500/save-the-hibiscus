"""Minimal cloud-deployable agent — proves the Agent Engine path.

Deliberately tiny: Claude (via LiteLLM) + one trivial tool. The point of this
first deploy is to exercise Agent Runtime -> Agent Registry -> Agent Gateway,
not to ship product logic. We grow it into the real copilot once the platform
plumbing is proven.

Claude runs via LiteLLM, so the deployed engine needs ANTHROPIC_API_KEY in its
environment (set in this folder's .env) and outbound egress to api.anthropic.com
— which is exactly the egress the Agent Gateway will govern.
"""

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm


def ping() -> dict:
    """Health check. Returns a simple status payload."""
    return {"status": "ok", "service": "hibiscus-copilot"}


root_agent = Agent(
    name="hibiscus_copilot",
    model=LiteLlm(model="anthropic/claude-sonnet-4-6"),
    description="Minimal cloud-deployed Hibiscus copilot (Claude via LiteLLM).",
    instruction="You are the Hibiscus Guard cloud copilot. Be brief and helpful.",
    tools=[ping],
)
