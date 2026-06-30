"""Hibiscus Guard — the agent that decides what to do about a confirmed squirrel.

Division of labor (see perception/events.py for the full reasoning):

  PERCEPTION  (YOLO + ByteTrack + zone geometry, or the stub) already did the
  high-frequency, deterministic work: confidence thresholding, frame dedupe via
  track identity, and confirming the animal is in the hibiscus zone. It hands us
  sparse, cooked TrackEvents.

  THIS AGENT does the low-frequency, judgmental work that benefits from
  reasoning and context: how serious is THIS incursion given recent history?
  what urgency? what should the alert say? (Channel selection / governed egress
  arrives with the MCP phase.)

So there is no confidence check or debounce here anymore — an "entered_zone"
event is already trustworthy. State tracks INCIDENTS (real visits), not frames.
"""

import os
import sys
import time

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools import ToolContext
from google.adk.tools.mcp_tool import MCPToolset, StdioConnectionParams
from mcp import StdioServerParameters

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# --- Tools -------------------------------------------------------------------


def get_incident_history(camera: str, tool_context: ToolContext) -> dict:
    """Look up recent confirmed squirrel incursions for a camera.

    Call this before deciding urgency, so escalation reflects how persistent the
    squirrel has been in the last hour.

    Args:
        camera: The camera id the event came from, e.g. "webcam-1".

    Returns:
        incidents_last_hour: int, confirmed incursions in the past 60 minutes
            (NOT counting the current event yet).
        seconds_since_last: float or None, time since the previous incursion.
    """
    now = time.time()
    timestamps = tool_context.state.get(f"incidents:{camera}", [])
    recent = [t for t in timestamps if now - t < 3600]
    last = max(recent) if recent else None
    return {
        "incidents_last_hour": len(recent),
        "seconds_since_last": (now - last) if last is not None else None,
    }


def log_incident(camera: str, urgency: str, tool_context: ToolContext) -> dict:
    """Record that a confirmed incursion happened (local bookkeeping only).

    This does NOT notify anyone — it just updates the agent's memory so future
    escalation decisions are correct. Sending the actual notification is a
    separate, governed action: the `send_push` MCP tool.

    Args:
        camera: The camera id the event came from.
        urgency: One of "low", "medium", "high" — the level you alerted at.

    Returns:
        status and the running incident count for this camera.
    """
    timestamps = tool_context.state.get(f"incidents:{camera}", [])
    timestamps.append(time.time())
    tool_context.state[f"incidents:{camera}"] = timestamps
    return {"status": "logged", "urgency": urgency, "incidents_total": len(timestamps)}


# The EGRESS lives in a separate MCP server process. The agent launches it over
# stdio. This is the boundary the Agent Gateway will later govern: the agent may
# reach THIS server (to push notifications) and nothing else. We launch it with
# the same interpreter running the agent, so it shares this venv's deps.
alert_egress = MCPToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command=sys.executable,
            args=["-m", "hibiscus_guard.alert_mcp_server"],
            cwd=_PROJECT_ROOT,
        ),
    ),
)


# --- Agent -------------------------------------------------------------------

root_agent = Agent(
    name="hibiscus_guard",
    model=LiteLlm(model="anthropic/claude-sonnet-4-6"),
    description="Decides urgency and alerts when a confirmed squirrel enters the hibiscus zone.",
    instruction=(
        "You protect a hibiscus from squirrels. The perception layer has ALREADY "
        "filtered noise, de-duplicated frames, and confirmed zone entry, so every "
        "event you receive is trustworthy. Do not second-guess confidence.\n\n"
        "Events look like: 'event=<entered_zone|left_zone> label=squirrel "
        "camera=<id> zone=hibiscus track_id=<n> confidence=<0..1>'.\n\n"
        "Rules:\n"
        "1. If event is 'left_zone', reply 'noted: squirrel left' and call no tool.\n"
        "2. If event is 'entered_zone', call get_incident_history for that camera.\n"
        "3. Decide urgency by incidents_last_hour:\n"
        "     0  -> 'low'\n"
        "     1  -> 'medium'\n"
        "     2+ -> 'high'\n"
        "4. Call send_alert to notify the household. ALWAYS pass recipient="
        "'household' (the only allowlisted recipient). Use a short title like "
        "'Squirrel on the hibiscus' and a message noting how many times it's "
        "struck this hour. Pass the urgency as the priority. If send_alert "
        "returns status 'blocked', report the reason and do not retry.\n"
        "5. Then call log_incident with the same camera and urgency to record it.\n"
        "Be terse. You are a sensor's brain, not a chatbot."
    ),
    tools=[get_incident_history, log_incident, alert_egress],
)
