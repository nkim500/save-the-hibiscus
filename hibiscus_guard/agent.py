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

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.mcp_tool import MCPToolset, StdioConnectionParams
from mcp import StdioServerParameters

from hibiscus_guard import actuator, store

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# --- Tools -------------------------------------------------------------------
# State now lives in the SQLite incident store (hibiscus_guard/store.py), not in
# session memory — so it survives restarts and the daily digest agent can read
# it. The tools are the agent's read/write window onto that store.


def get_incident_history(camera: str) -> dict:
    """Look up recent confirmed squirrel alerts for a camera (read-only).

    Call this before deciding urgency, so escalation reflects how persistent the
    squirrel has been in the last hour.

    Args:
        camera: The camera id the event came from, e.g. "webcam-1".

    Returns:
        alerts_last_hour: int, confirmed alerts in the past 60 minutes (NOT
            counting the current event yet).
        seconds_since_last_alert: float or None, time since the previous alert.
    """
    return store.history(camera)


def record_sighting(
    camera: str, track_id: int, tier: str, confidence: float, alerted: bool, urgency: str = ""
) -> dict:
    """Record this sighting in the durable incident store.

    Call this for EVERY event you decide on — both confident incursions you
    alerted about AND candidates ("maybes") you passed on. This is what lets the
    daily digest say "I saw N maybes and passed on them".

    Args:
        camera: The camera id.
        track_id: The track id from the event.
        tier: "confident" or "candidate".
        confidence: The event's confidence (0..1).
        alerted: True if you sent an alert for this, False if you passed.
        urgency: "low"/"medium"/"high" if you alerted, else "".
    """
    store.record(camera, track_id, tier, confidence, alerted, urgency or None)
    return {"status": "recorded", "tier": tier, "alerted": alerted}


def sound_alarm(urgency: str) -> dict:
    """Trigger the local deterrent (make a sound on the laptop) to scare the
    squirrel in real time. Use for confident incursions, louder for higher
    urgency.

    Args:
        urgency: "low", "medium", or "high".
    """
    return actuator.buzz(urgency)


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
        "You protect a hibiscus from squirrels. Perception has ALREADY filtered "
        "noise, de-duplicated frames, and confirmed zone entry, so every event is "
        "trustworthy. Do not second-guess confidence.\n\n"
        "Events look like: 'event=<entered_zone|left_zone> tier=<confident|candidate> "
        "label=squirrel camera=<id> zone=hibiscus track_id=<n> confidence=<0..1>'.\n\n"
        "Handle each event:\n"
        "A) event 'left_zone' -> reply 'noted: squirrel left'. No tools.\n\n"
        "B) event 'entered_zone' with tier 'candidate' (a MAYBE) -> do NOT alert "
        "or sound the alarm. Just call record_sighting with alerted=false, "
        "urgency='' (so the daily digest knows you saw it and passed). Reply "
        "'logged maybe, passed'.\n\n"
        "C) event 'entered_zone' with tier 'confident' -> act:\n"
        "   1. call get_incident_history for the camera.\n"
        "   2. pick urgency by alerts_last_hour: 0->'low', 1->'medium', 2+->'high'.\n"
        "   3. call send_alert (recipient ALWAYS 'household', the only allowlisted "
        "one; title like 'Squirrel on the hibiscus'; message notes how many times "
        "it struck this hour; priority = urgency). If it returns 'blocked', report "
        "the reason and stop.\n"
        "   4. if urgency is 'high', call sound_alarm to scare it off.\n"
        "   5. call record_sighting with alerted=true and the chosen urgency.\n\n"
        "Be terse. You are a sensor's brain, not a chatbot."
    ),
    tools=[get_incident_history, record_sighting, sound_alarm, alert_egress],
)
