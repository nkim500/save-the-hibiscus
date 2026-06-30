"""Wire a perception EventSource into the agent and watch it decide.

This is the integration seam. Today the source is the stub; to go live you
change ONE line — `source = demo_afternoon()` becomes `source = YoloEventSource(...)`
— and nothing else here moves.

Run it:  uv run python -m hibiscus_guard.simulate
"""

import asyncio
import os

from dotenv import load_dotenv
from google.adk.runners import InMemoryRunner
from google.genai import types

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

from hibiscus_guard.agent import root_agent          # noqa: E402
from hibiscus_guard.perception import demo_afternoon  # noqa: E402

APP = "hibiscus_guard"
USER = "nick"


async def handle_event(runner, session_id, event):
    """Send one TrackEvent to the agent; print tool calls + final reply."""
    msg = types.Content(role="user", parts=[types.Part(text=event.to_prompt())])
    final = ""
    async for ev in runner.run_async(
        user_id=USER, session_id=session_id, new_message=msg
    ):
        if ev.content and ev.content.parts:
            for part in ev.content.parts:
                if part.function_call:
                    print(f"    ↳ tool: {part.function_call.name}({dict(part.function_call.args)})")
        if ev.is_final_response() and ev.content and ev.content.parts:
            final = "".join(p.text or "" for p in ev.content.parts)
    return final


async def main():
    runner = InMemoryRunner(agent=root_agent, app_name=APP)
    session = await runner.session_service.create_session(app_name=APP, user_id=USER)

    source = demo_afternoon()          # <-- the only line that changes for real YOLO
    async for event in source.events():
        print(f"\n=== {event.event_type} (track {event.track_id}) ===")
        print(f"    event: {event.to_prompt()}")
        reply = await handle_event(runner, session.id, event)
        print(f"    agent: {reply.strip()}")


if __name__ == "__main__":
    asyncio.run(main())
