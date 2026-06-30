"""The ambient agent — a long-running process that watches a live event stream
and reacts in real time.

This is the heart of the "ambient agent" pattern:
  * It STAYS ALIVE, subscribed to a perception EventSource.
  * Each event wakes the agent, which decides and fires reactions (alert /
    local alarm) and records the sighting to the durable SQLite store.
  * It holds ONE long-lived session for the process; the durable state lives in
    the store (hibiscus_guard/store.py), not in memory — so a restart, or the
    separate daily-digest process, loses nothing.

With the stub source the stream is finite and we exit at the end. With a real
YoloEventSource over a webcam, `source.events()` simply never stops — the same
loop becomes a true always-on guard. That ONE line is the only thing that
changes to go live.

Run it:  uv run python -m hibiscus_guard.ambient
"""

import asyncio
import os

from dotenv import load_dotenv
from google.adk.runners import InMemoryRunner
from google.genai import types

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

from hibiscus_guard.agent import root_agent  # noqa: E402
from hibiscus_guard.perception import demo_afternoon  # noqa: E402

APP = "hibiscus_guard"
USER = "nick"


async def handle_event(runner, session_id, event):
    """Wake the agent on one TrackEvent; surface tool calls + final reply."""
    msg = types.Content(role="user", parts=[types.Part(text=event.to_prompt())])
    final = ""
    async for ev in runner.run_async(user_id=USER, session_id=session_id, new_message=msg):
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

    # Swap this single line for a real YoloEventSource(...) to go live; the loop
    # below — the ambient pattern — does not change.
    source = demo_afternoon()

    print("🌺 Hibiscus Guard is watching... (Ctrl-C to stop)\n")
    async for event in source.events():
        print(f"=== {event.event_type} tier={event.tier} (track {event.track_id}) ===")
        print(f"    event: {event.to_prompt()}")
        reply = await handle_event(runner, session.id, event)
        print(f"    agent: {reply.strip()}\n")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🌺 Hibiscus Guard stopped.")
