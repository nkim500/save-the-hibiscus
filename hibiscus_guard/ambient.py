"""The ambient agent — a long-running process that watches a live event stream
and reacts in real time.

This is the heart of the "ambient agent" pattern:
  * It STAYS ALIVE, subscribed to a perception EventSource.
  * Each event wakes the agent, which decides and fires reactions (alert /
    local alarm) and records the sighting to the durable SQLite store.
  * It holds ONE long-lived session for the process; the durable state lives in
    the store (hibiscus_guard/store.py), not in memory — so a restart, or the
    separate daily-digest process, loses nothing.

With the stub source the stream is finite and we exit at the end. With the
trained DetectorEventSource over a webcam, `source.events()` simply never
stops — the same loop becomes a true always-on guard.

Which source runs is chosen by environment (so the copilot — or you — can
deploy this process without editing code):

  HIBISCUS_SOURCE=stub       (default) the scripted demo afternoon
  HIBISCUS_SOURCE=detector   the trained detector over a live camera; needs:
      DETECTOR_MODEL   path to the trained .joblib bundle
      TARGET_LABEL     what the detector was trained on, e.g. "squirrel"
      CAMERA           camera spec: webcam index ("0" / "webcam:0"), a
                       directory to watch ("folder:/path" or a bare path),
                       an "rtsp://…" stream, or a "file:clip.mp4" replay
      ZONE             zone name for events (default "hibiscus")
      POLL_SECONDS     seconds between frames (default "1.0")
      CONFIDENT_T / CANDIDATE_T   tier thresholds (default 0.8 / 0.6)

Run it:  uv run --group detector python -m hibiscus_guard.ambient
"""

import asyncio
import os

from dotenv import load_dotenv
from google.adk.runners import InMemoryRunner
from google.genai import types

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

from hibiscus_guard.agent import root_agent  # noqa: E402
from hibiscus_guard.perception import (  # noqa: E402
    DetectorEventSource,
    camera_from_spec,
    demo_afternoon,
)

APP = "hibiscus_guard"
USER = "nick"


def source_from_env():
    """Build the EventSource the environment asks for (default: the stub)."""
    kind = os.environ.get("HIBISCUS_SOURCE", "stub")
    if kind == "stub":
        return demo_afternoon()
    if kind != "detector":
        raise SystemExit(f"unknown HIBISCUS_SOURCE={kind!r} (want 'stub' or 'detector')")

    model = os.environ.get("DETECTOR_MODEL", "")
    if not os.path.isfile(model):
        raise SystemExit(f"DETECTOR_MODEL={model!r} is not a file")
    try:
        camera, camera_name = camera_from_spec(os.environ.get("CAMERA", "0"))
    except ValueError as e:
        raise SystemExit(str(e)) from e
    return DetectorEventSource(
        camera=camera,
        model_path=model,
        label=os.environ.get("TARGET_LABEL", "target"),
        camera_name=camera_name,
        zone=os.environ.get("ZONE", "hibiscus"),
        confident_threshold=float(os.environ.get("CONFIDENT_T", "0.8")),
        candidate_threshold=float(os.environ.get("CANDIDATE_T", "0.6")),
        interval_seconds=float(os.environ.get("POLL_SECONDS", "1.0")),
    )


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

    # Stub or trained detector — chosen by env (see module docstring); the loop
    # below — the ambient pattern — does not change.
    source = source_from_env()

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
