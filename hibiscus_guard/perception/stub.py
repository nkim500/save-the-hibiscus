"""A fake EventSource that replays a scripted afternoon in the garden.

It emits the SAME TrackEvents a real YOLO+ByteTrack pipeline would emit AFTER
filtering — so the leaf and the blurry 0.41-confidence blip simply never appear
here; perception would have dropped them upstream. What reaches the agent is
only confirmed squirrels entering/leaving the hibiscus zone.

Swapping this for the real thing later means writing a YoloEventSource(EventSource)
and changing one line in simulate.py.
"""

import asyncio
from typing import AsyncIterator

from .base import EventSource
from .events import TrackEvent


class StubEventSource(EventSource):
    def __init__(self, script: list[tuple[float, TrackEvent]]):
        # script: list of (seconds_to_wait_before_emitting, event)
        self._script = script

    async def events(self) -> AsyncIterator[TrackEvent]:
        for delay, event in self._script:
            await asyncio.sleep(delay)
            yield event


def _ev(event_type: str, track_id: int, confidence: float, tier: str = "confident") -> TrackEvent:
    return TrackEvent(
        event_type=event_type,
        track_id=track_id,
        label="squirrel",
        camera="webcam-1",
        zone="hibiscus",
        confidence=confidence,
        tier=tier,
    )


def demo_afternoon() -> StubEventSource:
    """One squirrel's escalating campaign against the hibiscus, plus a "maybe".

    Note what is NOT here: the wind-blown leaf and the low-confidence blur — those
    fell below even the 'candidate' bar and perception dropped them. What reaches
    the agent:
      * a 'candidate' (a maybe) -> the agent should LOG and pass, no alert.
      * three 'confident' incursions -> escalate low -> medium -> high, and the
        high one also fires the local alarm.
    """
    return StubEventSource(
        [
            (0.5, _ev("entered_zone", track_id=5, confidence=0.55, tier="candidate")),
            (2.0, _ev("entered_zone", track_id=7, confidence=0.93)),
            (1.0, _ev("left_zone", track_id=7, confidence=0.93)),
            (3.0, _ev("entered_zone", track_id=12, confidence=0.88)),
            (3.0, _ev("entered_zone", track_id=15, confidence=0.97)),
        ]
    )
