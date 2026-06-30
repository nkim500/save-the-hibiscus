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


def demo_afternoon() -> StubEventSource:
    """One squirrel's escalating campaign against the hibiscus.

    Note what is NOT here: the wind-blown leaf and the low-confidence blur. The
    perception layer already discarded those, and ByteTrack collapsed each
    multi-second visit into a single track. The agent sees three real incursions
    (with one departure in between) — and should escalate low -> medium -> high.
    """
    return StubEventSource(
        [
            (
                0.5,
                TrackEvent(
                    "entered_zone",
                    track_id=7,
                    label="squirrel",
                    camera="webcam-1",
                    zone="hibiscus",
                    confidence=0.93,
                ),
            ),
            (
                1.0,
                TrackEvent(
                    "left_zone",
                    track_id=7,
                    label="squirrel",
                    camera="webcam-1",
                    zone="hibiscus",
                    confidence=0.93,
                ),
            ),
            (
                3.0,
                TrackEvent(
                    "entered_zone",
                    track_id=12,
                    label="squirrel",
                    camera="webcam-1",
                    zone="hibiscus",
                    confidence=0.88,
                ),
            ),
            (
                3.0,
                TrackEvent(
                    "entered_zone",
                    track_id=15,
                    label="squirrel",
                    camera="webcam-1",
                    zone="hibiscus",
                    confidence=0.97,
                ),
            ),
        ]
    )
