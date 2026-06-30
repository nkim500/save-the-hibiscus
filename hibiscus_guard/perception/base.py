"""Abstract seams for the perception layer.

Two swappable interfaces, two different axes of change:

  CameraSource  -- WHERE frames come from (laptop webcam, dashcam, RTSP, a video
                   file). Swap this to change hardware.
  EventSource   -- WHAT the agent subscribes to: a stream of cooked TrackEvents.
                   Swap this to change the whole detection strategy (stub today,
                   YOLO+ByteTrack tomorrow, a hosted Roboflow workflow later).

The agent only ever depends on EventSource. The real detector (next phase) will
*hold* a CameraSource internally, run YOLO+ByteTrack+zone logic over its frames,
and expose the result as an EventSource. The stub skips cameras entirely.
"""

from abc import ABC, abstractmethod
from typing import AsyncIterator, Iterator

from .events import TrackEvent


class CameraSource(ABC):
    """A source of video frames. Implement once per hardware type."""

    @abstractmethod
    def frames(self) -> Iterator:
        """Yield frames (e.g. numpy BGR arrays) until the source is exhausted."""

    @abstractmethod
    def release(self) -> None:
        """Free the underlying device/handle."""


class EventSource(ABC):
    """A source of cooked TrackEvents — the agent's actual input."""

    @abstractmethod
    def events(self) -> AsyncIterator[TrackEvent]:
        """Async-iterate TrackEvents as they occur."""
