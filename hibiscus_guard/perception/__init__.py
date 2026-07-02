from .base import CameraSource, EventSource
from .detector_source import DetectorEventSource, FolderSource, WebcamSource
from .events import TrackEvent
from .stub import StubEventSource, demo_afternoon

__all__ = [
    "CameraSource",
    "EventSource",
    "TrackEvent",
    "StubEventSource",
    "demo_afternoon",
    "DetectorEventSource",
    "FolderSource",
    "WebcamSource",
]
