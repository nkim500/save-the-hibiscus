"""The real perception pipeline: camera frames -> trained detector -> TrackEvents.

This replaces the stub with the detector the copilot trains (detector/).
The division of labor stays exactly as events.py promises: everything here is
deterministic — score thresholds, debounce, track identity — so the agent
downstream still receives only cooked, trustworthy events.

Two CameraSource implementations:

  WebcamSource  -- a real camera via OpenCV (the live deployment).
  FolderSource  -- images appearing in a directory. Zero hardware: great for
                   tests, and honest too — a security cam that drops JPEGs into
                   a folder is exactly this.

DetectorEventSource holds a CameraSource, scores each frame with the trained
classifier (detector.classify), and runs a small state machine:

  score >= confident_threshold  -> a "confident" hit
  score >= candidate_threshold  -> a "candidate" hit (a maybe)
  below                         -> a miss

  enter: `enter_frames` consecutive hits  -> emit entered_zone (new track id)
  exit:  `exit_frames` consecutive misses -> emit left_zone (same track id)

The tier and confidence reported on entry are the best seen during the entry
streak. No LLM anywhere in this file — that's the point.
"""

import asyncio
import glob
import os
import time
from typing import AsyncIterator, Iterator

from .base import CameraSource, EventSource
from .events import TrackEvent

_IMG_EXT = (".jpg", ".jpeg", ".png", ".webp")


class WebcamSource(CameraSource):
    """Frames from a local camera via OpenCV. Yields PIL Images."""

    def __init__(self, index: int = 0):
        self._index = index
        self._cap = None

    def frames(self) -> Iterator:
        import cv2  # heavy import deferred; only the live path needs it
        from PIL import Image

        self._cap = cv2.VideoCapture(self._index)
        if not self._cap.isOpened():
            raise RuntimeError(f"webcam {self._index} could not be opened")
        while True:
            ok, frame = self._cap.read()
            if not ok:
                return
            yield Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class FolderSource(CameraSource):
    """Frames from image files appearing in a directory, in sorted order.

    watch=True keeps polling for new files forever (a drop-folder camera);
    watch=False yields what's there and stops (deterministic, for tests).
    """

    def __init__(self, directory: str, watch: bool = True, poll_seconds: float = 1.0):
        self._dir = directory
        self._watch = watch
        self._poll = poll_seconds

    def _listing(self) -> list[str]:
        return sorted(
            p for p in glob.glob(os.path.join(self._dir, "*")) if p.lower().endswith(_IMG_EXT)
        )

    def frames(self) -> Iterator:
        from PIL import Image

        seen: set[str] = set()
        while True:
            new = [p for p in self._listing() if p not in seen]
            for path in new:
                seen.add(path)
                try:
                    yield Image.open(path)
                except OSError:
                    continue  # partially-written or corrupt file; skip
            if not self._watch:
                return
            time.sleep(self._poll)

    def release(self) -> None:
        pass


class DetectorEventSource(EventSource):
    """Score frames with a trained detector and emit cooked TrackEvents."""

    def __init__(
        self,
        camera: CameraSource,
        model_path: str,
        label: str,
        camera_name: str = "webcam-1",
        zone: str = "hibiscus",
        confident_threshold: float = 0.8,
        candidate_threshold: float = 0.6,
        enter_frames: int = 2,
        exit_frames: int = 3,
        interval_seconds: float = 1.0,
    ):
        if not 0.0 < candidate_threshold <= confident_threshold <= 1.0:
            raise ValueError("need 0 < candidate_threshold <= confident_threshold <= 1")
        self._camera = camera
        self._model_path = model_path
        self._label = label
        self._camera_name = camera_name
        self._zone = zone
        self._confident = confident_threshold
        self._candidate = candidate_threshold
        self._enter_frames = max(1, enter_frames)
        self._exit_frames = max(1, exit_frames)
        self._interval = interval_seconds

    def _event(self, event_type: str, track_id: int, confidence: float, tier: str) -> TrackEvent:
        return TrackEvent(
            event_type=event_type,
            track_id=track_id,
            label=self._label,
            camera=self._camera_name,
            zone=self._zone,
            confidence=confidence,
            tier=tier,
        )

    async def events(self) -> AsyncIterator[TrackEvent]:
        from detector import classify  # heavy import deferred to runtime

        bundle = classify.load(self._model_path)
        frames = self._camera.frames()

        present = False
        track_id = 0
        hit_streak = 0
        miss_streak = 0
        best_score = 0.0

        try:
            while True:
                # Frame grab + DINOv2 embed are blocking CPU work; keep them off
                # the event loop so the agent can run concurrently.
                frame = await asyncio.to_thread(next, frames, None)
                if frame is None:
                    break
                result = await asyncio.to_thread(classify.predict, frame, bundle)
                score = result["score"]

                if score >= self._candidate:
                    hit_streak += 1
                    miss_streak = 0
                    best_score = max(best_score, score)
                    if not present and hit_streak >= self._enter_frames:
                        present = True
                        track_id += 1
                        tier = "confident" if best_score >= self._confident else "candidate"
                        yield self._event("entered_zone", track_id, best_score, tier)
                else:
                    miss_streak += 1
                    hit_streak = 0
                    if present and miss_streak >= self._exit_frames:
                        present = False
                        yield self._event("left_zone", track_id, best_score, "confident")
                        best_score = 0.0

                if self._interval > 0:
                    await asyncio.sleep(self._interval)

            if present:  # camera ended mid-visit; close the track
                yield self._event("left_zone", track_id, best_score, "confident")
        finally:
            self._camera.release()
