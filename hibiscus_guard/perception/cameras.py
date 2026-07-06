"""Config-addressable camera sources.

One string names any camera, so the copilot (or an env var) can pick hardware
without code changes:

  "webcam:0"  / "0"        local camera via OpenCV
  "folder:/path" / "/path" images appearing in a directory (drop-folder cam)
  "rtsp://user:pw@host/…"  a network camera stream
  "file:clip.mp4"          replay a recorded clip (deterministic, for tests)

`camera_from_spec` builds the source; `validate_spec` checks a spec without
opening any device (for pre-flight validation in tools). The returned display
name never includes RTSP userinfo — credentials stay out of logs and events.
"""

import os
from typing import Iterator
from urllib.parse import urlparse

from .base import CameraSource
from .detector_source import FolderSource, WebcamSource

_VIDEO_EXT = (".mp4", ".mov", ".avi", ".mkv", ".webm")


class _Cv2CaptureSource(CameraSource):
    """Shared cv2.VideoCapture loop for streams and files. Yields PIL Images."""

    def __init__(self, target, live: bool):
        self._target = target  # url string or file path
        self._live = live  # live streams raise if they can't open; files too
        self._cap = None

    def frames(self) -> Iterator:
        import cv2  # heavy import deferred; only the live path needs it
        from PIL import Image

        self._cap = cv2.VideoCapture(self._target)
        if not self._cap.isOpened():
            raise RuntimeError(f"could not open video source ({self.name()})")
        while True:
            ok, frame = self._cap.read()
            if not ok:
                return
            yield Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def name(self) -> str:
        raise NotImplementedError


class RtspSource(_Cv2CaptureSource):
    """A network camera stream (RTSP/RTSPS) via OpenCV."""

    def __init__(self, url: str):
        super().__init__(url, live=True)
        self._host = urlparse(url).hostname or "stream"

    def name(self) -> str:
        # host only — the URL may carry credentials in its userinfo.
        return f"rtsp:{self._host}"


class VideoFileSource(_Cv2CaptureSource):
    """Replay a recorded clip frame by frame; ends when the file does."""

    def __init__(self, path: str):
        super().__init__(path, live=False)
        self._basename = os.path.basename(path)

    def name(self) -> str:
        return f"file:{self._basename}"


def _parse(spec: str) -> tuple[str, str]:
    """Normalize a spec into (kind, target). Raises ValueError on junk."""
    spec = str(spec).strip()
    if spec.startswith(("rtsp://", "rtsps://")):
        return "rtsp", spec
    for prefix in ("webcam:", "folder:", "file:"):
        if spec.startswith(prefix):
            return prefix[:-1], spec[len(prefix) :]
    # bare forms, kept for backwards compatibility with the old CAMERA contract
    if spec.isdigit():
        return "webcam", spec
    if os.path.isdir(os.path.expanduser(spec)):
        return "folder", spec
    if spec.lower().endswith(_VIDEO_EXT):
        return "file", spec
    raise ValueError(
        f"unrecognized camera spec {spec!r} "
        "(want webcam:N, folder:/path, rtsp://…, or file:clip.mp4)"
    )


def validate_spec(spec: str) -> str:
    """Check a spec is well-formed and its target exists; return the kind.

    Opens no device — safe to call from a tool before launching a daemon.
    """
    kind, target = _parse(spec)
    if kind == "webcam" and not target.isdigit():
        raise ValueError(f"webcam index must be a number, got {target!r}")
    if kind == "folder" and not os.path.isdir(os.path.expanduser(target)):
        raise ValueError(f"{target!r} is not a directory")
    if kind == "file" and not os.path.isfile(os.path.expanduser(target)):
        raise ValueError(f"{target!r} is not a file")
    return kind


def sanitize_spec(spec: str) -> str:
    """A spec safe to store/echo: RTSP userinfo (user:password@) is stripped.

    The full spec should only ever travel to the process that opens the
    stream (via its environment) — never into saved state, logs, or an
    agent's context.
    """
    kind, target = _parse(spec)
    if kind != "rtsp":
        return spec
    parsed = urlparse(target)
    netloc = parsed.hostname or ""
    if parsed.port:
        netloc += f":{parsed.port}"
    return parsed._replace(netloc=netloc).geturl()


def camera_from_spec(spec: str) -> tuple[CameraSource, str]:
    """Build a CameraSource from a config string. Returns (source, display_name)."""
    validate_spec(spec)
    kind, target = _parse(spec)
    if kind == "webcam":
        return WebcamSource(int(target)), f"webcam-{target}"
    if kind == "folder":
        path = os.path.expanduser(target)
        return FolderSource(path), f"folder:{os.path.basename(path.rstrip('/'))}"
    if kind == "file":
        path = os.path.expanduser(target)
        src = VideoFileSource(path)
        return src, src.name()
    src = RtspSource(target)
    return src, src.name()
