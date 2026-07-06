"""The stay-and-capture daemon — a camera that keeps only interesting frames.

The copilot's collect step assumed positives exist on demand; a squirrel
disagrees. So this process watches the camera continuously and KEEPS a frame
only when something interesting happens:

  * no trained model yet  -> motion: the frame differs from a rolling
    background average (cheap, no ML) — "something moved, maybe your target".
  * trained model exists  -> uncertainty: the classifier scores the frame in
    the candidate band (candidate_t <= score < confident_t) — the model
    harvests exactly the examples it is unsure about, its own hard cases.

Kept frames land in the project's captures/ dir with one manifest line each
(read -> act -> log), and — if CAPTURES_BUCKET is set — are uploaded through
copilot.gcs (governed egress: fixed bucket, validated names, hourly budget,
audited). The copilot later shows them to the user: "is THIS your squirrel?"

Everything here is deterministic; no LLM anywhere near a camera or a bucket.

Env contract (see copilot/capture.py, which launches this):

  CAPTURE_CAMERA   camera spec (webcam:0 / folder:/p / rtsp://… / file:c.mp4)
  CAPTURE_DIR      where kept frames + manifest.jsonl go
  CAPTURE_SLUG     project slug (namespaces the GCS prefix)
  DETECTOR_MODEL   optional .joblib — presence switches motion -> uncertainty
  CANDIDATE_T / CONFIDENT_T   uncertain band (default 0.6 / 0.8)
  MOTION_T         mean |diff| threshold on 0-255 gray (default 12)
  POLL_SECONDS     pause between frames (default 1.0)
  KEEP_COOLDOWN    min seconds between kept frames (default 5)
  MAX_KEPT         stop when captures/ holds this many frames (default 500)
  CAPTURES_BUCKET  optional GCS bucket for candidate review
"""

import json
import os
import time

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), "hibiscus_guard", ".env"))

from hibiscus_guard.perception import camera_from_spec  # noqa: E402

_IMG_EXT = (".jpg", ".jpeg", ".png", ".webp")


def _kept_count(capture_dir: str) -> int:
    try:
        return sum(1 for n in os.listdir(capture_dir) if n.lower().endswith(_IMG_EXT))
    except FileNotFoundError:
        return 0


def _log_manifest(capture_dir: str, record: dict) -> None:
    with open(os.path.join(capture_dir, "manifest.jsonl"), "a") as f:
        f.write(json.dumps({"ts": time.time(), **record}) + "\n")


class MotionGate:
    """Keep a frame when it differs enough from a rolling background average."""

    def __init__(self, threshold: float):
        self._threshold = threshold
        self._background = None  # 64x64 float, EMA of gray frames

    def interesting(self, frame) -> tuple[bool, float]:
        import numpy as np

        gray = np.asarray(frame.convert("L").resize((64, 64)), dtype=float)
        if self._background is None:
            self._background = gray
            return False, 0.0  # first frame defines the background
        diff = float(np.abs(gray - self._background).mean())
        self._background = 0.9 * self._background + 0.1 * gray
        return diff >= self._threshold, round(diff, 2)


class UncertaintyGate:
    """Keep a frame the trained classifier is unsure about (candidate band)."""

    def __init__(self, model_path: str, candidate_t: float, confident_t: float):
        from detector import classify

        self._predict = classify.predict
        self._bundle = classify.load(model_path)
        self._lo, self._hi = candidate_t, confident_t

    def interesting(self, frame) -> tuple[bool, float]:
        score = self._predict(frame, self._bundle)["score"]
        return self._lo <= score < self._hi, score


def main() -> None:
    capture_dir = os.environ["CAPTURE_DIR"]
    slug = os.environ["CAPTURE_SLUG"]
    os.makedirs(capture_dir, exist_ok=True)

    poll = max(0.0, float(os.environ.get("POLL_SECONDS", "1.0")))
    cooldown = max(0.0, float(os.environ.get("KEEP_COOLDOWN", "5")))
    max_kept = max(1, int(os.environ.get("MAX_KEPT", "500")))

    model = os.environ.get("DETECTOR_MODEL", "")
    if model and os.path.isfile(model):
        gate = UncertaintyGate(
            model,
            float(os.environ.get("CANDIDATE_T", "0.6")),
            float(os.environ.get("CONFIDENT_T", "0.8")),
        )
        mode = "uncertain"
    else:
        gate = MotionGate(float(os.environ.get("MOTION_T", "12")))
        mode = "motion"

    bucket = os.environ.get("CAPTURES_BUCKET", "")
    gcs_project = {"slug": slug, "root": os.path.dirname(capture_dir)}
    if bucket:
        from copilot import gcs

    camera, camera_name = camera_from_spec(os.environ["CAPTURE_CAMERA"])
    print(f"capture: mode={mode} camera={camera_name} dir={capture_dir} bucket={bucket or '-'}")

    kept = _kept_count(capture_dir)
    last_keep = 0.0
    seq = int(time.time())  # unique across restarts into the same dir
    try:
        for frame in camera.frames():
            if kept >= max_kept:
                print(f"capture: MAX_KEPT={max_kept} reached — stopping")
                return
            interesting, measure = gate.interesting(frame)
            now = time.time()
            if interesting and now - last_keep >= cooldown:
                seq += 1
                name = f"cap-{seq}.jpg"
                frame.convert("RGB").save(os.path.join(capture_dir, name), "JPEG")
                kept += 1
                last_keep = now
                # local keep is logged immediately; the upload gets its own
                # record after — a slow bucket must never delay local truth
                _log_manifest(
                    capture_dir,
                    {"name": name, "mode": mode, "measure": measure, "camera": camera_name},
                )
                print(f"capture: kept {name} ({mode}={measure}) total={kept}")
                if bucket:
                    result = gcs.upload_captures(gcs_project, [os.path.join(capture_dir, name)])
                    _log_manifest(
                        capture_dir, {"name": name, "uploaded": name in result["uploaded"]}
                    )
            if poll:
                time.sleep(poll)
        print("capture: camera stream ended")
    finally:
        camera.release()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("capture: stopped")
