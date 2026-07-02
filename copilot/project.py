"""Durable project state + dataset operations for the copilot.

The copilot's whole job is a multi-step flow (define -> collect -> train ->
review -> deploy), and each step must survive a restart and be inspectable.
So the state lives in a small JSON file on disk, not in the LLM's session —
same reasoning as hibiscus_guard/store.py.

Everything here is deterministic and validates its own inputs (the LLM is
untrusted): kinds are whitelisted, counts are clamped, and imported files must
actually open as images before they enter the dataset.

Layout (under data/, which is git-ignored):

  data/copilot/project.json          the single active project
  data/copilot/<slug>/positive/      scene WITH the target
  data/copilot/<slug>/negative/      the user's own scene WITHOUT it
  data/copilot/<slug>/model.joblib   trained head (written by the job)
"""

import json
import os
import re
import shutil
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOME = os.environ.get("COPILOT_HOME", os.path.join(_REPO_ROOT, "data", "copilot"))
_STATE = os.path.join(HOME, "project.json")

_IMG_EXT = (".jpg", ".jpeg", ".png", ".webp")
KINDS = ("positive", "negative")
MIN_PER_CLASS = 4  # detector.train's floor
MAX_BATCH = 100  # cap any single collect operation


def _slug(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    if not s:
        raise ValueError(f"cannot make a project name out of {name!r}")
    return s


def load() -> dict | None:
    if not os.path.isfile(_STATE):
        return None
    with open(_STATE) as f:
        return json.load(f)


def save(project: dict) -> None:
    os.makedirs(HOME, exist_ok=True)
    with open(_STATE, "w") as f:
        json.dump(project, f, indent=2)


def define(target: str, zone: str = "hibiscus") -> dict:
    """Create (or reset to) a project for one detection target."""
    slug = _slug(target)
    root = os.path.join(HOME, slug)
    for kind in KINDS:
        os.makedirs(os.path.join(root, kind), exist_ok=True)
    project = {
        "target": target,
        "slug": slug,
        "zone": zone,
        "root": root,
        "created": time.time(),
        "status": "collecting",  # collecting -> trained -> live
        "metrics": None,
        "surveillance": None,
    }
    save(project)
    return project


def _dir(project: dict, kind: str) -> str:
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    return os.path.join(project["root"], kind)


def _images_in(directory: str) -> list[str]:
    try:
        names = sorted(os.listdir(directory))
    except FileNotFoundError:
        return []
    return [os.path.join(directory, n) for n in names if n.lower().endswith(_IMG_EXT)]


def counts(project: dict) -> dict:
    per = {kind: len(_images_in(_dir(project, kind))) for kind in KINDS}
    per["ready_to_train"] = all(v >= MIN_PER_CLASS for v in per.values())
    return per


def _next_index(directory: str) -> int:
    return len(os.listdir(directory)) if os.path.isdir(directory) else 0


def import_from_dir(project: dict, src_dir: str, kind: str, limit: int = MAX_BATCH) -> dict:
    """Copy image files from a user-named folder into the dataset.

    Each file must open as a real image (Pillow) to be accepted — extension
    alone is not trusted.
    """
    from PIL import Image

    dst = _dir(project, kind)
    src_dir = os.path.expanduser(src_dir)
    if not os.path.isdir(src_dir):
        raise ValueError(f"{src_dir!r} is not a directory")
    limit = max(1, min(int(limit), MAX_BATCH))

    copied, skipped = 0, 0
    idx = _next_index(dst)
    for path in _images_in(src_dir)[:limit]:
        try:
            with Image.open(path) as im:
                im.verify()
        except Exception:
            skipped += 1
            continue
        ext = os.path.splitext(path)[1].lower()
        shutil.copyfile(path, os.path.join(dst, f"import-{idx + copied:03d}{ext}"))
        copied += 1
    return {"copied": copied, "skipped_non_images": skipped, "dest": dst}


def capture_from_webcam(
    project: dict, kind: str, count: int = 10, camera_index: int = 0, interval_seconds: float = 1.0
) -> dict:
    """Grab `count` frames from the local webcam into the dataset.

    This is how the user photographs their OWN scene — the in-domain negatives
    (and positives) that generic web data can't provide.
    """
    import cv2

    dst = _dir(project, kind)
    count = max(1, min(int(count), MAX_BATCH))
    interval_seconds = max(0.1, min(float(interval_seconds), 10.0))

    cap = cv2.VideoCapture(int(camera_index))
    if not cap.isOpened():
        raise RuntimeError(f"webcam {camera_index} could not be opened")
    try:
        idx = _next_index(dst)
        saved = 0
        while saved < count:
            ok, frame = cap.read()
            if not ok:
                break
            cv2.imwrite(os.path.join(dst, f"cam-{idx + saved:03d}.jpg"), frame)
            saved += 1
            if saved < count:
                time.sleep(interval_seconds)
    finally:
        cap.release()
    return {"captured": saved, "dest": dst}


def fetch_from_web(project: dict, query: str, kind: str, n: int = 10) -> dict:
    """Fetch CC0 images from Openverse (fixed API host) into the dataset.

    Useful for positives when the target rarely shows up on demand; negatives
    should still come from the user's own scene.
    """
    from tools.fetch_dataset import fetch

    dst = _dir(project, kind)
    n = max(1, min(int(n), MAX_BATCH))
    saved = fetch(query, dst, n, start=_next_index(dst))
    return {"fetched": saved, "dest": dst, "license": "cc0/pdm via Openverse"}
