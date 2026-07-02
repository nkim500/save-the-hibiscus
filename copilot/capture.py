"""Manage the stay-and-capture daemon and label what it caught.

Same shape as copilot/runtime.py: "start" launches copilot.capture_daemon as a
detached child configured entirely by environment variables, and we only ever
signal a PID that we recorded AND that still identifies itself as our daemon
module — a recycled PID can't make stop() kill an innocent process.

Labeling closes the data loop: a captured candidate the user confirms moves
into the dataset's positive/ class; a rejection becomes a hard NEGATIVE (the
model was unsure about it, so it's exactly the background example that fixes
over-prediction); discard just deletes. Every label is appended to the
capture manifest, and the GCS copy (if any) is deleted once labeled.

All inputs are validated here (the LLM is untrusted): labels are whitelisted,
names must be bare filenames that exist in the captures dir, camera specs are
parsed by the perception factory before any process launches.
"""

import json
import os
import shutil
import signal
import subprocess
import sys
import time

from copilot import project as proj
from hibiscus_guard.perception.cameras import sanitize_spec, validate_spec

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODULE = "copilot.capture_daemon"

LABELS = ("positive", "negative", "discard")


def _captures_dir(project: dict) -> str:
    return os.path.join(project["root"], "captures")


def _log_path(project: dict) -> str:
    return os.path.join(project["root"], "capture.log")


def _manifest_path(project: dict) -> str:
    return os.path.join(_captures_dir(project), "manifest.jsonl")


def _is_ours(pid: int) -> bool:
    """True if `pid` is alive AND is the capture daemon we launched."""
    try:
        out = subprocess.run(
            ["ps", "-p", str(pid), "-o", "command="], capture_output=True, text=True
        ).stdout
    except OSError:
        return False
    return _MODULE in out


def _manifest(project: dict) -> dict[str, dict]:
    """Latest manifest record per capture name (captures, uploads, labels)."""
    records: dict[str, dict] = {}
    try:
        with open(_manifest_path(project)) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("name"):
                    records.setdefault(rec["name"], {}).update(rec)
    except FileNotFoundError:
        pass
    return records


def start(project: dict, camera: str = "0") -> dict:
    """Launch the capture daemon. One instance per project at a time."""
    cap = project.get("capture") or {}
    if cap.get("pid") and _is_ours(cap["pid"]):
        return {"ok": False, "error": f"capture already running (pid {cap['pid']}) — stop first"}
    camera = str(camera)
    try:
        validate_spec(camera)
    except ValueError as e:
        return {"ok": False, "error": str(e)}

    capture_dir = _captures_dir(project)
    os.makedirs(capture_dir, exist_ok=True)
    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "CAPTURE_CAMERA": camera,
        "CAPTURE_DIR": capture_dir,
        "CAPTURE_SLUG": project["slug"],
    }
    # Once a model exists the daemon harvests the frames it is UNSURE about;
    # before that, motion vs a rolling background is the gate.
    model_path = (project.get("metrics") or {}).get("model_path", "")
    if model_path and os.path.isfile(model_path):
        env["DETECTOR_MODEL"] = model_path

    with open(_log_path(project), "a") as log:
        child = subprocess.Popen(
            [sys.executable, "-m", _MODULE],
            cwd=_REPO_ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,  # survives the copilot exiting
        )
    # stored state (and anything echoed back to the agent) gets the
    # credential-free form; only the child's env holds the full spec
    project["capture"] = {"pid": child.pid, "camera": sanitize_spec(camera), "started": time.time()}
    proj.save(project)
    mode = "uncertain-band (trained model)" if env.get("DETECTOR_MODEL") else "motion-diff"
    return {"ok": True, "pid": child.pid, "mode": mode, "log": _log_path(project)}


def status(project: dict) -> dict:
    cap = project.get("capture") or {}
    pid = cap.get("pid")
    running = bool(pid and _is_ours(pid))
    manifest = _manifest(project)
    unlabeled = [n for n, r in manifest.items() if not r.get("label")]
    tail = ""
    if os.path.isfile(_log_path(project)):
        with open(_log_path(project)) as f:
            tail = "".join(f.readlines()[-10:])
    return {
        "running": running,
        **cap,
        "kept_total": len(manifest),
        "awaiting_review": len(unlabeled),
        "log_tail": tail,
    }


def stop(project: dict) -> dict:
    cap = project.get("capture") or {}
    pid = cap.get("pid")
    project["capture"] = None
    proj.save(project)
    if not pid or not _is_ours(pid):
        return {"ok": True, "note": "nothing was running"}
    os.kill(pid, signal.SIGTERM)
    return {"ok": True, "stopped_pid": pid}


def review(project: dict, limit: int = 12) -> dict:
    """List unlabeled candidates, pulling down any that exist only in GCS."""
    from copilot import gcs

    limit = max(1, min(int(limit), 50))
    capture_dir = _captures_dir(project)
    manifest = _manifest(project)

    pulled = []
    try:
        os.makedirs(capture_dir, exist_ok=True)
        for name in gcs.list_captures(project):
            if not os.path.isfile(os.path.join(capture_dir, name)):
                pulled.append(gcs.download_capture(project, name, capture_dir))
                manifest.setdefault(name, {"name": name, "mode": "gcs"})
                with open(_manifest_path(project), "a") as f:
                    f.write(json.dumps({"ts": time.time(), "name": name, "mode": "gcs"}) + "\n")
    except gcs.GcsPolicyError:
        pass  # no bucket configured — local-only review is fine

    candidates = []
    for name in sorted(manifest):
        rec = manifest[name]
        path = os.path.join(capture_dir, name)
        if rec.get("label") or not os.path.isfile(path):
            continue
        candidates.append(
            {"name": name, "path": path, "mode": rec.get("mode"), "measure": rec.get("measure")}
        )
        if len(candidates) >= limit:
            break
    return {"awaiting_review": candidates, "pulled_from_gcs": len(pulled)}


def label(project: dict, name: str, label: str) -> dict:
    """File one reviewed candidate: positive / negative (hard) / discard."""
    from copilot import gcs

    if label not in LABELS:
        raise ValueError(f"label must be one of {LABELS}, got {label!r}")
    if os.path.basename(name) != name:
        raise ValueError(f"bad capture name {name!r}")
    src = os.path.join(_captures_dir(project), name)
    if not os.path.isfile(src):
        raise ValueError(f"no such capture {name!r}")

    if label == "discard":
        os.remove(src)
        dest = None
    else:
        dest_dir = os.path.join(project["root"], label)
        os.makedirs(dest_dir, exist_ok=True)
        dest = os.path.join(dest_dir, f"capture-{name}")
        shutil.move(src, dest)

    try:  # the bucket copy has served its purpose either way
        gcs.delete_capture(project, name)
    except gcs.GcsPolicyError:
        pass

    with open(_manifest_path(project), "a") as f:
        f.write(json.dumps({"ts": time.time(), "name": name, "label": label}) + "\n")
    return {"ok": True, "name": name, "label": label, "moved_to": dest}
