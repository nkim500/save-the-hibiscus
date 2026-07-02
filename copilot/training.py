"""Dispatch a training job and collect its result.

Training runs as a SUBPROCESS (detector/job.py), not in the copilot's process.
That's a deliberate rehearsal of the cloud shape — copilot dispatches a
sandboxed job, gets metrics back — and it keeps the heavy ML stack out of the
agent's memory. The subprocess is invoked with a fixed argv (no shell), with
paths that come only from the validated project state.
"""

import json
import os
import subprocess
import sys
import time

from copilot import project as proj

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TIMEOUT = 900  # embedding a few dozen images on CPU takes ~a minute; be generous


def dispatch(project: dict) -> dict:
    """Run the training job for the active project; store and return metrics."""
    c = proj.counts(project)
    if not c["ready_to_train"]:
        return {
            "ok": False,
            "error": f"need >= {proj.MIN_PER_CLASS} images per class first",
            "counts": c,
        }

    model_path = os.path.join(project["root"], "model.joblib")
    argv = [
        sys.executable,
        "-m",
        "detector.job",
        os.path.join(project["root"], "positive"),
        os.path.join(project["root"], "negative"),
        model_path,
    ]
    started = time.time()
    try:
        run = subprocess.run(argv, cwd=_REPO_ROOT, capture_output=True, text=True, timeout=_TIMEOUT)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"training job timed out after {_TIMEOUT}s"}

    line = run.stdout.strip().splitlines()[-1] if run.stdout.strip() else ""
    try:
        result = json.loads(line)
    except json.JSONDecodeError:
        return {
            "ok": False,
            "error": "job produced no result",
            "stderr_tail": run.stderr[-500:],
        }

    if result.get("ok"):
        project["status"] = "trained"
        project["metrics"] = {**result, "trained_at": started, "model_path": model_path}
        proj.save(project)
    return result
