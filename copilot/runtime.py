"""Deploy / inspect / stop the surveillance runtime.

"Deploy" here means: launch hibiscus_guard.ambient as a long-running child
process, configured by environment variables to use the trained detector
(see ambient.source_from_env). The copilot never touches the runtime's
internals — the env contract is the deployment surface, same as it will be
on Cloud Run.

Safety: we only ever signal a PID that (a) we recorded at launch and (b) still
identifies itself as our ambient module in the process table — so a recycled
PID can't make stop() kill an innocent process.
"""

import os
import signal
import subprocess
import sys
import time

from copilot import project as proj

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODULE = "hibiscus_guard.ambient"


def _log_path(project: dict) -> str:
    return os.path.join(project["root"], "surveillance.log")


def _is_ours(pid: int) -> bool:
    """True if `pid` is alive AND is the ambient runtime we launched."""
    try:
        out = subprocess.run(
            ["ps", "-p", str(pid), "-o", "command="], capture_output=True, text=True
        ).stdout
    except OSError:
        return False
    return _MODULE in out


def deploy(project: dict, camera: str = "0") -> dict:
    """Launch the ambient runtime on the trained model. One instance at a time."""
    if project.get("status") not in ("trained", "live"):
        return {"ok": False, "error": "no trained model yet — train and review first"}
    sv = project.get("surveillance") or {}
    if sv.get("pid") and _is_ours(sv["pid"]):
        return {"ok": False, "error": f"already live (pid {sv['pid']}) — stop it first"}

    model_path = project["metrics"]["model_path"]
    if not os.path.isfile(model_path):
        return {"ok": False, "error": f"model file missing: {model_path}"}
    camera = str(camera)
    if not (camera.isdigit() or os.path.isdir(camera)):
        return {"ok": False, "error": "camera must be a webcam index or an existing directory"}

    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",  # log lines reach the file as they happen
        "HIBISCUS_SOURCE": "detector",
        "DETECTOR_MODEL": model_path,
        "TARGET_LABEL": project["target"],
        "CAMERA": camera,
        "ZONE": project["zone"],
    }
    with open(_log_path(project), "a") as log:
        child = subprocess.Popen(
            [sys.executable, "-m", _MODULE],
            cwd=_REPO_ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,  # survives the copilot exiting
        )
    project["status"] = "live"
    project["surveillance"] = {"pid": child.pid, "camera": camera, "started": time.time()}
    proj.save(project)
    return {"ok": True, "pid": child.pid, "log": _log_path(project)}


def status(project: dict) -> dict:
    sv = project.get("surveillance") or {}
    pid = sv.get("pid")
    running = bool(pid and _is_ours(pid))
    tail = ""
    if os.path.isfile(_log_path(project)):
        with open(_log_path(project)) as f:
            tail = "".join(f.readlines()[-15:])
    return {"running": running, **sv, "log_tail": tail}


def stop(project: dict) -> dict:
    sv = project.get("surveillance") or {}
    pid = sv.get("pid")
    if not pid or not _is_ours(pid):
        project["surveillance"] = None
        if project.get("status") == "live":
            project["status"] = "trained"
        proj.save(project)
        return {"ok": True, "note": "nothing was running"}
    os.kill(pid, signal.SIGTERM)
    project["surveillance"] = None
    project["status"] = "trained"
    proj.save(project)
    return {"ok": True, "stopped_pid": pid}
