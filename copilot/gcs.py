"""Governed GCS access for capture candidates — upload is EGRESS.

Same posture as hibiscus_guard/egress/governance.py, applied to a new channel:

  1. Destination allowlist -- callers never name a bucket. The ONE allowed
     bucket comes from the CAPTURES_BUCKET env var, and every object lives
     under the project's own prefix (<slug>/captures/<basename>). Object
     names are validated server-side, so neither the LLM nor a filename can
     traverse outside that prefix.
  2. Batch caps            -- MAX_BATCH objects per call, and an hourly upload
     budget per slug (persisted in the audit file, so it survives restarts
     and applies across the daemon and the copilot alike).
  3. Audit logging         -- every attempt (uploaded, downloaded, deleted,
     blocked) is appended to data/copilot/<slug>/gcs_audit.jsonl.

Auth is ADC/IAM (google-cloud-storage) — no key material anywhere near here.
"""

import json
import os
import re
import time

BUCKET_ENV = "CAPTURES_BUCKET"
MAX_BATCH = 25  # objects per call
MAX_UPLOADS_PER_HOUR = 120  # per slug, across all processes

_SAFE_NAME = re.compile(r"^[A-Za-z0-9._-]{1,128}$")


class GcsPolicyError(Exception):
    """Raised when a transfer violates policy (no bucket, bad name, budget)."""


def bucket_name() -> str:
    name = os.environ.get(BUCKET_ENV, "")
    if not name:
        raise GcsPolicyError(f"no capture bucket configured (set {BUCKET_ENV})")
    return name


def _audit_path(project: dict) -> str:
    return os.path.join(project["root"], "gcs_audit.jsonl")


def _audit(project: dict, record: dict) -> None:
    """Append one line per attempt. Object names only — never file contents."""
    os.makedirs(project["root"], exist_ok=True)
    with open(_audit_path(project), "a") as f:
        f.write(json.dumps({"ts": time.time(), **record}) + "\n")


def _uploads_last_hour(project: dict) -> int:
    cutoff = time.time() - 3600
    n = 0
    try:
        with open(_audit_path(project)) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("action") == "upload" and rec.get("ok") and rec.get("ts", 0) > cutoff:
                    n += 1
    except FileNotFoundError:
        pass
    return n


def _safe_basename(name: str) -> str:
    """Accept only a bare, boring filename — no separators, no traversal."""
    base = os.path.basename(str(name))
    if base != name or not _SAFE_NAME.match(base):
        raise GcsPolicyError(f"unsafe object name {name!r}")
    return base


def _prefix(project: dict) -> str:
    return f"{project['slug']}/captures/"


def _client():
    from google.cloud import storage  # deferred: only transfers need it

    return storage.Client()


def _bucket():
    return _client().bucket(bucket_name())


def upload_captures(project: dict, paths: list[str]) -> dict:
    """Upload local capture files to the project's prefix. Read -> act -> log.

    Only files that really live in the project's captures/ dir may leave the
    machine — a path pointing anywhere else (however dressed up) is rejected,
    so this can never become an exfiltration channel for arbitrary files.
    """
    bucket = bucket_name()  # fail fast (and unaudited-but-harmless) if unset
    captures_root = os.path.realpath(os.path.join(project["root"], "captures"))
    for path in paths:
        if os.path.dirname(os.path.realpath(path)) != captures_root:
            raise GcsPolicyError(f"{path!r} is outside the captures dir — refusing to upload")
    paths = list(paths)[:MAX_BATCH]
    budget = MAX_UPLOADS_PER_HOUR - _uploads_last_hour(project)
    uploaded, blocked = [], []
    handle = None
    for path in paths:
        base = _safe_basename(os.path.basename(path))
        dest = _prefix(project) + base
        if budget <= 0:
            blocked.append(base)
            _audit(project, {"action": "upload", "object": dest, "ok": False, "why": "hourly cap"})
            continue
        try:
            if handle is None:
                handle = _bucket()
            handle.blob(dest).upload_from_filename(path)
        except Exception as e:  # noqa: BLE001 — audit the failure, keep going
            _audit(project, {"action": "upload", "object": dest, "ok": False, "why": str(e)[:200]})
            blocked.append(base)
            continue
        budget -= 1
        uploaded.append(base)
        _audit(project, {"action": "upload", "object": dest, "ok": True})
    return {"bucket": bucket, "uploaded": uploaded, "blocked": blocked}


def list_captures(project: dict, limit: int = 100) -> list[str]:
    """Object basenames currently in the project's capture prefix."""
    limit = max(1, min(int(limit), 1000))
    prefix = _prefix(project)
    blobs = _client().list_blobs(bucket_name(), prefix=prefix, max_results=limit)
    return [b.name[len(prefix) :] for b in blobs]


def download_capture(project: dict, name: str, dst_dir: str) -> str:
    """Pull one candidate down for review. Name validated against our prefix."""
    base = _safe_basename(name)
    dest = _prefix(project) + base
    os.makedirs(dst_dir, exist_ok=True)
    local = os.path.join(dst_dir, base)
    _bucket().blob(dest).download_to_filename(local)
    _audit(project, {"action": "download", "object": dest, "ok": True})
    return local


def delete_capture(project: dict, name: str) -> None:
    """Remove a candidate from the bucket once it has been labeled/discarded."""
    base = _safe_basename(name)
    dest = _prefix(project) + base
    try:
        _bucket().blob(dest).delete()
        _audit(project, {"action": "delete", "object": dest, "ok": True})
    except Exception as e:  # noqa: BLE001
        _audit(project, {"action": "delete", "object": dest, "ok": False, "why": str(e)[:200]})
