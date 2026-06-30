"""The policy layer every alert must pass through before it leaves the system.

This is where governance lives, deliberately in ONE place so it is auditable.
The agent calls a tool; that tool cannot bypass any of this:

  1. Recipient allowlist  -- the agent passes a KEY ("household"); we resolve it
     to a real channel address. An unknown key is rejected. The agent can never
     name an arbitrary phone number / chat id, so a hijacked agent cannot text
     strangers.
  2. Rate limiting        -- caps sends per recipient per hour, so a runaway loop
     (or 900 squirrel frames) cannot spam anyone.
  3. Audit logging        -- every attempt (sent, blocked, rate-limited) is
     appended to a JSONL file: non-repudiation, "what did the agent do".

Recipient ADDRESSES are read from the environment, not hardcoded, so nothing
sensitive is committed. The MAP of which keys exist is policy and lives here.
"""

import json
import os
import time
from collections import defaultdict, deque

# Which recipient keys are allowed, and which env var holds each channel address.
# Add a key here to authorize a new recipient; the agent still only ever sees
# the key, never the resolved address.
_RECIPIENTS = {
    "household": {
        "ntfy": "NTFY_TOPIC",
        "telegram": "TELEGRAM_CHAT_ID",
        "whatsapp": "WHATSAPP_TO",
    },
}

_AUDIT_PATH = os.path.join(os.path.dirname(__file__), "egress_audit.jsonl")
_RATE_LIMIT_PER_HOUR = int(os.environ.get("RATE_LIMIT_PER_HOUR", "8"))


class PolicyError(Exception):
    """Raised when a send violates policy (bad recipient, rate limit)."""


def resolve_recipient(key: str, channel_name: str) -> str:
    """Map a safe recipient key + channel to a concrete address, or reject."""
    if key not in _RECIPIENTS:
        raise PolicyError(f"Recipient '{key}' is not allowlisted. Allowed: {list(_RECIPIENTS)}")
    env_var = _RECIPIENTS[key].get(channel_name)
    address = os.environ.get(env_var) if env_var else None
    if not address:
        raise PolicyError(
            f"Recipient '{key}' has no address configured for channel "
            f"'{channel_name}' (expected env var {env_var})."
        )
    return address


class RateLimiter:
    """Sliding-window limiter, keyed by recipient. In-memory for now (resets on
    restart); becomes a shared store once deployed."""

    def __init__(self, per_hour: int = _RATE_LIMIT_PER_HOUR) -> None:
        self._per_hour = per_hour
        self._hits: dict[str, deque] = defaultdict(deque)

    def check_and_record(self, key: str) -> None:
        now = time.time()
        window = self._hits[key]
        while window and now - window[0] > 3600:
            window.popleft()
        if len(window) >= self._per_hour:
            raise PolicyError(f"Rate limit hit for '{key}': {self._per_hour}/hour. Try later.")
        window.append(now)


def audit(record: dict) -> None:
    """Append one tamper-evident-ish line per attempt. Never logs secrets."""
    record = {"ts": time.time(), **record}
    with open(_AUDIT_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")
