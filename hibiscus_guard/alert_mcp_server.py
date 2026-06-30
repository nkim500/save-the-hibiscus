"""MCP egress server — the single, governed exit for all alerts.

The agent calls exactly one tool here: send_alert(recipient, title, message,
priority). Notice what the agent does NOT get to provide: no token, no phone
number, no chat id, no channel choice. Those are decided by configuration and
policy on THIS side of the boundary. That asymmetry is the governance.

Every call flows through, in order:
    resolve recipient (allowlist)  ->  rate limit  ->  send  ->  audit log

Channel is chosen by the ALERT_CHANNEL env var (default 'ntfy'), so swapping
ntfy -> telegram -> whatsapp is a config change, not a code change, and the
agent is none the wiser.

Run standalone (debug):  uv run python -m hibiscus_guard.alert_mcp_server
"""

import os

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

from hibiscus_guard.egress import (  # noqa: E402
    PolicyError,
    RateLimiter,
    audit,
    get_channel,
    resolve_recipient,
)

_CHANNEL_NAME = os.environ.get("ALERT_CHANNEL", "ntfy")
_rate_limiter = RateLimiter()

mcp = FastMCP("hibiscus-alerts")


@mcp.tool()
def send_alert(recipient: str, title: str, message: str, priority: str = "medium") -> dict:
    """Notify an allowlisted recipient about the hibiscus.

    Args:
        recipient: An allowlisted recipient KEY, e.g. "household". NOT a phone
            number or address — those are resolved by policy on the server.
        title: Short notification title.
        message: The body text.
        priority: One of "low", "medium", "high".

    Returns:
        A status dict. On policy violation, status is "blocked" with a reason —
        the agent learns it cannot do that, without ever seeing a secret.
    """
    channel = get_channel(_CHANNEL_NAME)
    try:
        address = resolve_recipient(recipient, channel.name)  # 1. allowlist
        _rate_limiter.check_and_record(recipient)  # 2. rate limit
    except PolicyError as e:
        audit(
            {"event": "blocked", "recipient": recipient, "channel": channel.name, "reason": str(e)}
        )
        return {"status": "blocked", "reason": str(e)}

    result = channel.send(address, title, message, priority)  # 3. send
    audit(
        {
            "event": "sent",
            "recipient": recipient,
            "channel": channel.name,
            "priority": priority,
            "title": title,
        }
    )  # 4. audit
    return {"status": "sent", **result}


if __name__ == "__main__":
    mcp.run()
