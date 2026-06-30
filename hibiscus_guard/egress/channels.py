"""Notification channels — the swappable destinations an alert can go to.

Each channel knows how to deliver to ONE service (ntfy, Telegram, WhatsApp) and
nothing else. Crucially, each channel reads its OWN secret from the environment
*inside this server process*. The secret never travels to the agent, never
enters an LLM prompt. That is "credential isolation": the dangerous token lives
here, behind the MCP boundary, where a prompt-injected agent cannot reach it.

`address` is the channel-specific destination (an ntfy topic, a Telegram chat
id, a WhatsApp number). The governance layer resolves a safe recipient *key*
("household") into one of these — the agent never names a raw address.
"""

import os
from typing import Protocol

import httpx


class Channel(Protocol):
    name: str

    def send(self, address: str, title: str, message: str, priority: str) -> dict:
        ...


# --- ntfy (default; no secret — a public topic) ------------------------------

class NtfyChannel:
    name = "ntfy"
    _PRIORITY = {"low": 2, "medium": 3, "high": 5}

    def __init__(self) -> None:
        self._base = os.environ.get("NTFY_BASE", "https://ntfy.sh")

    def send(self, address: str, title: str, message: str, priority: str) -> dict:
        resp = httpx.post(
            self._base,
            json={
                "topic": address,
                "title": title,
                "message": message,
                "priority": self._PRIORITY.get(priority, 3),
                "tags": ["chipmunk"],
            },
            timeout=10.0,
        )
        resp.raise_for_status()
        return {"channel": self.name, "delivered": True}


# --- Telegram (real secret: a bot token) -------------------------------------

class TelegramChannel:
    name = "telegram"

    def __init__(self) -> None:
        # The token is THE secret. It is read here and nowhere else. If it is
        # absent we fail loudly rather than silently degrade.
        self._token = os.environ.get("TELEGRAM_BOT_TOKEN")
        if not self._token:
            raise RuntimeError(
                "TELEGRAM_BOT_TOKEN not set. Create a bot via @BotFather and put "
                "the token in hibiscus_guard/.env."
            )

    def send(self, address: str, title: str, message: str, priority: str) -> dict:
        # 'address' is the chat id. Low priority -> silent notification.
        prefix = {"low": "", "medium": "⚠️ ", "high": "🚨 "}.get(priority, "")
        resp = httpx.post(
            f"https://api.telegram.org/bot{self._token}/sendMessage",
            json={
                "chat_id": address,
                "text": f"{prefix}*{title}*\n{message}",
                "parse_mode": "Markdown",
                "disable_notification": priority == "low",
            },
            timeout=10.0,
        )
        resp.raise_for_status()
        return {"channel": self.name, "delivered": True}


# --- WhatsApp (scaffolded — Meta Cloud API; activate when you have creds) -----

class WhatsAppChannel:
    name = "whatsapp"

    def __init__(self) -> None:
        self._token = os.environ.get("WHATSAPP_TOKEN")
        self._phone_id = os.environ.get("WHATSAPP_PHONE_ID")
        if not (self._token and self._phone_id):
            raise RuntimeError(
                "WhatsApp not configured. Set WHATSAPP_TOKEN and WHATSAPP_PHONE_ID "
                "(Meta Cloud API) in .env. NOTE: business-initiated messages also "
                "require a pre-approved message TEMPLATE — plain text will be "
                "rejected by Meta outside the 24h customer-service window."
            )

    def send(self, address: str, title: str, message: str, priority: str) -> dict:
        # Real shape of the Meta Cloud API call. Left here as the integration
        # point; uncomment + supply an approved template to go live.
        #
        # resp = httpx.post(
        #     f"https://graph.facebook.com/v20.0/{self._phone_id}/messages",
        #     headers={"Authorization": f"Bearer {self._token}"},
        #     json={
        #         "messaging_product": "whatsapp",
        #         "to": address,
        #         "type": "template",
        #         "template": {"name": "hibiscus_alert", "language": {"code": "en"},
        #                      "components": [...]},
        #     }, timeout=10.0,
        # )
        # resp.raise_for_status()
        raise NotImplementedError(
            "WhatsAppChannel is scaffolded but not activated. Complete Meta Cloud "
            "API onboarding + template approval, then enable the call above."
        )


_REGISTRY = {
    NtfyChannel.name: NtfyChannel,
    TelegramChannel.name: TelegramChannel,
    WhatsAppChannel.name: WhatsAppChannel,
}


def get_channel(name: str) -> Channel:
    """Construct the configured channel. Secrets are read at construction time,
    inside this process — never passed in from the agent."""
    if name not in _REGISTRY:
        raise ValueError(f"Unknown channel '{name}'. Options: {list(_REGISTRY)}")
    return _REGISTRY[name]()
