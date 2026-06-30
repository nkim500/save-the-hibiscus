# save-the-hibiscus

A small **Google ADK** project for learning agent development + **agent governance**.
A camera detects squirrels approaching a hibiscus; an AI agent judges each
sighting and sends a phone alert through a **governed egress** (the agent never
holds secrets or picks raw recipients).

```
perception (YOLO+ByteTrack, or stub)  ──►  ambient agent (judges + escalates)  ──►  MCP egress (ntfy / Telegram)
   confidence + tracking + zone gating       urgency, dedupe, incident count        allowlist · rate limit · audit
```

## Run it

```bash
uv sync
uv run python -m hibiscus_guard.simulate   # replays scripted squirrel events
```

## Environment variables

Put these in `hibiscus_guard/.env` (git-ignored):

| Var | Needed for | Notes |
|-----|-----------|-------|
| `ANTHROPIC_API_KEY` | always | the agent's model (Claude via LiteLLM) |
| `ALERT_CHANNEL` | always | `ntfy` (default), `telegram`, or `whatsapp` |
| `NTFY_TOPIC` | ntfy channel | your unique, unguessable topic name |
| `TELEGRAM_BOT_TOKEN` | telegram channel | the bot's secret (see below) |
| `TELEGRAM_CHAT_ID` | telegram channel | where alerts land (your user id) |
| `RATE_LIMIT_PER_HOUR` | optional | max alerts per recipient/hour (default 8) |

`WHATSAPP_*` vars exist but the channel is scaffolded only (needs Meta onboarding).

## Telegram setup (2 minutes, free)

1. In Telegram, message **@BotFather** → `/newbot` → follow prompts → copy the **token**.
2. Send your new bot any message (so it's allowed to DM you).
3. Message **@userinfobot** → it replies with your numeric **id** = your `TELEGRAM_CHAT_ID`.
4. In `.env`: set `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`, and `ALERT_CHANNEL=telegram`.
5. Re-run the simulation — alerts now arrive in Telegram.
