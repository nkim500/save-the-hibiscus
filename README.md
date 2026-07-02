# save-the-hibiscus

Detect and deter squirrels from a hibiscus — or train a detector for anything else.

```
perception (trained detector, or stub)  ──►  ambient agent (judges + escalates)  ──►  MCP egress (ntfy / Telegram)
   confidence + tracking + zone gating        urgency, dedupe, incident count         allowlist · rate limit · audit
```

## Run it

```bash
uv sync
uv run python -m hibiscus_guard.ambient   # replays scripted squirrel events
```

## The copilot — train your own detector by chatting

The control plane: a chat agent that walks you from "I want to detect X" to a
live surveillance agent. It helps you collect example images (webcam capture
of your own scene, imports, CC0 web images), dispatches a training job
(DINOv2 embeddings + small classifier head, CPU, ~a minute), reports holdout
accuracy, and — after you say "go live" — launches the ambient runtime on the
trained model.

```bash
uv sync --group detector
uv run --group detector adk run copilot
```

Everything it builds lives under `data/copilot/<target>/` (git-ignored):
dataset, trained model, project state, surveillance log. The deployed runtime
is `hibiscus_guard.ambient` with `HIBISCUS_SOURCE=detector` (see its docstring
for the env contract) — same governed egress as the demo.

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
