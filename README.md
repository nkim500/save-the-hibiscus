# adk-weather

A tiny [Google ADK](https://google.github.io/adk-docs/) project to get acquainted
with agent building in Python. One agent, one tool, running on a Claude model
via LiteLLM (no Google API key needed).

## Setup

Add your API key in [weather_agent/.env](weather_agent/.env):
```
ANTHROPIC_API_KEY=sk-ant-...
```

## Run it

**In the terminal** (chat in your shell):
```bash
uv run adk run weather_agent
```

**In the browser** (ADK's dev UI — shows the tool calls visually):
```bash
uv run adk web
```
Then open the printed URL and pick `weather_agent`.

Try asking: *"What's the weather in Seoul?"*

## What's where

| File | What it is |
|------|------------|
| [weather_agent/agent.py](weather_agent/agent.py) | The agent + its `get_weather` tool. Start here. |
| [weather_agent/__init__.py](weather_agent/__init__.py) | Makes ADK discover the agent. |
| [weather_agent/.env](weather_agent/.env) | Your API key (git-ignored). |

## Next steps to learn ADK

- Make `get_weather` call a real weather API instead of the fake dict.
- Add a second tool and watch the model choose between them.
- Give the agent `sub_agents=[...]` to try multi-agent delegation.
- Switch the model to `"openai/gpt-4o"` to see LiteLLM swap providers.
