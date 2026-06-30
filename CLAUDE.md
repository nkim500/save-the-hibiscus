# Project instructions for Claude Code

## Red-team review before pushing

Before creating a PR or pushing to the remote, run an **adversarial
(red-team) code review** of the staged/changed code and report findings to the
user. Do NOT push until findings are surfaced and addressed (or explicitly
waived by the user).

This project's whole point is agent governance, so review through an attacker's
eyes. For every change, ask:

1. **Secret leakage** — can any new tool, log line, or error message expose a
   credential (the model key, Telegram/WhatsApp token)? Does anything put a
   secret into the agent's context or into a prompt? Is there any new tool that
   can read the process's own files/env/memory?
2. **Egress bypass** — does every outbound notification still go through
   `hibiscus_guard/egress/governance.py` (allowlist → rate limit → send →
   audit)? Any new path that reaches a network/service directly, around the
   policy layer?
3. **Recipient & input integrity** — can the agent be steered (prompt
   injection) into messaging a non-allowlisted recipient, escalating its own
   powers, or passing a raw address/number where a key is expected?
4. **Rate-limit / audit integrity** — can the limiter be skipped? Is every
   attempt (sent AND blocked) still audited? Any unbounded loop that could spam?
5. **Trust boundary** — is the LLM treated as untrusted? Are tool inputs
   validated server-side rather than trusting the model to behave?

Output a short findings list (severity + file:line + fix). If clean, say so
explicitly. Keep it terse — only real issues.

## Conventions

- New agent tools follow the **read → act → log** pattern.
- Counting/escalation logic lives in deterministic tools, not the LLM prompt.
- Secrets live in `.env` (local) or a separate process (MCP egress) — never in
  the agent's context.
