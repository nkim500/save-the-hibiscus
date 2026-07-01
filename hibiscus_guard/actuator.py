"""Local actuator — a physical reaction on the machine running the agent.

This is the seed of "connect to a reactive physical device." For now it just
makes a sound on the laptop (the same laptop that, later, runs the webcam). The
ambient agent calls this to scare the squirrel in real time — the tangible
payoff of the ambient pattern: an agent driving hardware, not just sending text.

Swap the body for a GPIO pin, a relay to a sprinkler, an MQTT message to an
ESP32, etc. The agent-facing interface (`sound_alarm`) stays the same.

NOTE for governance: a physical actuator is also an egress worth governing in a
real deployment ("the agent may make noise, but not unlock the door"). We keep
it a simple local tool here to focus on the ambient pattern.
"""

import platform
import subprocess
import sys


def buzz(urgency: str) -> dict:
    """Make a deterrent sound locally. Best-effort; never raises."""
    times = {"low": 1, "medium": 2, "high": 3}.get(urgency, 1)
    try:
        if platform.system() == "Darwin":
            # macOS: play a built-in system sound, once per urgency level.
            for _ in range(times):
                subprocess.run(
                    ["afplay", "/System/Library/Sounds/Sosumi.aiff"],
                    check=False,
                    timeout=5,
                )
        else:
            # Portable fallback: terminal bell.
            sys.stdout.write("\a" * times)
            sys.stdout.flush()
        return {"status": "buzzed", "beeps": times}
    except Exception as e:  # actuator must never crash the agent loop
        return {"status": "actuator_error", "error": str(e)}
