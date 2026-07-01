"""The contract between the perception layer and the agent.

This is the ONE thing both sides depend on. Perception (YOLO+ByteTrack, or the
stub) produces TrackEvents; the agent consumes them. Keeping this tiny and
explicit is what lets us swap a fake detector for a real one without touching
the agent.

A TrackEvent is already *cooked*: by the time one exists, perception has applied
the confidence threshold, the tracker has de-duplicated frames into a single
track identity, and the zone geometry has confirmed the animal is on/near the
hibiscus. So an "entered_zone" event means "a confirmed squirrel just entered
the hibiscus zone" — not "a pixel changed." That is why the agent no longer
needs to threshold or debounce.

The `tier` field carries perception's CONFIDENCE judgment as two bands:
  * "confident" — clearly a squirrel; worth alerting.
  * "candidate" — a "maybe": above noise, below the alert bar. The agent should
    LOG it (so the daily digest can say "I saw a maybe and passed") but NOT
    alert. This is how the two-tier idea reaches the agent.
"""

import time
from dataclasses import dataclass, field


@dataclass
class TrackEvent:
    event_type: str  # "entered_zone" | "left_zone"
    track_id: int  # stable identity from the tracker (e.g. ByteTrack)
    label: str  # "squirrel"
    camera: str  # which source, e.g. "webcam-1" — supports multi-cam later
    zone: str  # the zone crossed, e.g. "hibiscus"
    confidence: float  # peak/smoothed track confidence from perception
    tier: str = "confident"  # "confident" | "candidate" (a "maybe")
    timestamp: float = field(default_factory=time.time)

    def to_prompt(self) -> str:
        """Render the event as the line the agent reads."""
        return (
            f"event={self.event_type} tier={self.tier} label={self.label} "
            f"camera={self.camera} zone={self.zone} track_id={self.track_id} "
            f"confidence={self.confidence:.2f}"
        )
