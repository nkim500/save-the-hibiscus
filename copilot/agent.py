"""The copilot — the control-plane agent of the train-your-own-detector flow.

This is where an LLM actually earns its keep in this project. The runtime
(hibiscus_guard) barely needs one — deterministic rules beat it. But turning a
user's fuzzy "I want to know when squirrels raid my hibiscus" into a working
detector requires conversation: what's the target? which scene? are the
example images any good? is 87% holdout accuracy acceptable to YOU? That
judgment-and-dialogue layer is this agent.

The tools below are the only things that touch disk/processes, and each one
validates its own inputs (copilot/project.py, training.py, runtime.py). The
agent decides WHAT to do; the tools decide whether it's allowed.

Chat with it:  uv run --group detector adk run copilot
"""

import os

from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# The model key lives in the runtime's .env (never in this agent's context).
load_dotenv(os.path.join(_REPO_ROOT, "hibiscus_guard", ".env"))

from copilot import project as proj  # noqa: E402
from copilot import runtime, training  # noqa: E402


# --- Tools (read -> act -> log; all validation server-side) -------------------


def get_project() -> dict:
    """Read the active detector project: target, status, dataset counts, metrics.

    Call this first in any conversation to see where the flow stands.
    """
    p = proj.load()
    if p is None:
        return {"exists": False, "hint": "no project yet — use define_target"}
    return {"exists": True, **p, "counts": proj.counts(p)}


def define_target(target: str, zone: str = "hibiscus") -> dict:
    """Start a new detector project for one target (e.g. 'squirrel', 'raccoon').

    Creates the dataset folders. Replaces any previous project definition (the
    old project's files stay on disk).

    Args:
        target: What to detect, in the user's words.
        zone: Name of the protected zone for alert events.
    """
    return proj.define(target, zone)


def dataset_status() -> dict:
    """Count collected examples per class and whether training can start."""
    p = proj.load()
    if p is None:
        return {"error": "no project — use define_target first"}
    return proj.counts(p)


def capture_examples(kind: str, count: int = 10, interval_seconds: float = 1.0) -> dict:
    """Photograph the user's OWN scene with the local webcam.

    This is the most valuable data in the whole flow — especially kind=
    'negative' (the scene WITHOUT the target), which web data cannot provide.
    Tell the user what to stage before you call this (e.g. 'point the camera
    at the hibiscus, make sure no squirrel is around').

    Args:
        kind: 'positive' (target in view) or 'negative' (scene without it).
        count: How many frames to grab (1-100).
        interval_seconds: Pause between frames (0.1-10).
    """
    p = proj.load()
    if p is None:
        return {"error": "no project — use define_target first"}
    try:
        return proj.capture_from_webcam(p, kind, count, 0, interval_seconds)
    except Exception as e:  # noqa: BLE001 — surface as data, agent decides next step
        return {"error": str(e)}


def import_examples(src_dir: str, kind: str) -> dict:
    """Copy the user's existing photos from a folder they name into the dataset.

    Only files that verify as real images are accepted.

    Args:
        src_dir: Folder the user pointed at (e.g. '~/Pictures/squirrels').
        kind: 'positive' or 'negative'.
    """
    p = proj.load()
    if p is None:
        return {"error": "no project — use define_target first"}
    try:
        return proj.import_from_dir(p, src_dir, kind)
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


def fetch_web_examples(query: str, kind: str, n: int = 10) -> dict:
    """Fetch CC0/public-domain images from Openverse into the dataset.

    A fallback for POSITIVES when the target won't pose on demand. Negatives
    should come from the user's own scene instead (capture_examples).

    Args:
        query: Search terms, e.g. 'squirrel'.
        kind: 'positive' or 'negative'.
        n: How many to fetch (1-100).
    """
    p = proj.load()
    if p is None:
        return {"error": "no project — use define_target first"}
    try:
        return proj.fetch_from_web(p, query, kind, n)
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


def dispatch_training() -> dict:
    """Run the training job (DINOv2 embeddings + small classifier head).

    Takes a minute or two on CPU. Returns holdout accuracy and the model path;
    report the accuracy to the user honestly and discuss whether it's enough
    before proposing deployment.
    """
    p = proj.load()
    if p is None:
        return {"error": "no project — use define_target first"}
    return training.dispatch(p)


def deploy_surveillance(camera: str = "0") -> dict:
    """Go live: launch the ambient surveillance runtime on the trained model.

    ONLY call this after the user has explicitly confirmed they want to go
    live with the reviewed accuracy.

    Args:
        camera: Webcam index like '0', or a directory path to watch for images.
    """
    p = proj.load()
    if p is None:
        return {"error": "no project — use define_target first"}
    return runtime.deploy(p, camera)


def surveillance_status() -> dict:
    """Check whether the surveillance runtime is up, and read its recent log."""
    p = proj.load()
    if p is None:
        return {"error": "no project — use define_target first"}
    return runtime.status(p)


def stop_surveillance() -> dict:
    """Stop the running surveillance runtime (verifies the PID is really ours)."""
    p = proj.load()
    if p is None:
        return {"error": "no project — use define_target first"}
    return runtime.stop(p)


# --- Agent --------------------------------------------------------------------

root_agent = Agent(
    name="detector_copilot",
    model=LiteLlm(model="anthropic/claude-sonnet-4-6"),
    description="Guides a user from 'I want to detect X' to a live, trained surveillance agent.",
    instruction=(
        "You are the detector copilot. You take a user from an idea ('tell me "
        "when squirrels raid my hibiscus') to a live detector, in five steps. "
        "Start every conversation by calling get_project to see where things "
        "stand, and resume from there.\n\n"
        "1. DEFINE — ask what they want to detect and where; call define_target.\n"
        "2. COLLECT — build the dataset. Explain the two classes: 'positive' "
        "(scene WITH the target) and 'negative' (their scene WITHOUT it). Push "
        "hard for IN-DOMAIN negatives via capture_examples — generic web "
        "backgrounds make the detector over-predict; photos of their actual "
        "empty scene are what fix that. Web images (fetch_web_examples) are an "
        "acceptable fallback for positives only. Aim for 10+ images per class; "
        "4 is the hard minimum. Check progress with dataset_status.\n"
        "3. TRAIN — call dispatch_training. It's a background job; report the "
        "holdout accuracy verbatim when it returns.\n"
        "4. REVIEW — interpret the number honestly (small holdout => coarse "
        "estimate). If accuracy is weak, recommend more/better examples — "
        "usually more in-domain negatives — and retrain.\n"
        "5. DEPLOY — describe what going live means (a local process watching "
        "the camera, alerting through the governed egress channel). Ask "
        "explicitly: 'go live?'. Only after a clear yes, call "
        "deploy_surveillance. Afterwards use surveillance_status / "
        "stop_surveillance on request.\n\n"
        "Never invent tool results or accuracy numbers. If a tool returns an "
        "error, tell the user what failed and propose the next step. Be "
        "conversational but efficient — one question at a time."
    ),
    tools=[
        get_project,
        define_target,
        dataset_status,
        capture_examples,
        import_examples,
        fetch_web_examples,
        dispatch_training,
        deploy_surveillance,
        surveillance_status,
        stop_surveillance,
    ],
)
