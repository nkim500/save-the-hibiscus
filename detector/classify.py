"""Use a trained detector to score a new image.

This is what the surveillance runtime calls per zone-crop: load the tiny saved
classifier once, then for each crop return target/not + a confidence score. The
score is what maps onto the 'confident' vs 'candidate' tiers the agent already
understands.
"""

import os

import joblib

from detector.embed import embed

_DEFAULT_MODEL = os.path.join(os.path.dirname(__file__), "model.joblib")


def load(model_path: str = _DEFAULT_MODEL) -> dict:
    """Load a trained detector bundle (the linear head + which embedder)."""
    return joblib.load(model_path)


def predict(image, bundle: dict) -> dict:
    """Score one image (PIL.Image or path). Returns is_target + score 0..1."""
    vec = embed(image).reshape(1, -1)
    score = float(bundle["clf"].predict_proba(vec)[0, 1])
    return {"is_target": score >= 0.5, "score": round(score, 3)}
