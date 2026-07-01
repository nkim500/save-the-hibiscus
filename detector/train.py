"""Few-shot training: turn a folder of example images into a target detector.

The pipeline the copilot will eventually orchestrate, proven here as plain code:
  1. Embed every example image with DINOv2 (embed.py).
  2. Train a logistic-regression classifier on "target" vs "background".
  3. Evaluate on a held-out split and report accuracy.
  4. Save the classifier (the big DINOv2 model is frozen and reused, so the
     saved artifact is tiny — just the linear head).

No bounding boxes. No GPU. Seconds to train. ~10-20 images per class is enough
to get started; more improves it.

CLI:  uv run --group detector python -m detector.train data/squirrel data/background
"""

import glob
import os
import sys

import joblib
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from detector.embed import embed_paths

_IMG_EXT = (".jpg", ".jpeg", ".png", ".webp")
_DEFAULT_OUT = os.path.join(os.path.dirname(__file__), "model.joblib")


def _images_in(directory: str) -> list[str]:
    return sorted(
        p for p in glob.glob(os.path.join(directory, "*")) if p.lower().endswith(_IMG_EXT)
    )


def train(positive_dir: str, negative_dir: str, out_path: str = _DEFAULT_OUT) -> dict:
    """Train a background-vs-target classifier from two folders of images."""
    pos = _images_in(positive_dir)
    neg = _images_in(negative_dir)
    if len(pos) < 4 or len(neg) < 4:
        raise ValueError(
            f"Need >=4 images per class (got {len(pos)} positive, {len(neg)} negative)."
        )

    paths = pos + neg
    y = np.array([1] * len(pos) + [0] * len(neg))
    print(f"embedding {len(paths)} images with DINOv2...")
    X = embed_paths(paths)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
    # Small MLP head on top of the frozen DINOv2 features. Easy to grow later
    # (more layers/units) without touching the embedding or serving code.
    clf = MLPClassifier(hidden_layer_sizes=(128,), max_iter=1000, random_state=0).fit(X_tr, y_tr)
    acc = accuracy_score(y_te, clf.predict(X_te))

    joblib.dump({"clf": clf, "embed_model": "facebook/dinov2-small"}, out_path)
    return {
        "n_positive": len(pos),
        "n_negative": len(neg),
        "holdout_accuracy": round(float(acc), 3),
        "model_path": out_path,
    }


if __name__ == "__main__":
    pos_dir, neg_dir = sys.argv[1], sys.argv[2]
    metrics = train(pos_dir, neg_dir)
    print("\n=== training complete ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
