"""DINOv2 image embeddings — the frozen feature extractor.

DINOv2 (a self-supervised Vision Transformer from Meta) turns any image into a
fixed-length vector that captures *what's in it*, without us training the big
model at all. We just call it. A handful of example images, embedded, are then
enough to train a tiny classifier (see train.py) to tell "background" from
"target" — no bounding boxes, no GPU, no fine-tuning.

We use the small variant (~21M params, 384-dim) so it runs fast on a laptop CPU.
"""

import functools
import os

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

_MODEL_NAME = "facebook/dinov2-small"


@functools.lru_cache(maxsize=1)
def _load():
    """Load the model + processor once and cache them for the process."""
    processor = AutoImageProcessor.from_pretrained(_MODEL_NAME)
    model = AutoModel.from_pretrained(_MODEL_NAME)
    model.eval()
    return processor, model


@torch.no_grad()
def embed(image) -> np.ndarray:
    """Embed one image (a PIL.Image or a path) into an L2-normalized vector."""
    if isinstance(image, (str, bytes, os.PathLike)):
        image = Image.open(image)
    image = image.convert("RGB")

    processor, model = _load()
    inputs = processor(images=image, return_tensors="pt")
    out = model(**inputs)
    # pooler_output is the CLS token after a final norm; fall back to CLS if absent.
    feat = out.pooler_output if out.pooler_output is not None else out.last_hidden_state[:, 0]
    vec = feat[0].numpy().astype(np.float32)
    return vec / (np.linalg.norm(vec) + 1e-8)


def embed_paths(paths: list[str]) -> np.ndarray:
    """Embed many image paths into a (N, dim) matrix."""
    return np.stack([embed(p) for p in paths])
