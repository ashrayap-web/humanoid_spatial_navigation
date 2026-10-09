"""Lazy, cached loaders for the heavy models (each loaded once per process).

Checkpoints are Hugging Face repos cached under ``checkpoints/hf`` (git-ignored); fetch them
with ``scripts/download_models.sh``.
"""

from __future__ import annotations

import functools
import gc
from pathlib import Path

from changedet.core.logging import get_logger

log = get_logger("models")

CHECKPOINT_DIR = Path(__file__).resolve().parents[1] / "checkpoints" / "hf"


def resolve_device(preference: str = "auto") -> str:
    """``"auto"`` -> ``"cuda"`` if available, else ``"cpu"``."""
    import torch

    if preference == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return preference


def is_downloaded(model_id: str) -> bool:
    """True if the Hugging Face repo ``model_id`` is in the local checkpoint cache."""
    return (CHECKPOINT_DIR / f"models--{model_id.replace('/', '--')}").exists()


@functools.cache
def grounding_dino(model_id: str, device: str):
    """(processor, model) for open-vocabulary box detection."""
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    log.info("Loading %s on %s", model_id, device)
    processor = AutoProcessor.from_pretrained(model_id, cache_dir=CHECKPOINT_DIR)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id, cache_dir=CHECKPOINT_DIR)
    return processor, model.to(device).eval()


@functools.cache
def sam2(model_id: str, device: str):
    """(processor, model) for box-prompted segmentation."""
    from transformers import Sam2Model, Sam2Processor

    log.info("Loading %s on %s", model_id, device)
    processor = Sam2Processor.from_pretrained(model_id, cache_dir=CHECKPOINT_DIR)
    model = Sam2Model.from_pretrained(model_id, cache_dir=CHECKPOINT_DIR)
    return processor, model.to(device).eval()


@functools.cache
def clip(model_id: str, device: str):
    """(processor, model) for CLIP image/text embeddings (semantic similarity)."""
    from transformers import CLIPModel, CLIPProcessor

    log.info("Loading %s on %s", model_id, device)
    processor = CLIPProcessor.from_pretrained(model_id, cache_dir=CHECKPOINT_DIR)
    model = CLIPModel.from_pretrained(model_id, cache_dir=CHECKPOINT_DIR)
    return processor, model.to(device).eval()


@functools.cache
def dinov2(model_id: str, device: str):
    """(processor, model) for DINOv2 image embeddings (appearance similarity)."""
    from transformers import AutoImageProcessor, AutoModel

    log.info("Loading %s on %s", model_id, device)
    processor = AutoImageProcessor.from_pretrained(model_id, cache_dir=CHECKPOINT_DIR)
    model = AutoModel.from_pretrained(model_id, cache_dir=CHECKPOINT_DIR)
    return processor, model.to(device).eval()


def embed_images(images: list, clip_id: str, dino_id: str, device: str, batch_size: int = 32):
    """L2-normalised (CLIP, DINOv2) embeddings, each (N, D) float32, for RGB uint8 images."""
    import numpy as np
    import torch

    clip_proc, clip_model = clip(clip_id, device)
    dino_proc, dino_model = dinov2(dino_id, device)
    clip_out, dino_out = [], []
    with torch.inference_mode():
        for start in range(0, len(images), batch_size):
            batch = images[start : start + batch_size]
            x = clip_proc(images=batch, return_tensors="pt").to(device)
            clip_out.append(clip_model.get_image_features(**x).pooler_output.float().cpu())
            x = dino_proc(images=batch, return_tensors="pt").to(device)
            dino_out.append(dino_model(**x).pooler_output.float().cpu())

    def normalise(chunks):
        if not chunks:
            return np.zeros((0, 0), np.float32)
        e = torch.cat(chunks).numpy()
        return (e / np.linalg.norm(e, axis=1, keepdims=True)).astype(np.float32)

    return normalise(clip_out), normalise(dino_out)


def unload() -> None:
    """Drop all cached models and free GPU memory (between stages)."""
    import torch

    for loader in (grounding_dino, sam2, clip, dinov2):
        loader.cache_clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def download(model_ids: list[str]) -> None:
    """Fetch the given Hugging Face repos into ``checkpoints/hf``."""
    from huggingface_hub import snapshot_download

    for model_id in model_ids:
        log.info("Downloading %s", model_id)
        snapshot_download(model_id, cache_dir=CHECKPOINT_DIR)


if __name__ == "__main__":
    from changedet.core.config import load_config
    from changedet.core.logging import setup_logging

    setup_logging()
    cfg = load_config()
    download(
        [
            cfg.detect.detector_model,
            cfg.detect.segmenter_model,
            cfg.fusion.clip_model,
            cfg.fusion.dino_model,
        ]
    )
