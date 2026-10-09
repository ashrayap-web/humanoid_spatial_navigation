"""C3 — 2D detection & segmentation (Grounding DINO boxes -> SAM 2 masks).

Per kept frame:

1. Grounding DINO scores every box against each vocabulary phrase separately (max token
   probability within the phrase) and takes the best phrase as the label. This avoids the
   run-together labels ("chair office chair") of the library's phrase decoding. Structural
   classes are in the prompt too, so e.g. a window competes with "picture frame".
2. SAM 2 turns the boxes into masks.
3. Filters: score, mask size, cut off by the image border, mask NMS, structural classes.
4. Masks are made exclusive: a pixel belongs to the smallest mask covering it, so an object
   lying on a larger one (a bag on the bed) stays a separate segment.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass

import cv2
import numpy as np

from changedet import models
from changedet.core.cache import cached_stage, load_json, run_dir, run_path, save_json
from changedet.core.config import load_run_config
from changedet.core.logging import get_logger, timed
from changedet.core.types import Detection2D, Frame
from changedet.stages.c1_frames import load_frames

log = get_logger("c3")

DETECTIONS_JSON = "detections/detections.json"
SESSIONS = ("A", "B")


# --------------------------------------------------------------------------------------------
# Pure helpers (unit-tested)
# --------------------------------------------------------------------------------------------


def phrase_spans(input_ids: list[int], sep_id: int, n_phrases: int) -> list[list[int]]:
    """Token positions of each phrase in a ``"[CLS] a b . c . d . [SEP]"`` prompt."""
    spans, current = [], []
    for pos, token in enumerate(input_ids[1:], start=1):  # skip [CLS]
        if token == sep_id:
            spans.append(current)
            current = []
        else:
            current.append(pos)
    spans = [s for s in spans if s][:n_phrases]
    if len(spans) != n_phrases:
        raise ValueError(f"Found {len(spans)} phrases in the prompt, expected {n_phrases}")
    return spans


def assign_labels(logits: np.ndarray, spans: list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
    """Per query: (best phrase index, its score), where a phrase's score is the max sigmoid
    probability over its tokens. ``logits`` is (queries, tokens)."""
    probs = 1.0 / (1.0 + np.exp(-logits))
    per_phrase = np.stack([probs[:, span].max(axis=1) for span in spans], axis=1)
    best = per_phrase.argmax(axis=1)
    return best, per_phrase[np.arange(len(best)), best]


def cxcywh_to_xyxy(boxes: np.ndarray, width: int, height: int) -> np.ndarray:
    """Normalised (cx, cy, w, h) -> pixel (x0, y0, x1, y1), clipped to the image."""
    cx, cy, w, h = boxes.T
    xyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    xyxy *= [width, height, width, height]
    return np.clip(xyxy, 0, [width, height, width, height])


def border_fraction(mask: np.ndarray) -> float:
    """Fraction of the mask's boundary that lies on the image border (1.0 = all cut off)."""
    padded = np.pad(mask, 1, constant_values=False)
    inner = padded[1:-1, 1:-1]
    interior = padded[:-2, 1:-1] & padded[2:, 1:-1] & padded[1:-1, :-2] & padded[1:-1, 2:]
    boundary = inner & ~interior  # mask pixels with an outside 4-neighbour (incl. off-image)
    on_border = np.zeros_like(mask)
    on_border[0, :] = on_border[-1, :] = on_border[:, 0] = on_border[:, -1] = True
    n = boundary.sum()
    return float((boundary & on_border).sum() / n) if n else 0.0


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def mask_nms(masks: list[np.ndarray], scores: np.ndarray, iou_thresh: float) -> list[int]:
    """Indices kept by greedy NMS on mask IoU, highest score first."""
    keep: list[int] = []
    for i in np.argsort(-np.asarray(scores)):
        if all(mask_iou(masks[i], masks[j]) <= iou_thresh for j in keep):
            keep.append(int(i))
    return keep


def make_exclusive(masks: list[np.ndarray]) -> list[np.ndarray]:
    """Give every pixel to the smallest mask that covers it (objects on top of larger ones win)."""
    order = np.argsort([m.sum() for m in masks])  # smallest first
    taken = np.zeros_like(masks[0]) if masks else None
    out: list[np.ndarray] = [None] * len(masks)  # type: ignore[list-item]
    for i in order:
        out[i] = masks[i] & ~taken
        taken |= masks[i]
    return out


def tight_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    """(x0, y0, x1, y1) of a non-empty mask, inclusive-exclusive."""
    ys, xs = np.nonzero(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


# --------------------------------------------------------------------------------------------
# Model inference
# --------------------------------------------------------------------------------------------


@dataclass
class FrameDetections:
    labels: list[str]
    scores: list[float]
    masks: list[np.ndarray]


def _detect_boxes(images: list[np.ndarray], phrases: list[str], cfg, device: str):
    """Grounding DINO on a batch of RGB images -> per image (boxes xyxy, scores, phrase index)."""
    import torch

    processor, model = models.grounding_dino(cfg.detector_model, device)
    with torch.inference_mode():
        inputs = processor(images=images, text=[phrases] * len(images), return_tensors="pt")
        outputs = model(**inputs.to(device))
    sep_id = processor.tokenizer.convert_tokens_to_ids(".")
    spans = phrase_spans(inputs.input_ids[0].tolist(), sep_id, len(phrases))
    results = []
    for k, image in enumerate(images):
        best, scores = assign_labels(outputs.logits[k].float().cpu().numpy(), spans)
        keep = scores >= cfg.min_score
        h, w = image.shape[:2]
        boxes = cxcywh_to_xyxy(outputs.pred_boxes[k].float().cpu().numpy()[keep], w, h)
        results.append((boxes, scores[keep], best[keep]))
    return results


def _segment(image: np.ndarray, boxes: np.ndarray, cfg, device: str) -> list[np.ndarray]:
    """SAM 2 masks (bool, image size) for the given boxes."""
    import torch

    processor, model = models.sam2(cfg.segmenter_model, device)
    with torch.inference_mode():
        inputs = processor(images=image, input_boxes=[boxes.tolist()], return_tensors="pt")
        outputs = model(**inputs.to(device), multimask_output=False)
        masks = processor.post_process_masks(
            outputs.pred_masks.cpu(), inputs["original_sizes"].cpu()
        )
    return [m[0].numpy().astype(bool) for m in masks[0]]


def detect_frame(
    image: np.ndarray, boxes, scores, phrase_idx, phrases, cfg, device
) -> FrameDetections:
    """Segment and filter one frame's boxes into exclusive, labelled object masks."""
    if len(boxes) == 0:
        return FrameDetections([], [], [])
    masks = _segment(image, boxes, cfg, device)
    labels = [phrases[i] for i in phrase_idx]
    ok = [
        i
        for i, m in enumerate(masks)
        if m.sum() >= cfg.min_mask_px and border_fraction(m) <= cfg.max_border_frac
    ]
    keep = [ok[i] for i in mask_nms([masks[i] for i in ok], scores[ok], cfg.mask_nms_iou)]
    keep = [i for i in keep if labels[i] not in cfg.structural]
    exclusive = make_exclusive([masks[i] for i in keep])
    out = FrameDetections([], [], [])
    for i, mask in zip(keep, exclusive, strict=True):
        if mask.sum() >= cfg.min_mask_px:
            out.labels.append(labels[i])
            out.scores.append(float(scores[i]))
            out.masks.append(mask)
    return out


# --------------------------------------------------------------------------------------------
# Stage
# --------------------------------------------------------------------------------------------


def load_detections(run: str) -> list[Detection2D]:
    """Read ``detections/detections.json``."""
    return [Detection2D.from_dict(d) for d in load_json(run_path(run, DETECTIONS_JSON))]


def overlay(image: np.ndarray, dets: FrameDetections) -> np.ndarray:
    """RGB image with coloured masks and labels (for the debug output)."""
    vis = image.copy()
    rng = np.random.default_rng(0)
    colours = rng.integers(60, 255, size=(max(1, len(dets.masks)), 3))
    for mask, colour in zip(dets.masks, colours, strict=False):
        vis[mask] = (0.45 * vis[mask] + 0.55 * colour).astype(np.uint8)
    for mask, label, score, colour in zip(
        dets.masks, dets.labels, dets.scores, colours, strict=False
    ):
        x0, y0, x1, y1 = tight_bbox(mask)
        cv2.rectangle(vis, (x0, y0), (x1, y1), colour.tolist(), 2)
        text = f"{label} {score:.2f}"
        cv2.putText(vis, text, (x0 + 3, y0 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
        cv2.putText(vis, text, (x0 + 3, y0 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return vis


def _detect_session(
    run: str, session: str, frames: list[Frame], cfg, device: str
) -> list[Detection2D]:
    phrases = list(cfg.vocabulary) + [s for s in cfg.structural if s not in cfg.vocabulary]
    mask_dir = run_path(run, f"detections/{session}/masks")
    viz_dir = run_path(run, f"viz/c3_overlays_{session}")
    for d in (mask_dir, viz_dir):
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)

    detections = []
    for start in range(0, len(frames), cfg.batch_size):
        batch = frames[start : start + cfg.batch_size]
        images = [cv2.imread(str(run_dir(run) / f.image_path))[..., ::-1].copy() for f in batch]
        for frame, image, (boxes, scores, idx) in zip(
            batch, images, _detect_boxes(images, phrases, cfg, device), strict=True
        ):
            dets = detect_frame(image, boxes, scores, idx, phrases, cfg, device)
            for k, (label, score, mask) in enumerate(
                zip(dets.labels, dets.scores, dets.masks, strict=True)
            ):
                relpath = f"detections/{session}/masks/{frame.index:06d}_{k:02d}.png"
                cv2.imwrite(str(run_dir(run) / relpath), mask.astype(np.uint8) * 255)
                detections.append(
                    Detection2D(session, frame.index, label, score, relpath, tight_bbox(mask))
                )
            if frame.index % cfg.debug_every == 0:
                vis = overlay(image, dets)
                cv2.imwrite(str(viz_dir / f"{frame.index:06d}.jpg"), vis[..., ::-1])

    counts: dict[str, int] = {}
    for d in detections:
        counts[d.label] = counts.get(d.label, 0) + 1
    log.info(
        "  %d detections in %d frames (%.1f / frame): %s",
        len(detections),
        len(frames),
        len(detections) / max(1, len(frames)),
        ", ".join(f"{k} {v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1])),
    )
    return detections


@cached_stage(DETECTIONS_JSON, load=load_detections)
def detect(run: str, force: bool = False) -> list[Detection2D]:
    """Open-vocabulary instance masks for every kept frame of both sessions.

    Args:
        run: run name; needs C1 output.
        force: recompute even if ``detections/detections.json`` exists.
    """
    cfg = load_run_config(run).detect
    if cfg.mode != "grounded_sam":
        raise NotImplementedError(f"detect.mode={cfg.mode}: TODO — only grounded_sam so far")
    device = models.resolve_device(cfg.device)
    frames = load_frames(run)
    detections = []
    for s in SESSIONS:
        with timed(f"C3 session {s}", log):
            detections += _detect_session(run, s, frames[s], cfg, device)
    models.unload()
    save_json(run_path(run, DETECTIONS_JSON), detections)  # written last: marks the stage done
    return detections
