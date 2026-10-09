"""C7 — VLM change verification: a vision-language model as a second opinion.

For every candidate that is not UNCHANGED or already REJECTED, build a before/after image pair:

- REMOVED: the object's best crop in A + the same place seen by a B camera,
- ADDED: the same place seen by an A camera + the object's best crop in B,
- MOVED / REPLACED: the best crops from A and from B.

The "same place" view comes from the C6 cameras that saw the spot (else the camera with the best
view of it): the object's points are projected into that frame and the region is cropped.
The VLM answers a fixed question with JSON; a confident "no change here" rejects the candidate.
The VLM never adds or confirms changes — geometry stays in charge.
"""

from __future__ import annotations

import textwrap

import cv2
import numpy as np

from changedet import llm
from changedet.core import geometry as geo
from changedet.core.cache import cached_stage, exists, load_ply, run_dir, run_path, save_json
from changedet.core.config import load_run_config
from changedet.core.logging import get_logger
from changedet.core.types import CameraFrame, Change, ChangeType, Confidence, Object3D
from changedet.stages.c5_match import CHANGES_RAW, load_changes

log = get_logger("c7")

CHANGES_VERIFIED = "changes/changes_verified.json"
CHANGES_VIS = "changes/changes_vis.json"

# JSON schema for structured output. `same_object` is a string enum rather than bool|null so the
# schema stays within plain types; it is mapped back to True/False/None.
SCHEMA = {
    "type": "object",
    "properties": {
        "same_object": {"type": "string", "enum": ["yes", "no", "not_applicable"]},
        "change_present": {"type": "boolean"},
        "label": {"type": "string"},
        "short_description": {"type": "string"},
        "confidence": {"type": "number"},
    },
    "required": ["same_object", "change_present", "label", "short_description", "confidence"],
    "additionalProperties": False,
}

_INTRO = (
    "You are checking a candidate change found by a 3D change-detection system that compared "
    "two recordings of the same room made at different times. The system can be wrong: "
    "detectors mislabel things, and the second camera may have looked from a different angle. "
    "Image 1 is from the FIRST (before) recording, image 2 from the SECOND (after) recording.\n\n"
)
_QUESTIONS = {
    ChangeType.REMOVED: (
        "Image 1 shows the object the system calls '{label}'. Image 2 shows the same location "
        "in the second recording. Claimed change: this object was REMOVED.\n"
        "Is the object from image 1 really gone from that place in image 2?"
    ),
    ChangeType.ADDED: (
        "Image 2 shows the object the system calls '{label}'. Image 1 shows the same location "
        "in the first recording. Claimed change: this object was ADDED.\n"
        "Was this object really absent from that place in image 1?"
    ),
    ChangeType.MOVED: (
        "Both images show the object the system calls '{label}', before and after. Claimed "
        "change: the same object was MOVED to a different place.\n"
        "Do the two images show the same physical object?"
    ),
    ChangeType.REPLACED: (
        "Image 1 shows '{old}' and image 2 shows '{new}' at the same place. Claimed change: the "
        "first object was REPLACED by a different one.\nAre these really different objects?"
    ),
}
_ANSWER = (
    "\n\nAnswer with JSON only:\n"
    "- change_present: true if the claimed change really happened, false if it did not "
    "(e.g. the object is clearly still there, or it was never a real object).\n"
    "- same_object: for a move or replacement, 'yes' if both images show the same physical "
    "object, 'no' if not; otherwise 'not_applicable'.\n"
    "- label: a short, specific name for the object (e.g. 'black backpack').\n"
    "- short_description: one sentence on what you see.\n"
    "- confidence: 0 to 1. If image 2 (or image 1) does not clearly show the location — wrong "
    "angle, blur, occlusion — use a low confidence rather than guessing."
)


def build_prompt(change: Change) -> str:
    if change.type is ChangeType.REPLACED:
        old, new = change.label.split("->")
        question = _QUESTIONS[change.type].format(old=old, new=new)
    else:
        question = _QUESTIONS[change.type].format(label=change.label)
    return _INTRO + question + _ANSWER


def normalise_verdict(raw: dict) -> dict:
    """Map the VLM JSON to the spec's verdict shape (same_object: bool | None)."""
    same = {"yes": True, "no": False}.get(str(raw.get("same_object")).lower())
    return {
        "same_object": same,
        "change_present": bool(raw.get("change_present", True)),
        "label": str(raw.get("label", "")).strip(),
        "short_description": str(raw.get("short_description", "")).strip(),
        "confidence": float(np.clip(float(raw.get("confidence", 0.0)), 0.0, 1.0)),
    }


def decide(change: Change, verdict: dict, reject_conf: float) -> None:
    """Apply a verdict: reject on a confident "no change"; keep a more specific label."""
    rejected = not verdict["change_present"] and verdict["confidence"] >= reject_conf
    if rejected:
        change.confidence = Confidence.REJECTED
    refined = verdict["label"]
    base = change.label.split("->")[-1]
    if refined and refined.lower() != base.lower() and base.lower() in refined.lower():
        verdict["refined_label"] = refined  # e.g. "chair" -> "office chair"
    verdict["decision"] = "rejected" if rejected else "kept"
    change.vlm_verdict = verdict


# --------------------------------------------------------------------------------------------
# Evidence images
# --------------------------------------------------------------------------------------------


def _limit(image: np.ndarray, max_side: int) -> np.ndarray:
    h, w = image.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1:
        return image
    return cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)


def projected_box(points: np.ndarray, cam: CameraFrame, width: int, height: int):
    """(x0, y0, x1, y1, fraction of points in view) of the points projected into a camera."""
    uv, z = geo.project(points, cam.K, cam.T_world_cam)
    front = z > 0.05
    if not front.any():
        return None
    uv = uv[front]
    inside = (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
    if inside.mean() < 0.5 or inside.sum() < 10:
        return None
    x0, y0 = uv[inside].min(axis=0)
    x1, y1 = uv[inside].max(axis=0)
    return x0, y0, x1, y1, float(inside.sum() / len(points))


def view_of_place(
    run: str, points: np.ndarray, cameras: list[CameraFrame], preferred: list[int], pad: float
) -> tuple[np.ndarray | None, int | None]:
    """Crop of the region where ``points`` project, from the best camera of a session.

    ``preferred`` (C6 cameras that saw the spot) are tried first; otherwise the camera that
    sees most of the points with the largest projection is used.
    """
    by_index = {c.frame.index: c for c in cameras}
    candidates = [by_index[i] for i in preferred if i in by_index] or cameras
    best = None
    for cam in candidates:
        w, h = int(round(2 * cam.K[0, 2] + 1)), int(round(2 * cam.K[1, 2] + 1))
        box = projected_box(points, cam, w, h)
        if box is None:
            continue
        x0, y0, x1, y1, frac = box
        score = frac * (x1 - x0) * (y1 - y0)
        if best is None or score > best[0]:
            best = (score, cam, box)
        if preferred and best is not None:
            break  # first preferred camera that sees it (they are ordered by evidence)
    if best is None:
        return None, None
    _, cam, (x0, y0, x1, y1, _) = best
    image = cv2.imread(str(run_dir(run) / cam.frame.image_path))[..., ::-1]
    h, w = image.shape[:2]
    px, py = pad * (x1 - x0) + 20, pad * (y1 - y0) + 20
    x0, y0 = int(max(0, x0 - px)), int(max(0, y0 - py))
    x1, y1 = int(min(w, x1 + px)), int(min(h, y1 + py))
    return image[y0:y1, x0:x1].copy(), cam.frame.index


def evidence_pair(run: str, change: Change, objects: dict[str, Object3D], recon, cfg):
    """(before RGB, after RGB, captions) for a change, or None if a view is missing."""

    def crop(obj_id: str) -> np.ndarray:
        return cv2.imread(str(run_dir(run) / objects[obj_id].best_crop_path))[..., ::-1]

    def place(obj_id: str, session: str, check: str):
        points, _ = load_ply(run_path(run, objects[obj_id].points_path))
        vis = (change.visibility or {}).get(check, {})
        preferred = vis.get("free_cameras", []) + vis.get("occupied_cameras", [])
        return view_of_place(run, points, recon.cameras[session], preferred, cfg.crop_pad)

    if change.type is ChangeType.REMOVED:
        after, frame = place(change.object_a, "B", "old_location")
        pair = (
            crop(change.object_a),
            after,
            (f"BEFORE: {change.object_a} (best view)", f"AFTER: same place, B frame {frame}"),
        )
    elif change.type is ChangeType.ADDED:
        before, frame = place(change.object_b, "A", "new_location")
        pair = (
            before,
            crop(change.object_b),
            (f"BEFORE: same place, A frame {frame}", f"AFTER: {change.object_b} (best view)"),
        )
    else:
        pair = (
            crop(change.object_a),
            crop(change.object_b),
            (f"BEFORE: {change.object_a}", f"AFTER: {change.object_b}"),
        )
    if pair[0] is None or pair[1] is None:
        return None
    return _limit(pair[0], cfg.max_side), _limit(pair[1], cfg.max_side), pair[2]


def evidence_image(
    before: np.ndarray,
    after: np.ndarray,
    captions,
    change: Change,
    verdict: dict | None,
    height: int = 360,
) -> np.ndarray:
    """Side-by-side before/after with the claim and the verdict written underneath (RGB)."""

    def fit(img, min_width: int = 340):
        h, w = img.shape[:2]
        img = cv2.resize(img, (max(1, round(w * height / h)), height))
        if img.shape[1] < min_width:  # leave room for the caption
            pad = np.full((height, min_width - img.shape[1], 3), 255, np.uint8)
            img = np.hstack([img, pad])
        return img

    panels = [fit(before), fit(after)]
    width = sum(p.shape[1] for p in panels) + 10
    canvas = np.full((height + 150, max(width, 640), 3), 255, np.uint8)
    x = 0
    for panel, caption in zip(panels, captions, strict=True):
        canvas[30 : 30 + height, x : x + panel.shape[1]] = panel
        cv2.putText(canvas, caption, (x + 4, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
        x += panel.shape[1] + 10
    lines = [
        f"{change.id}: {change.type.value} {change.label} (geometry: {change.confidence.value})"
    ]
    if verdict is None:
        lines.append("VLM: no verdict (not run)")
    else:
        lines.append(
            f"VLM: change_present={verdict['change_present']} "
            f"conf={verdict['confidence']:.2f} -> {verdict['decision']}"
        )
        lines += textwrap.wrap(f'"{verdict["short_description"]}"', 95)[:2]
    for k, line in enumerate(lines):
        cv2.putText(
            canvas, line, (6, height + 58 + 26 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1
        )
    return canvas


# --------------------------------------------------------------------------------------------
# Stage
# --------------------------------------------------------------------------------------------


@cached_stage(CHANGES_VERIFIED, load=lambda run: load_changes(run, CHANGES_VERIFIED))
def verify_changes(run: str, force: bool = False) -> list[Change]:
    """Second opinion from a VLM on every open candidate; may only reject.

    Args:
        run: run name; needs C5 output (uses C6 output if present).
        force: recompute even if ``changes/changes_verified.json`` exists (cached VLM
            answers are reused).
    """
    from changedet.stages.c2_reconstruct import load_reconstruction
    from changedet.stages.c4_fuse import load_objects

    cfg = load_run_config(run).vlm
    source = CHANGES_VIS if exists(run, CHANGES_VIS) else CHANGES_RAW
    changes = load_changes(run, source)
    objects = {o.id: o for s in ("A", "B") for o in load_objects(run)[s]}
    recon = load_reconstruction(run)
    out_dir = run_path(run, "viz/c7_evidence")
    out_dir.mkdir(parents=True, exist_ok=True)

    available = cfg.enabled and llm.credentials_available(cfg.provider)
    if not cfg.enabled:
        log.warning("vlm.enabled is false: changes passed through unchanged")
    elif not available:
        log.warning(
            "No %s credentials found (e.g. ANTHROPIC_API_KEY): changes passed through "
            "unchanged; evidence images are still written",
            cfg.provider,
        )
    calls = 0
    for change in changes:
        if change.type is ChangeType.UNCHANGED or change.confidence is Confidence.REJECTED:
            continue
        pair = evidence_pair(run, change, objects, recon, cfg)
        if pair is None:
            log.warning("%s: no usable view of the place; not verified", change.id)
            continue
        before, after, captions = pair
        verdict = None
        if available and calls < cfg.max_calls:
            try:
                raw = llm.ask_json(
                    cfg,
                    build_prompt(change),
                    SCHEMA,
                    [llm.encode_jpeg(before), llm.encode_jpeg(after)],
                    run_path(run, "changes/vlm_cache"),
                )
                calls += 1
                verdict = normalise_verdict(raw)
                verdict["model"] = cfg.model
                decide(change, verdict, cfg.reject_conf)
                log.info(
                    "%s %-8s %-14s VLM: present=%s conf=%.2f -> %s  (%s)",
                    change.id,
                    change.type.value,
                    change.label,
                    verdict["change_present"],
                    verdict["confidence"],
                    verdict["decision"],
                    verdict["short_description"],
                )
            except llm.LLMUnavailable as e:
                log.warning("VLM unavailable (%s): remaining changes passed through", e)
                available = False
            except (llm.LLMRefused, ValueError) as e:
                log.warning("%s: no usable VLM verdict (%s)", change.id, e)
        elif available:
            log.warning("%s: vlm.max_calls=%d reached, not verified", change.id, cfg.max_calls)
        image = evidence_image(before, after, captions, change, verdict)
        cv2.imwrite(str(out_dir / f"{change.id}.jpg"), image[..., ::-1])

    n_rejected = sum(
        c.vlm_verdict is not None and c.vlm_verdict.get("decision") == "rejected" for c in changes
    )
    log.info(
        "C7: %d VLM verdicts (new or cached), %d rejected by the VLM; evidence in %s",
        calls,
        n_rejected,
        out_dir,
    )
    save_json(run_path(run, CHANGES_VERIFIED), changes)  # written last: marks the stage as done
    return changes
