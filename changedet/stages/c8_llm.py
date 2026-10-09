"""C8 full — spatial facts per change, and sentences written by an LLM from those facts only.

Facts come from the 3D map: what the object rests on, the nearest unchanged landmark objects (with
"on" / "next to" / distance and a direction — left, right, in front of, behind — as seen from where
the recordings were made), the distance to the nearest wall, its size, how far it moved, and the
C6/C7 evidence behind its confidence. The LLM gets these as a compact list and must use nothing
else. Every sentence is checked: each number in it must appear in that change's facts and the
object must be named; otherwise that change keeps its template sentence.
"""

from __future__ import annotations

import re

import numpy as np
from scipy.spatial import cKDTree

from changedet import llm
from changedet.core.logging import get_logger
from changedet.core.types import Change, ChangeType, Confidence, Object3D
from changedet.stages.c8_describe import box_gap, evidence, vlm_note, xy_overlap

log = get_logger("c8")

VIEWPOINT = "as seen from where the recordings were made"

SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "sentences": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"id": {"type": "string"}, "sentence": {"type": "string"}},
                "required": ["id", "sentence"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["summary", "sentences"],
    "additionalProperties": False,
}


# --------------------------------------------------------------------------------------------
# Spatial facts (pure, unit-tested)
# --------------------------------------------------------------------------------------------


def direction(
    obj_centre: np.ndarray,
    landmark_centre: np.ndarray,
    viewpoint: np.ndarray,
    min_offset: float = 0.15,
) -> str | None:
    """Where the object is relative to a landmark, seen from ``viewpoint`` (all world xy/z-up).

    Forward = from the viewpoint towards the landmark. Returns "to the left of",
    "to the right of", "in front of" (closer to the viewer), "behind", or None if the offset is
    smaller than ``min_offset``.
    """
    forward = landmark_centre[:2] - viewpoint[:2]
    norm = np.linalg.norm(forward)
    if norm < 1e-6:
        return None
    forward = forward / norm
    right = np.array([forward[1], -forward[0]])  # z-up: right = forward rotated by -90 deg
    offset = obj_centre[:2] - landmark_centre[:2]
    along, side = float(offset @ forward), float(offset @ right)
    if max(abs(along), abs(side)) < min_offset:
        return None
    if abs(side) >= abs(along):
        return "to the right of" if side > 0 else "to the left of"
    return "behind" if along > 0 else "in front of"


def relation(obj: Object3D, lm: Object3D, cfg) -> tuple[str, float]:
    """("on" | "next to" | "about d m from", gap in metres) between an object and a landmark."""
    lo, hi = obj.bbox_min, obj.bbox_max
    gap = box_gap(lo, hi, lm.bbox_min, lm.bbox_max)
    if (
        xy_overlap(lo, hi, lm.bbox_min, lm.bbox_max) > 0.5
        and lm.bbox_min[2] + cfg.on_tol <= lo[2] <= lm.bbox_max[2] + cfg.on_tol
    ):
        return "on", gap
    if gap < cfg.next_to_dist:
        return "next to", gap
    return f"about {gap:.1f} m from", gap


def location_facts(
    obj: Object3D,
    landmarks: list[Object3D],
    viewpoint: np.ndarray,
    wall_tree: cKDTree | None,
    points: np.ndarray | None,
    cfg,
) -> dict:
    """Rounded, coordinate-free description of where an object is."""
    near = sorted(
        landmarks, key=lambda lm: box_gap(obj.bbox_min, obj.bbox_max, lm.bbox_min, lm.bbox_max)
    )
    near = [
        lm
        for lm in near
        if box_gap(obj.bbox_min, obj.bbox_max, lm.bbox_min, lm.bbox_max) < cfg.near_dist
    ]
    facts: dict = {"size_m": [round(float(x), 1) for x in obj.bbox_max - obj.bbox_min]}
    landmark_facts = []
    for lm in near[: cfg.landmark_k]:
        rel, _ = relation(obj, lm, cfg)
        name = f"another {lm.label}" if lm.label == obj.label else f"the {lm.label}"
        item = {"landmark": name, "relation": rel}
        if rel != "on":
            d = direction(obj.centroid, lm.centroid, viewpoint)
            if d:
                item["direction"] = f"{d} {name} ({VIEWPOINT})"
        landmark_facts.append(item)
    on = next((f["landmark"] for f in landmark_facts if f["relation"] == "on"), None)
    if on:
        facts["rests_on"] = on
    elif obj.bbox_min[2] < cfg.on_tol:
        facts["rests_on"] = "the floor"
    else:
        facts["height_above_floor_m"] = round(float(obj.bbox_min[2]), 1)
    facts["landmarks"] = landmark_facts
    if wall_tree is not None:
        xy = (points if points is not None else obj.centroid[None])[:, :2]
        facts["nearest_wall_m"] = round(float(wall_tree.query(xy)[0].min()), 1)
    return facts


def change_facts(change: Change, location: dict[str, dict], cfg_match) -> dict:
    """Everything the writer may say about one change."""
    vlm = change.vlm_verdict or {}
    facts = {
        "id": change.id,
        "type": change.type.value,
        "object": change.label,
        "confidence": change.confidence.value,
        "evidence": " ".join(x for x in (evidence(change), vlm_note(change)) if x),
    }
    if vlm.get("label"):
        facts["visual_check_name"] = vlm["label"]
    if change.object_a:
        facts["before"] = location[change.object_a]
    if change.object_b:
        facts["after"] = location[change.object_b]
    if change.type is ChangeType.MOVED:
        facts["moved_m"] = round(float(np.linalg.norm(change.translation)), 1)
        facts["turned_deg"] = round(abs(float(change.rotation_deg)))
        facts["moved_counts_as_turn_in_place"] = facts["moved_m"] < cfg_match.move_thresh_m
    return facts


def facts_text(facts: dict) -> str:
    """Compact human-readable fact list (also shown in report.md)."""
    lines = [
        f"- {facts['id']}: {facts['type'].upper()} {facts['object']}"
        + (
            f" (the visual check calls it: {facts['visual_check_name']})"
            if "visual_check_name" in facts
            else ""
        )
        + f". Confidence: {facts['confidence']}."
    ]
    if facts["evidence"]:
        lines.append(f"  Evidence: {facts['evidence']}")
    for side in ("before", "after"):
        if side not in facts:
            continue
        f = facts[side]
        parts = []
        if "rests_on" in f:
            parts.append(f"rests on {f['rests_on']}")
        else:
            parts.append(f"{f['height_above_floor_m']} m above the floor")
        for lm in f["landmarks"]:
            if lm["relation"] == "on":
                continue
            text = f"{lm['relation']} {lm['landmark']}"
            if "direction" in lm:
                text += f", {lm['direction']}"
            parts.append(text)
        if "nearest_wall_m" in f:
            parts.append(f"{f['nearest_wall_m']} m from the nearest wall")
        parts.append("size {} x {} x {} m".format(*f["size_m"]))
        lines.append(f"  {side.capitalize()}: " + "; ".join(parts) + ".")
    if "moved_m" in facts:
        if facts["moved_counts_as_turn_in_place"]:
            lines.append(f"  Turned {facts['turned_deg']} degrees in place.")
        else:
            lines.append(f"  Moved {facts['moved_m']} m and turned {facts['turned_deg']} degrees.")
    return "\n".join(lines)


# --------------------------------------------------------------------------------------------
# Writing and grounding check
# --------------------------------------------------------------------------------------------

_NUMBER = re.compile(r"(?<![\w.])\d+(?:\.\d+)?")


def numbers_in(text: str) -> set[str]:
    """Numbers in a text, normalised ("0.40" -> "0.4", "84%" -> "84")."""
    out = set()
    for m in _NUMBER.findall(text):
        x = float(m)
        out.add(str(int(x)) if x == int(x) else f"{x:g}")
    return out


def grounded(sentence: str, facts_block: str, names: list[str] | None) -> str | None:
    """None if the sentence only uses numbers from its facts and (if ``names`` is given) names
    the object, else the reason it fails."""
    extra = numbers_in(sentence) - numbers_in(facts_block)
    if extra:
        return f"numbers not in the facts: {', '.join(sorted(extra))}"
    if not names:
        return None
    words = sentence.lower()
    if not any(w in words for n in names for w in re.findall(r"[a-z]+", n.lower())):
        return "does not name the object"
    return None


def build_prompt(blocks: list[str], counts: dict, style: str) -> str:
    return (
        "You write the change report for a system that compared two 3D scans of the same room, "
        "recorded at different times (the 'first' and the 'second' recording).\n\n"
        "Facts about each change:\n" + "\n".join(blocks) + "\n\n"
        f"Overall: {counts['confirmed']} confirmed change(s), {counts['unverified']} unverified, "
        f"{counts['rejected']} false alarm(s) rejected, "
        f"{counts['unchanged']} unchanged objects.\n\n"
        "Write:\n"
        "1. For every change id above, exactly one sentence (two short ones at most) saying what "
        "changed and where, for someone who knows the room. Name the object and place it using "
        "the landmarks, the surface it rests on, or the nearest wall.\n"
        "2. A summary of 2-3 sentences for the whole comparison.\n\n"
        "Rules:\n"
        "- Use ONLY the facts above. Do not invent objects, colours, materials, distances or "
        "causes. Any number you write must appear in the facts, unchanged.\n"
        "- Match the confidence: state confirmed changes plainly and end with their key "
        "evidence in a short clause (e.g. how much of the space was seen empty); for unverified "
        "ones say the change could not be verified and give the reason from its evidence.\n"
        "- When you give a left/right/in front/behind direction, keep the 'as seen from where "
        "the recordings were made' qualifier (you may shorten it).\n"
        "- Do not mention change ids, coordinates or internal names inside the sentences.\n"
        f"- Style: {style}, plain English.\n"
        "Return JSON only."
    )


def write_with_llm(
    main: list[Change], facts: dict[str, dict], counts: dict, cfg, cache_dir
) -> tuple[dict[str, str], str | None, list[str]]:
    """LLM sentences for ``main`` changes that pass the grounding check, a grounded summary
    (or None), and warnings. Raises llm.LLMUnavailable if there is no provider/credentials."""
    blocks = {c.id: facts_text(facts[c.id]) for c in main}
    prompt = build_prompt(list(blocks.values()), counts, cfg.language_style)
    raw = llm.ask_json(cfg.llm, prompt, SCHEMA, [], cache_dir)
    written = {item["id"]: item["sentence"].strip() for item in raw.get("sentences", [])}
    sentences, warnings = {}, []
    for c in main:
        text = written.get(c.id)
        names = [c.label] + [facts[c.id].get("visual_check_name", "")]
        problem = "missing" if not text else grounded(text, blocks[c.id], names)
        if problem:
            warnings.append(f"{c.id}: LLM sentence rejected ({problem}); used the template")
            log.warning(warnings[-1])
        else:
            sentences[c.id] = text
    summary = raw.get("summary", "").strip()
    count_text = " ".join(str(v) for v in counts.values())
    problem = (
        grounded(summary, "\n".join(blocks.values()) + " " + count_text, None)
        if summary
        else "missing"
    )
    if problem:
        warnings.append(f"summary: LLM summary rejected ({problem}); used the template")
        log.warning(warnings[-1])
        summary = None
    return sentences, summary, warnings


def counts_of(changes: list[Change]) -> dict:
    real = [c for c in changes if c.type is not ChangeType.UNCHANGED]
    return {
        "confirmed": sum(c.confidence is Confidence.CONFIRMED for c in real),
        "unverified": sum(c.confidence is Confidence.UNVERIFIED for c in real),
        "rejected": sum(c.confidence is Confidence.REJECTED for c in real),
        "unchanged": len(changes) - len(real),
    }
