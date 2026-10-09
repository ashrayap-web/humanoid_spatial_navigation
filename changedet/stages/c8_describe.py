"""C8 — Semantic description & report (template mode).

Every change gets one sentence built from its geometry: what it is, where it is (relative to an
unchanged landmark object: "on the bed", "next to the desk"), how far it moved, and how sure we
are (the C6 visibility evidence). The report lists confirmed and unverified changes; rejected
candidates go to an appendix.
"""

from __future__ import annotations

import numpy as np

from changedet.core.cache import cached_stage, exists, load_json, run_path, save_json
from changedet.core.config import load_run_config
from changedet.core.logging import get_logger
from changedet.core.types import Change, ChangeReport, ChangeType, Confidence, Object3D

log = get_logger("c8")

REPORT_JSON = "report/changes.json"
REPORT_MD = "report/report.md"
# Latest available change list first.
CHANGE_SOURCES = (
    "changes/changes_verified.json",
    "changes/changes_vis.json",
    "changes/changes_raw.json",
)


# --------------------------------------------------------------------------------------------
# Spatial relations (pure, unit-tested)
# --------------------------------------------------------------------------------------------


def box_gap(lo_a, hi_a, lo_b, hi_b) -> float:
    """Shortest distance between two axis-aligned boxes (0 if they touch or overlap)."""
    gap = np.maximum(0.0, np.maximum(lo_a - hi_b, lo_b - hi_a))
    return float(np.linalg.norm(gap))


def xy_overlap(lo_a, hi_a, lo_b, hi_b) -> float:
    """Fraction of box A's footprint covered by box B's footprint."""
    inter = np.prod(
        np.clip(np.minimum(hi_a[:2], hi_b[:2]) - np.maximum(lo_a[:2], lo_b[:2]), 0, None)
    )
    area = np.prod(np.maximum(hi_a[:2] - lo_a[:2], 1e-6))
    return float(inter / area)


def where(obj: Object3D, landmarks: list[Object3D], cfg) -> str:
    """Short location phrase, e.g. "on the bed", "on the floor next to the desk"."""
    lo, hi = obj.bbox_min, obj.bbox_max

    def name(lm: Object3D) -> str:
        return f"another {lm.label}" if lm.label == obj.label else f"the {lm.label}"

    for lm in landmarks:  # resting on it: footprint mostly over it, starting clearly above its base
        if (
            xy_overlap(lo, hi, lm.bbox_min, lm.bbox_max) > 0.5
            and lm.bbox_min[2] + cfg.on_tol <= lo[2] <= lm.bbox_max[2] + cfg.on_tol
        ):
            return f"on {name(lm)}"
    near = sorted(landmarks, key=lambda lm: box_gap(lo, hi, lm.bbox_min, lm.bbox_max))
    near = [lm for lm in near if box_gap(lo, hi, lm.bbox_min, lm.bbox_max) < cfg.near_dist]
    parts = ["on the floor"] if lo[2] < cfg.on_tol else []
    if near:
        gap = box_gap(lo, hi, near[0].bbox_min, near[0].bbox_max)
        parts.append(
            f"next to {name(near[0])}"
            if gap < cfg.next_to_dist
            else f"about {gap:.1f} m from {name(near[0])}"
        )
    return " ".join(parts)


# --------------------------------------------------------------------------------------------
# Sentences (pure, unit-tested)
# --------------------------------------------------------------------------------------------


def _recording(session: str) -> str:
    return "second" if session == "B" else "first"


def evidence(change: Change) -> str:
    """The visibility evidence behind the confidence, in words."""
    if not change.visibility:
        return ""
    name, vis = next(iter(change.visibility.items()))
    rec = _recording(vis["seen_from"])
    if change.confidence is Confidence.CONFIRMED:
        return (
            f"Confirmed: {vis['free_frac']:.0%} of that space was seen empty in the {rec} "
            "recording."
        )
    if change.confidence is Confidence.UNVERIFIED:
        return (
            f"Could not be verified: {vis['unknown_frac']:.0%} of that space was never clearly "
            f"observed in the {rec} recording."
        )
    return (
        f"Rejected: the {rec} recording still shows a surface in {vis['occupied_frac']:.0%} "
        f"of that space, so the object is most likely still there."
    )


# Verb per change type for (confirmed, unverified, rejected).
VERBS = {
    ChangeType.REMOVED: (
        "has been removed",
        "may have been removed",
        "seemed to have been removed",
    ),
    ChangeType.ADDED: ("has appeared", "may have appeared", "seemed to have appeared"),
    ChangeType.REPLACED: (
        "was replaced by",
        "may have been replaced by",
        "seemed to have been replaced by",
    ),
    ChangeType.MOVED: ("moved", "may have moved", "seemed to have moved"),
    "turned": ("was turned", "may have been turned", "seemed to have been turned"),
}
_CONF = {Confidence.CONFIRMED: 0, Confidence.UNVERIFIED: 1, Confidence.REJECTED: 2}


def sentence(change: Change, where_a: str, where_b: str, cfg_match) -> str:
    """One-sentence description of a change, worded by its confidence, plus the evidence."""
    a = f" {where_a}" if where_a else ""
    b = f" {where_b}" if where_b else ""
    k = _CONF[change.confidence]
    t = change.type
    if t is ChangeType.REMOVED:
        text = f"The {change.label}{(' that was' + a) if a else ''} {VERBS[t][k]}."
    elif t is ChangeType.ADDED:
        text = f"A new {change.label} {VERBS[t][k]}{b}."
    elif t is ChangeType.REPLACED:
        old, new = change.label.split("->")
        text = f"The {old}{a} {VERBS[t][k]} a {new}."
    elif t is ChangeType.MOVED:
        dist = float(np.linalg.norm(change.translation))
        turn = abs(change.rotation_deg)
        if dist < cfg_match.move_thresh_m:
            text = f"The {change.label}{a} {VERBS['turned'][k]} {turn:.0f}° in place."
        else:
            text = f"The {change.label} {VERBS[t][k]} {dist:.1f} m"
            if turn >= cfg_match.rot_thresh_deg:
                text += f" and turned {turn:.0f}°"
            if a and b and a != b:
                text += f", from{a} to{b}"
            elif b:
                text += f", now{b}"
            text += "."
    else:
        text = f"The {change.label}{a} is unchanged."
    return f"{text} {evidence(change)}".strip()


def summary(changes: list[Change]) -> str:
    """2-3 sentence overview of the whole comparison."""
    real = [c for c in changes if c.type is not ChangeType.UNCHANGED]
    confirmed = [c for c in real if c.confidence is Confidence.CONFIRMED]
    unverified = [c for c in real if c.confidence is Confidence.UNVERIFIED]
    rejected = [c for c in real if c.confidence is Confidence.REJECTED]
    unchanged = len(changes) - len(real)
    if confirmed:
        what = [c.description.split(" Confirmed:")[0].rstrip(".") for c in confirmed]
        head = (
            f"{len(confirmed)} change{'s were' if len(confirmed) > 1 else ' was'} confirmed: "
            + "; ".join(w[0].lower() + w[1:] for w in what)
            + "."
        )
    else:
        head = "No change could be confirmed."
    parts = [head]
    if unverified:
        parts.append(
            f"{len(unverified)} further candidate{'s' if len(unverified) > 1 else ''} "
            f"could not be verified because the area was barely observed in one of the "
            f"recordings."
        )
    if rejected:
        parts.append(
            f"{len(rejected)} false alarm{'s were' if len(rejected) > 1 else ' was'} "
            f"rejected because the other recording still shows the object."
        )
    parts.append(f"{unchanged} objects were matched as unchanged.")
    return " ".join(parts)


# --------------------------------------------------------------------------------------------
# Stage
# --------------------------------------------------------------------------------------------


def latest_changes(run: str) -> tuple[list[Change], str]:
    """The most processed change list available (C7 > C6 > C5) and its path."""
    for relpath in CHANGE_SOURCES:
        if exists(run, relpath):
            return [Change.from_dict(d) for d in load_json(run_path(run, relpath))], relpath
    raise FileNotFoundError(f"No change list in runs/{run}/changes — run C5 first")


def load_report(run: str) -> ChangeReport:
    return ChangeReport.from_dict(load_json(run_path(run, REPORT_JSON)))


def _row(c: Change, where_text: str) -> str:
    move = ""
    if c.type is ChangeType.MOVED:
        move = f"{np.linalg.norm(c.translation):.2f} m, {c.rotation_deg:+.0f}°"
    conf = {"confirmed": "✅ confirmed", "unverified": "❓ unverified", "rejected": "✖ rejected"}[
        c.confidence.value
    ]
    ids = " → ".join(i for i in (c.object_a, c.object_b) if i)
    return f"| {c.id} | **{c.type.value}** | {c.label} | {conf} | {where_text} | {move} | {ids} |"


def write_markdown(
    run: str, report: ChangeReport, objects: dict[str, Object3D], where_of: dict[str, str]
) -> str:
    def crop(obj_id: str | None) -> str:
        if not obj_id:
            return ""
        o = objects[obj_id]
        session = "before" if o.session == "A" else "after"
        return f"![{obj_id} ({session})](../{o.best_crop_path})"

    real = [c for c in report.changes if c.type is not ChangeType.UNCHANGED]
    main = [c for c in real if c.confidence is not Confidence.REJECTED]
    rejected = [c for c in real if c.confidence is Confidence.REJECTED]
    unchanged = [c for c in report.changes if c.type is ChangeType.UNCHANGED]
    header = "| id | change | object | confidence | where | movement | objects (A → B) |"
    lines = [
        f"# What changed in the room? — run `{report.run_name}`",
        "",
        report.summary,
        "",
        "## Changes",
        "",
    ]
    if main:
        lines += [header, "|---|---|---|---|---|---|---|"]
        lines += [_row(c, where_of[c.id]) for c in main]
        lines += [""]
        for c in main:
            lines += [
                f"### {c.id}: {c.label} {c.type.value}",
                "",
                c.description,
                "",
                " ".join(x for x in (crop(c.object_a), crop(c.object_b)) if x),
                "",
            ]
    else:
        lines += ["No changes.", ""]
    lines += [
        "## Unchanged objects",
        "",
        ", ".join(sorted({c.label for c in unchanged})) or "none",
        "",
    ]
    lines += ["## Appendix: rejected candidates", ""]
    if rejected:
        lines += [header, "|---|---|---|---|---|---|---|"]
        lines += [_row(c, where_of[c.id]) for c in rejected]
        lines += [""] + [f"- **{c.id}** {c.description}" for c in rejected] + [""]
    else:
        lines += ["None.", ""]
    st = report.stats
    lines += [
        "## Run information",
        "",
        f"- change list: `{st['source']}`",
        f"- objects: {st['n_objects']['A']} before, {st['n_objects']['B']} after",
        f"- before/after alignment RMSE: {st['alignment_rmse'] * 100:.1f} cm",
        f"- warnings: {'; '.join(st['warnings']) or 'none'}",
        "",
    ]
    text = "\n".join(lines)
    run_path(run, REPORT_MD).parent.mkdir(parents=True, exist_ok=True)
    run_path(run, REPORT_MD).write_text(text)
    return text


@cached_stage(REPORT_JSON, load=load_report)
def describe(run: str, force: bool = False) -> ChangeReport:
    """Write ``report/changes.json`` (ChangeReport) and ``report/report.md``.

    Args:
        run: run name; needs C5 output (uses C7 / C6 output if present).
        force: recompute even if ``report/changes.json`` exists.
    """
    from changedet.stages.c2_reconstruct import load_reconstruction
    from changedet.stages.c4_fuse import load_objects

    full = load_run_config(run)
    cfg = full.report
    warnings: list[str] = []
    if cfg.mode != "template":
        # TODO(C8-full): LLM sentences grounded in spatial facts.
        warnings.append(f"report.mode={cfg.mode} not implemented yet; used templates")
        log.warning(warnings[-1])
    changes, source = latest_changes(run)
    if source != CHANGE_SOURCES[0]:
        warnings.append(f"used {source} (later stages not run)")
    objects_by_session = load_objects(run)
    objects = {o.id: o for s in ("A", "B") for o in objects_by_session[s]}
    unchanged_ids = {
        i for c in changes if c.type is ChangeType.UNCHANGED for i in (c.object_a, c.object_b) if i
    }

    def landmarks(session: str, exclude: str) -> list[Object3D]:
        return [o for o in objects_by_session[session] if o.id in unchanged_ids and o.id != exclude]

    where_of = {}
    for c in changes:
        wa = where(objects[c.object_a], landmarks("A", c.object_a), cfg) if c.object_a else ""
        wb = where(objects[c.object_b], landmarks("B", c.object_b), cfg) if c.object_b else ""
        c.description = sentence(c, wa, wb, full.match)
        where_of[c.id] = wa or wb

    stats = {
        "source": source,
        "counts": {
            f"{t.value}/{k.value}": sum(c.type is t and c.confidence is k for c in changes)
            for t in ChangeType
            for k in Confidence
            if any(c.type is t and c.confidence is k for c in changes)
        },
        "n_objects": {s: len(v) for s, v in objects_by_session.items()},
        "alignment_rmse": load_reconstruction(run).alignment_rmse,
        "warnings": warnings,
    }
    report = ChangeReport(run, changes, summary(changes), stats)
    write_markdown(run, report, objects, where_of)
    log.info("Summary: %s", report.summary)
    for c in changes:
        if c.type is not ChangeType.UNCHANGED:
            log.info("  %s %s", c.id, c.description)
    save_json(run_path(run, REPORT_JSON), report)  # written last: marks the stage as done
    return report
