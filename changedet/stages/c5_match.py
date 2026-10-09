"""C5 — Cross-session matching & change classification.

1. Cost between every A object and B object from CLIP, DINOv2, distance, size and label.
2. Hungarian assignment with a cost gate; the gap to the second-best option is kept as an
   ambiguity margin.
3. Each matched pair gets a rigid yaw + translation from ICP (staying put is preferred unless a
   move fits clearly better) -> UNCHANGED or MOVED.
4. Unmatched objects whose space is still occupied by a similar-looking object of the other
   session (fused at a different granularity, e.g. two bottles vs one) -> UNCHANGED;
   the rest -> REMOVED (A) / ADDED (B).
5. A REMOVED and an ADDED close together -> REPLACED.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from changedet.core import geometry as geo
from changedet.core.cache import cached_stage, load_json, load_ply, run_path, save_json
from changedet.core.config import load_run_config
from changedet.core.logging import get_logger
from changedet.core.types import Change, ChangeType, Confidence, Object3D
from changedet.stages.c4_fuse import load_objects

log = get_logger("c5")

CHANGES_RAW = "changes/changes_raw.json"


# --------------------------------------------------------------------------------------------
# Matching (pure, unit-tested)
# --------------------------------------------------------------------------------------------


def object_volume(o: Object3D, min_side: float = 0.02) -> float:
    return float(np.prod(np.maximum(o.bbox_max - o.bbox_min, min_side)))


def cost_matrix(objects_a: list[Object3D], objects_b: list[Object3D], cfg) -> np.ndarray:
    """(len(A), len(B)) matching cost; see the module docstring and ``configs/default.yaml``."""
    if not objects_a or not objects_b:
        return np.zeros((len(objects_a), len(objects_b)))
    clip_a = np.array([o.clip_embedding for o in objects_a])
    clip_b = np.array([o.clip_embedding for o in objects_b])
    dino_a = np.array([o.dino_embedding for o in objects_a])
    dino_b = np.array([o.dino_embedding for o in objects_b])
    cent_a = np.array([o.centroid for o in objects_a])
    cent_b = np.array([o.centroid for o in objects_b])
    vol_a = np.array([object_volume(o) for o in objects_a])
    vol_b = np.array([object_volume(o) for o in objects_b])
    labels_a = np.array([o.label for o in objects_a])
    labels_b = np.array([o.label for o in objects_b])
    distance = np.linalg.norm(cent_a[:, None] - cent_b[None], axis=2)
    return (
        cfg.w_sem * (1 - clip_a @ clip_b.T)
        + cfg.w_app * (1 - dino_a @ dino_b.T)
        + cfg.w_geo * np.minimum(distance / cfg.max_move_dist, 1.0)
        + cfg.w_size * np.abs(np.log(vol_a[:, None] / vol_b[None]))
        + cfg.w_label * (labels_a[:, None] != labels_b[None])
    )


def assign(cost: np.ndarray, max_cost: float) -> list[tuple[int, int, float | None]]:
    """Hungarian matches ``(i, j, margin)`` with cost <= ``max_cost``.

    ``margin`` = cheapest alternative for either side minus the match cost (None if there is no
    alternative). A small margin means the match is ambiguous (e.g. two identical chairs).
    """
    if cost.size == 0:
        return []
    rows, cols = linear_sum_assignment(cost)
    matches = []
    for i, j in zip(rows, cols, strict=True):
        if cost[i, j] > max_cost:
            continue
        alternatives = np.concatenate([np.delete(cost[i], j), np.delete(cost[:, j], i)])
        margin = float(alternatives.min() - cost[i, j]) if len(alternatives) else None
        matches.append((int(i), int(j), margin))
    return matches


def estimate_motion(source: np.ndarray, target: np.ndarray, cfg) -> tuple[np.ndarray, float]:
    """Yaw + translation taking ``source`` points onto ``target`` points.

    Tries staying put and moving the centroid with each of ``icp_yaw_inits_deg``; a move is only
    preferred if its inlier fraction beats staying put by ``stay_margin`` (so partial views and
    symmetric objects are not reported as moved). Returns (T, inlier fraction).
    """
    normals = geo.estimate_normals(target)
    dists = (0.2, 0.1, cfg.icp_max_dist)
    stay_T, _, stay_fit = geo.icp_yaw(source, target, max_dists=dists, target_normals=normals)
    best_T, best_fit = stay_T, -1.0
    c_src, c_dst = source.mean(axis=0), target.mean(axis=0)
    for yaw in cfg.icp_yaw_inits_deg:
        init = geo.make_T(t=c_dst) @ geo.make_T(geo.rot_z(np.radians(yaw))) @ geo.make_T(t=-c_src)
        T, _, fit = geo.icp_yaw(source, target, init=init, max_dists=dists, target_normals=normals)
        if fit > best_fit:
            best_T, best_fit = T, fit
    if best_fit > stay_fit + cfg.stay_margin:
        return best_T, best_fit
    return stay_T, stay_fit


def motion_a_to_b(points_a: np.ndarray, points_b: np.ndarray, cfg) -> tuple[np.ndarray, float]:
    """``T_a_to_b`` via ICP, using the smaller cloud as the source (partial views fit inside)."""
    if len(points_a) <= len(points_b):
        return estimate_motion(points_a, points_b, cfg)
    T, fit = estimate_motion(points_b, points_a, cfg)
    return geo.invert_T(T), fit


def wrap_deg(angle: float) -> float:
    return float((angle + 180.0) % 360.0 - 180.0)


# --------------------------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------------------------


@dataclass
class _Obj:
    obj: Object3D
    points: np.ndarray  # downsampled
    keys: np.ndarray
    dilated: np.ndarray


def _prepare(run: str, objects: list[Object3D], cfg) -> list[_Obj]:
    out = []
    for o in objects:
        points, _ = load_ply(run_path(run, o.points_path))
        points = geo.voxel_downsample(points, cfg.icp_voxel)
        keys = geo.voxel_keys(points, cfg.voxel_size)
        out.append(_Obj(o, points, keys, geo.dilate_keys(keys)))
    return out


def _change(kind: ChangeType, a: Object3D | None, b: Object3D | None, **kw) -> Change:
    label = kw.pop("label", None) or (a.label if a is not None else b.label)
    return Change(
        id="",
        type=kind,
        label=label,
        object_a=a.id if a else None,
        object_b=b.id if b else None,
        position_a=a.centroid if a else None,
        position_b=b.centroid if b else None,
        translation=kw.get("translation"),
        rotation_deg=kw.get("rotation_deg"),
        T_a_to_b=kw.get("T_a_to_b"),
        match_cost=kw.get("match_cost"),
        confidence=Confidence.CONFIRMED,
        match_margin=kw.get("match_margin"),
    )


def classify(objs_a: list[_Obj], objs_b: list[_Obj], cfg) -> list[Change]:
    """All changes (including UNCHANGED) between two prepared object maps."""
    cost = cost_matrix([o.obj for o in objs_a], [o.obj for o in objs_b], cfg)
    matches = assign(cost, cfg.max_cost)
    changes: list[Change] = []
    for i, j, margin in matches:
        a, b = objs_a[i], objs_b[j]
        T, _ = motion_a_to_b(a.points, b.points, cfg)
        translation = geo.transform_points(T, a.obj.centroid[None])[0] - a.obj.centroid
        yaw = wrap_deg(np.degrees(geo.yaw_of(T)))
        moved = np.linalg.norm(translation) >= cfg.move_thresh_m or abs(yaw) >= cfg.rot_thresh_deg
        changes.append(
            _change(
                ChangeType.MOVED if moved else ChangeType.UNCHANGED,
                a.obj,
                b.obj,
                translation=translation,
                rotation_deg=yaw,
                T_a_to_b=T,
                match_cost=float(cost[i, j]),
                match_margin=margin,
            )
        )
        if margin is not None and margin < cfg.ambiguous_margin:
            log.info("  ambiguous match %s <-> %s (margin %.2f)", a.obj.id, b.obj.id, margin)

    matched_a = {i for i, _, _ in matches}
    matched_b = {j for _, j, _ in matches}
    removed, added = [], []
    for pool, matched, others, is_a in (
        (objs_a, matched_a, objs_b, True),
        (objs_b, matched_b, objs_a, False),
    ):
        for k, o in enumerate(pool):
            if k in matched:
                continue
            # Same space still occupied by a similar-looking object of the other session (fused
            # at a different granularity) -> not a change. A different-looking object in the
            # same space is left for REPLACED below.
            overlaps = [
                geo.overlap_fraction(o.keys, other.dilated)
                if float(o.obj.dino_embedding @ other.obj.dino_embedding) >= cfg.same_space_min_dino
                else 0.0
                for other in others
            ]
            if overlaps and max(overlaps) >= cfg.same_space_frac:
                other = others[int(np.argmax(overlaps))].obj
                a_obj, b_obj = (o.obj, other) if is_a else (other, o.obj)
                changes.append(_change(ChangeType.UNCHANGED, a_obj, b_obj))
                log.info(
                    "  %s occupies the same space as %s (overlap %.2f) -> unchanged",
                    o.obj.id,
                    other.id,
                    max(overlaps),
                )
            elif is_a:
                removed.append(o.obj)
            else:
                added.append(o.obj)

    # REMOVED + ADDED at (almost) the same place -> REPLACED.
    for a in list(removed):
        if not added:
            break
        dists = [np.linalg.norm(a.centroid - b.centroid) for b in added]
        k = int(np.argmin(dists))
        if dists[k] < cfg.replace_dist:
            b = added.pop(k)
            removed.remove(a)
            changes.append(_change(ChangeType.REPLACED, a, b, label=f"{a.label}->{b.label}"))
    changes += [_change(ChangeType.REMOVED, a, None) for a in removed]
    changes += [_change(ChangeType.ADDED, None, b) for b in added]

    order = [
        ChangeType.MOVED,
        ChangeType.REPLACED,
        ChangeType.REMOVED,
        ChangeType.ADDED,
        ChangeType.UNCHANGED,
    ]
    changes.sort(key=lambda c: (order.index(c.type), c.label))
    for n, c in enumerate(changes):
        c.id = f"chg_{n:03d}"
    return changes


# --------------------------------------------------------------------------------------------
# Debug output
# --------------------------------------------------------------------------------------------

COLOURS = {
    ChangeType.REMOVED: (229, 57, 53),
    ChangeType.ADDED: (67, 160, 71),
    ChangeType.MOVED: (30, 136, 229),
    ChangeType.REPLACED: (142, 36, 170),
    ChangeType.UNCHANGED: (150, 160, 175),
}


def _describe(c: Change) -> str:
    if c.type is ChangeType.MOVED:
        return f"moved {np.linalg.norm(c.translation):.2f} m, yaw {c.rotation_deg:+.0f} deg"
    if c.type is ChangeType.UNCHANGED and c.translation is not None:
        return f"stayed ({np.linalg.norm(c.translation):.2f} m, {c.rotation_deg:+.0f} deg)"
    return ""


def print_table(changes: list[Change]) -> None:
    log.info(
        "%-8s %-10s %-22s %-7s %-7s %6s %6s  %s",
        "id",
        "type",
        "label",
        "A",
        "B",
        "cost",
        "margin",
        "motion",
    )
    for c in changes:
        log.info(
            "%-8s %-10s %-22s %-7s %-7s %6s %6s  %s",
            c.id,
            c.type.value,
            c.label[:22],
            c.object_a or "-",
            c.object_b or "-",
            f"{c.match_cost:.2f}" if c.match_cost is not None else "-",
            f"{c.match_margin:.2f}" if c.match_margin is not None else "-",
            _describe(c),
        )


def write_debug(run: str, changes: list[Change], objects: dict[str, list[Object3D]], recon) -> None:
    """``viz/c5_changes.rrd`` and ``viz/c5_changes_topdown.png``."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import rerun as rr

    by_id = {o.id: o for s in ("A", "B") for o in objects[s]}
    points = {i: load_ply(run_path(run, o.points_path))[0] for i, o in by_id.items()}
    background, _ = load_ply(run_path(run, recon.background_cloud_path))

    rec = rr.RecordingStream("changedet_c5")
    rec.save(run_path(run, "viz/c5_changes.rrd"))
    rec.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    rec.log(
        "world/background",
        rr.Points3D(background, colors=(170, 170, 170), radii=0.004),
        static=True,
    )
    fig, ax = plt.subplots(figsize=(10, 10))
    sel = (background[:, 2] > 0.1) & (background[:, 2] < 2.0)
    ax.scatter(background[sel, 0], background[sel, 1], s=0.2, color="0.85")
    for c in changes:
        colour = np.array(COLOURS[c.type])
        is_change = c.type is not ChangeType.UNCHANGED
        name = f"{c.id} {c.type.value} {c.label}"
        for side, oid in (("A", c.object_a), ("B", c.object_b)):
            if oid is None:
                continue
            faded = c.type in (ChangeType.MOVED, ChangeType.REPLACED) and side == "A"
            col = (colour * 0.4 + 255 * 0.6).astype(np.uint8) if faded else colour.astype(np.uint8)
            o, p = by_id[oid], points[oid]
            rec.log(
                f"world/{c.type.value}/{c.id}/{side}",
                rr.Points3D(p, colors=col, radii=0.008 if is_change else 0.004),
                static=True,
            )
            rec.log(
                f"world/{c.type.value}/{c.id}/{side}/box",
                rr.Boxes3D(
                    mins=[o.bbox_min],
                    sizes=[o.bbox_max - o.bbox_min],
                    colors=col,
                    labels=[name] if is_change else None,
                ),
                static=True,
            )
            ax.scatter(p[:, 0], p[:, 1], s=1.0 if is_change else 0.3, color=col / 255)
            if is_change:
                (x0, y0), (x1, y1) = o.bbox_min[:2], o.bbox_max[:2]
                ax.add_patch(
                    plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, lw=1.5, color=col / 255)
                )
                ax.text(
                    x0,
                    y1 + 0.03,
                    f"{c.id} {c.type.value} {c.label} ({oid})",
                    fontsize=7,
                    color=colour / 255 * 0.7,
                )
        if c.type is ChangeType.MOVED:
            rec.log(
                f"world/moved/{c.id}/arrow",
                rr.Arrows3D(
                    origins=[c.position_a],
                    vectors=[c.position_b - c.position_a],
                    colors=colour.astype(np.uint8),
                ),
                static=True,
            )
            ax.annotate(
                "",
                xy=c.position_b[:2],
                xytext=c.position_a[:2],
                arrowprops={"arrowstyle": "->", "color": colour / 255, "lw": 2},
            )
    counts = {t.value: sum(c.type is t for c in changes) for t in ChangeType}
    ax.set_title("C5 candidates (top-down): " + ", ".join(f"{k} {v}" for k, v in counts.items()))
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(run_path(run, "viz/c5_changes_topdown.png"), dpi=110)
    plt.close(fig)


# --------------------------------------------------------------------------------------------
# Stage
# --------------------------------------------------------------------------------------------


def load_changes(run: str, relpath: str = CHANGES_RAW) -> list[Change]:
    return [Change.from_dict(d) for d in load_json(run_path(run, relpath))]


@cached_stage(CHANGES_RAW, load=load_changes)
def match_and_classify(run: str, force: bool = False) -> list[Change]:
    """Match A and B objects and classify each as unchanged / moved / removed / added / replaced.

    Args:
        run: run name; needs C4 output.
        force: recompute even if ``changes/changes_raw.json`` exists.
    """
    from changedet.stages.c2_reconstruct import load_reconstruction

    cfg = load_run_config(run).match
    objects = load_objects(run)
    objs_a, objs_b = _prepare(run, objects["A"], cfg), _prepare(run, objects["B"], cfg)
    changes = classify(objs_a, objs_b, cfg)
    print_table(changes)
    counts = {t.value: sum(c.type is t for c in changes) for t in ChangeType}
    log.info("C5: %s", ", ".join(f"{v} {k}" for k, v in counts.items()))
    write_debug(run, changes, objects, load_reconstruction(run))
    save_json(run_path(run, CHANGES_RAW), changes)  # written last: marks the stage as done
    return changes
