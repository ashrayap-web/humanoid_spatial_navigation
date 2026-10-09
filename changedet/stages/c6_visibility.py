"""C6 — Visibility reasoning: was the space of a change actually observed in the other session?

For an object of session S, its points are projected into every camera of the other session S'
and compared with that camera's depth ``D`` at the pixel (point depth ``d``):

- ``D > d + margin``: the camera saw **past** the point -> free,
- ``|D - d| <= margin``: the camera saw a surface **at** the point -> occupied,
- ``D < d - margin``, invalid depth or outside the image: no information (occluded / unseen).

Depth under C3 masks of glossy objects (TV, monitor) is treated as invalid: LiDAR returns a mirror
image there, which would otherwise "see past" anything behind the screen.

A point is free / occupied if its votes for that state outnumber the other by ``vote_ratio``;
otherwise unknown. REMOVED / ADDED / REPLACED / MOVED changes are then CONFIRMED (space seen
empty), REJECTED (space seen occupied — the object is still there and matching failed) or
UNVERIFIED (not enough observation).
"""

from __future__ import annotations

from collections import defaultdict

import cv2
import numpy as np

from changedet.core import geometry as geo
from changedet.core.cache import cached_stage, load_ply, run_dir, run_path, save_json
from changedet.core.config import load_run_config
from changedet.core.logging import get_logger
from changedet.core.types import CameraFrame, Change, ChangeType, Confidence, Object3D
from changedet.stages.c5_match import CHANGES_RAW, load_changes

log = get_logger("c6")

CHANGES_VIS = "changes/changes_vis.json"
UNKNOWN, FREE, OCCUPIED = 0, 1, 2
_RANK = {Confidence.REJECTED: 0, Confidence.UNVERIFIED: 1, Confidence.CONFIRMED: 2}


# --------------------------------------------------------------------------------------------
# Core test (pure, unit-tested)
# --------------------------------------------------------------------------------------------


def count_votes(
    points: np.ndarray, cameras: list[CameraFrame], depths: list[np.ndarray], margin: float
) -> tuple[np.ndarray, np.ndarray]:
    """Per point and per camera: (free votes (M,N), occupied votes (M,N)) as booleans."""
    free = np.zeros((len(cameras), len(points)), bool)
    occupied = np.zeros_like(free)
    for k, (cam, depth) in enumerate(zip(cameras, depths, strict=True)):
        uv, z = geo.project(points, cam.K, cam.T_world_cam)
        h, w = depth.shape
        inside = (z > 0.05) & np.all(np.isfinite(uv), axis=1)
        u = np.full(len(points), -1)
        v = np.full(len(points), -1)
        u[inside], v[inside] = (
            np.round(uv[inside, 0]).astype(int),
            np.round(uv[inside, 1]).astype(int),
        )
        inside &= (u >= 0) & (u < w) & (v >= 0) & (v < h)
        observed = np.zeros(len(points))
        observed[inside] = depth[v[inside], u[inside]]
        valid = inside & (observed > 0)
        free[k] = valid & (observed > z + margin)
        occupied[k] = valid & (np.abs(observed - z) <= margin)
    return free, occupied


def point_states(free: np.ndarray, occupied: np.ndarray, vote_ratio: float) -> np.ndarray:
    """UNKNOWN / FREE / OCCUPIED per point from per-camera votes."""
    n_free, n_occ = free.sum(axis=0), occupied.sum(axis=0)
    states = np.full(free.shape[1], UNKNOWN)
    states[(n_free > 0) & (n_free >= vote_ratio * n_occ)] = FREE
    states[(n_occ > 0) & (n_occ >= vote_ratio * n_free)] = OCCUPIED
    return states


def assess_points(
    points: np.ndarray, cameras: list[CameraFrame], depths: list[np.ndarray], cfg
) -> dict:
    """Visibility of an object's space in the other session -> stats + confidence."""
    free, occupied = count_votes(points, cameras, depths, cfg.margin)
    states = point_states(free, occupied, cfg.vote_ratio)
    n = max(1, len(points))
    stats = {
        "n_points": int(len(points)),
        "free_frac": float((states == FREE).sum() / n),
        "occupied_frac": float((states == OCCUPIED).sum() / n),
        "unknown_frac": float((states == UNKNOWN).sum() / n),
    }
    if stats["occupied_frac"] >= cfg.reject_occ_frac:
        confidence = Confidence.REJECTED
    elif stats["free_frac"] >= cfg.confirm_free_frac:
        confidence = Confidence.CONFIRMED
    else:
        confidence = Confidence.UNVERIFIED
    stats["confidence"] = confidence.value

    def supporting(votes: np.ndarray) -> list[int]:
        counts = votes.sum(axis=1)
        order = np.argsort(-counts)
        keep = [k for k in order if counts[k] >= max(5, 0.05 * len(points))][: cfg.max_cameras]
        return [cameras[k].frame.index for k in keep]

    stats["free_cameras"] = supporting(free & (states == FREE)[None])
    stats["occupied_cameras"] = supporting(occupied & (states == OCCUPIED)[None])
    stats["_states"] = states  # popped before saving to JSON
    return stats


def combine(confidences: list[Confidence]) -> Confidence:
    """The weakest of several checks (REJECTED < UNVERIFIED < CONFIRMED)."""
    return min(confidences, key=_RANK.get)


# --------------------------------------------------------------------------------------------
# Stage
# --------------------------------------------------------------------------------------------


def load_session_depths(run: str, cameras: list[CameraFrame], unreliable: dict[int, list[str]]):
    """Depth maps of a session with glossy-object pixels (C3 masks) set to 0 (unknown)."""
    depths = []
    for cam in cameras:
        depth = np.load(run_dir(run) / cam.depth_path)
        for mask_path in unreliable.get(cam.frame.index, []):
            depth[cv2.imread(str(run_dir(run) / mask_path), cv2.IMREAD_GRAYSCALE) > 0] = 0
        depths.append(depth)
    return depths


def _sample(points: np.ndarray, n: int, seed: int) -> np.ndarray:
    if len(points) <= n:
        return points
    return points[np.random.default_rng(seed).choice(len(points), n, replace=False)]


def checks_for(change: Change) -> list[tuple[str, str | None, str]]:
    """(name, object id, session whose cameras look at it) for each space to check."""
    old = ("old_location", change.object_a, "B")  # A's object should be gone in B
    new = ("new_location", change.object_b, "A")  # B's object should not have been there in A
    return {
        ChangeType.REMOVED: [old],
        ChangeType.ADDED: [new],
        ChangeType.MOVED: [old, new],
        ChangeType.REPLACED: [new],  # A's old spot is legitimately occupied by the new object
    }.get(change.type, [])


def write_debug(run: str, results: list[tuple[Change, dict]], recon) -> None:
    """``viz/c6_visibility.rrd`` and ``viz/c6_visibility_topdown.png``."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import rerun as rr

    colours = {UNKNOWN: (150, 150, 150), FREE: (67, 160, 71), OCCUPIED: (229, 57, 53)}
    rec = rr.RecordingStream("changedet_c6")
    rec.save(run_path(run, "viz/c6_visibility.rrd"))
    rec.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    background, _ = load_ply(run_path(run, recon.background_cloud_path))
    rec.log(
        "world/background",
        rr.Points3D(background, colors=(190, 190, 190), radii=0.004),
        static=True,
    )
    fig, ax = plt.subplots(figsize=(10, 10))
    sel = (background[:, 2] > 0.1) & (background[:, 2] < 2.0)
    ax.scatter(background[sel, 0], background[sel, 1], s=0.2, color="0.88")
    cams = {s: {c.frame.index: c for c in recon.cameras[s]} for s in ("A", "B")}
    for change, checks in results:
        for name, (points, stats, session) in checks.items():
            states = stats["_states"]
            col = np.array([colours[s] for s in states], np.uint8)
            path = f"world/{change.id}_{change.label.replace(' ', '_')}/{name}"
            rec.log(path, rr.Points3D(points, colors=col, radii=0.01), static=True)
            for k in stats["free_cameras"][:5]:
                cam = cams[session][k]
                h, w = int(2 * cam.K[1, 2] + 1), int(2 * cam.K[0, 2] + 1)
                rec.log(
                    f"{path}/cameras/{k}",
                    rr.Transform3D(
                        translation=cam.T_world_cam[:3, 3], mat3x3=cam.T_world_cam[:3, :3]
                    ),
                    static=True,
                )
                rec.log(
                    f"{path}/cameras/{k}",
                    rr.Pinhole(
                        image_from_camera=cam.K,
                        resolution=[w, h],
                        camera_xyz=rr.ViewCoordinates.RDF,
                        image_plane_distance=0.1,
                    ),
                    static=True,
                )
            ax.scatter(points[:, 0], points[:, 1], s=1.5, c=col / 255)
            lo = points[:, :2].min(axis=0)
            ax.text(
                lo[0],
                points[:, 1].max() + 0.04,
                f"{change.id} {change.type.value} {change.label}: {stats['confidence']}\n"
                f"free {stats['free_frac']:.2f} occ {stats['occupied_frac']:.2f} "
                f"unk {stats['unknown_frac']:.2f} ({name}, seen from {session})",
                fontsize=6.5,
            )
    ax.set_title("C6 visibility: green = seen empty, red = seen occupied, grey = not observed")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(run_path(run, "viz/c6_visibility_topdown.png"), dpi=110)
    plt.close(fig)


@cached_stage(CHANGES_VIS, load=lambda run: load_changes(run, CHANGES_VIS))
def assess_visibility(run: str, force: bool = False) -> list[Change]:
    """Set the confidence of every candidate change from what the other session observed.

    Args:
        run: run name; needs C2, C3, C4 and C5 output.
        force: recompute even if ``changes/changes_vis.json`` exists.
    """
    from changedet.stages.c2_reconstruct import load_reconstruction
    from changedet.stages.c3_detect import load_detections
    from changedet.stages.c4_fuse import load_objects

    full = load_run_config(run)
    cfg = full.visibility
    recon = load_reconstruction(run)
    objects: dict[str, Object3D] = {o.id: o for s in ("A", "B") for o in load_objects(run)[s]}
    changes = load_changes(run, CHANGES_RAW)

    unreliable: dict[str, dict[int, list[str]]] = {"A": defaultdict(list), "B": defaultdict(list)}
    for d in load_detections(run):
        if d.label in cfg.unreliable_labels:
            unreliable[d.session][d.frame_index].append(d.mask_path)
    depths = {s: load_session_depths(run, recon.cameras[s], unreliable[s]) for s in ("A", "B")}

    run_path(run, "changes/visibility").mkdir(parents=True, exist_ok=True)
    results = []
    for change in changes:
        checks = {}
        for name, obj_id, session in checks_for(change):
            points, _ = load_ply(run_path(run, objects[obj_id].points_path))
            points = _sample(points, cfg.point_subsample, full.seed)
            stats = assess_points(points, recon.cameras[session], depths[session], cfg)
            checks[name] = (points, stats, session)
            np.savez_compressed(
                run_path(run, f"changes/visibility/{change.id}_{name}.npz"),
                points=points,
                states=stats["_states"],
            )
        if not checks:
            continue
        change.confidence = combine([Confidence(s["confidence"]) for _, s, _ in checks.values()])
        change.visibility = {
            name: {"seen_from": session, **{k: v for k, v in stats.items() if k != "_states"}}
            for name, (_, stats, session) in checks.items()
        }
        results.append((change, checks))
        for name, (_, stats, session) in checks.items():
            log.info(
                "%s %-8s %-14s %s seen from %s: free %.2f occupied %.2f unknown %.2f -> %s",
                change.id,
                change.type.value,
                change.label,
                name,
                session,
                stats["free_frac"],
                stats["occupied_frac"],
                stats["unknown_frac"],
                stats["confidence"],
            )

    counts: dict[str, int] = defaultdict(int)
    for c in changes:
        if c.type is not ChangeType.UNCHANGED:
            counts[f"{c.type.value}/{c.confidence.value}"] += 1
    log.info("C6: %s", ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    write_debug(run, results, recon)
    save_json(run_path(run, CHANGES_VIS), changes)  # written last: marks the stage as done
    return changes
