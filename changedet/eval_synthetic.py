"""Synthetic evaluation suite for change classification (C5) — every change type, many scenes.

The real data has a single removal, so moved / added / replaced / rotated-in-place are only
measured here. Each scene places asymmetric L-shaped objects in a 6 x 6 m room, applies 1-3 known
changes, and gives each session its own partial, noisy view of every object. Embeddings mimic the
statistics measured on the real recordings: CLIP cosine ~0.97 for the same object but ~0.8 for
different ones (high baseline), DINOv2 ~0.9 vs ~0.15. Some scenes contain identical twins.

What it measures: matching + classification + motion estimation (C5) given fused objects. It does
**not** exercise detection, fusion or visibility (C3/C4/C6) — those need real recordings.
"""

from __future__ import annotations

import numpy as np

from changedet.core import geometry as geo
from changedet.core.config import Config
from changedet.core.types import ChangeType, Object3D
from changedet.stages.c5_match import _Obj, classify
from changedet.stages.c11_evaluate import TYPES, GTChange, LabelMatcher, metrics_table, prf, score

LABELS = [
    "chair",
    "box",
    "lamp",
    "bin",
    "backpack",
    "plant",
    "stool",
    "suitcase",
    "basket",
    "speaker",
    "guitar",
    "fan",
]
ROOM = 6.0


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


class Scene:
    """Object identities (shape, label, embedding) and their placements in sessions A and B."""

    def __init__(self, rng: np.random.Generator, cfg_match):
        self.rng, self.cfg = rng, cfg_match
        self.clip_base, self.dino_base = _unit(rng.normal(size=64)), _unit(rng.normal(size=64))
        self.identities: list[dict] = []

    def new_identity(self, label: str | None = None, like: dict | None = None) -> dict:
        if like is not None:  # identical twin: same shape, label and appearance
            ident = dict(like)
        else:
            r = self.rng
            arm1 = r.uniform([0.3, 0.08, 0.3], [0.9, 0.25, 1.2])
            arm2 = np.array([r.uniform(0.08, 0.25), r.uniform(0.3, 0.8), arm1[2]])
            e = _unit(r.normal(size=64))
            ident = {
                "label": label or str(r.choice(LABELS)),
                "arms": (arm1, arm2),
                "clip": _unit(self.clip_base + 0.5 * e),  # cos(different) ~ 0.8
                "dino": _unit(0.4 * self.dino_base + e),
            }  # cos(different) ~ 0.15
        self.identities.append(ident)
        return ident

    def points(self, ident: dict, xy, yaw_deg: float, n: int = 1500) -> np.ndarray:
        """A partial, noisy view of the object placed at ``xy`` with ``yaw``."""
        r = self.rng
        arm1, arm2 = ident["arms"]
        pts = np.vstack(
            [r.uniform(0, arm1, size=(n // 2, 3)), r.uniform(0, arm2, size=(n // 2, 3))]
        )
        pts[:, :2] -= pts[:, :2].mean(axis=0)
        if r.random() < 0.5:  # this session saw only part of it
            d = _unit(np.append(r.normal(size=2), 0))
            pts = pts[(pts - pts.mean(axis=0)) @ d > -0.3 * np.ptp(pts[:, :2], axis=0).max()]
        T = geo.make_T(geo.rot_z(np.radians(yaw_deg)), [xy[0], xy[1], 0])
        return geo.transform_points(T, pts) + r.normal(scale=0.004, size=pts.shape)

    def observe(self, obj_id: str, ident: dict, xy, yaw_deg: float) -> _Obj:
        pts = self.points(ident, xy, yaw_deg)
        clip = _unit(ident["clip"] + self.rng.normal(scale=0.01, size=64))
        dino = _unit(ident["dino"] + self.rng.normal(scale=0.05, size=64))
        obj = Object3D(
            obj_id,
            obj_id[0],
            ident["label"],
            {ident["label"]: 1.0},
            "",
            pts.mean(axis=0),
            pts.min(axis=0),
            pts.max(axis=0),
            10,
            clip,
            dino,
            (obj_id[0], 0),
            "",
        )
        sampled = geo.voxel_downsample(pts, self.cfg.icp_voxel)
        keys = geo.voxel_keys(sampled, self.cfg.voxel_size)
        return _Obj(obj, sampled, keys, geo.dilate_keys(keys))


def _free_spot(rng, taken: list[np.ndarray], min_gap: float = 1.3, tries: int = 200):
    for _ in range(tries):
        xy = rng.uniform(-ROOM / 2 + 0.6, ROOM / 2 - 0.6, size=2)
        if all(np.linalg.norm(xy - t) >= min_gap for t in taken):
            return xy
    return None


def make_scene(rng: np.random.Generator, cfg_match):
    """Returns (objects A, objects B, ground truth list, {gt index: true yaw change})."""
    scene = Scene(rng, cfg_match)
    placements = []  # (identity, xy, yaw)
    for _ in range(int(rng.integers(5, 9))):
        xy = _free_spot(rng, [p[1] for p in placements])
        if xy is None:
            break
        twin = placements and rng.random() < 0.15
        ident = scene.new_identity(like=placements[-1][0] if twin else None)
        placements.append((ident, xy, float(rng.uniform(0, 360))))

    after = list(placements)
    gts: list[GTChange] = []
    yaw_truth: dict[int, float] = {}
    kinds = rng.choice(
        ["moved", "rotated", "removed", "added", "replaced"],
        size=int(rng.integers(1, 4)),
        replace=False,
    )
    changed = rng.choice(len(placements), size=len(kinds), replace=False)
    for kind, k in zip(kinds, changed, strict=True):
        ident, xy, yaw = placements[k]
        if kind == "moved":
            new_xy = _free_spot(rng, [p[1] for p in after if p is not None])
            if new_xy is None:
                continue
            dyaw = float(rng.uniform(-90, 90))
            after[k] = (ident, new_xy, yaw + dyaw)
            yaw_truth[len(gts)] = dyaw
            gts.append(
                GTChange(
                    "moved",
                    ident["label"],
                    position=np.append(new_xy, 0),
                    distance_m=float(np.linalg.norm(new_xy - xy)),
                )
            )
        elif kind == "rotated":
            dyaw = float(rng.choice([-1, 1]) * rng.uniform(60, 150))
            after[k] = (ident, xy, yaw + dyaw)
            yaw_truth[len(gts)] = dyaw
            gts.append(GTChange("moved", ident["label"], position=np.append(xy, 0), distance_m=0.0))
        elif kind == "removed":
            after[k] = None
            gts.append(GTChange("removed", ident["label"], position=np.append(xy, 0)))
        elif kind == "added":
            new_xy = _free_spot(rng, [p[1] for p in after if p is not None])
            if new_xy is None:
                continue
            new = scene.new_identity()
            after.append((new, new_xy, float(rng.uniform(0, 360))))
            gts.append(GTChange("added", new["label"], position=np.append(new_xy, 0)))
        else:  # replaced: a different object in the same spot
            new = scene.new_identity(
                label=str(rng.choice([x for x in LABELS if x != ident["label"]]))
            )
            after[k] = (new, xy + rng.normal(scale=0.05, size=2), float(rng.uniform(0, 360)))
            gts.append(GTChange("replaced", f"{ident['label']}->{new['label']}"))

    objs_a = [scene.observe(f"A_{i:03d}", *p) for i, p in enumerate(placements)]
    objs_b = [scene.observe(f"B_{i:03d}", *p) for i, p in enumerate(q for q in after if q)]
    return objs_a, objs_b, gts, yaw_truth


def run_suite(cfg, n_scenes: int = 30, seed: int = 0) -> dict:
    """Score C5 on ``n_scenes`` synthetic scenes; returns totals, per-type metrics, errors."""
    rng = np.random.default_rng(seed)
    matcher = LabelMatcher(Config.from_dict({**cfg.eval, "use_clip": False}))  # exact labels
    totals = {t: [0, 0, 0] for t in TYPES}
    dist_err, yaw_err, n_gt = [], [], 0
    for _ in range(n_scenes):
        objs_a, objs_b, gts, yaw_truth = make_scene(rng, cfg.match)
        n_gt += len(gts)
        preds = [
            c for c in classify(objs_a, objs_b, cfg.match) if c.type is not ChangeType.UNCHANGED
        ]
        result = score(preds, gts, matcher, cfg.eval.pos_tol)
        for t, m in result["per_type"].items():
            for i, key in enumerate(("tp", "fp", "fn")):
                totals[t][i] += m[key]
        by_id = {p.id: p for p in preds}
        gt_index = {f"{g.type} {g.label}": k for k, g in enumerate(gts)}
        for m in result["matches"]:
            k, p = gt_index.get(m["gt"]), by_id[m["pred"][0]]
            if k is not None and k in yaw_truth and p.rotation_deg is not None:
                err = (p.rotation_deg - yaw_truth[k] + 180) % 360 - 180
                yaw_err.append(abs(err))
                dist_err.append(abs(float(np.linalg.norm(p.translation)) - gts[k].distance_m))
    per_type = {t: prf(*v) for t, v in totals.items() if any(v)}
    overall = prf(*np.sum([v for v in totals.values()], axis=0).tolist())
    return {
        "n_scenes": n_scenes,
        "n_ground_truth": n_gt,
        "overall": overall,
        "per_type": per_type,
        "moved_distance_error_m": float(np.mean(dist_err)) if dist_err else None,
        "moved_yaw_error_deg": float(np.median(yaw_err)) if yaw_err else None,
    }


def markdown(result: dict) -> str:
    lines = [
        f"## Synthetic suite (C5 only): {result['n_scenes']} scenes, "
        f"{result['n_ground_truth']} ground-truth changes",
        "",
    ]
    lines += metrics_table([("overall", result["overall"])] + list(result["per_type"].items()))
    if result["moved_distance_error_m"] is not None:
        lines += [
            "",
            f"Moved / rotated objects: mean distance error "
            f"{result['moved_distance_error_m']:.3f} m, median yaw error "
            f"{result['moved_yaw_error_deg']:.1f}°.",
        ]
    return "\n".join(lines) + "\n"
