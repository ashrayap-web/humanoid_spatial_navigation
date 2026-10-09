from __future__ import annotations

import numpy as np

from changedet.core import geometry as geo
from changedet.core.config import load_config
from changedet.core.types import ChangeType, Object3D
from changedet.stages.c5_match import _Obj, assign, classify, cost_matrix, wrap_deg

CFG = load_config().match


def unit(v):
    return v / np.linalg.norm(v)


def l_shape(rng, n=3000) -> np.ndarray:
    """Surface points of an L-shaped object (no rotational symmetry), centred near the origin."""
    a = rng.uniform([0, 0, 0], [0.6, 0.15, 0.8], size=(n // 2, 3))
    b = rng.uniform([0, 0, 0], [0.15, 0.5, 0.8], size=(n // 2, 3))
    pts = np.vstack([a, b])
    return pts - [0.2, 0.15, 0.0] + rng.normal(scale=0.003, size=pts.shape)


def make_obj(obj_id: str, label: str, points: np.ndarray, emb: np.ndarray, rng) -> _Obj:
    noisy = unit(emb + rng.normal(scale=0.02, size=emb.shape))
    obj = Object3D(
        id=obj_id,
        session=obj_id[0],
        label=label,
        label_scores={label: 1.0},
        points_path="",
        centroid=points.mean(axis=0),
        bbox_min=points.min(axis=0),
        bbox_max=points.max(axis=0),
        n_observations=10,
        clip_embedding=noisy,
        dino_embedding=noisy,
        best_view=(obj_id[0], 0),
        best_crop_path="",
    )
    pts = geo.voxel_downsample(points, CFG.icp_voxel)
    keys = geo.voxel_keys(pts, CFG.voxel_size)
    return _Obj(obj, pts, keys, geo.dilate_keys(keys))


def at(points, x, y, yaw_deg=0.0) -> np.ndarray:
    """Rotate about the object's own vertical axis, then place it at (x, y)."""
    centre = points.mean(axis=0)
    T = (
        geo.make_T(t=[x, y, 0])
        @ geo.make_T(geo.rot_z(np.radians(yaw_deg)))
        @ geo.make_T(t=[-centre[0], -centre[1], 0])
    )
    return geo.transform_points(T, points)


def test_scene_with_every_change_type() -> None:
    rng = np.random.default_rng(0)
    shape = l_shape(rng)
    emb = {k: unit(rng.normal(size=64)) for k in ("chair", "lamp", "box", "bag", "bin", "stool")}
    a = [
        make_obj("A_0", "chair", at(shape, 0, 0), emb["chair"], rng),  # moves + turns
        make_obj("A_1", "lamp", at(shape, 3, 0), emb["lamp"], rng),  # rotated in place
        make_obj("A_2", "bag", at(shape, 0, 3), emb["bag"], rng),  # removed
        make_obj("A_3", "stool", at(shape, -2, -2), emb["stool"], rng),  # identical pair ...
        make_obj("A_4", "stool", at(shape, -2, 0), emb["stool"], rng),  # ... swapped in B
        make_obj("A_5", "bin", at(shape, 3, 3), emb["bin"], rng),  # unchanged
    ]
    b = [
        make_obj("B_0", "chair", at(shape, 1.2, 0.5, yaw_deg=35), emb["chair"], rng),
        make_obj("B_1", "lamp", at(shape, 3, 0, yaw_deg=90), emb["lamp"], rng),
        make_obj("B_2", "box", at(shape, -3, 3), emb["box"], rng),  # added
        make_obj("B_3", "stool", at(shape, -2, 0), emb["stool"], rng),
        make_obj("B_4", "stool", at(shape, -2, -2), emb["stool"], rng),
        make_obj("B_5", "bin", at(shape, 3, 3), emb["bin"], rng),
    ]
    changes = classify(a, b, CFG)
    by_a = {c.object_a: c for c in changes if c.object_a}
    by_b = {c.object_b: c for c in changes if c.object_b}

    chair = by_a["A_0"]
    assert chair.type is ChangeType.MOVED and chair.object_b == "B_0"
    expected = b[0].obj.centroid - a[0].obj.centroid
    np.testing.assert_allclose(chair.translation[:2], expected[:2], atol=0.03)
    assert abs(chair.rotation_deg - 35) < 3

    lamp = by_a["A_1"]
    assert lamp.type is ChangeType.MOVED and abs(lamp.rotation_deg - 90) < 3
    assert np.linalg.norm(lamp.translation) < 0.05

    assert by_a["A_2"].type is ChangeType.REMOVED and by_a["A_2"].object_b is None
    assert by_b["B_2"].type is ChangeType.ADDED and by_b["B_2"].object_a is None
    assert by_a["A_5"].type is ChangeType.UNCHANGED

    # Identical stools that swapped places are indistinguishable: both "unchanged", matched by
    # position, with an equal (small) ambiguity margin.
    for i, j in (("A_3", "B_4"), ("A_4", "B_3")):
        assert by_a[i].type is ChangeType.UNCHANGED and by_a[i].object_b == j
    assert np.isclose(by_a["A_3"].match_margin, by_a["A_4"].match_margin, atol=0.05)
    assert [c.id for c in changes] == [f"chg_{k:03d}" for k in range(len(changes))]


def test_replaced_and_same_space() -> None:
    rng = np.random.default_rng(1)
    shape = l_shape(rng)
    small = shape * [0.4, 0.4, 0.3]
    mug, bottle, bottle2 = (unit(rng.normal(size=64)) for _ in range(3))
    a = [
        make_obj("A_0", "mug", at(small, 0, 0), mug, rng),  # replaced by a bottle
        make_obj("A_1", "bottle", at(small, 2, 0), bottle2, rng),  # two bottles in A ...
        make_obj("A_2", "bottle", at(small, 2.08, 0), bottle2, rng),
    ]
    b = [
        make_obj("B_0", "bottle", at(small, 0.05, 0), bottle, rng),
        make_obj("B_1", "bottle", np.vstack([at(small, 2, 0), at(small, 2.08, 0)]), bottle2, rng),
    ]  # ... fused into one object in B
    changes = classify(a, b, CFG)
    types = {(c.object_a, c.object_b): c.type for c in changes}
    assert types[("A_0", "B_0")] is ChangeType.REPLACED
    assert [c.label for c in changes if c.type is ChangeType.REPLACED] == ["mug->bottle"]
    assert {k for k, t in types.items() if t is ChangeType.UNCHANGED} == {
        ("A_1", "B_1"),
        ("A_2", "B_1"),
    } or {k for k, t in types.items() if t is ChangeType.UNCHANGED} == {
        ("A_2", "B_1"),
        ("A_1", "B_1"),
    }
    assert not any(t in (ChangeType.REMOVED, ChangeType.ADDED) for t in types.values())


def test_cost_matrix_and_assign() -> None:
    rng = np.random.default_rng(2)
    shape = l_shape(rng)
    e1, e2 = unit(rng.normal(size=64)), unit(rng.normal(size=64))
    a = [
        make_obj("A_0", "chair", at(shape, 0, 0), e1, rng).obj,
        make_obj("A_1", "bin", at(shape, 2, 0), e2, rng).obj,
    ]
    b = [make_obj("B_0", "bin", at(shape, 2, 0.1), e2, rng).obj]
    cost = cost_matrix(a, b, CFG)
    assert cost.shape == (2, 1) and cost[1, 0] < 0.2 < CFG.max_cost < cost[0, 0]
    assert assign(cost, CFG.max_cost) == [(1, 0, assign(cost, CFG.max_cost)[0][2])]
    assert assign(np.zeros((0, 3)), 1.0) == []


def test_wrap_deg() -> None:
    assert wrap_deg(270) == -90 and wrap_deg(-190) == 170 and wrap_deg(10) == 10
