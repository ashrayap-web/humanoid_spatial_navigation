from __future__ import annotations

import numpy as np

from changedet.core.config import load_config
from changedet.stages.c4_fuse import (
    Segment,
    associate,
    box_iou,
    dilate_keys,
    drop_see_through,
    lift_mask,
    overlap_fraction,
    postprocess,
    voxel_keys,
)

CFG = load_config().fusion


def unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def box_surface(lo, hi, rng, n=4000) -> np.ndarray:
    """Points sampled uniformly on the faces of an axis-aligned box."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    pts = rng.uniform(lo, hi, size=(n, 3))
    axis = rng.integers(0, 3, n)
    side = rng.integers(0, 2, n)
    pts[np.arange(n), axis] = np.where(side == 1, hi[axis], lo[axis])
    return pts


def views_of(points, label, embedding, rng, n_views=6, frame0=0, noise=0.005):
    """Fake segments: each view sees a random 60% half-space of the object, with noise."""
    segs = []
    centre = points.mean(axis=0)
    for k in range(n_views):
        direction = unit(rng.normal(size=3))
        visible = (points - centre) @ direction > -0.2 * np.ptp(points, axis=0).max()
        p = points[visible] + rng.normal(scale=noise, size=(visible.sum(), 3))
        clip = unit(embedding + rng.normal(scale=0.05, size=embedding.shape))
        segs.append(Segment(frame0 + k, label, 0.6, 1000 + k, p, np.full_like(p, 0.5), clip, clip))
    return segs


# --------------------------------------------------------------------------------------------
# Voxel helpers
# --------------------------------------------------------------------------------------------


def test_voxel_keys_and_dilation() -> None:
    pts = np.array([[0.01, 0.01, 0.01], [0.015, 0.0, 0.0], [0.05, 0.0, 0.0]])
    keys = voxel_keys(pts, 0.02)
    assert len(keys) == 2
    dilated = dilate_keys(keys[:1])
    assert len(dilated) == 27
    neighbour = voxel_keys(np.array([[0.03, 0.01, 0.01]]), 0.02)  # next voxel along x
    far = voxel_keys(np.array([[0.09, 0.01, 0.01]]), 0.02)
    assert overlap_fraction(neighbour, dilated) == 1.0
    assert overlap_fraction(far, dilated) == 0.0


def test_box_iou() -> None:
    assert box_iou(np.zeros(3), np.ones(3), np.zeros(3), np.ones(3)) == 1.0
    assert box_iou(np.zeros(3), np.ones(3), np.full(3, 2.0), np.full(3, 3.0)) == 0.0
    assert np.isclose(box_iou(np.zeros(3), np.ones(3), [0.5, 0, 0], [1.5, 1, 1]), 1 / 3)


# --------------------------------------------------------------------------------------------
# Lifting
# --------------------------------------------------------------------------------------------


def test_drop_see_through_removes_reflections_only() -> None:
    depth = np.full((60, 60), 2.0, np.float32)  # wall
    tv = np.zeros((60, 60), bool)
    tv[20:40, 20:40] = True
    depth[20:40, 20:40] = 3.5  # mirror image "behind" the wall
    assert not drop_see_through(tv, depth, ring_px=5, margin=0.25).any()

    depth[20:40, 20:40] = 1.2  # a real object in front of the wall
    assert (drop_see_through(tv, depth, ring_px=5, margin=0.25) == tv).all()


def test_lift_mask_keeps_largest_cluster() -> None:
    K = np.array([[100.0, 0, 50], [0, 100.0, 50], [0, 0, 1]])
    depth = np.full((100, 100), 2.0, np.float32)
    mask = np.zeros((100, 100), bool)
    mask[30:70, 30:70] = True
    depth[30:70, 30:38] = 1.0  # a strip of a nearer surface bleeding into the mask
    cfg = load_config(overrides=["fusion.erode_px=0", "fusion.pixel_stride=1"]).fusion
    points, pixels = lift_mask(mask, depth, K, np.eye(4), cfg)
    assert np.allclose(points[:, 2], 2.0)  # the larger cluster at 2 m wins
    assert len(points) == len(pixels) == 40 * 32


# --------------------------------------------------------------------------------------------
# Association and post-processing
# --------------------------------------------------------------------------------------------


def test_two_boxes_fuse_into_two_objects() -> None:
    rng = np.random.default_rng(0)
    e_chair, e_box = unit(rng.normal(size=64)), unit(rng.normal(size=64))
    chair = box_surface([0, 0, 0], [0.5, 0.5, 0.9], rng)
    box = box_surface([1.5, 0.2, 0], [1.9, 0.6, 0.4], rng)
    segments = views_of(chair, "chair", e_chair, rng) + views_of(box, "box", e_box, rng, frame0=10)
    objects = postprocess(associate(segments, CFG), CFG)
    assert sorted(o.label for o in objects) == ["box", "chair"]
    for obj, truth in [(o, chair if o.label == "chair" else box) for o in objects]:
        lo, hi = obj.bbox
        np.testing.assert_allclose(
            (lo + hi) / 2, (truth.min(0) + truth.max(0)) / 2, atol=2 * CFG.voxel_size
        )
        assert len(obj.frames) == 6


def test_object_on_top_of_another_stays_separate() -> None:
    """A bag lying on a bed must not be merged into the bed."""
    rng = np.random.default_rng(1)
    bed = box_surface([0, 0, 0], [2.0, 1.6, 0.5], rng, n=20000)
    bag = box_surface([0.6, 0.5, 0.5], [1.1, 1.0, 0.7], rng)
    segments = views_of(bed, "bed", unit(rng.normal(size=64)), rng) + views_of(
        bag, "bag", unit(rng.normal(size=64)), rng, frame0=10
    )
    objects = postprocess(associate(segments, CFG), CFG)
    assert sorted(o.label for o in objects) == ["bag", "bed"]


def test_duplicate_tracks_are_merged() -> None:
    """The same drawers fused twice (e.g. near and far views) with different labels -> one."""
    rng = np.random.default_rng(2)
    e = unit(rng.normal(size=64))
    drawers = box_surface([0, 0, 0], [0.8, 0.5, 0.9], rng)
    first = associate(views_of(drawers, "chest of drawers", e, rng), CFG)
    second = associate(views_of(drawers + 0.03, "desk", e, rng, frame0=10), CFG)
    objects = postprocess(first + second, CFG)
    assert len(objects) == 1 and len(objects[0].frames) == 12


def test_picture_inside_frame_is_absorbed() -> None:
    """A "jacket" detected on the photo inside a picture frame belongs to the frame."""
    rng = np.random.default_rng(3)
    frame = box_surface([2.0, 0, 1.2], [2.02, 0.5, 1.8], rng)  # thin frame on a wall
    photo = frame[(frame[:, 1] > 0.15) & (frame[:, 1] < 0.35) & (frame[:, 2] > 1.4)]
    segments = views_of(frame, "picture frame", unit(rng.normal(size=64)), rng) + views_of(
        photo, "jacket", unit(rng.normal(size=64)), rng, frame0=10
    )
    objects = postprocess(associate(segments, CFG), CFG)
    assert [o.label for o in objects] == ["picture frame"]


def test_postprocess_filters() -> None:
    rng = np.random.default_rng(3)
    e = unit(rng.normal(size=64))
    rare = views_of(box_surface([0, 0, 0], [0.3, 0.3, 0.3], rng), "box", e, rng, n_views=2)
    huge = views_of(box_surface([3, 0, 0], [7, 1, 1], rng), "sofa", -e, rng, frame0=10)
    under = views_of(
        box_surface([0, 3, -1], [0.3, 3.3, -0.5], rng), "bin", unit(e + 1), rng, frame0=20
    )
    assert postprocess(associate(rare + huge + under, CFG), CFG) == []
