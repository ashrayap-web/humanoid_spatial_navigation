from __future__ import annotations

import numpy as np
import pytest

from changedet.core import geometry as geo
from changedet.core.config import load_config
from changedet.core.types import CameraFrame, Change, ChangeType, Confidence, Frame
from changedet.stages.c6_visibility import (
    FREE,
    OCCUPIED,
    UNKNOWN,
    assess_points,
    checks_for,
    combine,
    point_states,
)
from tests.conftest import REPO_ROOT

CFG = load_config().visibility
K = np.array([[80.0, 0, 49.5], [0, 80.0, 49.5], [0, 0, 1]])  # 100 x 100 image


def camera(T: np.ndarray, index: int = 0) -> CameraFrame:
    return CameraFrame(Frame("B", index, index, 0.0, "", 0.0), K, T, "")


def render(T_world_cam: np.ndarray, boxes: list[tuple]) -> np.ndarray:
    """Z-depth image of axis-aligned boxes (world frame) by ray casting (slab method)."""
    v, u = np.mgrid[0:100, 0:100]
    rays_cam = np.stack(
        [(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u, float)], -1
    ).reshape(-1, 3)
    dirs = rays_cam @ T_world_cam[:3, :3].T
    origin = T_world_cam[:3, 3]
    best = np.full(len(dirs), np.inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        for lo, hi in boxes:
            t1 = (np.asarray(lo) - origin) / dirs
            t2 = (np.asarray(hi) - origin) / dirs
            t_near = np.nanmax(np.minimum(t1, t2), axis=1)
            t_far = np.nanmin(np.maximum(t1, t2), axis=1)
            hit = (t_near <= t_far) & (t_near > 0)
            best = np.where(hit & (t_near < best), t_near, best)
    depth = np.where(np.isfinite(best), best, 0.0)  # t along a ray with z_cam = 1 is z-depth
    return depth.reshape(100, 100).astype(np.float32)


# The object (seen in session A): a box below and in front of the camera; we have points on the
# surfaces A observed (front and top), as C4 would.
OBJ_LO, OBJ_HI = np.array([-0.3, 0.3, 1.8]), np.array([0.3, 0.6, 2.2])
BACK_WALL = ([-5, -5, 4.0], [5, 5, 4.1])


def object_points(rng, n=1500) -> np.ndarray:
    front = np.column_stack([rng.uniform(-0.3, 0.3, n), rng.uniform(0.3, 0.6, n), np.full(n, 1.8)])
    top = np.column_stack([rng.uniform(-0.3, 0.3, n), np.full(n, 0.3), rng.uniform(1.8, 2.2, n)])
    return np.vstack([front, top])


@pytest.mark.parametrize(
    "case, T, boxes, expected",
    [
        ("seen empty", np.eye(4), [BACK_WALL], Confidence.CONFIRMED),
        (
            "not covered",
            geo.make_T(geo.rot_z(0) @ np.diag([-1.0, 1, -1])),
            [BACK_WALL],
            Confidence.UNVERIFIED,
        ),  # camera turned around
        (
            "occluded by a wall",
            np.eye(4),
            [BACK_WALL, ([-5, -5, 1.0], [5, 5, 1.1])],
            Confidence.UNVERIFIED,
        ),
        ("still there", np.eye(4), [BACK_WALL, (OBJ_LO, OBJ_HI)], Confidence.REJECTED),
    ],
)
def test_removed_object_cases(case, T, boxes, expected) -> None:
    points = object_points(np.random.default_rng(0))
    cams = [camera(T, 0), camera(geo.make_T(t=[0.2, 0, 0]) @ T, 1)]
    depths = [render(c.T_world_cam, boxes) for c in cams]
    stats = assess_points(points, cams, depths, CFG)
    assert Confidence(stats["confidence"]) is expected, (case, stats)
    if expected is Confidence.CONFIRMED:
        assert stats["free_frac"] > 0.9 and stats["free_cameras"] == [0, 1]
    if expected is Confidence.REJECTED:
        # the top face is seen at a grazing angle: pixel rounding exceeds the margin for some
        assert stats["occupied_frac"] > 0.7 and stats["occupied_cameras"]


def test_point_states_vote_ratio() -> None:
    free = np.array([[1, 1, 0, 1, 0], [1, 0, 0, 1, 0], [1, 0, 0, 0, 0]], bool)
    occ = np.array([[0, 0, 1, 0, 0], [0, 1, 1, 0, 0], [0, 0, 1, 1, 0]], bool)
    np.testing.assert_array_equal(
        point_states(free, occ, 3.0), [FREE, UNKNOWN, OCCUPIED, UNKNOWN, UNKNOWN]
    )


def test_combine_and_checks() -> None:
    assert combine([Confidence.CONFIRMED, Confidence.UNVERIFIED]) is Confidence.UNVERIFIED
    assert combine([Confidence.REJECTED, Confidence.CONFIRMED]) is Confidence.REJECTED

    def change(kind):
        return Change(
            "c", kind, "x", "A_0", "B_0", None, None, None, None, None, None, Confidence.CONFIRMED
        )

    assert [c[0] for c in checks_for(change(ChangeType.MOVED))] == ["old_location", "new_location"]
    assert checks_for(change(ChangeType.REMOVED)) == [("old_location", "A_0", "B")]
    assert checks_for(change(ChangeType.ADDED)) == [("new_location", "B_0", "A")]
    assert [c[0] for c in checks_for(change(ChangeType.REPLACED))] == ["new_location"]
    assert checks_for(change(ChangeType.UNCHANGED)) == []


@pytest.mark.real_data
def test_bag_unverified_without_b_frames_that_see_it(monkeypatch) -> None:
    """Spec acceptance: dropping every B frame that sees the bed turns the bag UNVERIFIED."""
    from changedet.core.cache import RUNS_DIR_ENV, load_ply, run_path
    from changedet.stages.c2_reconstruct import load_reconstruction
    from changedet.stages.c4_fuse import load_objects
    from changedet.stages.c6_visibility import load_session_depths

    monkeypatch.setenv(RUNS_DIR_ENV, str(REPO_ROOT / "runs"))
    if not run_path("demo", "objects/objects_A.json").exists():
        pytest.skip("runs/demo has no C4 output")
    recon = load_reconstruction("demo")
    bag = next((o for o in load_objects("demo")["A"] if o.label == "bag"), None)
    if bag is None:
        pytest.skip("no bag object in runs/demo")
    points, _ = load_ply(run_path("demo", bag.points_path))
    points = points[:: max(1, len(points) // 1000)]

    def sees(cam):
        uv, z = geo.project(points, cam.K, cam.T_world_cam)
        w, h = 2 * cam.K[0, 2] + 1, 2 * cam.K[1, 2] + 1
        return np.any((z > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h))

    all_cams = recon.cameras["B"]
    stats = assess_points(points, all_cams, load_session_depths("demo", all_cams, {}), CFG)
    assert stats["confidence"] == "confirmed"
    blind = [c for c in all_cams if not sees(c)]
    stats = assess_points(points, blind, load_session_depths("demo", blind, {}), CFG)
    assert stats["confidence"] == "unverified" and stats["unknown_frac"] > 0.9
