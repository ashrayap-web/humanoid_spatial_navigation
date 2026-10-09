from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from changedet.core import geometry as geo
from changedet.core.config import load_config
from changedet.stages.c2_reconstruct import (
    SessionData,
    arkit_pose_to_opencv,
    background_mask,
    clean_depth,
    find_floor,
    register_sessions,
    reprojection_residual,
    scale_intrinsics,
    upsample_nearest,
)

# --------------------------------------------------------------------------------------------
# Synthetic room: floor, four walls and asymmetric furniture (so 180 deg is not ambiguous)
# --------------------------------------------------------------------------------------------


def sample_box(lo, hi, spacing, rng) -> np.ndarray:
    """Points on the surface of an axis-aligned box (all six faces)."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    faces = []
    for axis in range(3):
        u, v = [a for a in range(3) if a != axis]
        n_u = max(2, int((hi[u] - lo[u]) / spacing))
        n_v = max(2, int((hi[v] - lo[v]) / spacing))
        uu, vv = np.meshgrid(np.linspace(lo[u], hi[u], n_u), np.linspace(lo[v], hi[v], n_v))
        for value in (lo[axis], hi[axis]):
            face = np.zeros((uu.size, 3))
            face[:, u], face[:, v], face[:, axis] = uu.ravel(), vv.ravel(), value
            faces.append(face)
    return np.vstack(faces) + rng.normal(scale=0.005, size=(sum(len(f) for f in faces), 3))


def make_room(rng, include_chair: bool = True) -> np.ndarray:
    parts = [
        sample_box([0, 0, 0], [4, 3, 2.5], 0.04, rng),  # floor, ceiling, walls
        sample_box([0, 0, 0], [1.2, 0.6, 0.8], 0.03, rng),  # desk in a corner
        sample_box([3.0, 1.0, 0], [4.0, 3.0, 0.5], 0.03, rng),  # bed along a wall
        sample_box([0, 2.5, 0], [0.4, 3.0, 1.8], 0.03, rng),  # wardrobe
    ]
    if include_chair:
        parts.append(sample_box([1.6, 1.2, 0], [2.1, 1.7, 0.9], 0.03, rng))
    room = np.vstack(parts)
    return room[room[:, 2] < 2.3]  # most sessions don't see the ceiling


@pytest.mark.parametrize("yaw_deg, t", [(127.0, [0.8, -0.5, 0.0]), (-35.0, [-1.2, 0.4, 0.01])])
def test_register_sessions_recovers_yaw_and_translation(yaw_deg, t) -> None:
    rng = np.random.default_rng(0)
    target = make_room(rng)
    # Session B: same room, chair removed, only part of the room seen, in its own frame.
    source_world = make_room(rng, include_chair=False)
    source_world = source_world[source_world[:, 0] < 3.6]
    T_true = geo.make_T(geo.rot_z(np.radians(yaw_deg)), t)  # maps B frame -> A frame
    source = geo.transform_points(geo.invert_T(T_true), source_world)

    T, rmse, fitness = register_sessions(source, target, load_config().recon)
    yaw_err = np.degrees(abs(geo.yaw_of(T @ geo.invert_T(T_true))))
    np.testing.assert_allclose(T[:3, 3], T_true[:3, 3], atol=0.02)
    assert yaw_err < 1.0
    assert rmse < 0.02 and fitness > 0.8


def test_find_floor_picks_lowest_plane_and_levels_it() -> None:
    rng = np.random.default_rng(1)
    room = make_room(rng)  # floor at z=0, desk top at 0.8, bed top at 0.5
    tilt = geo.make_T(
        Rotation.from_euler("xy", [[4, -3]], degrees=True).as_matrix()[0], [0.3, -0.2, 1.4]
    )
    tilted = geo.transform_points(tilt, room)
    plane = find_floor(tilted, bin_size=0.02, min_frac=0.3)
    levelled = geo.transform_points(geo.level_plane_transform(plane), tilted)
    floor = np.abs(room[:, 2]) < 0.01
    assert np.abs(levelled[floor, 2]).max() < 0.03
    assert np.median(levelled[~floor, 2]) > 0.2  # "up" is +z, everything else above the floor


# --------------------------------------------------------------------------------------------
# Per-frame helpers
# --------------------------------------------------------------------------------------------


def test_arkit_pose_to_opencv_axes() -> None:
    # Identity ARKit pose: the camera looks along world -z, with its "up" along world +y.
    T = arkit_pose_to_opencv(np.array([0, 0, 0, 1.0, 1, 2, 3]))
    np.testing.assert_allclose(T[:3, 2], [0, 0, -1])  # OpenCV forward (+z_cam)
    np.testing.assert_allclose(T[:3, 1], [0, -1, 0])  # OpenCV down (+y_cam)
    np.testing.assert_allclose(T[:3, 3], [1, 2, 3])
    assert np.isclose(np.linalg.det(T[:3, :3]), 1.0)


def test_scale_intrinsics_keeps_pixel_centres() -> None:
    K = np.array([[600.0, 0, 359.5], [0, 600.0, 479.5], [0, 0, 1]])  # centre of a 720x960 image
    small = scale_intrinsics(K, 192 / 720, 256 / 960)
    np.testing.assert_allclose(small[:2, 2], [95.5, 127.5])  # centre of a 192x256 image
    np.testing.assert_allclose(small[0, 0], 600 * 192 / 720)


def test_clean_depth() -> None:
    depth = np.full((6, 6), 2.0, np.float32)
    depth[0, 0] = np.nan
    depth[5, 5] = 9.0  # beyond max_depth
    depth[2:4, 2:4] = 1.0  # foreground block: its border and the background around it jump
    cleaned = clean_depth(depth, max_depth=4.0, edge_rel_thresh=0.05)
    assert cleaned[0, 0] == 0 and cleaned[5, 5] == 0
    assert cleaned[2, 2] == 0 and cleaned[2, 1] == 0  # both sides of the depth edge removed
    assert cleaned[0, 3] == 2.0 and cleaned.dtype == np.float32


def test_upsample_nearest() -> None:
    depth = np.arange(12, dtype=np.float32).reshape(4, 3)
    up = upsample_nearest(depth, 8, 6)
    assert up.shape == (8, 6)
    assert set(np.unique(up)) == set(depth.ravel())  # no interpolated values
    np.testing.assert_array_equal(up[::2, ::2], depth)


def test_background_mask() -> None:
    a = np.array([[0, 0, 0], [1, 0, 0], [5, 5, 5.0]])
    b = np.array([[0.01, 0, 0], [1.2, 0, 0]])
    np.testing.assert_array_equal(background_mask(a, b, 0.05), [True, False, False])


def test_reprojection_residual_catches_wrong_conventions() -> None:
    """Two views of a wall: correct poses agree, a missing axis flip does not."""
    K = np.array([[100.0, 0, 31.5], [0, 100.0, 23.5], [0, 0, 1]])
    poses = [
        np.array([0, 0, 0, 1.0, 0, 0, 0]),
        np.r_[Rotation.from_euler("y", [[10]], degrees=True).as_quat()[0], 0.3, 0, 0],
    ]
    T_true = [arkit_pose_to_opencv(p) for p in poses]
    wall_z = -2.0  # wall at world z = -2, both cameras look along -z

    def render(T):
        v, u = np.mgrid[0:48, 0:64]
        rays = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], -1)
        rays_world = rays.reshape(-1, 3) @ T[:3, :3].T
        depth = (wall_z - T[2, 3]) / rays_world[:, 2]  # z_cam = 1 along each ray
        return depth.reshape(48, 64).astype(np.float32)

    def session(Ts):
        data = SessionData(frames=[None, None])
        data.depth = [render(T) for T in T_true]
        data.K_depth = [K, K]
        data.T_world_cam = Ts
        return data

    assert reprojection_residual(session(T_true)) < 5e-3  # pixel rounding only
    no_flip = [geo.make_T(Rotation.from_quat(p[:4]).as_matrix(), p[4:]) for p in poses]
    assert reprojection_residual(session(no_flip)) > load_config().recon.reproj_tol
