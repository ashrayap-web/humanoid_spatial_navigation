from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from changedet.core import geometry as g

K = np.array([[500.0, 0, 320], [0, 510.0, 240], [0, 0, 1]])


def random_T(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return g.make_T(Rotation.random(random_state=seed).as_matrix(), rng.normal(size=3))


def test_invert_and_transform() -> None:
    T = random_T()
    np.testing.assert_allclose(T @ g.invert_T(T), np.eye(4), atol=1e-12)
    pts = np.random.default_rng(1).normal(size=(20, 3))
    np.testing.assert_allclose(
        g.transform_points(g.invert_T(T), g.transform_points(T, pts)), pts, atol=1e-12
    )


def test_project_backproject_round_trip() -> None:
    rng = np.random.default_rng(0)
    depth = rng.uniform(0.5, 4.0, size=(48, 64)).astype(np.float32)
    depth[0, :5] = 0  # invalid
    depth[1, :5] = np.nan  # invalid
    T = random_T(3)
    points, pixels = g.backproject(depth, K, T, return_pixels=True)
    assert len(points) == depth.size - 10
    uv, z = g.project(points, K, T)
    np.testing.assert_allclose(uv, pixels, atol=1e-6)
    np.testing.assert_allclose(z, depth[pixels[:, 1], pixels[:, 0]], rtol=1e-6)


def test_backproject_mask_and_camera_frame() -> None:
    depth = np.full((4, 4), 2.0)
    mask = np.zeros((4, 4), bool)
    mask[1, 2] = True
    points = g.backproject(depth, K, mask=mask)
    np.testing.assert_allclose(points, [[(2 - 320) * 2 / 500, (1 - 240) * 2 / 510, 2.0]])


def test_project_marks_points_behind_camera() -> None:
    _, z = g.project(np.array([[0, 0, -1.0], [0, 0, 1.0]]), K, np.eye(4))
    assert z[0] < 0 < z[1]


def test_fit_plane_ransac_on_tilted_plane_with_outliers() -> None:
    rng = np.random.default_rng(0)
    xy = rng.uniform(-2, 2, size=(2000, 2))
    normal = np.array([0.1, -0.2, 1.0])
    normal /= np.linalg.norm(normal)
    d = -0.7
    z = -(normal[0] * xy[:, 0] + normal[1] * xy[:, 1] + d) / normal[2]
    plane_pts = np.column_stack([xy, z]) + rng.normal(scale=0.003, size=(2000, 3))
    outliers = rng.uniform(-2, 2, size=(500, 3))
    plane, inliers = g.fit_plane_ransac(np.vstack([plane_pts, outliers]), dist_thresh=0.02)
    sign = np.sign(plane[2])
    np.testing.assert_allclose(plane * sign, np.append(normal, d), atol=0.01)
    assert inliers[:2000].mean() > 0.98
    assert inliers[2000:].mean() < 0.1


def test_voxel_downsample() -> None:
    points = np.array([[0.01, 0.01, 0.01], [0.03, 0.03, 0.03], [0.51, 0.0, 0.0]])
    colors = np.array([[0, 0, 0], [1, 1, 1], [0.5, 0.5, 0.5]])
    down, down_colors = g.voxel_downsample(points, 0.1, colors)
    order = np.argsort(down[:, 0])
    np.testing.assert_allclose(down[order], [[0.02, 0.02, 0.02], [0.51, 0, 0]])
    np.testing.assert_allclose(down_colors[order], [[0.5] * 3, [0.5] * 3])
    assert g.voxel_downsample(points, 0.1).shape == (2, 3)
