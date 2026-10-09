"""Shared 3D math. Conventions (spec 5.1): metres, OpenCV camera axes, ``T_world_cam`` maps
camera coordinates to world coordinates."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def make_T(R: np.ndarray | None = None, t: np.ndarray | None = None) -> np.ndarray:
    """Build a 4x4 rigid transform from a 3x3 rotation and a 3-vector translation."""
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = R
    if t is not None:
        T[:3, 3] = t
    return T


def invert_T(T: np.ndarray) -> np.ndarray:
    """Inverse of a 4x4 rigid transform."""
    R, t = T[:3, :3], T[:3, 3]
    return make_T(R.T, -R.T @ t)


def transform_points(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a 4x4 transform to (N,3) points."""
    points = np.asarray(points, dtype=np.float64)
    return points @ T[:3, :3].T + T[:3, 3]


def backproject(
    depth: np.ndarray,
    K: np.ndarray,
    T_world_cam: np.ndarray | None = None,
    mask: np.ndarray | None = None,
    return_pixels: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Lift valid depth pixels (finite, > 0, inside ``mask``) to 3D points.

    Args:
        depth: (H,W) depth in metres.
        K: (3,3) intrinsics.
        T_world_cam: optional camera-to-world transform; points are in camera frame if None.
        mask: optional (H,W) boolean mask of pixels to use.
        return_pixels: also return the (N,2) integer (u, v) pixel of each point.

    Returns:
        (N,3) points, and (N,2) pixels if ``return_pixels``.
    """
    valid = np.isfinite(depth) & (depth > 0)
    if mask is not None:
        valid &= mask.astype(bool)
    v, u = np.nonzero(valid)
    z = depth[v, u].astype(np.float64)
    x = (u - K[0, 2]) * z / K[0, 0]
    y = (v - K[1, 2]) * z / K[1, 1]
    points = np.stack([x, y, z], axis=1)
    if T_world_cam is not None:
        points = transform_points(T_world_cam, points)
    if return_pixels:
        return points, np.stack([u, v], axis=1)
    return points


def project(
    points_world: np.ndarray, K: np.ndarray, T_world_cam: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Project (N,3) world points into a camera.

    Returns:
        uv: (N,2) float pixel coordinates (meaningless where z <= 0),
        z: (N,) depth along the camera's optical axis (<= 0 means behind the camera).
    """
    p_cam = transform_points(invert_T(T_world_cam), points_world)
    z = p_cam[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = K[0, 0] * p_cam[:, 0] / z + K[0, 2]
        v = K[1, 1] * p_cam[:, 1] / z + K[1, 2]
    return np.stack([u, v], axis=1), z


def fit_plane_ransac(
    points: np.ndarray, dist_thresh: float = 0.02, iters: int = 1000, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a plane ``ax + by + cz + d = 0`` with RANSAC, refined by least squares on inliers.

    The normal (a, b, c) has unit length; its sign is arbitrary (callers orient it).

    Returns:
        plane: (4,) coefficients, inliers: (N,) boolean mask.
    """
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 3:
        raise ValueError("Need at least 3 points to fit a plane")
    rng = np.random.default_rng(seed)
    best_inliers = np.zeros(len(points), dtype=bool)
    for _ in range(iters):
        p0, p1, p2 = points[rng.choice(len(points), 3, replace=False)]
        normal = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(normal)
        if norm < 1e-12:
            continue
        normal /= norm
        inliers = np.abs((points - p0) @ normal) < dist_thresh
        if inliers.sum() > best_inliers.sum():
            best_inliers = inliers
    if best_inliers.sum() < 3:
        raise ValueError("RANSAC found no plane")
    # Least-squares refinement: normal = smallest singular vector of the centred inliers.
    inlier_pts = points[best_inliers]
    centroid = inlier_pts.mean(axis=0)
    normal = np.linalg.svd(inlier_pts - centroid)[2][-1]
    plane = np.append(normal, -normal @ centroid)
    inliers = np.abs(points @ normal + plane[3]) < dist_thresh
    return plane, inliers


def voxel_downsample(
    points: np.ndarray, voxel_size: float, colors: np.ndarray | None = None
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Average points (and colours) falling into the same voxel.

    Returns:
        (M,3) points, plus (M,3) colours if ``colors`` was given.
    """
    points = np.asarray(points, dtype=np.float64)
    cells = np.floor(points / voxel_size).astype(np.int64)
    cells -= cells.min(axis=0)
    dims = cells.max(axis=0) + 1
    keys = (cells[:, 0] * dims[1] + cells[:, 1]) * dims[2] + cells[:, 2]  # one int per voxel
    _, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)

    def voxel_mean(values: np.ndarray) -> np.ndarray:
        sums = [
            np.bincount(inverse, weights=values[:, k], minlength=len(counts))
            for k in range(values.shape[1])
        ]
        return np.stack(sums, axis=1) / counts[:, None]

    if colors is None:
        return voxel_mean(points)
    return voxel_mean(points), voxel_mean(np.asarray(colors, dtype=np.float64))


# --------------------------------------------------------------------------------------------
# Rotations about z, normals, floor levelling, yaw-only ICP
# --------------------------------------------------------------------------------------------


def rot_z(yaw: float) -> np.ndarray:
    """3x3 rotation by ``yaw`` radians about +z."""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def yaw_of(R: np.ndarray) -> float:
    """Yaw (radians) of a rotation, i.e. the angle its x-axis makes in the xy-plane."""
    return float(np.arctan2(R[1, 0], R[0, 0]))


def rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Smallest 3x3 rotation taking direction ``a`` onto direction ``b``."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v, c = np.cross(a, b), float(a @ b)
    if np.linalg.norm(v) < 1e-12:
        if c > 0:
            return np.eye(3)
        # 180 degrees: rotate about any axis perpendicular to a
        axis = np.cross(a, [1.0, 0, 0] if abs(a[0]) < 0.9 else [0, 1.0, 0])
        axis /= np.linalg.norm(axis)
        return 2 * np.outer(axis, axis) - np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)


def level_plane_transform(plane: np.ndarray) -> np.ndarray:
    """4x4 transform that maps ``plane`` onto z = 0 with its normal along +z.

    The plane normal is flipped first if it points down, so "up" stays up.
    """
    normal, d = plane[:3], plane[3]
    scale = np.linalg.norm(normal)
    normal, d = normal / scale, d / scale
    if normal[2] < 0:
        normal, d = -normal, -d
    R = rotation_between(normal, np.array([0.0, 0, 1]))
    point_on_plane = -d * normal
    return make_T(R, -R @ point_on_plane)


def estimate_normals(points: np.ndarray, k: int = 16) -> np.ndarray:
    """Unit normals from the PCA of each point's ``k`` nearest neighbours (sign arbitrary)."""
    points = np.asarray(points, dtype=np.float64)
    k = min(k, len(points))
    _, idx = cKDTree(points).query(points, k=k)
    neighbours = points[idx] - points[idx].mean(axis=1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", neighbours, neighbours)
    return np.linalg.eigh(cov)[1][:, :, 0]  # eigenvector of the smallest eigenvalue


def icp_yaw(
    source: np.ndarray,
    target: np.ndarray,
    init: np.ndarray | None = None,
    max_dists: tuple[float, ...] = (0.2, 0.1, 0.05),
    iters: int = 20,
    target_normals: np.ndarray | None = None,
) -> tuple[np.ndarray, float, float]:
    """Robust point-to-plane ICP restricted to a yaw rotation about z plus a translation.

    Aligns ``source`` onto ``target``, coarse to fine over ``max_dists`` (correspondence
    cut-offs). Residuals are Huber-weighted so moved/changed objects pull little.

    Returns:
        T (4,4) mapping source into target, inlier RMSE (m) and inlier fraction at the last
        (finest) cut-off.
    """
    tree = cKDTree(target)
    normals = estimate_normals(target) if target_normals is None else target_normals
    T = np.eye(4) if init is None else init.copy()
    for max_dist in max_dists:
        delta = max_dist / 3  # Huber threshold
        for _ in range(iters):
            moved = transform_points(T, source)
            dist, j = tree.query(moved, distance_upper_bound=max_dist)
            ok = np.isfinite(dist)
            if ok.sum() < 10:
                break
            p, q, n = moved[ok], target[j[ok]], normals[j[ok]]
            r = np.einsum("ij,ij->i", n, p - q)
            # d/d(yaw) of n . (Rz(yaw) p) at yaw = 0 is n . (-p_y, p_x, 0)
            J = np.column_stack([n[:, 1] * p[:, 0] - n[:, 0] * p[:, 1], n])
            w = np.sqrt(np.minimum(1.0, delta / np.maximum(np.abs(r), 1e-12)))
            x = np.linalg.lstsq(J * w[:, None], -r * w, rcond=None)[0]
            T = make_T(rot_z(x[0]), x[1:]) @ T
            if np.abs(x).max() < 1e-6:
                break
    dist, _ = tree.query(transform_points(T, source), distance_upper_bound=max_dists[-1])
    ok = np.isfinite(dist)
    rmse = float(np.sqrt(np.mean(dist[ok] ** 2))) if ok.any() else float("inf")
    return T, rmse, float(ok.mean())


# --------------------------------------------------------------------------------------------
# Voxel keys (one int64 per 3D voxel) for fast overlap tests
# --------------------------------------------------------------------------------------------

_OFFSET = 1 << 19  # voxel indices in [-2^19, 2^19) -> ~10 km at 2 cm, plenty
_SPAN = 1 << 20
_NEIGHBOURS = np.array([(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)])


def voxel_keys(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """Sorted unique int64 keys of the voxels containing ``points``."""
    idx = np.floor(points / voxel_size).astype(np.int64) + _OFFSET
    return np.unique((idx[:, 0] * _SPAN + idx[:, 1]) * _SPAN + idx[:, 2])


def dilate_keys(keys: np.ndarray) -> np.ndarray:
    """Keys of the voxels plus their 26 neighbours."""
    idx = np.stack([keys // (_SPAN * _SPAN), (keys // _SPAN) % _SPAN, keys % _SPAN], axis=1)
    grown = (idx[:, None, :] + _NEIGHBOURS[None]).reshape(-1, 3)
    return np.unique((grown[:, 0] * _SPAN + grown[:, 1]) * _SPAN + grown[:, 2])


def overlap_fraction(keys: np.ndarray, other_dilated: np.ndarray) -> float:
    """Fraction of ``keys`` that touch the (dilated) voxels of another object."""
    return float(np.isin(keys, other_dilated, assume_unique=True).mean()) if len(keys) else 0.0


def box_iou(min_a, max_a, min_b, max_b) -> float:
    """IoU of two axis-aligned 3D boxes."""
    min_a, max_a, min_b, max_b = map(np.asarray, (min_a, max_a, min_b, max_b))
    inter = np.prod(np.clip(np.minimum(max_a, max_b) - np.maximum(min_a, min_b), 0, None))
    union = np.prod(max_a - min_a) + np.prod(max_b - min_b) - inter
    return float(inter / union) if union > 0 else 0.0
