"""C2 — Reconstruction & cross-session alignment (Record3D path).

Puts both sessions into one metric, z-up world with the floor at z = 0:

1. per kept frame: ARKit pose -> OpenCV ``T_world_cam``, intrinsics scaled to the C1 image,
   LiDAR depth cleaned (NaN, far, flying pixels) and saved upsampled to image resolution;
   a reprojection check between neighbouring frames guards the pose conventions,
2. per session: ARKit y-up -> z-up, fuse a coloured cloud, find the floor and level it to z = 0,
3. register B onto A with yaw + translation only (gravity is shared): exhaustive yaw search
   with 2D occupancy cross-correlation, then robust point-to-plane ICP,
4. centre the world on A's floor; write the clouds and a background cloud of points both
   sessions agree on.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import fftconvolve
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from changedet.core import geometry as geo
from changedet.core.cache import cached_stage, load_json, run_dir, run_path, save_json, save_ply
from changedet.core.config import load_run_config
from changedet.core.logging import get_logger, timed
from changedet.core.types import CameraFrame, Frame, Reconstruction
from changedet.io.record3d import Record3DExport, load_record3d
from changedet.stages.c1_frames import load_frames

log = get_logger("c2")

RECON_JSON = "recon/reconstruction.json"
SESSIONS = ("A", "B")

# ARKit camera axes (x right, y up, z backward) -> OpenCV (x right, y down, z forward).
GL_TO_CV = np.diag([1.0, -1.0, -1.0, 1.0])
# ARKit world (y up) -> our world (z up): (x, y, z) -> (x, -z, y).
Y_UP_TO_Z_UP = geo.make_T(np.array([[1.0, 0, 0], [0, 0, -1], [0, 1, 0]]))


# --------------------------------------------------------------------------------------------
# Per-frame helpers (pure, unit-tested)
# --------------------------------------------------------------------------------------------


def arkit_pose_to_opencv(pose: np.ndarray) -> np.ndarray:
    """Record3D ``[qx, qy, qz, qw, tx, ty, tz]`` -> 4x4 camera-to-world with OpenCV camera axes."""
    R = Rotation.from_quat(pose[:4]).as_matrix()
    return geo.make_T(R, pose[4:]) @ GL_TO_CV


def scale_intrinsics(K: np.ndarray, sx: float, sy: float) -> np.ndarray:
    """Intrinsics for an image resized by (sx, sy), using pixel-centre conventions."""
    K = K.copy()
    K[0, 0] *= sx
    K[1, 1] *= sy
    K[0, 2] = (K[0, 2] + 0.5) * sx - 0.5
    K[1, 2] = (K[1, 2] + 0.5) * sy - 0.5
    return K


def clean_depth(depth: np.ndarray, max_depth: float, edge_rel_thresh: float) -> np.ndarray:
    """NaN/inf/far -> 0, and zero 'flying pixels' whose depth jumps to a valid 4-neighbour by
    more than ``edge_rel_thresh`` x their own depth (they sit between foreground and background)."""
    d = np.where(np.isfinite(depth), depth, 0).astype(np.float32)
    d[d > max_depth] = 0
    pad = np.pad(d, 1, mode="edge")
    jump = np.zeros_like(d)
    for nb in (pad[:-2, 1:-1], pad[2:, 1:-1], pad[1:-1, :-2], pad[1:-1, 2:]):
        jump = np.maximum(jump, np.where(nb > 0, np.abs(d - nb), 0))
    d[jump > edge_rel_thresh * d] = 0
    return d


def upsample_nearest(depth: np.ndarray, height: int, width: int) -> np.ndarray:
    """Nearest-neighbour upsampling with pixel-centre alignment (never invents depth at edges)."""
    rows = np.minimum(
        ((np.arange(height) + 0.5) * depth.shape[0] / height).astype(int), depth.shape[0] - 1
    )
    cols = np.minimum(
        ((np.arange(width) + 0.5) * depth.shape[1] / width).astype(int), depth.shape[1] - 1
    )
    return depth[rows[:, None], cols[None, :]]


# --------------------------------------------------------------------------------------------
# Floor and registration (pure, unit-tested)
# --------------------------------------------------------------------------------------------


def find_floor(points: np.ndarray, bin_size: float, min_frac: float, seed: int = 0) -> np.ndarray:
    """Floor plane of a z-up cloud: the lowest well-populated height, refined with RANSAC.

    Returns plane coefficients (4,) with a unit normal.
    """
    z = points[:, 2]
    hist, edges = np.histogram(z, bins=np.arange(z.min(), z.max() + bin_size, bin_size))
    k = int(np.flatnonzero(hist >= min_frac * hist.max())[0])
    height = (edges[k] + edges[k + 1]) / 2
    near = points[np.abs(z - height) < 2.5 * bin_size]
    if len(near) > 200_000:
        near = near[np.random.default_rng(seed).choice(len(near), 200_000, replace=False)]
    plane, _ = geo.fit_plane_ransac(near, dist_thresh=bin_size, seed=seed)
    return plane


def coarse_yaw_search(
    source_xy: np.ndarray,
    target_xy: np.ndarray,
    cell: float,
    yaw_step_deg: float,
    top_k: int = 3,
    min_sep_deg: float = 20.0,
) -> list[tuple[float, np.ndarray]]:
    """Exhaustive yaw search: for each yaw, the best 2D shift by cross-correlating top-down
    occupancy grids (FFT). Returns up to ``top_k`` (score, 4x4 transform) with distinct yaws,
    best first."""

    def raster(xy: np.ndarray, origin: np.ndarray) -> np.ndarray:
        idx = np.floor((xy - origin) / cell).astype(int)
        grid = np.zeros(idx.max(axis=0) + 1)
        grid[idx[:, 0], idx[:, 1]] = 1.0
        return gaussian_filter(grid, 1.0)

    target_origin = target_xy.min(axis=0)
    target_grid = raster(target_xy, target_origin)
    results = []
    for yaw in np.deg2rad(np.arange(0.0, 360.0, yaw_step_deg)):
        rotated = source_xy @ geo.rot_z(yaw)[:2, :2].T
        origin = rotated.min(axis=0)
        source_grid = raster(rotated, origin)
        corr = fftconvolve(target_grid, source_grid[::-1, ::-1], mode="full")
        i, j = np.unravel_index(np.argmax(corr), corr.shape)
        shift = np.array([i - (source_grid.shape[0] - 1), j - (source_grid.shape[1] - 1)])
        t = target_origin + shift * cell - origin
        results.append((float(corr.max()), yaw, t))

    results.sort(key=lambda r: -r[0])
    picked: list[tuple[float, float, np.ndarray]] = []
    for score, yaw, t in results:
        separation = [abs((np.degrees(yaw - y) + 180) % 360 - 180) for _, y, _ in picked]
        if all(s >= min_sep_deg for s in separation):
            picked.append((score, yaw, t))
        if len(picked) == top_k:
            break
    return [(score, geo.make_T(geo.rot_z(yaw), [t[0], t[1], 0.0])) for score, yaw, t in picked]


def register_sessions(
    source: np.ndarray, target: np.ndarray, cfg
) -> tuple[np.ndarray, float, float]:
    """Align z-up, floor-levelled cloud ``source`` (B) onto ``target`` (A): yaw + translation.

    Returns (T 4x4, inlier RMSE, inlier fraction).
    """
    src = geo.voxel_downsample(source, cfg.icp_voxel)
    dst = geo.voxel_downsample(target, cfg.icp_voxel)
    z_lo, z_hi = cfg.coarse_z_range
    band = lambda p: p[(p[:, 2] > z_lo) & (p[:, 2] < z_hi), :2]  # noqa: E731 — walls & furniture
    candidates = coarse_yaw_search(
        band(src), band(dst), cfg.coarse_cell, cfg.yaw_step_deg, top_k=cfg.coarse_top_k
    )
    normals = geo.estimate_normals(dst)
    best = None
    for score, T0 in candidates:
        T, rmse, fitness = geo.icp_yaw(
            src, dst, init=T0, target_normals=normals, max_dists=(0.3, 0.15, cfg.icp_max_dist)
        )
        log.info(
            "  candidate yaw %6.1f deg (corr %.0f) -> ICP yaw %6.1f deg, fitness %.3f, rmse %.4f m",
            np.degrees(geo.yaw_of(T0)),
            score,
            np.degrees(geo.yaw_of(T)),
            fitness,
            rmse,
        )
        if best is None or (fitness, -rmse) > (best[2], -best[1]):
            best = (T, rmse, fitness)
    return best


def background_mask(points: np.ndarray, other: np.ndarray, thresh: float) -> np.ndarray:
    """True for points that have a neighbour in ``other`` within ``thresh`` (static scene)."""
    dist, _ = cKDTree(other).query(points, distance_upper_bound=thresh)
    return np.isfinite(dist)


# --------------------------------------------------------------------------------------------
# Session processing
# --------------------------------------------------------------------------------------------


@dataclass
class SessionData:
    frames: list[Frame]
    K: list[np.ndarray] = field(default_factory=list)  # at C1 image resolution
    K_depth: list[np.ndarray] = field(default_factory=list)  # at native depth resolution
    T_world_cam: list[np.ndarray] = field(default_factory=list)  # ARKit world until levelled
    depth: list[np.ndarray] = field(default_factory=list)  # cleaned, native resolution
    image_size: list[tuple[int, int]] = field(default_factory=list)  # (h, w)
    rgb_small: list[np.ndarray] = field(default_factory=list)  # RGB at depth resolution
    points: np.ndarray | None = None
    colors: np.ndarray | None = None


def _load_session(run: str, frames: list[Frame], rec: Record3DExport, cfg) -> SessionData:
    data = SessionData(frames)
    dh, dw = rec.depth_height, rec.depth_width
    for f in frames:
        image = cv2.imread(str(run_dir(run) / f.image_path))
        h, w = image.shape[:2]
        if rec.intrinsics is not None:
            fx, fy, cx, cy = rec.intrinsics[f.source_index]
            K_src = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
        else:
            K_src = rec.K
        data.K.append(scale_intrinsics(K_src, w / rec.width, h / rec.height))
        data.K_depth.append(scale_intrinsics(K_src, dw / rec.width, dh / rec.height))
        data.T_world_cam.append(arkit_pose_to_opencv(rec.poses[f.source_index]))
        data.depth.append(
            clean_depth(rec.read_depth(f.source_index), cfg.max_depth, cfg.edge_rel_thresh)
        )
        data.image_size.append((h, w))
        data.rgb_small.append(cv2.resize(image[..., ::-1], (dw, dh), interpolation=cv2.INTER_AREA))
    return data


def reprojection_residual(data: SessionData, max_pairs: int = 20) -> float:
    """Median |depth disagreement| when reprojecting each frame's depth into the next kept frame.

    A few millimetres means poses, intrinsics and axis conventions are right; a wrong quaternion
    order or missing axis flip gives ~0.1 m or more.
    """
    pairs = range(0, len(data.frames) - 1, max(1, (len(data.frames) - 1) // max_pairs))
    medians = []
    for i in pairs:
        j = i + 1
        pts = geo.backproject(data.depth[i], data.K_depth[i], data.T_world_cam[i])
        uv, z = geo.project(pts, data.K_depth[j], data.T_world_cam[j])
        u, v = np.round(uv[:, 0]).astype(int), np.round(uv[:, 1]).astype(int)
        h, w = data.depth[j].shape
        ok = (z > 0) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        observed = data.depth[j][v[ok], u[ok]]
        valid = observed > 0
        if valid.sum() > 100:
            medians.append(np.median(np.abs(observed[valid] - z[ok][valid])))
    return float(np.median(medians)) if medians else float("inf")


def _apply(data: SessionData, T: np.ndarray) -> None:
    """Left-multiply every camera and the cloud by world transform ``T``."""
    data.T_world_cam = [T @ Tc for Tc in data.T_world_cam]
    if data.points is not None:
        data.points = geo.transform_points(T, data.points)


def _fuse_cloud(data: SessionData, cfg) -> None:
    import open3d as o3d

    pts, cols = [], []
    for depth, K, T, rgb in zip(
        data.depth, data.K_depth, data.T_world_cam, data.rgb_small, strict=True
    ):
        p, pix = geo.backproject(depth, K, T, return_pixels=True)
        pts.append(p)
        cols.append(rgb[pix[:, 1], pix[:, 0]] / 255.0)
    points, colors = geo.voxel_downsample(np.vstack(pts), cfg.voxel_size, np.vstack(cols))
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    _, keep = pcd.remove_statistical_outlier(nb_neighbors=cfg.outlier_nb, std_ratio=cfg.outlier_std)
    data.points, data.colors = points[keep], colors[keep]


def _process_session(run: str, session: str, source: dict, frames: list[Frame], cfg) -> SessionData:
    if source["kind"] != "record3d":
        raise ValueError(
            f"recon.method=record3d needs Record3D input; session {session} is "
            f"{source['kind']} (TODO: learned reconstruction for plain video)"
        )
    rec = load_record3d(source["path"])
    data = _load_session(run, frames, rec, cfg)

    residual = reprojection_residual(data)
    log.info("  pose sanity check: median reprojection residual %.4f m", residual)
    if residual > cfg.reproj_tol:
        raise RuntimeError(
            f"Session {session}: depth reprojects inconsistently between neighbouring frames "
            f"(median {residual:.3f} m > {cfg.reproj_tol} m). Pose/axis conventions are wrong."
        )

    _apply(data, Y_UP_TO_Z_UP)
    _fuse_cloud(data, cfg)
    plane = find_floor(data.points, cfg.floor_bin, cfg.floor_min_frac, seed=cfg.seed)
    tilt = np.degrees(np.arccos(min(1.0, abs(plane[2]) / np.linalg.norm(plane[:3]))))
    _apply(data, geo.level_plane_transform(plane))
    log.info("  %d points, floor tilt vs ARKit gravity %.2f deg", len(data.points), tilt)
    return data


# --------------------------------------------------------------------------------------------
# Debug outputs
# --------------------------------------------------------------------------------------------

SESSION_COLORS = {"A": (66, 133, 244), "B": (255, 152, 0)}


def _log_rerun(path, sessions: dict[str, SessionData], background: np.ndarray) -> None:
    import rerun as rr

    rec = rr.RecordingStream("changedet_c2")
    rec.save(path)
    rec.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    for s, data in sessions.items():
        rec.log(
            f"world/{s}/cloud_rgb",
            rr.Points3D(data.points, colors=data.colors, radii=0.006),
            static=True,
        )
        rec.log(
            f"world/{s}/cloud_session_colour",
            rr.Points3D(data.points, colors=SESSION_COLORS[s], radii=0.006),
            static=True,
        )
        centres = np.array([T[:3, 3] for T in data.T_world_cam])
        rec.log(
            f"world/{s}/trajectory",
            rr.LineStrips3D([centres], colors=SESSION_COLORS[s]),
            static=True,
        )
        for k in range(0, len(data.frames), 5):
            h, w = data.image_size[k]
            cam = f"world/{s}/cameras/{k:03d}"
            T = data.T_world_cam[k]
            rec.log(cam, rr.Transform3D(translation=T[:3, 3], mat3x3=T[:3, :3]), static=True)
            rec.log(
                cam,
                rr.Pinhole(
                    image_from_camera=data.K[k],
                    resolution=[w, h],
                    camera_xyz=rr.ViewCoordinates.RDF,
                    image_plane_distance=0.15,
                    color=SESSION_COLORS[s],
                ),
                static=True,
            )
    rec.log(
        "world/background",
        rr.Points3D(background, colors=(150, 150, 150), radii=0.006),
        static=True,
    )
    lo, hi = sessions["A"].points[:, :2].min(0), sessions["A"].points[:, :2].max(0)
    rec.log(
        "world/floor",
        rr.Boxes3D(
            centers=[[*(lo + hi) / 2, 0.0]],
            half_sizes=[[*(hi - lo) / 2, 0.001]],
            colors=(120, 120, 120),
        ),
        static=True,
    )


def _top_down_png(path, sessions: dict[str, SessionData], z_range, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 9))
    rng = np.random.default_rng(0)
    for s, data in sessions.items():
        p = data.points[(data.points[:, 2] > z_range[0]) & (data.points[:, 2] < z_range[1])]
        p = p[rng.choice(len(p), min(len(p), 60_000), replace=False)]
        colour = np.array(SESSION_COLORS[s]) / 255
        ax.scatter(p[:, 0], p[:, 1], s=0.3, color=colour, alpha=0.35, label=f"session {s}")
        centres = np.array([T[:3, 3] for T in data.T_world_cam])
        ax.plot(centres[:, 0], centres[:, 1], color=colour, lw=1.5)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title)
    ax.legend(markerscale=20, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# --------------------------------------------------------------------------------------------
# Stage
# --------------------------------------------------------------------------------------------


def load_reconstruction(run: str) -> Reconstruction:
    """Read ``recon/reconstruction.json``."""
    return Reconstruction.from_dict(load_json(run_path(run, RECON_JSON)))


@cached_stage(RECON_JSON, load=load_reconstruction)
def reconstruct(run: str, force: bool = False) -> Reconstruction:
    """Cameras, depth and clouds for both sessions in one metric, z-up, floor-at-zero world.

    Args:
        run: run name; needs C1 output.
        force: recompute even if ``recon/reconstruction.json`` exists.
    """
    full_cfg = load_run_config(run)
    cfg = full_cfg.recon
    cfg["seed"] = full_cfg.seed
    if cfg.method != "record3d":
        raise NotImplementedError(f"recon.method={cfg.method}: TODO(post-MVP) — only record3d")
    inputs = load_json(run_path(run, "inputs.json"))
    frames = load_frames(run)

    sessions = {}
    for s in SESSIONS:
        with timed(f"C2 session {s}", log):
            log.info("Session %s: %d frames", s, len(frames[s]))
            sessions[s] = _process_session(run, s, inputs[s], frames[s], cfg)

    with timed("C2 registration B -> A", log):
        T_ba, rmse, fitness = register_sessions(sessions["B"].points, sessions["A"].points, cfg)
    _apply(sessions["B"], T_ba)
    log.info(
        "B -> A: yaw %.1f deg, t = %s m, inlier RMSE %.4f m, inlier fraction %.3f",
        np.degrees(geo.yaw_of(T_ba)),
        np.round(T_ba[:3, 3], 3),
        rmse,
        fitness,
    )
    if rmse > 0.03 or fitness < 0.5:
        log.warning(
            "Alignment looks poor (rmse %.3f m, fitness %.2f) — check viz/c2_topdown.png",
            rmse,
            fitness,
        )

    # Centre the world on A's cloud (x, y); the floor stays at z = 0.
    centre = sessions["A"].points[:, :2].mean(axis=0)
    for data in sessions.values():
        _apply(data, geo.make_T(t=[-centre[0], -centre[1], 0.0]))

    a, b = sessions["A"], sessions["B"]
    static_a = background_mask(a.points, b.points, cfg.static_thresh)
    static_b = background_mask(b.points, a.points, cfg.static_thresh)
    background, background_colors = geo.voxel_downsample(
        np.vstack([a.points[static_a], b.points[static_b]]),
        cfg.voxel_size,
        np.vstack([a.colors[static_a], b.colors[static_b]]),
    )
    log.info(
        "Background: %.1f%% of A and %.1f%% of B points agree within %.2f m (%d points)",
        100 * static_a.mean(),
        100 * static_b.mean(),
        cfg.static_thresh,
        len(background),
    )

    # Re-fit the floor on both sessions together as a check (should be ~[0, 0, 1, 0]).
    floor_points = np.vstack([d.points[np.abs(d.points[:, 2]) < 0.05] for d in sessions.values()])
    floor_plane, _ = geo.fit_plane_ransac(floor_points, dist_thresh=cfg.floor_bin, seed=cfg.seed)
    floor_plane = -floor_plane if floor_plane[2] < 0 else floor_plane
    log.info(
        "Joint floor plane %s (tilt %.2f deg)",
        np.round(floor_plane, 4),
        np.degrees(np.arccos(min(1.0, floor_plane[2]))),
    )

    cameras: dict[str, list[CameraFrame]] = {}
    for s, data in sessions.items():
        cameras[s] = []
        for k, f in enumerate(data.frames):
            h, w = data.image_size[k]
            relpath = f"recon/{s}/depth/{f.index:06d}.npy"
            path = run_dir(run) / relpath
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, upsample_nearest(data.depth[k], h, w))
            cameras[s].append(CameraFrame(f, data.K[k], data.T_world_cam[k], relpath, None))
        save_ply(run_path(run, f"recon/cloud_{s}.ply"), data.points, data.colors)
    save_ply(run_path(run, "recon/background.ply"), background, background_colors)

    viz = run_path(run, "viz")
    viz.mkdir(parents=True, exist_ok=True)
    _top_down_png(
        viz / "c2_topdown.png",
        sessions,
        cfg.coarse_z_range,
        f"C2 top-down (z {cfg.coarse_z_range[0]}-{cfg.coarse_z_range[1]} m): "
        f"A blue, B orange — RMSE {rmse * 100:.1f} cm",
    )
    _log_rerun(viz / "c2_recon.rrd", sessions, background)
    log.info("Debug: %s, %s", viz / "c2_topdown.png", viz / "c2_recon.rrd")

    recon = Reconstruction(
        cameras=cameras,
        cloud_paths={s: f"recon/cloud_{s}.ply" for s in SESSIONS},
        background_cloud_path="recon/background.ply",
        floor_plane=floor_plane,
        scale_is_metric=True,
        alignment_rmse=rmse,
    )
    save_json(run_path(run, RECON_JSON), recon)  # written last: marks the stage as done
    return recon
