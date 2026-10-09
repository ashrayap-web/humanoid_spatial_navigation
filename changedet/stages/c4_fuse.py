"""C4 — 3D lifting & object fusion.

Per session, independently:

1. **Lift** each C3 mask: erode, back-project the C2 depth under it into world points, voxelise,
   keep the largest DBSCAN cluster (drops depth that bled in from behind the object).
2. **Embed** the masked crop with CLIP (semantics) and DINOv2 (appearance).
3. **Associate** segments frame by frame: a segment joins the object whose voxels it overlaps most
   (dilated by one voxel, so 1-2 cm of noise still counts) if overlap > ``overlap_thresh`` and
   CLIP cosine > ``sim_thresh``; otherwise it starts a new object.
4. **Post-process:** merge over-segmented objects (overlapping 3D boxes or voxels, and agreeing
   label or CLIP), drop objects seen in too few frames, below the floor, or implausibly large.
"""

from __future__ import annotations

import shutil
from collections import defaultdict
from dataclasses import dataclass, field

import cv2
import numpy as np

from changedet import models
from changedet.core import geometry as geo
from changedet.core.cache import cached_stage, load_json, run_dir, run_path, save_json, save_ply
from changedet.core.config import load_run_config
from changedet.core.geometry import box_iou, dilate_keys, overlap_fraction, voxel_keys
from changedet.core.logging import get_logger, timed
from changedet.core.types import CameraFrame, Detection2D, Object3D
from changedet.stages.c2_reconstruct import load_reconstruction
from changedet.stages.c3_detect import load_detections

log = get_logger("c4")

SESSIONS = ("A", "B")


def objects_json(session: str) -> str:
    return f"objects/objects_{session}.json"


# --------------------------------------------------------------------------------------------
# Segments and objects under construction
# --------------------------------------------------------------------------------------------


@dataclass
class Segment:
    """One C3 detection lifted to 3D."""

    frame_index: int
    label: str
    score: float
    mask_area: int
    points: np.ndarray  # (M,3) voxelised world points
    colors: np.ndarray  # (M,3) in [0,1]
    clip: np.ndarray
    dino: np.ndarray
    keys: np.ndarray = field(default_factory=lambda: np.empty(0, np.int64))


@dataclass
class Track:
    """An object being fused from segments."""

    points: np.ndarray
    colors: np.ndarray
    clip_sum: np.ndarray
    dino_sum: np.ndarray
    label_scores: dict[str, float]
    frames: set[int]
    best: Segment
    keys: np.ndarray = field(default_factory=lambda: np.empty(0, np.int64))
    dilated: np.ndarray = field(default_factory=lambda: np.empty(0, np.int64))

    @property
    def clip(self) -> np.ndarray:
        return self.clip_sum / np.linalg.norm(self.clip_sum)

    @property
    def dino(self) -> np.ndarray:
        return self.dino_sum / np.linalg.norm(self.dino_sum)

    @property
    def label(self) -> str:
        return max(self.label_scores, key=self.label_scores.get)

    @property
    def bbox(self) -> tuple[np.ndarray, np.ndarray]:
        return self.points.min(axis=0), self.points.max(axis=0)

    def refresh(self, voxel_size: float) -> None:
        self.points, self.colors = geo.voxel_downsample(self.points, voxel_size, self.colors)
        self.keys = voxel_keys(self.points, voxel_size)
        self.dilated = dilate_keys(self.keys)

    def absorb(self, other: Track | Segment, voxel_size: float) -> None:
        """Merge another track or a segment into this one."""
        if isinstance(other, Segment):
            other = new_track(other, voxel_size, refresh=False)
        self.points = np.vstack([self.points, other.points])
        self.colors = np.vstack([self.colors, other.colors])
        self.clip_sum = self.clip_sum + other.clip_sum
        self.dino_sum = self.dino_sum + other.dino_sum
        for label, score in other.label_scores.items():
            self.label_scores[label] = self.label_scores.get(label, 0.0) + score
        self.frames |= other.frames
        if other.best.mask_area > self.best.mask_area:
            self.best = other.best
        self.refresh(voxel_size)


def new_track(seg: Segment, voxel_size: float, refresh: bool = True) -> Track:
    track = Track(
        seg.points,
        seg.colors,
        seg.clip.astype(np.float64),
        seg.dino.astype(np.float64),
        {seg.label: seg.score},
        {seg.frame_index},
        seg,
    )
    if refresh:
        track.refresh(voxel_size)
    return track


# --------------------------------------------------------------------------------------------
# Lifting, association, post-processing (pure, unit-tested)
# --------------------------------------------------------------------------------------------


def drop_see_through(
    mask: np.ndarray, depth: np.ndarray, ring_px: int, margin: float
) -> np.ndarray:
    """Remove mask pixels whose depth is behind everything around the mask.

    An object cannot be farther than the surfaces just outside its outline (median depth of a
    ``ring_px``-wide ring) plus ``margin``. This removes "depth" seen into glossy TV screens
    (LiDAR records a mirror image of the room behind the wall) without touching normal objects,
    whose surroundings are at the same depth or farther away. The median, not a high percentile,
    so a bright window next to the TV cannot vouch for the reflection.
    """
    kernel = np.ones((2 * ring_px + 1,) * 2, np.uint8)
    ring = cv2.dilate(mask.astype(np.uint8), kernel).astype(bool) & ~mask
    ring_depth = depth[ring & (depth > 0)]
    if len(ring_depth) < 20:
        return mask
    return mask & (depth <= np.median(ring_depth) + margin)


def lift_mask(mask: np.ndarray, depth: np.ndarray, K: np.ndarray, T_world_cam: np.ndarray, cfg):
    """World points (largest DBSCAN cluster) under an eroded mask, without see-through depth.

    Returns (points (M,3), pixels (M,2) as (u, v) of each point) or None if too few remain.
    """
    import open3d as o3d

    mask = drop_see_through(mask, depth, cfg.ring_px, cfg.behind_margin)
    if cfg.erode_px > 0:
        kernel = np.ones((2 * cfg.erode_px + 1,) * 2, np.uint8)
        mask = cv2.erode(mask.astype(np.uint8), kernel).astype(bool)
    stride = np.zeros_like(mask)
    stride[:: cfg.pixel_stride, :: cfg.pixel_stride] = True
    points, pixels = geo.backproject(depth, K, T_world_cam, mask=mask & stride, return_pixels=True)
    if len(points) < cfg.min_segment_points:
        return None
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    labels = np.asarray(pcd.cluster_dbscan(eps=cfg.dbscan_eps, min_points=cfg.dbscan_min_points))
    if (labels >= 0).sum() < cfg.min_segment_points:
        return None
    largest = np.bincount(labels[labels >= 0]).argmax()
    keep = labels == largest
    return points[keep], pixels[keep]


def associate(segments: list[Segment], cfg) -> list[Track]:
    """Incrementally fuse segments (in frame order) into tracks."""
    tracks: list[Track] = []
    for seg in sorted(segments, key=lambda s: s.frame_index):
        seg.keys = voxel_keys(seg.points, cfg.voxel_size)
        lo, hi = seg.points.min(axis=0) - cfg.voxel_size, seg.points.max(axis=0) + cfg.voxel_size
        best, best_overlap = None, cfg.overlap_thresh
        for track in tracks:
            t_lo, t_hi = track.bbox
            if np.any(t_lo > hi) or np.any(t_hi < lo):
                continue
            overlap = overlap_fraction(seg.keys, track.dilated)
            if overlap > best_overlap and float(seg.clip @ track.clip) > cfg.sim_thresh:
                best, best_overlap = track, overlap
        if best is None:
            tracks.append(new_track(seg, cfg.voxel_size))
        else:
            best.absorb(seg, cfg.voxel_size)
    return tracks


def should_merge(a: Track, b: Track, cfg) -> bool:
    """Same physical object seen as two tracks?

    Yes if they overlap geometrically AND agree semantically, or if the smaller one sits inside
    the larger one's box and lies almost entirely on its surface (a part, or a picture inside a
    frame), whatever its label.
    """
    (a_lo, a_hi), (b_lo, b_hi) = a.bbox, b.bbox
    if np.any(a_lo > b_hi + cfg.voxel_size) or np.any(b_lo > a_hi + cfg.voxel_size):
        return False
    small, large = (a, b) if len(a.keys) <= len(b.keys) else (b, a)
    overlap = overlap_fraction(small.keys, large.dilated)
    s_lo, s_hi = small.bbox
    l_lo, l_hi = large.bbox
    inside = np.all(s_lo >= l_lo - cfg.voxel_size) and np.all(s_hi <= l_hi + cfg.voxel_size)
    if overlap > cfg.contain_frac and inside:
        return True
    geometric = box_iou(a_lo, a_hi, b_lo, b_hi) > cfg.merge_iou or overlap > cfg.overlap_thresh
    semantic = a.label == b.label or float(a.clip @ b.clip) > cfg.sim_thresh
    return geometric and semantic


def postprocess(tracks: list[Track], cfg) -> list[Track]:
    """Merge over-segmented tracks, then drop rare, sub-floor and oversized ones."""
    tracks = list(tracks)
    merged = True
    while merged:
        merged = False
        for i in range(len(tracks)):
            for j in range(i + 1, len(tracks)):
                if should_merge(tracks[i], tracks[j], cfg):
                    tracks[i].absorb(tracks.pop(j), cfg.voxel_size)
                    merged = True
                    break
            if merged:
                break
    kept = []
    for t in tracks:
        lo, hi = t.bbox
        if len(t.frames) < cfg.min_observations:
            continue
        if hi[2] < 0.0 or np.max(hi - lo) > cfg.max_object_extent:
            continue
        kept.append(t)
    return kept


# --------------------------------------------------------------------------------------------
# Stage
# --------------------------------------------------------------------------------------------


def load_objects(run: str) -> dict[str, list[Object3D]]:
    """Read ``objects/objects_{A,B}.json``."""
    return {
        s: [Object3D.from_dict(d) for d in load_json(run_path(run, objects_json(s)))]
        for s in SESSIONS
    }


def _crop(image: np.ndarray, mask: np.ndarray, pad: float, grey_background: bool) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    px, py = int(pad * (x1 - x0)), int(pad * (y1 - y0))
    h, w = mask.shape
    x0, x1, y0, y1 = max(0, x0 - px), min(w, x1 + px), max(0, y0 - py), min(h, y1 + py)
    crop = image[y0:y1, x0:x1].copy()
    if grey_background:
        crop[~mask[y0:y1, x0:x1]] = 127
    return crop


def _lift_session(run: str, cams: list[CameraFrame], dets: list[Detection2D], cfg, device):
    """Lift and embed every detection of one session -> (segments, mask path per segment)."""
    by_frame: dict[int, list[Detection2D]] = defaultdict(list)
    for d in dets:
        by_frame[d.frame_index].append(d)
    segments, crops, mask_paths = [], [], []
    for cam in cams:
        frame_dets = by_frame.get(cam.frame.index, [])
        if not frame_dets:
            continue
        image = cv2.imread(str(run_dir(run) / cam.frame.image_path))[..., ::-1]
        depth = np.load(run_dir(run) / cam.depth_path)
        for d in frame_dets:
            mask = cv2.imread(str(run_dir(run) / d.mask_path), cv2.IMREAD_GRAYSCALE) > 0
            lifted = lift_mask(mask, depth, cam.K, cam.T_world_cam, cfg)
            if lifted is None:
                continue
            points, pixels = lifted
            colors = image[pixels[:, 1], pixels[:, 0]] / 255.0
            points, colors = geo.voxel_downsample(points, cfg.voxel_size, colors)
            segments.append(
                Segment(
                    d.frame_index,
                    d.label,
                    d.score,
                    int(mask.sum()),
                    points,
                    colors,
                    np.empty(0),
                    np.empty(0),
                )
            )
            crops.append(_crop(image, mask, cfg.crop_pad, grey_background=True))
            mask_paths.append(d.mask_path)
    clip_emb, dino_emb = models.embed_images(
        crops, cfg.clip_model, cfg.dino_model, device, cfg.embed_batch
    )
    for seg, c, e in zip(segments, clip_emb, dino_emb, strict=True):
        seg.clip, seg.dino = c, e
    log.info("  lifted %d / %d detections", len(segments), len(dets))
    return segments, dict(zip(map(id, segments), mask_paths, strict=True))


def _to_object(
    run: str, session: str, k: int, track: Track, cams: dict[int, CameraFrame], mask_path: str
) -> Object3D:
    obj_id = f"{session}_{k:03d}"
    points_path = f"objects/{session}/{obj_id}.ply"
    save_ply(run_path(run, points_path), track.points, track.colors)
    best = track.best
    cam = cams[best.frame_index]
    image = cv2.imread(str(run_dir(run) / cam.frame.image_path))
    mask = cv2.imread(str(run_dir(run) / mask_path), cv2.IMREAD_GRAYSCALE) > 0
    crop_path = f"objects/crops/{obj_id}.jpg"
    cv2.imwrite(str(run_dir(run) / crop_path), _crop(image, mask, 0.15, grey_background=False))
    lo, hi = track.bbox
    return Object3D(
        id=obj_id,
        session=session,
        label=track.label,
        label_scores={
            k_: round(v, 4) for k_, v in sorted(track.label_scores.items(), key=lambda kv: -kv[1])
        },
        points_path=points_path,
        centroid=track.points.mean(axis=0),
        bbox_min=lo,
        bbox_max=hi,
        n_observations=len(track.frames),
        clip_embedding=track.clip.astype(np.float32),
        dino_embedding=track.dino.astype(np.float32),
        best_view=(session, best.frame_index),
        best_crop_path=crop_path,
    )


def _fuse_session(
    run: str, session: str, cams: list[CameraFrame], dets: list[Detection2D], cfg, device
) -> list[Object3D]:
    segments, mask_of = _lift_session(run, cams, dets, cfg, device)
    tracks = associate(segments, cfg)
    n_assoc = len(tracks)
    tracks = postprocess(tracks, cfg)
    tracks.sort(key=lambda t: -len(t.frames))
    log.info(
        "  %d segments -> %d tracks -> %d objects after merge/filter",
        len(segments),
        n_assoc,
        len(tracks),
    )
    shutil.rmtree(run_path(run, f"objects/{session}"), ignore_errors=True)
    cam_by_index = {c.frame.index: c for c in cams}
    objects = [
        _to_object(run, session, k, t, cam_by_index, mask_of[id(t.best)])
        for k, t in enumerate(tracks)
    ]
    for o in objects:
        size = o.bbox_max - o.bbox_min
        log.info(
            "    %s %-16s seen %3d  centre %s  size %s",
            o.id,
            o.label,
            o.n_observations,
            np.round(o.centroid, 2),
            np.round(size, 2),
        )
    return objects


def write_debug(run: str, objects: dict[str, list[Object3D]], recon) -> None:
    """``viz/c4_objects.rrd`` (3D) and ``viz/c4_objects_topdown.png`` (A | B side by side)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import rerun as rr

    from changedet.core.cache import load_ply

    palette = (plt.get_cmap("tab20")(np.arange(20))[:, :3] * 255).astype(np.uint8)
    rec = rr.RecordingStream("changedet_c4")
    rec.save(run_path(run, "viz/c4_objects.rrd"))
    rec.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    background, _ = load_ply(run_path(run, recon.background_cloud_path))
    rec.log(
        "world/background",
        rr.Points3D(background, colors=(170, 170, 170), radii=0.004),
        static=True,
    )
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharex=True, sharey=True)
    rng = np.random.default_rng(0)
    for ax, s in zip(axes, SESSIONS, strict=True):
        cloud, _ = load_ply(run_path(run, recon.cloud_paths[s]))
        cloud = cloud[(cloud[:, 2] > 0.1) & (cloud[:, 2] < 2.0)]
        cloud = cloud[rng.choice(len(cloud), min(len(cloud), 40_000), replace=False)]
        ax.scatter(cloud[:, 0], cloud[:, 1], s=0.2, color="0.8")
        for k, o in enumerate(objects[s]):
            colour = palette[k % len(palette)]
            points, _ = load_ply(run_path(run, o.points_path))
            name = f"{o.id} {o.label}"
            rec.log(
                f"world/{s}/{o.id}", rr.Points3D(points, colors=colour, radii=0.008), static=True
            )
            rec.log(
                f"world/{s}/{o.id}/box",
                rr.Boxes3D(
                    mins=[o.bbox_min], sizes=[o.bbox_max - o.bbox_min], colors=colour, labels=[name]
                ),
                static=True,
            )
            ax.scatter(points[:, 0], points[:, 1], s=0.5, color=colour / 255)
            (x0, y0), (x1, y1) = o.bbox_min[:2], o.bbox_max[:2]
            ax.add_patch(
                plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, lw=1, color=colour / 255)
            )
            ax.text(x0, y1, name, fontsize=6, color=colour / 255 * 0.7)
        ax.set_title(f"Session {s}: {len(objects[s])} objects (top-down)")
        ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(run_path(run, "viz/c4_objects_topdown.png"), dpi=110)
    plt.close(fig)


@cached_stage([objects_json(s) for s in SESSIONS], load=load_objects)
def fuse_objects(run: str, force: bool = False) -> dict[str, list[Object3D]]:
    """Fuse each session's 2D detections into one 3D object per physical object.

    Args:
        run: run name; needs C2 and C3 output.
        force: recompute even if ``objects/objects_{A,B}.json`` exist.
    """
    cfg = load_run_config(run).fusion
    device = models.resolve_device(load_run_config(run).detect.device)
    recon = load_reconstruction(run)
    detections = load_detections(run)
    shutil.rmtree(run_path(run, "objects/crops"), ignore_errors=True)  # no stale crops
    run_path(run, "objects/crops").mkdir(parents=True)
    result = {}
    for s in SESSIONS:
        with timed(f"C4 session {s}", log):
            log.info("Session %s", s)
            dets = [d for d in detections if d.session == s]
            result[s] = _fuse_session(run, s, recon.cameras[s], dets, cfg, device)
    models.unload()
    write_debug(run, result, recon)
    for s in SESSIONS:  # written last: marks the stage as done
        save_json(run_path(run, objects_json(s)), result[s])
    return result
