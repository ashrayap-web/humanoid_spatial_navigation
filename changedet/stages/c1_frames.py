"""C1 — Frame extraction & quality filtering.

Turns each recording (~4000 frames at 60 fps) into ~50-100 sharp, well-spread keyframes:

1. score every frame's sharpness (variance of Laplacian on a half-resolution grayscale decode),
2. keep the sharpest frame in each ``1 / frames.fps`` second window,
3. drop frames much blurrier than their temporal neighbours (relative, so low-texture views such
   as a plain wall are not mistaken for blur),
4. drop frames where the camera barely moved since the last kept frame (Record3D poses),
5. cap the count by uniform subsampling, resize, and write ``frames/{A,B}/<index>.jpg``.
"""

from __future__ import annotations

import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from changedet.core.cache import cached_stage, load_json, run_dir, run_path, save_json
from changedet.core.config import load_run_config
from changedet.core.logging import get_logger, timed
from changedet.core.types import Frame
from changedet.io.record3d import Record3DExport, load_record3d

log = get_logger("c1")

FRAMES_JSON = "frames/frames.json"
SESSIONS = ("A", "B")


# --------------------------------------------------------------------------------------------
# Selection steps (pure functions, unit-tested)
# --------------------------------------------------------------------------------------------


def sharpness(gray: np.ndarray) -> float:
    """Variance of the Laplacian: higher means sharper."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def best_in_windows(timestamps: np.ndarray, scores: np.ndarray, fps: float) -> np.ndarray:
    """Index of the highest-scoring frame in each ``1/fps``-second window, in time order."""
    windows = np.floor((timestamps - timestamps[0]) * fps).astype(np.int64)
    best = {}
    for i, w in enumerate(windows):
        if w not in best or scores[i] > scores[best[w]]:
            best[w] = i
    return np.array([best[w] for w in sorted(best)], dtype=np.int64)


def relative_blur_keep(scores: np.ndarray, ratio: float, half_window: int) -> np.ndarray:
    """Keep mask: False where a frame's score is below ``ratio`` x the median of its neighbours
    (up to ``half_window`` frames on each side, excluding itself)."""
    keep = np.ones(len(scores), dtype=bool)
    for i in range(len(scores)):
        lo, hi = max(0, i - half_window), min(len(scores), i + half_window + 1)
        neighbours = np.delete(scores[lo:hi], i - lo)
        if len(neighbours) and scores[i] < ratio * np.median(neighbours):
            keep[i] = False
    return keep


def motion_keep(
    quats_xyzw: np.ndarray, positions: np.ndarray, min_trans_m: float, min_rot_deg: float
) -> np.ndarray:
    """Keep mask: a frame is kept if, since the last kept frame, the camera moved at least
    ``min_trans_m`` **or** rotated at least ``min_rot_deg``. The first frame is always kept."""
    rotations = Rotation.from_quat(quats_xyzw)
    keep = np.zeros(len(positions), dtype=bool)
    last = 0
    keep[0] = len(positions) > 0
    for i in range(1, len(positions)):
        angle = np.degrees((rotations[last].inv() * rotations[i]).magnitude())
        moved = np.linalg.norm(positions[i] - positions[last])
        if moved >= min_trans_m or angle >= min_rot_deg:
            keep[i] = True
            last = i
    return keep


def uniform_cap(n: int, max_n: int) -> np.ndarray:
    """Indices of at most ``max_n`` items spread uniformly over ``range(n)``."""
    if n <= max_n:
        return np.arange(n)
    return np.unique(np.round(np.linspace(0, n - 1, max_n)).astype(np.int64))


# --------------------------------------------------------------------------------------------
# Stage
# --------------------------------------------------------------------------------------------


def load_frames(run: str) -> dict[str, list[Frame]]:
    """Read ``frames/frames.json`` back into Frame objects, keyed by session."""
    data = load_json(run_path(run, FRAMES_JSON))
    return {session: [Frame.from_dict(d) for d in frames] for session, frames in data.items()}


def _score_all(rec: Record3DExport) -> np.ndarray:
    """Sharpness of every frame (half-resolution decode — fast, and fine for ranking)."""

    def score(i: int) -> float:
        return sharpness(cv2.imread(str(rec.rgb_path(i)), cv2.IMREAD_REDUCED_GRAYSCALE_2))

    with ThreadPoolExecutor() as pool:  # OpenCV releases the GIL
        return np.array(list(pool.map(score, range(len(rec)))))


def _write_image(src: Path, dst: Path, max_side: int) -> None:
    """Copy the frame (no re-encode) or downscale it so the long side is ``max_side``."""
    image = cv2.imread(str(src))
    h, w = image.shape[:2]
    if max(h, w) <= max_side:
        shutil.copyfile(src, dst)
        return
    scale = max_side / max(h, w)
    image = cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(dst), image, [cv2.IMWRITE_JPEG_QUALITY, 95])


def _select_record3d(rec: Record3DExport, cfg) -> tuple[np.ndarray, np.ndarray]:
    """Return (selected source indices, sharpness of every source frame)."""
    scores = _score_all(rec)
    candidates = best_in_windows(rec.timestamps, scores, cfg.fps)
    sharp = candidates[relative_blur_keep(scores[candidates], cfg.blur_ratio, cfg.blur_window)]
    moving = sharp[
        motion_keep(rec.poses[sharp, :4], rec.poses[sharp, 4:], cfg.min_trans_m, cfg.min_rot_deg)
    ]
    selected = moving[uniform_cap(len(moving), cfg.max_per_session)]
    log.info(
        "  %d source frames -> %d windows -> %d after blur filter -> %d after motion filter "
        "-> %d kept",
        len(rec),
        len(candidates),
        len(sharp),
        len(moving),
        len(selected),
    )
    return selected, scores


def _extract_session(run: str, session: str, source: dict, cfg) -> list[Frame]:
    if source["kind"] != "record3d":
        # TODO(C12): plain video input (OpenCV decode honouring rotation metadata, grayscale-diff
        # de-duplication with frames.min_diff instead of poses).
        raise NotImplementedError(f"C1 supports Record3D input only so far (got {source['kind']})")
    rec = load_record3d(source["path"])
    selected, scores = _select_record3d(rec, cfg)

    out_dir = run_path(run, f"frames/{session}")
    shutil.rmtree(out_dir, ignore_errors=True)  # no stale frames from an earlier, larger run
    out_dir.mkdir(parents=True)
    frames = []
    for index, src in enumerate(selected):
        relpath = f"frames/{session}/{index:06d}.jpg"
        _write_image(rec.rgb_path(int(src)), run_dir(run) / relpath, cfg.max_side)
        frames.append(
            Frame(
                session=session,
                index=index,
                source_index=int(src),
                timestamp=float(rec.timestamps[src] - rec.timestamps[0]),
                image_path=relpath,
                sharpness=float(scores[src]),
            )
        )
    return frames


def contact_sheet(
    run: str, frames: list[Frame], path: Path, cols: int = 10, thumb_w: int = 160
) -> Path:
    """Grid of thumbnails labelled ``index / source_index / sharpness``."""
    thumbs = []
    for f in frames:
        image = cv2.imread(str(run_dir(run) / f.image_path))
        h, w = image.shape[:2]
        thumb = cv2.resize(image, (thumb_w, round(h * thumb_w / w)), interpolation=cv2.INTER_AREA)
        cv2.rectangle(thumb, (0, 0), (thumb_w, 16), (0, 0, 0), -1)
        label = f"{f.index} src{f.source_index} s{f.sharpness:.0f}"
        cv2.putText(thumb, label, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
        thumbs.append(thumb)
    thumbs += [np.zeros_like(thumbs[0])] * (-len(thumbs) % cols)
    rows = [np.hstack(thumbs[r : r + cols]) for r in range(0, len(thumbs), cols)]
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.vstack(rows))
    return path


@cached_stage(FRAMES_JSON, load=load_frames)
def extract_frames(run: str, force: bool = False) -> dict[str, list[Frame]]:
    """Select sharp, well-spread keyframes for both sessions.

    Args:
        run: run name (``runs/<run>/`` must contain ``inputs.json`` and ``config.yaml``).
        force: recompute even if ``frames/frames.json`` exists.

    Returns:
        Kept frames keyed by session ("A", "B").
    """
    cfg = load_run_config(run).frames
    inputs = load_json(run_path(run, "inputs.json"))
    result = {}
    for session in SESSIONS:
        with timed(f"C1 session {session}", log):
            log.info("Session %s: %s", session, inputs[session]["path"])
            result[session] = _extract_session(run, session, inputs[session], cfg)
        sheet = contact_sheet(
            run, result[session], run_path(run, f"viz/c1_contact_sheet_{session}.png")
        )
        log.info("  contact sheet: %s", sheet)
    save_json(run_path(run, FRAMES_JSON), result)  # written last: marks the stage as done
    return result
