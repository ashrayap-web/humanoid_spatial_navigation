"""Reader for Record3D exports (spec Section 8.1). Pure I/O: no coordinate conversions.

Layout::

    <session>/
      rgb/0.jpg … N.jpg       # 720x960, numeric (not zero-padded) names
      depth/0.exr … N.exr     # 192x256 float32 metres, channel "R", NaN = invalid
      metadata*.json          # name varies, e.g. "metadata (1).json"
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import OpenEXR


def find_metadata(root: str | Path) -> Path:
    """Return the single ``metadata*.json`` file in a Record3D export."""
    matches = sorted(Path(root).glob("metadata*.json"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one metadata*.json in {root}, found {matches}")
    return matches[0]


def is_record3d_dir(path: str | Path) -> bool:
    """True if ``path`` looks like a Record3D export (rgb/, depth/ and metadata*.json)."""
    path = Path(path)
    return (
        path.is_dir()
        and (path / "rgb").is_dir()
        and (path / "depth").is_dir()
        and any(path.glob("metadata*.json"))
    )


def _numeric_ids(folder: Path, suffix: str) -> list[int]:
    return sorted(int(p.stem) for p in folder.glob(f"*{suffix}") if p.stem.isdigit())


@dataclass
class Record3DExport:
    """One Record3D recording. Frame ``i`` is ``rgb/i.jpg``, ``depth/i.exr`` and row ``i`` of the
    per-frame metadata arrays."""

    root: Path
    metadata_path: Path
    width: int  # RGB size
    height: int
    depth_width: int
    depth_height: int
    fps: float
    K: np.ndarray  # (3,3) at RGB resolution
    intrinsics: np.ndarray | None  # (N,4) per-frame [fx, fy, cx, cy] at RGB resolution
    timestamps: np.ndarray  # (N,) seconds
    poses: np.ndarray  # (N,7) raw [qx, qy, qz, qw, tx, ty, tz], ARKit world, OpenGL camera

    def __len__(self) -> int:
        return len(self.timestamps)

    def rgb_path(self, i: int) -> Path:
        return self.root / "rgb" / f"{i}.jpg"

    def depth_path(self, i: int) -> Path:
        return self.root / "depth" / f"{i}.exr"

    def read_rgb(self, i: int) -> np.ndarray:
        """(H,W,3) uint8 image in **RGB** channel order."""
        image = cv2.imread(str(self.rgb_path(i)), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(self.rgb_path(i))
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def read_depth(self, i: int) -> np.ndarray:
        """(depth_height, depth_width) float32 depth in metres, as stored (NaN = invalid)."""
        path = self.depth_path(i)
        if not path.exists():
            raise FileNotFoundError(path)
        channels = OpenEXR.File(str(path)).channels()
        channel = channels["R"] if "R" in channels else next(iter(channels.values()))
        return np.asarray(channel.pixels, dtype=np.float32)


def load_record3d(root: str | Path) -> Record3DExport:
    """Read a Record3D export's metadata and check it is consistent with the frame files."""
    root = Path(root)
    if not is_record3d_dir(root):
        raise FileNotFoundError(
            f"{root} is not a Record3D export (needs rgb/, depth/, metadata*.json)"
        )
    metadata_path = find_metadata(root)
    meta = json.loads(metadata_path.read_text())

    poses = np.asarray(meta["poses"], dtype=np.float64)
    timestamps = np.asarray(meta["frameTimestamps"], dtype=np.float64)
    n = len(poses)
    rgb_ids = _numeric_ids(root / "rgb", ".jpg")
    depth_ids = _numeric_ids(root / "depth", ".exr")
    expected = list(range(n))
    if rgb_ids != expected or depth_ids != expected or len(timestamps) != n:
        raise ValueError(
            f"{root}: inconsistent export — {n} poses, {len(timestamps)} timestamps, "
            f"{len(rgb_ids)} rgb frames, {len(depth_ids)} depth frames (ids must be 0..N-1)"
        )

    # Record3D stores K as a flat column-major 3x3 (cx, cy are elements 6 and 7).
    K = np.asarray(meta["K"], dtype=np.float64).reshape(3, 3).T
    intrinsics = meta.get("perFrameIntrinsicCoeffs")

    return Record3DExport(
        root=root,
        metadata_path=metadata_path,
        width=int(meta["w"]),
        height=int(meta["h"]),
        depth_width=int(meta["dw"]),
        depth_height=int(meta["dh"]),
        fps=float(meta["fps"]),
        K=K,
        intrinsics=np.asarray(intrinsics, dtype=np.float64) if intrinsics else None,
        timestamps=timestamps,
        poses=poses,
    )
