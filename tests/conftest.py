"""Shared fixtures: an isolated runs/ directory and a tiny synthetic Record3D export."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import OpenEXR
import pytest

from changedet.core.cache import RUNS_DIR_ENV

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def runs_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the run cache at a temporary directory for every test."""
    runs = tmp_path / "runs"
    monkeypatch.setenv(RUNS_DIR_ENV, str(runs))
    return runs


def write_record3d_export(
    root: Path,
    n_frames: int = 3,
    metadata_name: str = "metadata (1).json",
    images: list[np.ndarray] | None = None,
    poses: list[list[float]] | None = None,
):
    """Create a minimal Record3D export with column-major K and 4x3 EXR depth.

    By default frames are 8x6 pure-red images moving 0.1 m apart along x. Pass ``images`` (BGR)
    and/or ``poses`` (``[qx, qy, qz, qw, tx, ty, tz]``) to control them; ``n_frames`` then
    follows their length.
    """
    if images is not None:
        n_frames = len(images)
    elif poses is not None:
        n_frames = len(poses)
    if images is None:
        red = np.zeros((8, 6, 3), np.uint8)
        red[..., 2] = 200  # pure red in BGR order
        images = [red] * n_frames
    if poses is None:
        poses = [[0, 0, 0, 1, 0.1 * i, 0, 0] for i in range(n_frames)]
    (root / "rgb").mkdir(parents=True)
    (root / "depth").mkdir()
    for i in range(n_frames):
        cv2.imwrite(str(root / "rgb" / f"{i}.jpg"), images[i], [cv2.IMWRITE_JPEG_QUALITY, 98])
        depth = np.full((4, 3), 1.0 + i, np.float32)
        depth[0, 0] = np.nan
        header = {"compression": OpenEXR.ZIP_COMPRESSION, "type": OpenEXR.scanlineimage}
        OpenEXR.File(header, {"R": depth}).write(str(root / "depth" / f"{i}.exr"))
    h, w = images[0].shape[:2]
    fx, cx, cy = 7.0, w / 2, h / 2
    metadata = {
        "w": w,
        "h": h,
        "dw": 3,
        "dh": 4,
        "fps": 60,
        "cameraType": 1,
        "K": [fx, 0, 0, 0, fx, 0, cx, cy, 1],  # column-major, as Record3D writes it
        "perFrameIntrinsicCoeffs": [[fx, fx, cx, cy]] * n_frames,
        "frameTimestamps": [i / 60 for i in range(n_frames)],
        "poses": poses,
        "initPose": [0, 0, 0, 1, 0, 0, 0],
    }
    (root / metadata_name).write_text(json.dumps(metadata))
    return root


@pytest.fixture
def record3d_export(tmp_path: Path) -> Path:
    return write_record3d_export(tmp_path / "rec_a")
