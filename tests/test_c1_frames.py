from __future__ import annotations

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from changedet.core import cache
from changedet.core.config import load_config, save_config
from changedet.core.types import Frame
from changedet.stages.c1_frames import (
    best_in_windows,
    extract_frames,
    load_frames,
    motion_keep,
    relative_blur_keep,
    sharpness,
    uniform_cap,
)
from tests.conftest import write_record3d_export

# --------------------------------------------------------------------------------------------
# Selection steps
# --------------------------------------------------------------------------------------------


def test_sharpness_drops_with_blur() -> None:
    texture = np.random.default_rng(0).integers(0, 255, (64, 64), dtype=np.uint8)
    assert sharpness(cv2.GaussianBlur(texture, (9, 9), 3)) < 0.2 * sharpness(texture)


def test_best_in_windows() -> None:
    timestamps = np.arange(12) / 6.0  # 6 fps for 2 s
    scores = np.array([1, 5, 2, 0, 0, 9, 3, 3, 1, 0, 7, 0], dtype=float)
    # fps=2 -> windows of 3 frames
    np.testing.assert_array_equal(best_in_windows(timestamps, scores, fps=2), [1, 5, 6, 10])


def test_relative_blur_keeps_low_texture_runs() -> None:
    # A low-texture stretch (frames 4-9) is NOT blur; the isolated dip at 13 is.
    scores = np.array([500.0] * 4 + [150.0] * 6 + [500.0] * 3 + [100.0] + [500.0] * 4)
    keep = relative_blur_keep(scores, ratio=0.7, half_window=3)
    assert not keep[13]
    assert keep[[5, 6, 7, 8]].all()
    assert keep.sum() >= len(scores) - 3  # only the dip and the run's edges can go


def test_motion_keep() -> None:
    positions = np.array([[0, 0, 0], [0.01, 0, 0], [0.02, 0, 0], [0.08, 0, 0], [0.08, 0, 0]])
    quats = Rotation.from_euler("z", [[0], [0], [0], [0], [10]], degrees=True).as_quat()
    keep = motion_keep(quats, positions, min_trans_m=0.05, min_rot_deg=5)
    np.testing.assert_array_equal(keep, [True, False, False, True, True])


def test_motion_keep_accumulates_slow_drift() -> None:
    # Rotating 2 deg per frame: each step is small, but drift since the last kept frame adds up.
    quats = Rotation.from_euler("y", np.arange(10)[:, None] * 2.0, degrees=True).as_quat()
    keep = motion_keep(quats, np.zeros((10, 3)), min_trans_m=0.05, min_rot_deg=5)
    np.testing.assert_array_equal(np.flatnonzero(keep), [0, 3, 6, 9])


def test_uniform_cap() -> None:
    np.testing.assert_array_equal(uniform_cap(5, 10), np.arange(5))
    capped = uniform_cap(100, 10)
    assert len(capped) == 10 and capped[0] == 0 and capped[-1] == 99


# --------------------------------------------------------------------------------------------
# End to end on a synthetic recording
# --------------------------------------------------------------------------------------------

FPS = 60
SAMPLE_FPS = 6  # windows of 10 frames
BLURRED_WINDOWS = {3, 9}  # every frame in these windows is blurred
STATIC_WINDOWS = range(12, 16)  # camera does not move here


def make_recording(root):
    """2.67 s at 60 fps of a textured scene panning sideways, with fully blurred windows and a
    stationary stretch."""
    rng = np.random.default_rng(0)
    scene = cv2.resize(
        rng.integers(0, 255, (60, 400, 3), dtype=np.uint8),
        (1600, 240),
        interpolation=cv2.INTER_NEAREST,
    )
    images, poses, x = [], [], 0.0
    for i in range(160):
        window = i // (FPS // SAMPLE_FPS)
        if window not in STATIC_WINDOWS:
            x += 0.01  # 0.1 m per window
        offset = int(x * 1000) % 1200
        image = np.ascontiguousarray(scene[:, offset : offset + 320])
        if window in BLURRED_WINDOWS:
            image = cv2.GaussianBlur(image, (21, 21), 6)
        images.append(image)
        poses.append([0, 0, 0, 1, x, 0, 0])
    return write_record3d_export(root, images=images, poses=poses)


@pytest.fixture
def synthetic_run(tmp_path):
    rec = make_recording(tmp_path / "rec")
    cache.save_json(
        cache.run_path("synth", "inputs.json"),
        {s: {"path": str(rec), "kind": "record3d"} for s in "AB"},
    )
    cfg = load_config(
        overrides=[f"frames.fps={SAMPLE_FPS}", "frames.max_side=200", "frames.blur_window=3"]
    )
    save_config(cfg, cache.run_path("synth", "config.yaml"))
    return "synth"


def test_extract_frames_end_to_end(synthetic_run) -> None:
    frames = extract_frames(synthetic_run)
    for session in "AB":
        kept = frames[session]
        windows = [f.source_index // (FPS // SAMPLE_FPS) for f in kept]
        assert not BLURRED_WINDOWS & set(windows), "blurred windows must be dropped"
        assert sum(w in STATIC_WINDOWS for w in windows) <= 1, "stationary frames de-duplicated"
        assert len(kept) >= 10
        assert [f.index for f in kept] == list(range(len(kept)))
        assert all(a.timestamp < b.timestamp for a, b in zip(kept, kept[1:], strict=False))
        image = cv2.imread(str(cache.run_dir(synthetic_run) / kept[0].image_path))
        assert max(image.shape[:2]) == 200  # resized to max_side
        assert cache.exists(synthetic_run, f"viz/c1_contact_sheet_{session}.png")

    assert load_frames(synthetic_run) == frames
    assert isinstance(frames["A"][0], Frame)


def test_extract_frames_is_cached(synthetic_run, monkeypatch) -> None:
    first = extract_frames(synthetic_run)
    import changedet.stages.c1_frames as c1

    def fail(*args, **kwargs):
        raise AssertionError("should have been skipped")

    monkeypatch.setattr(c1, "_extract_session", fail)
    assert extract_frames(synthetic_run) == first
    with pytest.raises(AssertionError):
        extract_frames(synthetic_run, force=True)
