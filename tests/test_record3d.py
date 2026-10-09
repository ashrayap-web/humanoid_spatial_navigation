from __future__ import annotations

import numpy as np
import pytest

from changedet.io.record3d import is_record3d_dir, load_record3d
from tests.conftest import REPO_ROOT


def test_load_synthetic_export(record3d_export) -> None:
    rec = load_record3d(record3d_export)
    assert len(rec) == 3
    assert (rec.width, rec.height, rec.depth_width, rec.depth_height) == (6, 8, 3, 4)
    np.testing.assert_allclose(rec.K, [[7, 0, 3], [0, 7, 4], [0, 0, 1]])  # column-major decoded
    assert rec.intrinsics.shape == (3, 4)
    assert rec.poses.shape == (3, 7)

    rgb = rec.read_rgb(1)
    assert rgb.shape == (8, 6, 3) and rgb.dtype == np.uint8
    assert rgb[..., 0].mean() > 150 and rgb[..., 2].mean() < 50  # RGB order (red image)

    depth = rec.read_depth(2)
    assert depth.shape == (4, 3) and depth.dtype == np.float32
    assert np.isnan(depth[0, 0])  # raw: NaN kept, conversion is C2's job
    assert np.nanmax(depth) == pytest.approx(3.0)


def test_detects_export_dirs(record3d_export, tmp_path) -> None:
    assert is_record3d_dir(record3d_export)
    assert not is_record3d_dir(tmp_path)
    assert not is_record3d_dir(record3d_export / "rgb")


def test_inconsistent_export_fails(record3d_export) -> None:
    (record3d_export / "depth" / "1.exr").unlink()
    with pytest.raises(ValueError, match="inconsistent"):
        load_record3d(record3d_export)


def test_ambiguous_metadata_fails(record3d_export) -> None:
    (record3d_export / "metadata (2).json").write_text("{}")
    with pytest.raises(FileNotFoundError, match="exactly one"):
        load_record3d(record3d_export)


@pytest.mark.real_data
@pytest.mark.parametrize("session", ["vid1", "vid2"])
def test_real_recordings(session: str) -> None:
    root = REPO_ROOT / session
    if not root.exists():
        pytest.skip(f"{root} not present")
    rec = load_record3d(root)
    assert len(rec) > 1000
    assert rec.read_rgb(0).shape == (960, 720, 3)
    depth = rec.read_depth(0)
    assert depth.shape == (256, 192)
    assert 0.2 < np.nanmedian(depth) < 5.0
    assert rec.K[0, 2] == pytest.approx(rec.width / 2, rel=0.05)
    assert rec.K[1, 2] == pytest.approx(rec.height / 2, rel=0.05)
