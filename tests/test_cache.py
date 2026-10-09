from __future__ import annotations

import numpy as np

from changedet.core import cache
from changedet.core.types import Frame


def test_run_dir_uses_env(runs_dir) -> None:
    assert cache.run_dir("demo") == runs_dir / "demo"
    assert not cache.exists("demo", "frames/frames.json")


def test_json_round_trip_with_dataclasses(tmp_path) -> None:
    frames = [Frame("A", 0, 10, 0.5, "frames/A/000000.jpg", 100.0)]
    path = cache.save_json(tmp_path / "sub" / "frames.json", {"A": frames, "arr": np.arange(3)})
    data = cache.load_json(path)
    assert Frame.from_dict(data["A"][0]) == frames[0]
    assert data["arr"] == [0, 1, 2]
    assert not list(path.parent.glob("*.tmp"))  # atomic write leaves no temp file


def test_ply_round_trip(tmp_path) -> None:
    points = np.random.default_rng(0).normal(size=(50, 3))
    colors = np.full((50, 3), 255, np.uint8)
    path = cache.save_ply(tmp_path / "cloud.ply", points, colors)
    loaded, loaded_colors = cache.load_ply(path)
    np.testing.assert_allclose(loaded, points, atol=1e-6)
    np.testing.assert_allclose(loaded_colors, 1.0)

    cache.save_ply(tmp_path / "plain.ply", points)
    assert cache.load_ply(tmp_path / "plain.ply")[1] is None


def test_cached_stage_skips_and_forces() -> None:
    calls = []

    @cache.cached_stage(
        "out/result.json", load=lambda run: cache.load_json(cache.run_path(run, "out/result.json"))
    )
    def stage(run: str, force: bool = False) -> dict:
        calls.append(force)
        result = {"n": len(calls)}
        cache.save_json(cache.run_path(run, "out/result.json"), result)
        return result

    assert stage("demo") == {"n": 1}
    assert stage("demo") == {"n": 1}  # skipped, loaded from cache
    assert calls == [False]
    assert stage("demo", force=True) == {"n": 2}
    assert calls == [False, True]


def test_cached_stage_requires_all_outputs() -> None:
    calls = []

    @cache.cached_stage(["a.json", "b.json"])
    def stage(run: str, force: bool = False) -> None:
        calls.append(1)

    cache.save_json(cache.run_path("demo", "a.json"), {})
    stage("demo")
    assert calls == [1]  # b.json missing -> ran
    cache.save_json(cache.run_path("demo", "b.json"), {})
    assert stage("demo") is None
    assert calls == [1]
