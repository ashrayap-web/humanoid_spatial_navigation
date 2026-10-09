from __future__ import annotations

from changedet.cli import main
from changedet.core import cache
from changedet.core.config import load_run_config
from tests.conftest import write_record3d_export


def test_init_creates_run(record3d_export, tmp_path) -> None:
    video = tmp_path / "after.mp4"
    video.write_bytes(b"")
    assert (
        main(
            [
                "init",
                "--run",
                "demo",
                "--a",
                str(record3d_export),
                "--b",
                str(video),
                "--set",
                "frames.fps=5",
            ]
        )
        == 0
    )

    inputs = cache.load_json(cache.run_path("demo", "inputs.json"))
    assert inputs["A"] == {"path": str(record3d_export.resolve()), "kind": "record3d"}
    assert inputs["B"]["kind"] == "video"
    assert load_run_config("demo").frames.fps == 5
    assert cache.exists("demo", "logs/run.log")


def test_init_refuses_to_change_inputs_without_force(record3d_export, tmp_path) -> None:
    other = write_record3d_export(tmp_path / "rec_b")
    assert main(["init", "--run", "demo", "--a", str(record3d_export), "--b", str(other)]) == 0
    assert main(["init", "--run", "demo", "--a", str(other), "--b", str(record3d_export)]) == 1
    assert (
        main(["init", "--run", "demo", "--a", str(other), "--b", str(record3d_export), "--force"])
        == 0
    )
    assert cache.load_json(cache.run_path("demo", "inputs.json"))["A"]["path"].endswith("rec_b")


def test_init_rejects_missing_input(tmp_path) -> None:
    assert main(["init", "--run", "demo", "--a", str(tmp_path / "nope"), "--b", "x"]) == 1


def test_stage_stub_and_overrides(record3d_export) -> None:
    main(["init", "--run", "demo", "--a", str(record3d_export), "--b", str(record3d_export)])
    assert main(["c9", "--run", "demo", "--set", "detect.min_score=0.5"]) == 1  # not implemented
    assert load_run_config("demo").detect.min_score == 0.5
    assert main(["c1", "--run", "missing"]) == 1
