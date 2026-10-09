from __future__ import annotations

import pytest

from changedet.core.config import apply_overrides, load_config, save_config

SECTIONS = [
    "frames",
    "recon",
    "detect",
    "fusion",
    "match",
    "visibility",
    "vlm",
    "report",
    "nav",
    "viz",
    "eval",
]


def test_default_config_has_every_section() -> None:
    cfg = load_config()
    for section in SECTIONS:
        assert section in cfg, section
    assert cfg.recon.method == "record3d"
    assert cfg["frames"]["fps"] == cfg.frames.fps


def test_overrides_are_parsed_as_yaml() -> None:
    cfg = load_config(
        overrides=[
            "frames.fps=5",
            "recon.voxel_size=0.01",
            "vlm.enabled=false",
            "detect.vocabulary=[bag, bed]",
            "nav.start=[0.5, 1.0]",
        ]
    )
    assert cfg.frames.fps == 5
    assert cfg.recon.voxel_size == 0.01
    assert cfg.vlm.enabled is False
    assert cfg.detect.vocabulary == ["bag", "bed"]
    assert cfg.nav.start == [0.5, 1.0]


def test_override_does_not_mutate_input() -> None:
    data = {"frames": {"fps": 3}}
    assert apply_overrides(data, ["frames.fps=7"])["frames"]["fps"] == 7
    assert data["frames"]["fps"] == 3


@pytest.mark.parametrize("bad", ["frames.fsp=5", "nosection.key=1", "frames.fps.deeper=1"])
def test_unknown_keys_fail_loudly(bad: str) -> None:
    with pytest.raises(KeyError):
        load_config(overrides=[bad])


def test_malformed_override() -> None:
    with pytest.raises(ValueError):
        load_config(overrides=["frames.fps"])


def test_snapshot_round_trip(tmp_path) -> None:
    cfg = load_config(overrides=["match.max_cost=2.5"])
    path = save_config(cfg, tmp_path / "config.yaml")
    assert load_config(path) == cfg


def test_run_config_picks_up_new_default_keys(tmp_path) -> None:
    from changedet.core.cache import run_path
    from changedet.core.config import load_run_config

    # An old snapshot: missing a section and a key, with one customised value.
    save_config({"frames": {"fps": 7}}, run_path("old", "config.yaml"))
    cfg = load_run_config("old")
    assert cfg.frames.fps == 7  # the run's own value wins
    assert cfg.frames.max_side == load_config().frames.max_side  # new key from defaults
    assert cfg.detect.detector_model == load_config().detect.detector_model  # new section
