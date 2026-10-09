"""C10 visualization on a synthetic run, the `all` pipeline driver and `export`."""

from __future__ import annotations

import cv2
import numpy as np

import changedet.cli as cli
from changedet.core import cache
from changedet.core.config import load_config, save_config
from changedet.core.types import (
    CameraFrame,
    Change,
    ChangeType,
    Confidence,
    Frame,
    Object3D,
    Reconstruction,
)

RUN = "synth"


def make_synthetic_run(rng) -> None:
    """Background, two cameras, three objects and one change of each kind (C2/C4/C6 outputs)."""
    room = rng.uniform([-2, -2, 0], [2, 2, 2.4], size=(5000, 3))
    cache.save_ply(cache.run_path(RUN, "recon/background.ply"), room, np.full_like(room, 0.5))
    cams = {
        s: [
            CameraFrame(
                Frame(s, k, k, k, f"frames/{s}/{k:06d}.jpg", 1.0),
                np.array([[500.0, 0, 359.5], [0, 500.0, 479.5], [0, 0, 1]]),
                np.eye(4) + np.diag([0, 0, 0, 0]) * k,
                "",
                None,
            )
            for k in range(3)
        ]
        for s in "AB"
    }
    for s in "AB":
        for k, cam in enumerate(cams[s]):
            cam.T_world_cam[:3, 3] = [k * 0.3, 0, 1.3]
    recon = Reconstruction(
        cams, {"A": "", "B": ""}, "recon/background.ply", np.array([0, 0, 1.0, 0]), True, 0.01
    )
    cache.save_json(cache.run_path(RUN, "recon/reconstruction.json"), recon)

    crop = cache.run_path(RUN, "objects/crops/A_000.jpg")
    crop.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(crop), np.full((40, 60, 3), 90, np.uint8))
    objects = {"A": [], "B": []}
    for oid, centre in (
        ("A_000", [0.5, 0.5, 0.6]),
        ("A_001", [-1, 0, 0.4]),
        ("B_000", [-1, 0.05, 0.4]),
        ("B_001", [1, -1, 0.2]),
    ):
        pts = rng.normal(centre, 0.1, size=(300, 3))
        cache.save_ply(cache.run_path(RUN, f"objects/{oid[0]}/{oid}.ply"), pts)
        objects[oid[0]].append(
            Object3D(
                oid,
                oid[0],
                "thing",
                {"thing": 1.0},
                f"objects/{oid[0]}/{oid}.ply",
                pts.mean(0),
                pts.min(0),
                pts.max(0),
                5,
                np.zeros(4),
                np.zeros(4),
                (oid[0], 0),
                "objects/crops/A_000.jpg",
            )
        )
    for s in "AB":
        cache.save_json(cache.run_path(RUN, f"objects/objects_{s}.json"), objects[s])

    def ch(i, kind, conf, a, b):
        pa = next((o.centroid for o in objects["A"] if o.id == a), None)
        pb = next((o.centroid for o in objects["B"] if o.id == b), None)
        return Change(
            f"chg_{i:03d}",
            kind,
            "thing",
            a,
            b,
            pa,
            pb,
            None,
            None,
            None,
            None,
            conf,
            description="A sentence.",
        )

    changes = [
        ch(0, ChangeType.REMOVED, Confidence.CONFIRMED, "A_000", None),
        ch(1, ChangeType.ADDED, Confidence.UNVERIFIED, None, "B_001"),
        ch(2, ChangeType.UNCHANGED, Confidence.CONFIRMED, "A_001", "B_000"),
    ]
    cache.save_json(cache.run_path(RUN, "changes/changes_vis.json"), changes)
    save_config(
        load_config(overrides=["viz.gif_frames=3", "viz.gif_points=500"]),
        cache.run_path(RUN, "config.yaml"),
    )


def test_visualize_builds_rrd_and_hero_images() -> None:
    from changedet.stages.c10_visualize import visualize

    make_synthetic_run(np.random.default_rng(0))
    path = visualize(RUN, open_viewer=False)
    assert path.endswith("viz/final.rrd")
    for rel in ("viz/final.rrd", "viz/hero.png", "viz/hero.gif"):
        assert cache.run_path(RUN, rel).stat().st_size > 1000, rel


def test_all_runs_stages_in_order_with_from(monkeypatch, record3d_export) -> None:
    calls = []
    monkeypatch.setattr(
        cli, "_call", lambda stage, run, force, open_viewer=False: calls.append((stage, force))
    )
    args = ["all", "--run", "demo", "--a", str(record3d_export), "--b", str(record3d_export)]
    assert cli.main(args) == 0
    implemented = [s for s in cli.STAGES if s in cli.STAGE_ENTRIES]
    assert [s for s, _ in calls] == implemented  # c7, c9 skipped; c11 only with --gt
    assert not any(force for _, force in calls)

    calls.clear()
    assert cli.main(["all", "--run", "demo", "--from", "c4"]) == 0
    forced = {s for s, force in calls if force}
    assert forced == {s for s in implemented if list(cli.STAGES).index(s) >= 3}
    assert cli.main(["all", "--run", "nope"]) == 1  # not initialised


def test_export_rewrites_image_links(tmp_path) -> None:
    from changedet.export import export_run

    crop = cache.run_path(RUN, "objects/crops/A_011.jpg")
    crop.parent.mkdir(parents=True, exist_ok=True)
    crop.write_bytes(b"jpg")
    report = cache.run_path(RUN, "report/report.md")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("# r\n![A_011 (before)](../objects/crops/A_011.jpg)\n")
    out = export_run(RUN, tmp_path / "example")
    assert "(crops/A_011.jpg)" in (out / "report.md").read_text()
    assert (out / "crops" / "A_011.jpg").read_bytes() == b"jpg"
