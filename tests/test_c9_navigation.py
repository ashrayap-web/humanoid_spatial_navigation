from __future__ import annotations

import numpy as np

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
from changedet.stages.c9_navigation import (
    FREE,
    MARGIN,
    OBSTACLE,
    UNKNOWN,
    Grid,
    astar,
    farthest_pair,
    navigation_impact,
    occupancy,
    path_length,
    verdict,
    with_navigation_section,
)

RUN = "navtest"


def test_astar_and_length() -> None:
    free = np.ones((10, 10), bool)
    free[5, :9] = False  # wall with a gap at the end
    path = astar(free, (0, 0), (9, 0))
    assert path[0] == (0, 0) and path[-1] == (9, 0)
    assert all(free[c] for c in path)
    assert path_length(path, 0.1) > 0.9  # had to detour around the wall
    assert astar(free, (0, 0), (5, 0)) is None  # goal inside the wall
    assert np.isclose(path_length([(0, 0), (1, 1), (2, 1)], 1.0), np.sqrt(2) + 1)


def test_occupancy_and_farthest_pair() -> None:
    floor = np.zeros((20, 10), bool)
    floor[2:18, 2:8] = True
    obstacles = np.zeros_like(floor)
    obstacles[10, 5] = True
    occ = occupancy(floor, obstacles, radius_cells=1, unknown_is_free=False)
    assert occ[10, 5] == OBSTACLE and occ[10, 6] == MARGIN and occ[0, 0] == UNKNOWN
    assert occ[3, 3] == FREE
    u, v = farthest_pair(occ == FREE, (5, 5))
    assert abs(u[0] - v[0]) >= 14  # the two ends of the corridor


def test_verdict_sentences() -> None:
    assert verdict(2.0, 2.05, [], [], "near the desk", "near the bed", 0.1).startswith(
        "Navigation is not affected: the route from near the desk to near the bed is 2.0 m"
    )
    longer = verdict(2.0, 3.4, ["the new box (chg_001)"], [], "near the desk", "near the bed", 0.1)
    assert longer.startswith("The new box (chg_001) blocks the direct way")
    assert "1.4 m longer (2.0 m -> 3.4 m)" in longer
    assert "no longer a way through" in verdict(
        2.0, None, ["the new box (chg_001)"], [], "a", "b", 0.1
    )
    assert "shorter" in verdict(3.0, 2.0, [], ["the chair (chg_002)"], "a", "b", 0.1)


def _make_corridor_run(add_box: bool) -> None:
    """A 4 m x 1.6 m corridor room; in B a box may stand in the middle of it."""
    rng = np.random.default_rng(0)
    xs, ys = np.meshgrid(np.arange(0, 4, 0.02), np.arange(0, 1.6, 0.02))
    floor = np.column_stack([xs.ravel(), ys.ravel(), np.zeros(xs.size)])
    wall_y = np.column_stack(
        [
            np.repeat(np.arange(0, 4, 0.02), 30),
            np.tile(np.r_[np.full(15, -0.05), np.full(15, 1.65)], 200),
            rng.uniform(0.1, 1.0, 6000),
        ]
    )
    wall_x = np.column_stack(
        [np.tile([-0.05, 4.05], 1000), rng.uniform(0, 1.6, 2000), rng.uniform(0.1, 1.0, 2000)]
    )
    room = np.vstack([floor, wall_y, wall_x])
    box = rng.uniform([1.8, 0.0, 0.0], [2.2, 1.0, 0.6], size=(3000, 3))  # leaves a 0.6 m gap
    for s in "AB":
        cloud = np.vstack([room, box]) if (s == "B" and add_box) else room
        cache.save_ply(cache.run_path(RUN, f"recon/cloud_{s}.ply"), cloud)
    cache.save_ply(cache.run_path(RUN, "recon/background.ply"), room)
    K = np.array([[500.0, 0, 359.5], [0, 500.0, 479.5], [0, 0, 1]])
    cams = {}
    for s in "AB":
        cams[s] = []
        for k, x in enumerate(np.linspace(0.3, 3.7, 8)):
            T = np.eye(4)
            T[:3, 3] = [x, 0.8, 1.3]
            cams[s].append(CameraFrame(Frame(s, k, k, k, "", 1.0), K, T, ""))
    recon = Reconstruction(
        cams,
        {s: f"recon/cloud_{s}.ply" for s in "AB"},
        "recon/background.ply",
        np.array([0, 0, 1.0, 0]),
        True,
        0.01,
    )
    cache.save_json(cache.run_path(RUN, "recon/reconstruction.json"), recon)
    cache.save_ply(cache.run_path(RUN, "objects/B/B_000.ply"), box)
    obj = Object3D(
        "B_000",
        "B",
        "box",
        {"box": 1.0},
        "objects/B/B_000.ply",
        box.mean(0),
        box.min(0),
        box.max(0),
        5,
        np.zeros(4),
        np.zeros(4),
        ("B", 0),
        "",
    )
    cache.save_json(cache.run_path(RUN, "objects/objects_A.json"), [])
    cache.save_json(cache.run_path(RUN, "objects/objects_B.json"), [obj] if add_box else [])
    changes = (
        [
            Change(
                "chg_001",
                ChangeType.ADDED,
                "box",
                None,
                "B_000",
                None,
                box.mean(0),
                None,
                None,
                None,
                None,
                Confidence.CONFIRMED,
            )
        ]
        if add_box
        else []
    )
    cache.save_json(cache.run_path(RUN, "changes/changes_vis.json"), changes)
    save_config(
        load_config(overrides=["nav.start=[0.3, 0.4]", "nav.goal=[3.7, 0.4]"]),
        cache.run_path(RUN, "config.yaml"),
    )


def test_added_box_makes_a_detour_and_is_blamed() -> None:
    _make_corridor_run(add_box=True)
    nav = navigation_impact(RUN)
    assert nav["length_after_m"] > nav["length_before_m"] + 0.3
    assert nav["blocking_changes"] == ["chg_001"]
    assert "The new box (chg_001) blocks the direct way" in nav["sentence"]
    assert cache.exists(RUN, "nav/nav_diff.png") and cache.exists(RUN, "nav/occupancy_B.npy")


def test_no_change_no_impact() -> None:
    _make_corridor_run(add_box=False)
    nav = navigation_impact(RUN)
    assert nav["length_before_m"] == nav["length_after_m"]
    assert nav["blocking_changes"] == [] and nav["sentence"].startswith("Navigation is not")


def test_grid_index_round_trip() -> None:
    g = Grid(np.array([-1.0, -2.0]), 0.1, (30, 40))
    ij = g.index(np.array([[0.05, 0.05, 0.0]]))[0]
    np.testing.assert_allclose(g.centre(ij), [0.05, 0.05])


def test_with_navigation_section_replaces_in_place() -> None:
    report = "# R\n\n## Changes\n\nx\n\n## Navigation impact\n\nold\n\n## Run information\n\n- a\n"
    out = with_navigation_section(report, "## Navigation impact\n\nnew\n")
    assert out.count("## Navigation impact") == 1 and "new" in out and "old" not in out
    assert out.index("## Navigation impact") < out.index("## Run information")
    assert out.endswith("- a\n")
    fresh = with_navigation_section("# R\n\n## Run information\n\n- a\n", "## Navigation impact\n")
    assert fresh.index("## Navigation impact") < fresh.index("## Run information")
