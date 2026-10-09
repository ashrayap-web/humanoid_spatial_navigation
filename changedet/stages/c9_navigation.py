"""C9 — Navigation impact: what the changes mean for a robot moving through the room.

One static 2D occupancy grid is shared by both sessions, so that differences in *coverage*
(one recording saw more floor) cannot masquerade as navigation changes:

- free floor = floor seen in either recording, plus the cells the camera walked through,
- static obstacles = points between ``nav.min_z`` and ``nav.robot_height`` in either cloud,
  except points belonging to a changed object,
- "before" = static obstacles + the changed objects as they were (removed, moved-from, replaced),
  "after" = static obstacles + the changed objects as they are (added, moved-to, replacement).

Obstacles are inflated by the robot radius; unknown cells are blocked unless
``nav.unknown_is_free``. Start and goal default to the two cells farthest apart (walking distance)
in the free space shared by both grids, inside the region the camera walked through (this also
excludes the "mirror room" a TV's reflection adds behind the wall). A* runs on both grids; a longer
or blocked "after" route is attributed to the changes whose footprint cuts the "before" route.
"""

from __future__ import annotations

import heapq
import re
from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from changedet.core.cache import cached_stage, exists, load_json, load_ply, run_path, save_json
from changedet.core.config import load_run_config
from changedet.core.logging import get_logger
from changedet.core.types import Change, ChangeType, Confidence, Object3D

log = get_logger("c9")

PATHS_JSON = "nav/paths.json"
FREE, OBSTACLE, MARGIN, UNKNOWN = 0, 1, 2, -1
_STEPS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


# --------------------------------------------------------------------------------------------
# Grid helpers (pure, unit-tested)
# --------------------------------------------------------------------------------------------


@dataclass
class Grid:
    origin: np.ndarray  # world xy of cell (0, 0)'s corner
    cell: float
    shape: tuple[int, int]

    @classmethod
    def around(cls, points: np.ndarray, cell: float, margin: float = 0.3) -> Grid:
        lo, hi = points[:, :2].min(axis=0) - margin, points[:, :2].max(axis=0) + margin
        return cls(lo, cell, tuple(np.ceil((hi - lo) / cell).astype(int)))

    def index(self, xy: np.ndarray) -> np.ndarray:
        ij = np.floor((np.atleast_2d(xy)[:, :2] - self.origin) / self.cell).astype(int)
        return np.clip(ij, 0, np.array(self.shape) - 1)

    def centre(self, ij) -> np.ndarray:
        return self.origin + (np.asarray(ij, float) + 0.5) * self.cell

    def rasterize(self, points: np.ndarray, min_count: int = 1) -> np.ndarray:
        counts = np.zeros(self.shape, int)
        if len(points):
            ij = self.index(points)
            np.add.at(counts, (ij[:, 0], ij[:, 1]), 1)
        return counts >= min_count


def disk(radius_cells: float) -> np.ndarray:
    k = int(np.ceil(radius_cells))
    yy, xx = np.mgrid[-k : k + 1, -k : k + 1]
    return xx**2 + yy**2 <= radius_cells**2


def occupancy(
    floor: np.ndarray, obstacles: np.ndarray, radius_cells: float, unknown_is_free: bool
) -> np.ndarray:
    """int8 grid: FREE, OBSTACLE, MARGIN (within the robot radius of an obstacle), UNKNOWN."""
    grid = np.full(floor.shape, FREE if unknown_is_free else UNKNOWN, np.int8)
    grid[floor] = FREE
    margin = (
        ndimage.binary_dilation(obstacles, disk(radius_cells)) if obstacles.any() else obstacles
    )
    grid[margin] = MARGIN
    grid[obstacles] = OBSTACLE
    return grid


def astar(free: np.ndarray, start, goal) -> list[tuple[int, int]] | None:
    """Shortest 8-connected path over ``free`` cells (octile metric), or None."""
    start, goal = tuple(map(int, start)), tuple(map(int, goal))
    if not (free[start] and free[goal]):
        return None

    def h(c):
        dx, dy = abs(c[0] - goal[0]), abs(c[1] - goal[1])
        return max(dx, dy) + (np.sqrt(2) - 1) * min(dx, dy)

    open_heap = [(h(start), 0.0, start)]
    came, cost = {start: None}, {start: 0.0}
    while open_heap:
        _, g, cur = heapq.heappop(open_heap)
        if cur == goal:
            path = [cur]
            while came[path[-1]] is not None:
                path.append(came[path[-1]])
            return path[::-1]
        if g > cost[cur]:
            continue
        for dx, dy in _STEPS:
            nxt = (cur[0] + dx, cur[1] + dy)
            if not (0 <= nxt[0] < free.shape[0] and 0 <= nxt[1] < free.shape[1]) or not free[nxt]:
                continue
            if dx and dy and not (free[cur[0] + dx, cur[1]] and free[cur[0], cur[1] + dy]):
                continue  # no corner cutting
            ng = g + (np.sqrt(2) if dx and dy else 1.0)
            if ng < cost.get(nxt, np.inf):
                cost[nxt], came[nxt] = ng, cur
                heapq.heappush(open_heap, (ng + h(nxt), ng, nxt))
    return None


def path_length(path, cell: float) -> float:
    steps = np.abs(np.diff(np.asarray(path), axis=0))
    return float(np.sum(np.where(steps.sum(axis=1) == 2, np.sqrt(2), 1.0)) * cell)


def geodesic_from(free: np.ndarray, seed) -> np.ndarray:
    """Walking distance (in cells) from ``seed`` to every free cell (inf where unreachable)."""
    dist = np.full(free.shape, np.inf)
    seed = tuple(map(int, seed))
    dist[seed] = 0.0
    heap = [(0.0, seed)]
    while heap:
        d, cur = heapq.heappop(heap)
        if d > dist[cur]:
            continue
        for dx, dy in _STEPS:
            nxt = (cur[0] + dx, cur[1] + dy)
            if not (0 <= nxt[0] < free.shape[0] and 0 <= nxt[1] < free.shape[1]) or not free[nxt]:
                continue
            if dx and dy and not (free[cur[0] + dx, cur[1]] and free[cur[0], cur[1] + dy]):
                continue
            nd = d + (np.sqrt(2) if dx and dy else 1.0)
            if nd < dist[nxt]:
                dist[nxt] = nd
                heapq.heappush(heap, (nd, nxt))
    return dist


def farthest_pair(free: np.ndarray, seed) -> tuple[tuple[int, int], tuple[int, int]]:
    """Two cells far apart by walking distance (double sweep from ``seed``)."""
    d0 = geodesic_from(free, seed)
    u = np.unravel_index(np.argmax(np.where(np.isfinite(d0), d0, -1)), free.shape)
    d1 = geodesic_from(free, u)
    v = np.unravel_index(np.argmax(np.where(np.isfinite(d1), d1, -1)), free.shape)
    return tuple(map(int, u)), tuple(map(int, v))


# --------------------------------------------------------------------------------------------
# Change footprints and verdict
# --------------------------------------------------------------------------------------------


def footprints(changes: list[Change], points_of: dict[str, np.ndarray], grid: Grid, cfg):
    """Per change: (cells before, cells after) of its objects at obstacle height."""

    def cells(oid):
        if not oid:
            return np.zeros(grid.shape, bool)
        p = points_of[oid]
        return grid.rasterize(p[(p[:, 2] > cfg.min_z) & (p[:, 2] < cfg.robot_height)])

    out = {}
    for c in changes:
        before = cells(c.object_a) if c.type is not ChangeType.ADDED else np.zeros(grid.shape, bool)
        after = (
            cells(c.object_b) if c.type is not ChangeType.REMOVED else np.zeros(grid.shape, bool)
        )
        out[c.id] = (before, after)
    return out


def verdict(
    len_a: float | None,
    len_b: float | None,
    blockers: list[str],
    openers: list[str],
    start_name: str,
    goal_name: str,
    tol: float,
) -> str:
    """One or two sentences on the navigation impact."""
    route = f"the route from {start_name} to {goal_name}"
    if len_a is None and len_b is None:
        return f"No route could be planned from {start_name} to {goal_name} in either recording."
    if len_b is None:
        who = " and ".join(blockers) or "the changes"
        return (
            f"{who.capitalize()} block{'s' if len(blockers) == 1 else ''} {route}: there is "
            f"no longer a way through (it was {len_a:.1f} m)."
        )
    if len_a is None:
        who = " and ".join(openers) or "the changes"
        return (
            f"{route.capitalize()} only exists after the changes ({len_b:.1f} m), thanks to {who}."
        )
    diff = len_b - len_a
    if abs(diff) < tol:
        return f"Navigation is not affected: {route} is {len_a:.1f} m both before and after."
    if diff > 0:
        who = " and ".join(blockers) or "the changes"
        verb = "blocks" if len(blockers) == 1 else "block"
        return (
            f"{who.capitalize()} {verb} the direct way; {route} is now a detour "
            f"{diff:.1f} m longer ({len_a:.1f} m -> {len_b:.1f} m)."
        )
    who = " and ".join(openers) or "the changes"
    return f"With {who} gone, {route} is {-diff:.1f} m shorter ({len_a:.1f} m -> {len_b:.1f} m)."


def navigation_markdown(nav: dict) -> str:
    """The navigation section appended to report.md."""
    lines = ["## Navigation impact", "", nav["sentence"], ""]
    if nav.get("length_before_m") is not None or nav.get("length_after_m") is not None:
        fmt = lambda x: "no route" if x is None else f"{x:.2f} m"  # noqa: E731
        lines += [
            f"- route: {nav['start_name']} → {nav['goal_name']}; before "
            f"{fmt(nav['length_before_m'])}, after {fmt(nav['length_after_m'])}",
            f"- robot radius {nav['robot_radius_m']} m, height {nav['robot_height_m']} m; "
            f"walkable area before {nav['free_area_before_m2']:.1f} m², after "
            f"{nav['free_area_after_m2']:.1f} m²",
            f"- blocking changes: {', '.join(nav['blocking_changes']) or 'none'}",
            "",
        ]
    lines += ["![navigation before/after](../nav/nav_diff.png)", ""]
    return "\n".join(lines)


def _landmark_name(xy: np.ndarray, landmarks: list[Object3D]) -> str:
    if not landmarks:
        return "one end of the room"
    best = min(landmarks, key=lambda o: np.linalg.norm(o.centroid[:2] - xy))
    return f"near the {best.label}"


def _plot(path, grid: Grid, occ: dict, paths: dict, changes, fp, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = {
        UNKNOWN: (0.75, 0.75, 0.75),
        FREE: (1, 1, 1),
        MARGIN: (0.88, 0.88, 0.88),
        OBSTACLE: (0.25, 0.25, 0.25),
    }
    extent = [
        grid.origin[0],
        grid.origin[0] + grid.shape[0] * grid.cell,
        grid.origin[1],
        grid.origin[1] + grid.shape[1] * grid.cell,
    ]
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharex=True, sharey=True)
    for ax, key, name in zip(axes, ("before", "after"), ("Before", "After"), strict=True):
        img = np.zeros((*grid.shape, 3))
        for value, col in colours.items():
            img[occ[key] == value] = col
        for c in changes:
            cells = fp[c.id][0 if key == "before" else 1]
            img[cells] = (0.9, 0.3, 0.25) if key == "before" else (0.3, 0.7, 0.3)
        ax.imshow(img.transpose(1, 0, 2), origin="lower", extent=extent)
        p = paths.get(key)
        if p:
            xy = np.array([grid.centre(ij) for ij in p])
            ax.plot(
                xy[:, 0],
                xy[:, 1],
                color="tab:blue" if key == "before" else "tab:orange",
                lw=2.5,
                label=f"route ({path_length(p, grid.cell):.1f} m)",
            )
            ax.plot(*xy[0], "go", ms=8)
            ax.plot(*xy[-1], "rs", ms=8)
            ax.legend(loc="upper right")
        ax.set_title(f"{name}: white = free, grey = unknown, dark = obstacle")
        ax.set_aspect("equal")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


# --------------------------------------------------------------------------------------------
# Stage
# --------------------------------------------------------------------------------------------


def load_navigation(run: str) -> dict:
    return load_json(run_path(run, PATHS_JSON))


def _attach_to_report(run: str, nav: dict) -> None:
    """Add the navigation section to report/changes.json (stats) and report.md (if present)."""
    from changedet.core.types import ChangeReport
    from changedet.stages.c8_describe import REPORT_JSON, REPORT_MD

    if exists(run, REPORT_JSON):
        report = ChangeReport.from_dict(load_json(run_path(run, REPORT_JSON)))
        report.stats["navigation"] = {
            k: v for k, v in nav.items() if k not in ("path_before", "path_after")
        }
        save_json(run_path(run, REPORT_JSON), report)
    if exists(run, REPORT_MD):
        text = run_path(run, REPORT_MD).read_text()
        run_path(run, REPORT_MD).write_text(with_navigation_section(text, navigation_markdown(nav)))


def with_navigation_section(report_md: str, section: str) -> str:
    """Replace (or insert before "## Run information") the navigation section of a report."""
    text = re.sub(r"## Navigation impact\n.*?(?=^## |\Z)", "", report_md, flags=re.S | re.M)
    head, sep, tail = text.partition("## Run information")
    if not sep:
        return text.rstrip() + "\n\n" + section
    return head.rstrip() + "\n\n" + section + "\n" + sep + tail


@cached_stage(PATHS_JSON, load=load_navigation)
def navigation_impact(run: str, force: bool = False) -> dict:
    """Plan a route before and after the changes and report the difference.

    Args:
        run: run name; needs C2, C4 and C5 output (uses the latest change list).
        force: recompute even if ``nav/paths.json`` exists.
    """
    from changedet.stages.c2_reconstruct import load_reconstruction
    from changedet.stages.c4_fuse import load_objects
    from changedet.stages.c8_describe import latest_changes

    cfg = load_run_config(run).nav
    recon = load_reconstruction(run)
    changes, _ = latest_changes(run)
    objects = {o.id: o for s in ("A", "B") for o in load_objects(run)[s]}
    changed = [
        c
        for c in changes
        if c.type is not ChangeType.UNCHANGED and c.confidence is not Confidence.REJECTED
    ]
    clouds = {s: load_ply(run_path(run, recon.cloud_paths[s]))[0] for s in ("A", "B")}
    points_of = {
        oid: load_ply(run_path(run, objects[oid].points_path))[0]
        for c in changed
        for oid in (c.object_a, c.object_b)
        if oid
    }
    grid = Grid.around(np.vstack(list(clouds.values())), cfg.cell_size)

    # Static layers: floor seen in either recording + walked cells; obstacles minus changed objects.
    floor = np.zeros(grid.shape, bool)
    static_obstacles = np.zeros(grid.shape, bool)
    walked = np.zeros(grid.shape, bool)
    for s, cloud in clouds.items():
        floor |= grid.rasterize(cloud[np.abs(cloud[:, 2]) < cfg.floor_tol])
        tall = cloud[(cloud[:, 2] > cfg.min_z) & (cloud[:, 2] < cfg.robot_height)]
        mine = [
            points_of[c.object_a if s == "A" else c.object_b]
            for c in changed
            if (c.object_a if s == "A" else c.object_b)
        ]
        if mine and len(tall):
            d, _ = cKDTree(np.vstack(mine)).query(tall, distance_upper_bound=cfg.change_clearance)
            tall = tall[~np.isfinite(d)]
        static_obstacles |= grid.rasterize(tall, cfg.min_points)
        cams = np.array([c.T_world_cam[:3, 3] for c in recon.cameras[s]])
        walked |= grid.rasterize(cams)
    walked = ndimage.binary_dilation(walked, disk(cfg.walked_radius / cfg.cell_size))
    floor = ndimage.binary_closing(floor, np.ones((3, 3))) | walked

    fp = footprints(changed, points_of, grid, cfg)
    radius = cfg.robot_radius / cfg.cell_size
    before_obst, after_obst = static_obstacles.copy(), static_obstacles.copy()
    for before_cells, after_cells in fp.values():
        before_obst |= before_cells
        after_obst |= after_cells
    occ = {
        "before": occupancy(floor, before_obst, radius, cfg.unknown_is_free),
        "after": occupancy(floor, after_obst, radius, cfg.unknown_is_free),
    }
    run_path(run, "nav").mkdir(parents=True, exist_ok=True)
    np.save(run_path(run, "nav/occupancy_A.npy"), occ["before"])
    np.save(run_path(run, "nav/occupancy_B.npy"), occ["after"])

    free_a, free_b = occ["before"] == FREE, occ["after"] == FREE
    # Endpoints: shared free space, in the region connected to where the camera walked.
    common = free_a & free_b
    labels, _ = ndimage.label(common, np.ones((3, 3)))
    walked_labels = set(np.unique(labels[walked & common])) - {0}
    region = np.isin(labels, list(walked_labels)) if walked_labels else common
    if cfg.start is not None and cfg.goal is not None:
        start, goal = (
            tuple(grid.index(np.array(cfg.start))[0]),
            tuple(grid.index(np.array(cfg.goal))[0]),
        )
    elif region.any():
        seed = (
            np.argwhere(region & walked)[0] if (region & walked).any() else np.argwhere(region)[0]
        )
        start, goal = farthest_pair(region, seed)
    else:
        start = goal = None

    landmarks = [
        objects[i] for c in changes if c.type is ChangeType.UNCHANGED for i in (c.object_a,) if i
    ]
    paths, lengths = {}, {}
    if start is not None:
        for key, free in (("before", free_a), ("after", free_b)):
            paths[key] = astar(free, start, goal)
            lengths[key] = path_length(paths[key], cfg.cell_size) if paths[key] else None
        start_name = _landmark_name(grid.centre(start), landmarks)
        goal_name = _landmark_name(grid.centre(goal), landmarks)
    else:
        lengths = {"before": None, "after": None}
        start_name = goal_name = "the walkable area"

    def touching(path, which: int) -> list[str]:
        if not path:
            return []
        cells = np.zeros(grid.shape, bool)
        for ij in path:
            cells[ij] = True
        cells = ndimage.binary_dilation(cells, disk(radius))
        return [
            f"the {'new ' if c.type is ChangeType.ADDED else ''}{c.label} ({c.id})"
            for c in changed
            if (fp[c.id][which] & cells).any()
        ]

    blockers = touching(paths.get("before"), 1)  # after-state objects on the old route
    openers = touching(paths.get("after"), 0)  # before-state objects on the new route
    sentence = verdict(
        lengths["before"],
        lengths["after"],
        blockers,
        openers,
        start_name,
        goal_name,
        cfg.same_route_tol,
    )
    len_a, len_b = lengths["before"], lengths["after"]
    worse = len_a is not None and (len_b is None or len_b - len_a > cfg.same_route_tol)
    blocking_ids = [b.split("(")[-1].rstrip(")") for b in blockers] if worse else []
    nav = {
        "sentence": sentence,
        "start": grid.centre(start).tolist() if start else None,
        "goal": grid.centre(goal).tolist() if goal else None,
        "start_name": start_name,
        "goal_name": goal_name,
        "length_before_m": lengths["before"],
        "length_after_m": lengths["after"],
        "blocked": len_a is not None and len_b is None,
        "blocking_changes": blocking_ids,
        "path_before": [grid.centre(ij).tolist() for ij in paths.get("before") or []],
        "path_after": [grid.centre(ij).tolist() for ij in paths.get("after") or []],
        "free_area_before_m2": float(free_a.sum() * cfg.cell_size**2),
        "free_area_after_m2": float(free_b.sum() * cfg.cell_size**2),
        "robot_radius_m": cfg.robot_radius,
        "robot_height_m": cfg.robot_height,
        "grid": {"origin": grid.origin.tolist(), "cell": grid.cell, "shape": list(grid.shape)},
    }
    _plot(run_path(run, "nav/nav_diff.png"), grid, occ, paths, changed, fp, sentence)
    log.info("C9: %s", sentence)
    save_json(run_path(run, PATHS_JSON), nav)  # written last: marks the stage as done
    _attach_to_report(run, nav)
    return nav
