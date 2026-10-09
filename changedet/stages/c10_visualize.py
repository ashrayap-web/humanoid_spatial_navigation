"""C10 — Visualization: one rerun recording with a ready-made layout, plus README images.

``viz/final.rrd`` opens with the 3D scene, the report and a tab of evidence images per change
(the C7 before/after pair, or the object's crop). Two timelines:

- ``scene``: step 0 shows the full first scan and the changed objects as they were, step 1 the
  second scan and the objects as they are (removed ones vanish, added ones appear),
- ``animation``: removed objects dissolve, added ones materialise, moved ones glide and turn
  from their old to their new pose.

Context (grey background, camera paths, dimmed unchanged objects, labelled boxes, arrows) is
static. ``viz/hero.png`` (top-down + evidence) and ``viz/hero.gif`` (orbit) are for the README.
"""

from __future__ import annotations

import os
import shutil
import subprocess

import cv2
import numpy as np

from changedet.core import geometry as geo
from changedet.core.cache import exists, load_json, load_ply, run_dir, run_path
from changedet.core.config import load_run_config
from changedet.core.logging import get_logger
from changedet.core.types import Change, ChangeType, Confidence, Object3D

log = get_logger("c10")

FINAL_RRD = "viz/final.rrd"


def colour_of(change: Change, colours) -> np.ndarray:
    if change.confidence is Confidence.UNVERIFIED:
        return np.array(colours.unverified, np.uint8)
    return np.array(colours[change.type.value], np.uint8)


def caption(change: Change) -> str:
    text = f"{change.type.value.upper()}: {change.label}"
    return text + " (unverified)" if change.confidence is Confidence.UNVERIFIED else text


def report_markdown(summary: str, shown: list[Change], n_rejected: int) -> str:
    lines = ["# What changed?", "", summary, ""]
    for c in shown:
        lines.append(f"- **{caption(c)}** — {c.description or ''}")
    if n_rejected:
        lines += ["", f"_{n_rejected} rejected false alarm(s) are hidden (see report.md)._"]
    return "\n".join(lines)


def _inputs(run: str):
    from changedet.stages.c2_reconstruct import load_reconstruction
    from changedet.stages.c4_fuse import load_objects
    from changedet.stages.c8_describe import REPORT_JSON, latest_changes, load_report

    if exists(run, REPORT_JSON):
        report = load_report(run)
        changes, summary = report.changes, report.summary
    else:
        changes, _ = latest_changes(run)
        summary = "(run C8 for a written summary)"
    objects = {o.id: o for s in ("A", "B") for o in load_objects(run)[s]}
    return changes, summary, objects, load_reconstruction(run)


def _split(changes: list[Change]):
    shown = [
        c
        for c in changes
        if c.type is not ChangeType.UNCHANGED and c.confidence is not Confidence.REJECTED
    ]
    shown.sort(key=lambda c: c.confidence is not Confidence.CONFIRMED)  # confirmed first
    unchanged = [c for c in changes if c.type is ChangeType.UNCHANGED]
    rejected = [c for c in changes if c.confidence is Confidence.REJECTED]
    return shown, unchanged, rejected


def interpolate_motion(
    points: np.ndarray, centre_a: np.ndarray, T_a_to_b: np.ndarray, t: float
) -> np.ndarray:
    """Points of a moved object at fraction ``t`` (0 = before, 1 = after) of its motion.

    Translation and yaw are interpolated linearly about the object's own centre, so the object
    glides and turns in place instead of swinging around the world origin.
    """
    end = geo.transform_points(T_a_to_b, centre_a[None])[0]
    yaw = geo.yaw_of(T_a_to_b[:3, :3])
    R = geo.rot_z(t * yaw)
    return (points - centre_a) @ R.T + centre_a + t * (end - centre_a)


def _subsample(points: np.ndarray, fraction: float, seed: int = 0) -> np.ndarray:
    n = int(round(len(points) * float(np.clip(fraction, 0, 1))))
    return points[np.random.default_rng(seed).permutation(len(points))[:n]]


def _evidence_image(run: str, change: Change, objects: dict[str, Object3D]):
    """Path of the best evidence image for a change: C7 before/after pair, else the crop."""
    pair = run_path(run, f"viz/c7_evidence/{change.id}.jpg")
    if pair.exists():
        return pair
    return run_dir(run) / objects[change.object_a or change.object_b].best_crop_path


def write_rrd(run: str, changes, summary, objects: dict[str, Object3D], recon, cfg) -> str:
    """``viz/final.rrd``: static scene + a "scene" timeline (0 = before, 1 = after) + an
    "animation" timeline morphing before into after, with evidence images per change."""
    import rerun as rr
    import rerun.blueprint as rrb

    from changedet.core.cache import load_ply

    shown, unchanged, rejected = _split(changes)
    nav = load_json(run_path(run, "nav/paths.json")) if exists(run, "nav/paths.json") else None
    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(origin="world", name="Changes in 3D"),
            rrb.Vertical(
                rrb.TextDocumentView(origin="report", name="Report"),
                rrb.Tabs(
                    *[
                        rrb.Spatial2DView(origin=f"evidence/{c.id}", name=f"{c.id}: {c.label}")
                        for c in shown
                    ],
                    *(
                        [rrb.Spatial2DView(origin="navigation_map", name="Navigation")]
                        if nav
                        else []
                    ),
                    name="Evidence",
                )
                if shown or nav
                else rrb.TextDocumentView(origin="report"),
                row_shares=[1, 1],
            ),
            column_shares=[3, 1],
        ),
        rrb.TimePanel(timeline="scene", state="expanded"),
        rrb.BlueprintPanel(state="collapsed"),
        rrb.SelectionPanel(state="collapsed"),
    )
    path = run_path(run, FINAL_RRD)
    rec = rr.RecordingStream("changedet")
    rec.save(path, default_blueprint=blueprint)

    # --- static: context that is the same before and after
    rec.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    background, _ = load_ply(run_path(run, recon.background_cloud_path))
    background = geo.voxel_downsample(background, cfg.background_voxel)
    rec.log(
        "world/background",
        rr.Points3D(background, colors=cfg.colors.background, radii=cfg.point_size * 0.4),
        static=True,
    )
    for s, key in (("A", "session_a"), ("B", "session_b")):
        centres = [c.T_world_cam[:3, 3] for c in recon.cameras[s]]
        name = "before" if s == "A" else "after"
        rec.log(
            f"world/camera_path_{name}",
            rr.LineStrips3D([centres], colors=cfg.colors[key], labels=[f"{name} recording"]),
            static=True,
        )
    for oid in dict.fromkeys(c.object_b or c.object_a for c in unchanged):  # unique, in order
        o = objects[oid]
        points, _ = load_ply(run_path(run, o.points_path))
        rec.log(
            f"world/unchanged/{o.label.replace(' ', '_')}_{o.id}",
            rr.Points3D(points, colors=cfg.colors.unchanged, radii=cfg.point_size * 0.5),
            static=True,
        )

    points_of = {}
    for c in shown:
        for oid in (c.object_a, c.object_b):
            if oid:
                points_of[oid] = load_ply(run_path(run, objects[oid].points_path))[0]
    base_of = {
        c.id: f"world/changes/{c.id}_{c.label.replace(' ', '_').replace('->', '_to_')}"
        for c in shown
    }
    for c in shown:  # boxes, labels, arrows and evidence never change over time
        colour, base = colour_of(c, cfg.colors), base_of[c.id]
        for side, oid in (("before", c.object_a), ("after", c.object_b)):
            if oid is None:
                continue
            faded = c.type in (ChangeType.MOVED, ChangeType.REPLACED) and side == "before"
            col = (colour * 0.4 + 255 * 0.6).astype(np.uint8) if faded else colour
            o = objects[oid]
            rec.log(
                f"{base}/box_{side}",
                rr.Boxes3D(
                    mins=[o.bbox_min],
                    sizes=[o.bbox_max - o.bbox_min],
                    colors=col,
                    labels=[caption(c) if not faded else f"{c.label} (before)"],
                ),
                static=True,
            )
        if c.type is ChangeType.MOVED:
            rec.log(
                f"{base}/arrow",
                rr.Arrows3D(
                    origins=[c.position_a], vectors=[c.position_b - c.position_a], colors=colour
                ),
                static=True,
            )
        rec.log(
            f"evidence/{c.id}", rr.EncodedImage(path=_evidence_image(run, c, objects)), static=True
        )
    rec.log(
        "report",
        rr.TextDocument(
            report_markdown(summary, shown, len(rejected)), media_type=rr.MediaType.MARKDOWN
        ),
        static=True,
    )

    # --- "scene" timeline: the full scan and the changed objects before (0) and after (1)
    lo, hi = background.min(axis=0), background.max(axis=0)
    label_at = [[(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, hi[2] + 0.3]]
    for t, session, title in (
        (0, "A", "BEFORE (first recording)"),
        (1, "B", "AFTER (second recording)"),
    ):
        rec.set_time("scene", sequence=t)
        cloud, colours = load_ply(run_path(run, recon.cloud_paths[session]))
        cloud, colours = geo.voxel_downsample(cloud, cfg.background_voxel, colours)
        rec.log("world/scan", rr.Points3D(cloud, colors=colours, radii=cfg.point_size * 0.5))
        rec.log("world/scene_title", rr.Points3D(label_at, labels=[title], radii=0.001))
        for c in shown:
            oid = c.object_a if session == "A" else c.object_b
            entity = f"{base_of[c.id]}/object"
            if oid is None:
                rec.log(entity, rr.Clear(recursive=False))
            else:
                rec.log(
                    entity,
                    rr.Points3D(
                        points_of[oid], colors=colour_of(c, cfg.colors), radii=cfg.point_size
                    ),
                )

    # --- "animation" timeline: removed objects dissolve, added ones appear, moved ones travel
    for k in range(cfg.anim_frames):
        t = k / max(1, cfg.anim_frames - 1)
        rec.set_time("animation", sequence=k)
        for c in shown:
            colour, entity = colour_of(c, cfg.colors), f"{base_of[c.id]}/animated"
            if c.type is ChangeType.MOVED and c.T_a_to_b is not None:
                pts = interpolate_motion(points_of[c.object_a], c.position_a, c.T_a_to_b, t)
            elif c.type is ChangeType.REMOVED:
                pts = _subsample(points_of[c.object_a], 1 - t)
            elif c.type is ChangeType.ADDED:
                pts = _subsample(points_of[c.object_b], t)
            else:  # replaced: old dissolves while new appears
                pts = np.vstack(
                    [_subsample(points_of[c.object_a], 1 - t), _subsample(points_of[c.object_b], t)]
                )
            rec.log(entity, rr.Points3D(pts, colors=colour, radii=cfg.point_size))

    # --- navigation (C9): routes on the floor and the before/after map
    if nav:
        for key, colour in (("before", cfg.colors.session_a), ("after", cfg.colors.session_b)):
            route = nav[f"path_{key}"]
            if route:
                pts = [[x, y, 0.02] for x, y in route]
                rec.log(
                    f"world/navigation/route_{key}",
                    rr.LineStrips3D(
                        [pts], colors=colour, radii=0.015, labels=[f"robot route {key}"]
                    ),
                    static=True,
                )
        if exists(run, "nav/nav_diff.png"):
            rec.log(
                "navigation_map",
                rr.EncodedImage(path=run_path(run, "nav/nav_diff.png")),
                static=True,
            )
    return str(path)


def _top_down(ax, background, changes_points, unchanged_points, cfg) -> None:
    from matplotlib.patches import Rectangle

    sel = (background[:, 2] > 0.1) & (background[:, 2] < 2.0)
    ax.scatter(
        background[sel, 0], background[sel, 1], s=0.3, color=np.array(cfg.colors.background) / 255
    )
    for p in unchanged_points:
        ax.scatter(p[:, 0], p[:, 1], s=0.3, color=np.array(cfg.colors.unchanged) / 255)
    for c, colour, p, o in changes_points:
        ax.scatter(p[:, 0], p[:, 1], s=2, color=colour / 255)
        (x0, y0), (x1, y1) = o.bbox_min[:2], o.bbox_max[:2]
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, lw=2, color=colour / 255))
        ax.text(x0, y1 + 0.05, caption(c), fontsize=9, weight="bold", color=colour / 255 * 0.8)
    ax.set_aspect("equal")
    ax.axis("off")


def write_hero(run: str, changes, summary, objects, recon, cfg) -> tuple[str, str]:
    """``viz/hero.png`` (top-down + evidence crops) and ``viz/hero.gif`` (3D orbit)."""
    import matplotlib

    matplotlib.use("Agg")
    import textwrap

    import matplotlib.pyplot as plt
    from PIL import Image

    shown, unchanged, _ = _split(changes)
    background, _ = load_ply(run_path(run, recon.background_cloud_path))
    background = geo.voxel_downsample(background, cfg.background_voxel)
    changed = []
    for c in shown:
        o = objects[c.object_b or c.object_a]
        changed.append((c, colour_of(c, cfg.colors), load_ply(run_path(run, o.points_path))[0], o))
    unchanged_points = [
        load_ply(run_path(run, objects[c.object_b or c.object_a].points_path))[0] for c in unchanged
    ]

    # --- hero.png: top-down map + the evidence image of each confirmed change
    evidence = [c for c in shown if c.confidence is Confidence.CONFIRMED][: cfg.max_evidence]
    fig = plt.figure(figsize=(14, 8))
    grid = fig.add_gridspec(max(1, len(evidence)), 2, width_ratios=[2.2, 1])
    _top_down(fig.add_subplot(grid[:, 0]), background, changed, unchanged_points, cfg)
    for k, c in enumerate(evidence):
        ax = fig.add_subplot(grid[k, 1])
        o = objects[c.object_a or c.object_b]
        ax.imshow(cv2.imread(str(run_dir(run) / o.best_crop_path))[..., ::-1])
        ax.set_title(f"{caption(c)} — before", fontsize=10)
        ax.axis("off")
    fig.suptitle("\n".join(textwrap.wrap(summary, 130)), fontsize=10)
    fig.tight_layout()
    png = run_path(run, "viz/hero.png")
    fig.savefig(png, dpi=cfg.hero_dpi)
    plt.close(fig)

    # --- hero.gif: orbit around the scene from above; walls cut at gif_max_z so the inside shows
    rng = np.random.default_rng(0)
    bg = background[background[:, 2] < cfg.gif_max_z]
    bg = bg[rng.choice(len(bg), min(len(bg), cfg.gif_points), replace=False)]
    centre = np.median(bg, axis=0)
    radius = float(np.percentile(np.linalg.norm(bg[:, :2] - centre[:2], axis=1), 95))
    frames = []
    for k in range(cfg.gif_frames):
        fig = plt.figure(figsize=(6, 5), dpi=cfg.gif_dpi)
        ax = fig.add_subplot(projection="3d")
        ax.scatter(
            *bg.T, s=0.3, color=np.array(cfg.colors.background) / 255, alpha=0.6, depthshade=False
        )
        for p in unchanged_points:
            p = p[p[:, 2] < cfg.gif_max_z]
            ax.scatter(
                *p[:: max(1, len(p) // 800)].T,
                s=0.6,
                color=np.array(cfg.colors.unchanged) / 255,
                depthshade=False,
            )
        for c, colour, p, o in changed:
            confirmed = c.confidence is Confidence.CONFIRMED
            ax.scatter(
                *p[:: max(1, len(p) // 1500)].T,
                s=6 if confirmed else 2,
                color=colour / 255,
                depthshade=False,
            )
            if confirmed:
                ax.text(
                    *(o.bbox_max + [0, 0, 0.25]),
                    caption(c),
                    fontsize=9,
                    weight="bold",
                    color=colour / 255,
                )
        ax.set_xlim(centre[0] - radius, centre[0] + radius)
        ax.set_ylim(centre[1] - radius, centre[1] + radius)
        ax.set_zlim(0, 2 * radius * 0.5)
        ax.view_init(elev=55, azim=360 * k / cfg.gif_frames)
        ax.set_box_aspect((1, 1, 0.5))
        ax.axis("off")
        fig.tight_layout()
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3]))
        plt.close(fig)
    gif = run_path(run, "viz/hero.gif")
    frames[0].save(gif, save_all=True, append_images=frames[1:], duration=90, loop=0)
    return str(png), str(gif)


def open_in_viewer(path: str) -> None:
    """Start the rerun viewer if there is a display, otherwise say how to view the file."""
    viewer = shutil.which("rerun")
    if viewer and os.environ.get("DISPLAY"):
        subprocess.Popen([viewer, path])
        return
    log.info(
        "No display here. View with `rerun %s` on a machine with a screen, or run "
        "`uv run rerun --web-viewer %s` and open the printed URL (forward its port over "
        "SSH).",
        path,
        path,
    )


def visualize(run: str, open_viewer: bool = True, force: bool = False) -> str:
    """Build ``viz/final.rrd``, ``viz/hero.png`` and ``viz/hero.gif``; returns the .rrd path.

    Args:
        run: run name; needs C5 output (uses C6/C8 output if present).
        open_viewer: start the rerun viewer (or print how to open it when headless).
        force: unused — the visualization is cheap and always rebuilt.
    """
    cfg = load_run_config(run).viz
    run_path(run, "viz").mkdir(parents=True, exist_ok=True)
    changes, summary, objects, recon = _inputs(run)
    path = write_rrd(run, changes, summary, objects, recon, cfg)
    png, gif = write_hero(run, changes, summary, objects, recon, cfg)
    size = os.path.getsize(path) / 1e6
    log.info("Wrote %s (%.1f MB), %s, %s", path, size, png, gif)
    if open_viewer:
        open_in_viewer(path)
    return path
