"""Run cache: every stage reads from and writes to ``runs/<run>/`` (spec Section 6)."""

from __future__ import annotations

import functools
import json
import os
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from changedet.core.logging import get_logger
from changedet.core.types import to_jsonable

log = get_logger("cache")

RUNS_DIR_ENV = "CHANGEDET_RUNS_DIR"


def runs_root() -> Path:
    """Directory holding all runs (``runs/``, or ``$CHANGEDET_RUNS_DIR`` if set)."""
    return Path(os.environ.get(RUNS_DIR_ENV, "runs"))


def run_dir(run: str) -> Path:
    """Directory of one run: ``runs/<run>/``."""
    return runs_root() / run


def run_path(run: str, relpath: str | Path) -> Path:
    """Absolute-ish path of a file inside a run directory."""
    return run_dir(run) / relpath


def exists(run: str, relpath: str | Path) -> bool:
    """True if ``runs/<run>/<relpath>`` exists."""
    return run_path(run, relpath).exists()


def save_json(path: str | Path, obj: Any) -> Path:
    """Write ``obj`` (dataclasses, numpy and enums allowed) as JSON, atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(to_jsonable(obj), indent=2))
    os.replace(tmp, path)
    return path


def load_json(path: str | Path) -> Any:
    """Read a JSON file into plain Python values."""
    return json.loads(Path(path).read_text())


def save_ply(path: str | Path, points: np.ndarray, colors: np.ndarray | None = None) -> Path:
    """Write an (N,3) point cloud with optional (N,3) colours (uint8 or float in [0,1])."""
    import open3d as o3d

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(points, np.float64)))
    if colors is not None:
        colors = np.asarray(colors)
        if colors.dtype == np.uint8:
            colors = colors / 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    tmp = path.with_name(path.stem + ".tmp" + path.suffix)
    if not o3d.io.write_point_cloud(str(tmp), pcd):
        raise OSError(f"Failed to write point cloud {path}")
    os.replace(tmp, path)
    return path


def load_ply(path: str | Path) -> tuple[np.ndarray, np.ndarray | None]:
    """Read a point cloud; returns (points (N,3) float64, colours (N,3) float in [0,1] or None)."""
    import open3d as o3d

    if not Path(path).exists():
        raise FileNotFoundError(path)
    pcd = o3d.io.read_point_cloud(str(path))
    colors = np.asarray(pcd.colors) if pcd.has_colors() else None
    return np.asarray(pcd.points), colors


def cached_stage(
    outputs: str | Sequence[str], load: Callable[[str], Any] | None = None
) -> Callable:
    """Skip a stage when all its ``outputs`` (paths relative to the run dir) already exist.

    The decorated function must have the signature ``fn(run, ..., force=False)``. When skipped,
    returns ``load(run)`` (or None if no loader is given). Stages should write the listed outputs
    **last**, so their existence means the stage finished.
    """
    outs = [outputs] if isinstance(outputs, str) else list(outputs)

    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(run: str, *args: Any, force: bool = False, **kwargs: Any) -> Any:
            if not force and all(exists(run, o) for o in outs):
                log.info(
                    "%s: outputs exist (%s), skipping; use force to redo",
                    fn.__name__,
                    ", ".join(outs),
                )
                return load(run) if load else None
            return fn(run, *args, force=force, **kwargs)

        return wrapper

    return decorator
