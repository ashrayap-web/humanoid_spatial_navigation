"""Copy a run's presentable outputs into a self-contained folder (e.g. ``examples/demo``)."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

from changedet.core.cache import run_dir, run_path

FILES = ["report/report.md", "report/changes.json", "viz/hero.png", "viz/hero.gif", "viz/final.rrd"]


def export_run(run: str, out: Path) -> Path:
    """Copy report, hero images, the final .rrd and the crops the report links to into ``out``.

    Image links in ``report.md`` are rewritten to the copied ``crops/`` folder.
    """
    out.mkdir(parents=True, exist_ok=True)
    for relpath in FILES:
        src = run_path(run, relpath)
        if src.exists():
            shutil.copyfile(src, out / Path(relpath).name)
    report = out / "report.md"
    if report.exists():
        text = report.read_text()
        crops = re.findall(r"\(\.\./(objects/crops/[^)]+)\)", text)
        (out / "crops").mkdir(exist_ok=True)
        for crop in crops:
            shutil.copyfile(run_dir(run) / crop, out / "crops" / Path(crop).name)
        report.write_text(re.sub(r"\(\.\./objects/crops/", "(crops/", text))
    return out
