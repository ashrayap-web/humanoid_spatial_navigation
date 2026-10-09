"""Command line entry point.

python -m changedet.cli all --a BEFORE --b AFTER --run demo   # everything, cached
python -m changedet.cli init --run demo --a BEFORE --b AFTER   # just create the run
python -m changedet.cli c4 --run demo --force                  # one stage
python -m changedet.cli export --run demo --out examples/demo  # README example outputs
"""

from __future__ import annotations

import argparse
import importlib
import sys
import time
from pathlib import Path

from changedet.core import cache
from changedet.core.config import RUN_CONFIG_RELPATH, load_config, load_run_config, save_config
from changedet.core.logging import attach_file_log, get_logger, setup_logging
from changedet.io.record3d import is_record3d_dir

log = get_logger("cli")

INPUTS_RELPATH = "inputs.json"
LOG_RELPATH = "logs/run.log"

# Stage id -> description, in pipeline order. Each id becomes a subcommand.
STAGES = {
    "c1": "Frame extraction & quality filtering",
    "c2": "Reconstruction & cross-session alignment",
    "c3": "2D detection & segmentation",
    "c4": "3D lifting & object fusion",
    "c5": "Cross-session matching & change classification",
    "c6": "Visibility reasoning",
    "c7": "VLM change verification",
    "c8": "Semantic description & report",
    "c9": "Navigation impact analysis",
    "c10": "Interactive visualization",
    "c11": "Evaluation against ground truth",
}

# Implemented stages: id -> (module, entry function). Imported lazily so the CLI starts fast.
STAGE_ENTRIES = {
    "c1": ("changedet.stages.c1_frames", "extract_frames"),
    "c2": ("changedet.stages.c2_reconstruct", "reconstruct"),
    "c3": ("changedet.stages.c3_detect", "detect"),
    "c4": ("changedet.stages.c4_fuse", "fuse_objects"),
    "c5": ("changedet.stages.c5_match", "match_and_classify"),
    "c6": ("changedet.stages.c6_visibility", "assess_visibility"),
    "c7": ("changedet.stages.c7_verify", "verify_changes"),
    "c8": ("changedet.stages.c8_describe", "describe"),
    "c10": ("changedet.stages.c10_visualize", "visualize"),
}


def detect_input(path: str) -> dict:
    """Describe one session's input: a Record3D export dir or a video file."""
    p = Path(path).expanduser().resolve()
    if is_record3d_dir(p):
        return {"path": str(p), "kind": "record3d"}
    if p.is_file():
        return {"path": str(p), "kind": "video"}
    raise FileNotFoundError(f"{path} is neither a Record3D export directory nor a video file")


def cmd_init(args: argparse.Namespace) -> int:
    """Create a run: record the inputs and snapshot the resolved config."""
    inputs = {"A": detect_input(args.a), "B": detect_input(args.b)}
    inputs_path = cache.run_path(args.run, INPUTS_RELPATH)
    if inputs_path.exists() and cache.load_json(inputs_path) != inputs and not args.force:
        log.error(
            "Run '%s' already exists with different inputs; use --force to overwrite "
            "(cached stage outputs are NOT deleted)",
            args.run,
        )
        return 1
    cfg = load_config(args.config, args.set)
    cache.save_json(inputs_path, inputs)
    save_config(cfg, cache.run_path(args.run, RUN_CONFIG_RELPATH))
    attach_file_log(cache.run_path(args.run, LOG_RELPATH))
    log.info("Initialised run '%s' in %s", args.run, cache.run_dir(args.run))
    for session, info in inputs.items():
        log.info("  %s: %s (%s)", session, info["path"], info["kind"])
    return 0


def _apply_overrides(args: argparse.Namespace) -> None:
    """Save ``--set`` overrides into the run's config snapshot and start the run log."""
    cfg = load_run_config(args.run, args.set)
    if args.set:
        save_config(cfg, cache.run_path(args.run, RUN_CONFIG_RELPATH))
        log.info("Updated run config with overrides: %s", ", ".join(args.set))
    attach_file_log(cache.run_path(args.run, LOG_RELPATH))


def _call(stage: str, run: str, force: bool, open_viewer: bool = False) -> None:
    module, entry = STAGE_ENTRIES[stage]
    fn = getattr(importlib.import_module(module), entry)
    log.info("== %s: %s", stage, STAGES[stage])
    if stage == "c10":
        fn(run, open_viewer=open_viewer, force=force)
    else:
        fn(run, force=force)


def run_stage(stage: str, args: argparse.Namespace) -> int:
    """Run one pipeline stage on an initialised run."""
    _apply_overrides(args)
    if stage not in STAGE_ENTRIES:
        log.warning("%s (%s) is not implemented yet", stage, STAGES[stage])
        return 1
    _call(stage, args.run, args.force, getattr(args, "open", False))
    return 0


def cmd_all(args: argparse.Namespace) -> int:
    """Run the whole pipeline, skipping cached stages and stages that do not exist yet."""
    if args.a or args.b:
        if not (args.a and args.b):
            log.error("Give both --a and --b (or neither, to reuse an existing run)")
            return 1
        if cmd_init(args) != 0:
            return 1
    if not cache.exists(args.run, INPUTS_RELPATH):
        log.error("Run '%s' does not exist: pass --a and --b to create it", args.run)
        return 1
    _apply_overrides(args)
    if args.from_stage and args.from_stage not in STAGES:
        log.error("--from must be one of %s", ", ".join(STAGES))
        return 1
    order = list(STAGES)
    start = order.index(args.from_stage) if args.from_stage else len(order)
    total = time.perf_counter()
    for k, stage in enumerate(order):
        if stage == "c11" and not args.gt:
            continue
        if stage not in STAGE_ENTRIES:
            log.warning("== %s: %s — not implemented yet, skipped", stage, STAGES[stage])
            continue
        t0 = time.perf_counter()
        _call(stage, args.run, args.force or k >= start, args.open)
        log.info("== %s done in %.1f s", stage, time.perf_counter() - t0)
    log.info(
        "Pipeline finished in %.1f s. Report: %s",
        time.perf_counter() - total,
        cache.run_path(args.run, "report/report.md"),
    )
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    """Copy a run's presentable outputs (report, images, small .rrd) into a folder."""
    from changedet.export import export_run

    out = export_run(args.run, Path(args.out or f"examples/{args.run}"))
    log.info("Exported example outputs to %s", out)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="changedet", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--log-level", default="INFO")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--run", required=True, help="run name (directory under runs/)")
        p.add_argument(
            "--set",
            action="append",
            default=[],
            metavar="SECTION.KEY=VALUE",
            help="override a config value (repeatable)",
        )
        p.add_argument("--force", action="store_true", help="recompute even if outputs exist")

    p = sub.add_parser("init", help="create a run from two recordings")
    add_common(p)
    p.add_argument("--a", required=True, help="before: Record3D export dir (or video)")
    p.add_argument("--b", required=True, help="after: Record3D export dir (or video)")
    p.add_argument(
        "--config", default=None, help="base config YAML (default: configs/default.yaml)"
    )
    p.set_defaults(func=cmd_init)

    for stage, description in STAGES.items():
        p = sub.add_parser(stage, help=description)
        add_common(p)
        if stage == "c10":
            p.add_argument("--open", action="store_true", help="open the rerun viewer")
        p.set_defaults(func=lambda a, s=stage: run_stage(s, a))

    p = sub.add_parser("all", help="run the whole pipeline (cached stages are skipped)")
    add_common(p)
    p.add_argument("--a", help="before recording; creates the run if given with --b")
    p.add_argument("--b", help="after recording")
    p.add_argument("--config", default=None, help="base config YAML for a new run")
    p.add_argument(
        "--from",
        dest="from_stage",
        metavar="STAGE",
        help="recompute this stage and everything after it (e.g. c4)",
    )
    p.add_argument("--gt", help="ground-truth JSON: also run C11 evaluation")
    p.add_argument("--open", action="store_true", help="open the rerun viewer at the end")
    p.set_defaults(func=cmd_all)

    p = sub.add_parser("export", help="copy report, images and a small .rrd to a folder")
    p.add_argument("--run", required=True)
    p.add_argument("--out", help="output folder (default: examples/<run>)")
    p.set_defaults(func=cmd_export)
    return parser


def main(argv: list[str] | None = None) -> int:
    from dotenv import load_dotenv

    load_dotenv()  # e.g. ANTHROPIC_API_KEY from ./.env; never overrides the real environment
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)
    try:
        return args.func(args)
    except (FileNotFoundError, KeyError, ValueError) as e:
        log.error("%s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
