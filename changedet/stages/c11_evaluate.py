"""C11 — Evaluation against ground truth.

A predicted change (any non-REJECTED, non-UNCHANGED change) matches a ground-truth change if

- the type is the same (a REPLACED may also match a REMOVED + ADDED pair, and vice versa),
- the labels are compatible: one contains the other as a whole word ("office chair" ~ "chair"),
  or CLIP text similarity >= ``eval.label_sim_thresh`` ("bag" ~ "backpack"),
- and, if the ground truth gives a position, the prediction is within ``eval.pos_tol``
  horizontally (x, y).

Scores are reported twice: for every claim (confirmed + unverified, as the spec defines it) and
for confirmed claims only — what the system asserts with confidence.

Ground-truth file (``data/ground_truth/<pair>.json``)::

    {"pair": "main", "run": "demo",
     "changes": [{"type": "removed", "label": "bag", "note": "...",
                  "position": [x, y, z]  # optional, world frame of the run
                  "distance_m": 1.4}],   # optional, moved objects
     "expected_unverified": [{"type": "removed", "label": "bag"}]}
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from changedet.core.cache import load_json, run_path, save_json
from changedet.core.logging import get_logger
from changedet.core.types import Change, ChangeType, Confidence

log = get_logger("c11")

EVAL_JSON = "eval/eval.json"
EVAL_MD = "eval/eval.md"
TYPES = ["removed", "added", "moved", "replaced"]


@dataclass
class GTChange:
    type: str
    label: str
    note: str = ""
    position: np.ndarray | None = None
    distance_m: float | None = None


@dataclass
class GroundTruth:
    pair: str
    run: str
    changes: list[GTChange]
    expected_unverified: list[dict] = field(default_factory=list)


def load_ground_truth(path: str | Path) -> GroundTruth:
    data = json.loads(Path(path).read_text())
    changes = [
        GTChange(
            c["type"],
            c["label"],
            c.get("note", ""),
            np.asarray(c["position"], float) if "position" in c else None,
            c.get("distance_m"),
        )
        for c in data["changes"]
    ]
    return GroundTruth(
        data["pair"], data.get("run", data["pair"]), changes, data.get("expected_unverified", [])
    )


# --------------------------------------------------------------------------------------------
# Label compatibility
# --------------------------------------------------------------------------------------------


def _norm(label: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", label.lower()).strip()


def contains_word(a: str, b: str) -> bool:
    """True if one label contains the other as whole words ("office chair" ~ "chair")."""
    a, b = _norm(a), _norm(b)
    return bool(a and b) and (
        re.search(rf"\b{re.escape(b)}\b", a) is not None
        or re.search(rf"\b{re.escape(a)}\b", b) is not None
    )


class LabelMatcher:
    """Scores label compatibility: 1.0 for a word match, else CLIP text similarity (if enabled)."""

    def __init__(self, cfg, clip_model: str | None = None, device: str = "cpu"):
        self.cfg = cfg
        self.clip_model = clip_model if cfg.use_clip else None
        self.device = device
        self._emb: dict[str, np.ndarray] = {}

    def _embed(self, labels: list[str]) -> None:
        from changedet import models

        missing = sorted({_norm(x) for x in labels} - set(self._emb))
        if missing:
            for text, e in zip(
                missing, models.embed_texts(missing, self.clip_model, self.device), strict=True
            ):
                self._emb[text] = e

    def score(self, predicted: list[str], truth: str) -> float:
        """Best compatibility of any predicted label with the ground-truth label (0..1)."""
        if any(contains_word(p, truth) for p in predicted):
            return 1.0
        if not self.clip_model:
            return 0.0
        self._embed(predicted + [truth])
        sims = [float(self._emb[_norm(p)] @ self._emb[_norm(truth)]) for p in predicted]
        best = max(sims, default=0.0)
        return best if best >= self.cfg.label_sim_thresh else 0.0


def labels_of(change: Change) -> list[str]:
    """Every name the system gave this change: detector label + VLM labels."""
    labels = [change.label]
    vlm = change.vlm_verdict or {}
    labels += [vlm[k] for k in ("refined_label", "label") if vlm.get(k)]
    return labels


# --------------------------------------------------------------------------------------------
# Matching and metrics (pure, unit-tested)
# --------------------------------------------------------------------------------------------


def _position_ok(pred: Change, gt: GTChange, pos_tol: float) -> bool:
    if gt.position is None:
        return True
    pos = pred.position_a if gt.type == "removed" else pred.position_b
    if pos is None:
        pos = pred.position_a if pred.position_a is not None else pred.position_b
    # Horizontal distance: GT positions are usually marked on a floor plan, centroids are not.
    return pos is not None and float(np.linalg.norm(pos[:2] - gt.position[:2])) <= pos_tol


def match(preds: list[Change], gts: list[GTChange], matcher: LabelMatcher, pos_tol: float):
    """Greedy one-to-one matching. Returns (pairs [(gt index, [pred ids], label score)],
    unmatched pred ids, unmatched gt indices)."""
    free = {p.id: p for p in preds}
    pairs, unmatched_gt = [], []

    def best(gt: GTChange, kind: str, label: str):
        options = []
        for p in free.values():
            if p.type.value != kind or not _position_ok(p, gt, pos_tol):
                continue
            s = matcher.score(labels_of(p) if kind != "replaced" else [p.label], label)
            if s > 0:
                options.append((s, p.confidence is Confidence.CONFIRMED, p.id))
        return max(options)[0::2] if options else None

    for k, gt in enumerate(gts):
        hit = best(gt, gt.type, gt.label)
        if hit is not None:
            pairs.append((k, [free.pop(hit[1]).id], hit[0]))
            continue
        if gt.type == "replaced" and "->" in gt.label:  # GT replaced ~ predicted removed + added
            old, new = gt.label.split("->")
            r, a = best(gt, "removed", old), best(gt, "added", new)
            if r is not None and a is not None:
                pairs.append((k, [free.pop(r[1]).id, free.pop(a[1]).id], min(r[0], a[0])))
                continue
        unmatched_gt.append(k)

    # Predicted replaced ~ a GT removed + GT added pair that is still unmatched.
    for pid, p in list(free.items()):
        if p.type is not ChangeType.REPLACED or "->" not in p.label:
            continue
        old, new = p.label.split("->")
        r = next(
            (
                k
                for k in unmatched_gt
                if gts[k].type == "removed" and matcher.score([old], gts[k].label) > 0
            ),
            None,
        )
        a = next(
            (
                k
                for k in unmatched_gt
                if gts[k].type == "added" and matcher.score([new], gts[k].label) > 0
            ),
            None,
        )
        if r is not None and a is not None:
            free.pop(pid)
            unmatched_gt = [k for k in unmatched_gt if k not in (r, a)]
            pairs += [(r, [pid], 1.0), (a, [pid], 1.0)]
    return pairs, list(free), unmatched_gt


def prf(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    if precision is None or recall is None:
        f1 = None
    elif precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def score(preds: list[Change], gts: list[GTChange], matcher: LabelMatcher, pos_tol: float) -> dict:
    """Overall and per-type precision / recall / F1 plus the matches themselves."""
    pairs, fp_ids, fn_idx = match(preds, gts, matcher, pos_tol)
    by_id = {p.id: p for p in preds}
    matched_pred_ids = {pid for _, ids, _ in pairs for pid in ids}
    per_type = {}
    for t in TYPES:
        tp = sum(1 for k, _, _ in pairs if gts[k].type == t)
        fn = sum(1 for k in fn_idx if gts[k].type == t)
        fp = sum(1 for pid in fp_ids if by_id[pid].type.value == t)
        if tp or fn or fp:
            per_type[t] = prf(tp, fp, fn)
    errors = []
    for k, ids, _ in pairs:
        gt, p = gts[k], by_id[ids[0]]
        if gt.distance_m is not None and p.translation is not None:
            errors.append(abs(float(np.linalg.norm(p.translation)) - gt.distance_m))
    return {
        "overall": prf(len({k for k, _, _ in pairs}), len(fp_ids), len(fn_idx)),
        "per_type": per_type,
        "matches": [
            {"gt": f"{gts[k].type} {gts[k].label}", "pred": ids, "label_score": round(s, 3)}
            for k, ids, s in pairs
        ],
        "false_positives": [
            f"{pid} {by_id[pid].type.value} {by_id[pid].label} ({by_id[pid].confidence.value})"
            for pid in fp_ids
        ],
        "false_negatives": [f"{gts[k].type} {gts[k].label}" for k in fn_idx],
        "translation_error_m": errors,
        "n_matched_predictions": len(matched_pred_ids),
    }


# --------------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------------


def _fmt(x) -> str:
    return "–" if x is None else f"{x:.2f}"


def metrics_table(rows: list[tuple[str, dict]]) -> list[str]:
    lines = ["| | TP | FP | FN | precision | recall | F1 |", "|---|---|---|---|---|---|---|"]
    for name, m in rows:
        lines.append(
            f"| {name} | {m['tp']} | {m['fp']} | {m['fn']} | {_fmt(m['precision'])} | "
            f"{_fmt(m['recall'])} | {_fmt(m['f1'])} |"
        )
    return lines


def markdown(result: dict) -> str:
    a, c = result["all"], result["confirmed_only"]
    lines = [
        f"# Evaluation — pair `{result['pair']}` (run `{result['run']}`)",
        "",
        f"Based on **{result['n_ground_truth']} ground-truth change(s)** and "
        f"{result['n_predictions']} prediction(s) "
        f"({result['n_confirmed']} confirmed, {result['n_unverified']} unverified; "
        f"rejected candidates excluded).",
        "",
    ]
    lines += metrics_table(
        [("all claims (confirmed + unverified)", a["overall"]), ("confirmed only", c["overall"])]
    )
    if a["per_type"]:
        lines += ["", "Per type (all claims):", ""]
        lines += metrics_table(list(a["per_type"].items()))
    lines += [
        "",
        "Matches: "
        + ("; ".join(f"{m['gt']} ← {', '.join(m['pred'])}" for m in a["matches"]) or "none"),
        "False positives: " + ("; ".join(a["false_positives"]) or "none"),
        "False negatives: " + ("; ".join(a["false_negatives"]) or "none"),
        "",
    ]
    u = result["unverified"]
    lines.append(
        f"Unverified predictions: {u['count']} — expected: {u['expected']}, "
        f"found as unverified: {u['expected_found']}."
    )
    if a["translation_error_m"]:
        lines.append(
            f"Moved-object distance error: mean {np.mean(a['translation_error_m']):.2f} m."
        )
    return "\n".join(lines) + "\n"


def evaluate(run: str, gt_path: str, force: bool = False) -> dict:
    """Score a run's changes against a ground-truth file; writes ``eval/eval.{json,md}``.

    Args:
        run: run name; needs C5 output (uses the latest change list).
        gt_path: ground-truth JSON (see module docstring).
        force: unused — evaluation is cheap and always recomputed.
    """
    from changedet import models
    from changedet.core.config import load_run_config
    from changedet.stages.c8_describe import latest_changes

    full = load_run_config(run)
    gt = load_ground_truth(gt_path)
    changes, source = latest_changes(run)
    preds = [
        c
        for c in changes
        if c.type is not ChangeType.UNCHANGED and c.confidence is not Confidence.REJECTED
    ]
    confirmed = [c for c in preds if c.confidence is Confidence.CONFIRMED]
    matcher = LabelMatcher(
        full.eval, full.fusion.clip_model, models.resolve_device(full.detect.device)
    )
    unverified = [c for c in preds if c.confidence is Confidence.UNVERIFIED]
    expected_found = [
        e
        for e in gt.expected_unverified
        if any(
            c.type.value == e["type"] and matcher.score(labels_of(c), e["label"]) > 0
            for c in unverified
        )
    ]
    result = {
        "pair": gt.pair,
        "run": run,
        "source": source,
        "n_ground_truth": len(gt.changes),
        "n_predictions": len(preds),
        "n_confirmed": len(confirmed),
        "n_unverified": len(unverified),
        "all": score(preds, gt.changes, matcher, full.eval.pos_tol),
        "confirmed_only": score(confirmed, gt.changes, matcher, full.eval.pos_tol),
        "unverified": {
            "count": len(unverified),
            "expected": len(gt.expected_unverified),
            "expected_found": len(expected_found),
        },
    }
    models.unload()
    save_json(run_path(run, EVAL_JSON), result)
    run_path(run, EVAL_MD).write_text(markdown(result))
    a, c = result["all"]["overall"], result["confirmed_only"]["overall"]
    log.info(
        "C11 %s: all claims P=%s R=%s F1=%s | confirmed only P=%s R=%s F1=%s",
        gt.pair,
        _fmt(a["precision"]),
        _fmt(a["recall"]),
        _fmt(a["f1"]),
        _fmt(c["precision"]),
        _fmt(c["recall"]),
        _fmt(c["f1"]),
    )
    return result


def load_eval(run: str) -> dict:
    return load_json(run_path(run, EVAL_JSON))
