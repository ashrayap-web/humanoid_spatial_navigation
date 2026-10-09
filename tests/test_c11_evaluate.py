from __future__ import annotations

import json

import numpy as np
import pytest

from changedet.core.config import Config, load_config
from changedet.core.types import Change, ChangeType, Confidence
from changedet.stages.c11_evaluate import (
    GTChange,
    LabelMatcher,
    contains_word,
    load_ground_truth,
    match,
    prf,
    score,
)

CFG = load_config()
EXACT = LabelMatcher(Config.from_dict({**CFG.eval, "use_clip": False}))


def pred(i, kind, label, conf=Confidence.CONFIRMED, pos=(0.0, 0.0, 0.5), **kw) -> Change:
    p = np.array(pos)
    return Change(
        f"chg_{i:03d}",
        ChangeType(kind),
        label,
        "A_0",
        "B_0",
        p,
        p,
        kw.get("translation"),
        None,
        None,
        None,
        conf,
        vlm_verdict=kw.get("vlm"),
    )


def test_contains_word() -> None:
    assert contains_word("office chair", "chair") and contains_word("chair", "Office Chair")
    assert not contains_word("chairs", "chair") and not contains_word("bag", "backpack")
    assert not contains_word("", "bag")


def test_prf() -> None:
    assert prf(1, 0, 0) == {"tp": 1, "fp": 0, "fn": 0, "precision": 1.0, "recall": 1.0, "f1": 1.0}
    m = prf(2, 2, 1)
    assert m["precision"] == 0.5 and np.isclose(m["recall"], 2 / 3)
    assert np.isclose(m["f1"], 2 * 0.5 * (2 / 3) / (0.5 + 2 / 3))
    assert prf(0, 0, 0)["precision"] is None and prf(0, 3, 0)["f1"] is None
    assert prf(0, 1, 1)["f1"] == 0.0


def test_score_counts_tp_fp_fn() -> None:
    gts = [
        GTChange("removed", "bag"),
        GTChange("moved", "chair", distance_m=1.0),
        GTChange("added", "box"),
        GTChange("removed", "lamp"),
    ]
    preds = [
        pred(0, "removed", "bag"),
        pred(1, "moved", "office chair", translation=np.array([1.2, 0, 0])),
        pred(2, "added", "box", Confidence.UNVERIFIED),
        pred(3, "removed", "jacket"),  # false alarm
        pred(4, "added", "lamp"),
    ]  # right label, wrong type
    r = score(preds, gts, EXACT, CFG.eval.pos_tol)
    assert (r["overall"]["tp"], r["overall"]["fp"], r["overall"]["fn"]) == (3, 2, 1)
    assert r["per_type"]["removed"] == prf(1, 1, 1)
    assert r["per_type"]["added"] == prf(1, 1, 0)
    assert np.isclose(r["translation_error_m"], [0.2]).all()
    assert r["false_negatives"] == ["removed lamp"]


def test_vlm_label_and_position() -> None:
    gts = [GTChange("removed", "backpack", position=np.array([1.0, 1.0, 0.0]))]
    near = pred(0, "removed", "bag", pos=(1.2, 1.1, 0.6), vlm={"label": "black backpack"})
    far = pred(1, "removed", "backpack", pos=(3.0, 1.0, 0.6))
    pairs, fp, fn = match([far, near], gts, EXACT, pos_tol=0.5)
    assert [ids for _, ids, _ in pairs] == [["chg_000"]] and fp == ["chg_001"] and not fn


def test_confirmed_preferred_over_unverified() -> None:
    gts = [GTChange("removed", "bag")]
    preds = [pred(0, "removed", "bag", Confidence.UNVERIFIED), pred(1, "removed", "bag")]
    pairs, fp, _ = match(preds, gts, EXACT, 0.5)
    assert pairs[0][1] == ["chg_001"] and fp == ["chg_000"]


def test_replaced_matches_removed_plus_added_both_ways() -> None:
    # GT replaced <- predicted removed + added
    pairs, fp, fn = match(
        [pred(0, "removed", "mug"), pred(1, "added", "bottle")],
        [GTChange("replaced", "mug->bottle")],
        EXACT,
        0.5,
    )
    assert pairs[0][1] == ["chg_000", "chg_001"] and not fp and not fn
    # GT removed + added <- predicted replaced
    pairs, fp, fn = match(
        [pred(0, "replaced", "mug->bottle")],
        [GTChange("removed", "mug"), GTChange("added", "bottle")],
        EXACT,
        0.5,
    )
    assert len(pairs) == 2 and not fp and not fn


def test_load_ground_truth(tmp_path) -> None:
    path = tmp_path / "gt.json"
    path.write_text(
        json.dumps(
            {
                "pair": "p",
                "changes": [
                    {"type": "moved", "label": "chair", "position": [1, 2, 0], "distance_m": 1.5}
                ],
            }
        )
    )
    gt = load_ground_truth(path)
    assert gt.run == "p" and gt.changes[0].distance_m == 1.5
    np.testing.assert_allclose(gt.changes[0].position, [1, 2, 0])


@pytest.mark.models
def test_clip_label_similarity() -> None:
    from changedet import models

    if not models.is_downloaded(CFG.fusion.clip_model):
        pytest.skip("CLIP checkpoint not downloaded")
    matcher = LabelMatcher(CFG.eval, CFG.fusion.clip_model, models.resolve_device("auto"))
    assert matcher.score(["backpack"], "bag") >= CFG.eval.label_sim_thresh
    assert matcher.score(["table"], "chair") == 0.0
    assert matcher.score(["box"], "bag") == 0.0
    models.unload()


def test_synthetic_suite_smoke() -> None:
    from changedet.eval_synthetic import markdown, run_suite

    result = run_suite(CFG, n_scenes=3, seed=1)
    assert result["n_ground_truth"] >= 3
    assert result["overall"]["recall"] > 0.5
    assert "Synthetic suite" in markdown(result)
