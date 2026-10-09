from __future__ import annotations

import numpy as np
import pytest

import changedet.stages.c3_detect as c3
from changedet.core.config import load_config
from changedet.stages.c3_detect import (
    assign_labels,
    border_fraction,
    cxcywh_to_xyxy,
    make_exclusive,
    mask_nms,
    phrase_spans,
    tight_bbox,
)
from tests.conftest import REPO_ROOT

# --------------------------------------------------------------------------------------------
# Pure helpers
# --------------------------------------------------------------------------------------------


def test_phrase_spans() -> None:
    # [CLS] office chair . chest of drawers . tv . [SEP]
    ids = [101, 1, 2, 1012, 3, 4, 5, 1012, 6, 1012, 102]
    assert phrase_spans(ids, 1012, 3) == [[1, 2], [4, 5, 6], [8]]
    with pytest.raises(ValueError):
        phrase_spans(ids, 1012, 4)


def test_assign_labels_scores_each_phrase_separately() -> None:
    spans = [[1, 2], [4, 5, 6], [8]]
    logits = np.full((2, 11), -10.0)
    logits[0, 2] = 2.0  # query 0 lights up "chair" of "office chair"
    logits[0, 8] = 0.0  # ... and weakly "tv"
    logits[1, 5] = 1.0  # query 1: "of" in "chest of drawers"
    best, score = assign_labels(logits, spans)
    np.testing.assert_array_equal(best, [0, 1])
    np.testing.assert_allclose(score, 1 / (1 + np.exp([-2.0, -1.0])))


def test_cxcywh_to_xyxy_clips() -> None:
    boxes = np.array([[0.5, 0.5, 0.2, 0.4], [0.05, 0.5, 0.2, 0.2]])
    np.testing.assert_allclose(
        cxcywh_to_xyxy(boxes, 100, 200), [[40, 60, 60, 140], [0, 80, 15, 120]]
    )


def square(h=20, w=20, y0=5, y1=15, x0=5, x1=15) -> np.ndarray:
    mask = np.zeros((h, w), bool)
    mask[y0:y1, x0:x1] = True
    return mask


def test_border_fraction() -> None:
    assert border_fraction(square()) == 0.0
    assert border_fraction(np.ones((20, 20), bool)) == 1.0
    sliver = square(y0=0, y1=2, x0=0, x1=20)  # thin strip along the top edge
    assert border_fraction(sliver) > 0.5
    assert border_fraction(np.zeros((5, 5), bool)) == 0.0


def test_mask_nms_keeps_best_of_duplicates() -> None:
    a, a_dup, b = square(), square(x1=14), square(x0=16, x1=20, y0=0, y1=4)
    assert mask_nms([a, a_dup, b], np.array([0.4, 0.6, 0.5]), 0.7) == [1, 2]


def test_make_exclusive_small_object_wins() -> None:
    bed = square(y0=0, y1=20, x0=0, x1=20)
    bag = square(y0=5, y1=10, x0=5, x1=10)
    bed_out, bag_out = make_exclusive([bed, bag])
    assert bag_out.sum() == 25 and not (bed_out & bag_out).any()
    assert bed_out.sum() == 400 - 25


def test_tight_bbox() -> None:
    assert tight_bbox(square(y0=3, y1=7, x0=2, x1=9)) == (2, 3, 9, 7)


# --------------------------------------------------------------------------------------------
# Per-frame filtering with a fake segmenter
# --------------------------------------------------------------------------------------------


def test_detect_frame_filters(monkeypatch) -> None:
    h = w = 100
    masks = {
        "bed": square(h, w, 40, 95, 10, 90),
        "bag": square(h, w, 50, 65, 30, 50),
        "bag_dup": square(h, w, 50, 65, 30, 49),
        "window": square(h, w, 5, 30, 20, 80),
        "tiny": square(h, w, 2, 4, 2, 4),
        "cut_off": square(h, w, 0, 100, 97, 100),
    }
    names = list(masks)
    monkeypatch.setattr(c3, "_segment", lambda image, boxes, cfg, device: [masks[n] for n in names])
    phrases = ["bed", "bag", "window", "chair"]
    idx = np.array([0, 1, 1, 2, 3, 3])
    scores = np.array([0.8, 0.5, 0.45, 0.6, 0.9, 0.9])
    cfg = load_config(overrides=["detect.min_mask_px=10"]).detect
    out = c3.detect_frame(
        np.zeros((h, w, 3), np.uint8), np.zeros((6, 4)), scores, idx, phrases, cfg, "cpu"
    )
    assert sorted(out.labels) == ["bag", "bed"]  # dup suppressed, window structural, tiny/cut off
    bed, bag = out.masks[out.labels.index("bed")], out.masks[out.labels.index("bag")]
    assert not (bed & bag).any()
    assert out.scores[out.labels.index("bag")] == 0.5


# --------------------------------------------------------------------------------------------
# Smoke test with the real models on a real frame (skipped if unavailable)
# --------------------------------------------------------------------------------------------


@pytest.mark.models
@pytest.mark.real_data
def test_models_find_bag_on_bed() -> None:
    from changedet import models
    from changedet.io.record3d import load_record3d

    cfg = load_config().detect
    if not (models.is_downloaded(cfg.detector_model) and models.is_downloaded(cfg.segmenter_model)):
        pytest.skip("model checkpoints not downloaded (scripts/download_models.sh)")
    if not (REPO_ROOT / "vid1").exists():
        pytest.skip("vid1 recording not present")
    device = models.resolve_device(cfg.device)
    image = load_record3d(REPO_ROOT / "vid1").read_rgb(1340)  # bag lying on the bed
    phrases = list(cfg.vocabulary) + list(cfg.structural)
    ((boxes, scores, idx),) = c3._detect_boxes([image], phrases, cfg, device)
    out = c3.detect_frame(image, boxes, scores, idx, phrases, cfg, device)
    models.unload()
    assert "bag" in out.labels and "bed" in out.labels
    bag, bed = out.masks[out.labels.index("bag")], out.masks[out.labels.index("bed")]
    assert bag.sum() > 5000 and not (bag & bed).any()
