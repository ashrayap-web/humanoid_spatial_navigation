from __future__ import annotations

import numpy as np

from changedet.core.config import load_config
from changedet.core.types import Change, ChangeType, Confidence, Object3D
from changedet.stages.c8_describe import box_gap, sentence, summary, where, xy_overlap

CFG = load_config()


def box(obj_id: str, label: str, lo, hi) -> Object3D:
    lo, hi = np.array(lo, float), np.array(hi, float)
    return Object3D(
        obj_id,
        obj_id[0],
        label,
        {label: 1.0},
        "",
        (lo + hi) / 2,
        lo,
        hi,
        5,
        np.zeros(4),
        np.zeros(4),
        (obj_id[0], 0),
        "",
    )


BED = box("A_1", "bed", [0, 0, 0], [2, 1.6, 0.8])  # pillows make the box top higher than the
DESK = box("A_2", "desk", [3, 0, 0], [4.2, 0.7, 0.75])  # mattress the bag lies on
CHAIR = box("A_3", "chair", [3.1, 0.9, 0], [3.6, 1.4, 0.9])


def test_box_helpers() -> None:
    assert box_gap(np.zeros(3), np.ones(3), np.array([2.0, 0, 0]), np.array([3.0, 1, 1])) == 1.0
    assert box_gap(np.zeros(3), np.ones(3), np.full(3, 0.5), np.full(3, 2.0)) == 0.0
    assert xy_overlap(np.zeros(3), np.ones(3), np.array([0.5, 0, 0]), np.array([2.0, 1, 1])) == 0.5


def test_where() -> None:
    cfg = CFG.report
    bag = box("A_9", "bag", [0.5, 0.4, 0.55], [1.1, 1.0, 0.75])
    assert where(bag, [BED, DESK, CHAIR], cfg) == "on the bed"
    tucked_chair = box("A_8", "stool", [3.2, 0.2, 0], [3.7, 0.6, 0.5])  # under the desk, on floor
    assert where(tucked_chair, [BED, DESK], cfg) == "on the floor next to the desk"
    box_on_floor = box("A_7", "box", [2.4, 0.2, 0], [2.55, 0.5, 0.3])
    assert where(box_on_floor, [BED, DESK], cfg) == "on the floor about 0.4 m from the bed"
    far = box("A_6", "lamp", [8, 8, 0.5], [8.3, 8.3, 1.5])
    assert where(far, [BED, DESK], cfg) == ""
    frame_b = box("A_5", "picture frame", [5, 0, 1.2], [5.02, 0.3, 1.5])
    frame_a = box("A_4", "picture frame", [5, 0.35, 1.2], [5.02, 0.6, 1.5])
    assert where(frame_a, [frame_b], cfg) == "next to another picture frame"


def change(kind: ChangeType, conf: Confidence, label: str = "bag", **kw) -> Change:
    vis = kw.pop("visibility", None)
    return Change(
        "chg_000",
        kind,
        label,
        "A_9",
        "B_9",
        np.zeros(3),
        np.ones(3),
        kw.get("translation"),
        kw.get("rotation_deg"),
        None,
        None,
        conf,
        vis,
    )


def test_sentences() -> None:
    m = CFG.match
    vis_b = {
        "old_location": {
            "seen_from": "B",
            "free_frac": 0.84,
            "occupied_frac": 0.06,
            "unknown_frac": 0.10,
        }
    }
    assert sentence(
        change(ChangeType.REMOVED, Confidence.CONFIRMED, visibility=vis_b), "on the bed", "", m
    ) == (
        "The bag that was on the bed has been removed. Confirmed: 84% of that space was seen "
        "empty in the second recording."
    )
    vis_a = {
        "new_location": {
            "seen_from": "A",
            "free_frac": 0.2,
            "occupied_frac": 0.0,
            "unknown_frac": 0.8,
        }
    }
    assert sentence(
        change(ChangeType.ADDED, Confidence.UNVERIFIED, "box", visibility=vis_a),
        "",
        "on the floor next to the desk",
        m,
    ) == (
        "A new box may have appeared on the floor next to the desk. Could not be verified: 80% "
        "of that space was never clearly observed in the first recording."
    )
    moved = change(
        ChangeType.MOVED,
        Confidence.CONFIRMED,
        "chair",
        translation=np.array([1.2, 0.5, 0]),
        rotation_deg=-85.0,
    )
    assert sentence(moved, "next to the desk", "next to the bed", m) == (
        "The chair moved 1.3 m and turned 85°, from next to the desk to next to the bed."
    )
    turned = change(
        ChangeType.MOVED,
        Confidence.CONFIRMED,
        "lamp",
        translation=np.array([0.02, 0, 0]),
        rotation_deg=90.0,
    )
    assert sentence(turned, "on the desk", "on the desk", m) == (
        "The lamp on the desk was turned 90° in place."
    )
    replaced = change(
        ChangeType.REPLACED,
        Confidence.REJECTED,
        "mug->bottle",
        visibility={
            "new_location": {
                "seen_from": "A",
                "free_frac": 0.0,
                "occupied_frac": 0.9,
                "unknown_frac": 0.1,
            }
        },
    )
    text = sentence(replaced, "on the desk", "on the desk", m)
    assert text.startswith(
        "The mug on the desk seemed to have been replaced by a bottle. Rejected:"
    )


def test_summary_counts() -> None:
    confirmed = change(ChangeType.REMOVED, Confidence.CONFIRMED)
    confirmed.description = "The bag that was on the bed has been removed. Confirmed: 84% ..."
    unverified = change(ChangeType.REMOVED, Confidence.UNVERIFIED, "table")
    rejected = change(ChangeType.ADDED, Confidence.REJECTED, "pillow")
    unchanged = change(ChangeType.UNCHANGED, Confidence.CONFIRMED, "bed")
    text = summary([confirmed, unverified, rejected, unchanged, unchanged])
    assert text == (
        "1 change was confirmed: the bag that was on the bed has been removed. 1 further "
        "candidate could not be verified because the area was barely observed in one of the "
        "recordings. 1 false alarm was rejected because the other recording still shows the "
        "object. 2 objects were matched as unchanged."
    )
    assert summary([unchanged]).startswith("No change could be confirmed.")
