from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial import cKDTree

import changedet.stages.c8_llm as c8l
from changedet import llm
from changedet.core import cache
from changedet.core.config import load_config
from changedet.core.types import Change, ChangeType, Confidence
from tests.test_c8_describe import BED, CHAIR, DESK, box

CFG = load_config()


@pytest.mark.parametrize(
    "obj, expected",
    [
        ((1.0, 2.0), "to the right of"),
        ((-1.0, 2.0), "to the left of"),
        ((0.0, 3.0), "behind"),
        ((0.0, 1.0), "in front of"),
        ((0.05, 2.05), None),
    ],
)
def test_direction_seen_from_viewpoint(obj, expected) -> None:
    viewpoint, landmark = np.zeros(3), np.array([0.0, 2.0, 0.5])
    assert c8l.direction(np.array([*obj, 0.5]), landmark, viewpoint) == expected


def test_relation() -> None:
    cfg = CFG.report
    bag = box("A_9", "bag", [0.5, 0.4, 0.55], [1.1, 1.0, 0.75])
    assert c8l.relation(bag, BED, cfg)[0] == "on"
    stool = box("A_8", "stool", [3.2, 0.2, 0], [3.7, 0.6, 0.5])
    assert c8l.relation(stool, DESK, cfg)[0] == "next to"
    far = box("A_7", "lamp", [6, 0, 0], [6.3, 0.3, 1.5])
    assert c8l.relation(far, DESK, cfg)[0] == "about 1.8 m from"


def test_location_facts_and_text() -> None:
    cfg = CFG.report
    bag = box("A_9", "bag", [0.5, 0.4, 0.55], [1.1, 1.0, 0.75])
    walls = cKDTree(np.array([[x, -1.0] for x in np.linspace(-3, 6, 50)]))
    facts = c8l.location_facts(
        bag, [BED, DESK, CHAIR], np.array([2.0, -3.0, 1.3]), walls, None, cfg
    )
    assert facts["rests_on"] == "the bed"
    assert facts["nearest_wall_m"] == 1.7 and facts["size_m"] == [0.6, 0.6, 0.2]
    change = Change(
        "chg_000",
        ChangeType.REMOVED,
        "bag",
        "A_9",
        None,
        bag.centroid,
        None,
        None,
        None,
        None,
        None,
        Confidence.CONFIRMED,
        visibility={
            "old_location": {
                "seen_from": "B",
                "free_frac": 0.84,
                "occupied_frac": 0.0,
                "unknown_frac": 0.16,
            }
        },
    )
    text = c8l.facts_text(c8l.change_facts(change, {"A_9": facts}, CFG.match))
    assert "REMOVED bag" in text and "rests on the bed" in text
    assert "84% of that space was seen empty" in text and "1.7 m from the nearest wall" in text


def test_numbers_and_grounding() -> None:
    assert c8l.numbers_in("moved 1.40 m, 84% seen, 2 views") == {"1.4", "84", "2"}
    facts = "- chg_000: REMOVED bag. Evidence: 84% of that space was seen empty. 0.3 m from X."
    assert c8l.grounded("The bag, 0.3 m from X, is gone (84% seen empty).", facts, ["bag"]) is None
    assert "1.5" in c8l.grounded("The bag moved 1.5 m.", facts, ["bag"])
    assert c8l.grounded("Something is gone.", facts, ["bag"]) == "does not name the object"
    assert c8l.grounded("Something is gone.", facts, None) is None


def _changes():
    vis = {
        "old_location": {
            "seen_from": "B",
            "free_frac": 0.84,
            "occupied_frac": 0.0,
            "unknown_frac": 0.16,
        }
    }
    bag = Change(
        "chg_000",
        ChangeType.REMOVED,
        "bag",
        "A_9",
        None,
        np.zeros(3),
        None,
        None,
        None,
        None,
        None,
        Confidence.CONFIRMED,
        vis,
    )
    lamp = Change(
        "chg_001",
        ChangeType.REMOVED,
        "lamp",
        "A_8",
        None,
        np.zeros(3),
        None,
        None,
        None,
        None,
        None,
        Confidence.UNVERIFIED,
        vis,
    )
    facts = {
        "chg_000": {
            "id": "chg_000",
            "type": "removed",
            "object": "bag",
            "confidence": "confirmed",
            "evidence": "84% seen empty.",
        },
        "chg_001": {
            "id": "chg_001",
            "type": "removed",
            "object": "lamp",
            "confidence": "unverified",
            "evidence": "16% never observed.",
        },
    }
    return [bag, lamp], facts


def test_write_with_llm_keeps_only_grounded_sentences(monkeypatch, tmp_path) -> None:
    changes, facts = _changes()
    monkeypatch.setattr(
        llm,
        "ask_json",
        lambda cfg, prompt, schema, images, cache_dir: {
            "summary": "One change is confirmed and 1 could not be verified.",
            "sentences": [
                {"id": "chg_000", "sentence": "The bag is gone; 84% of its spot was seen."},
                {"id": "chg_001", "sentence": "The lamp, 2.5 m from the door, may be gone."},
            ],
        },
    )
    counts = {"confirmed": 1, "unverified": 1, "rejected": 0, "unchanged": 3}
    sentences, summary, warnings = c8l.write_with_llm(changes, facts, counts, CFG.report, tmp_path)
    assert sentences == {"chg_000": "The bag is gone; 84% of its spot was seen."}
    assert summary.startswith("One change") and len(warnings) == 1 and "2.5" in warnings[0]


def test_write_with_llm_rejects_invented_summary_numbers(monkeypatch, tmp_path) -> None:
    changes, facts = _changes()
    monkeypatch.setattr(
        llm, "ask_json", lambda *a, **k: {"summary": "7 things changed.", "sentences": []}
    )
    counts = {"confirmed": 1, "unverified": 1, "rejected": 0, "unchanged": 3}
    sentences, summary, warnings = c8l.write_with_llm(changes, facts, counts, CFG.report, tmp_path)
    assert sentences == {} and summary is None and len(warnings) == 3  # 2 missing + summary


@pytest.fixture
def synthetic_run():
    from tests.test_c10_c12 import RUN, make_synthetic_run

    make_synthetic_run(np.random.default_rng(0))
    return RUN


def test_describe_llm_mode_falls_back_without_credentials(synthetic_run, monkeypatch) -> None:
    from changedet.stages.c8_describe import describe

    monkeypatch.setattr(llm, "credentials_available", lambda provider: False)
    report = describe(synthetic_run)
    assert report.stats["report_mode"] == "template (no credentials)"
    assert any("no LLM credentials" in w for w in report.stats["warnings"])
    text = cache.run_path(synthetic_run, "report/report.md").read_text()
    assert "Spatial facts the sentence is based on" in text


def test_describe_llm_mode_uses_llm_sentences(synthetic_run, monkeypatch) -> None:
    from changedet.stages.c8_describe import describe

    monkeypatch.setattr(llm, "credentials_available", lambda provider: True)
    monkeypatch.setattr(
        llm,
        "ask_json",
        lambda *a, **k: {
            "summary": "A thing was removed; another may have appeared.",
            "sentences": [
                {"id": "chg_000", "sentence": "The thing has been removed."},
                {"id": "chg_001", "sentence": "A new thing may have appeared."},
            ],
        },
    )
    report = describe(synthetic_run, force=True)
    by_id = {c.id: c for c in report.changes}
    assert by_id["chg_000"].description == "The thing has been removed."
    assert report.summary == "A thing was removed; another may have appeared."
    assert report.stats["report_mode"].startswith("llm")
