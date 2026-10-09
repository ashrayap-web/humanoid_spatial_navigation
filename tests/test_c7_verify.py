from __future__ import annotations

import json

import numpy as np
import pytest

import changedet.stages.c7_verify as c7
from changedet import llm
from changedet.core import cache
from changedet.core.config import load_config, save_config
from changedet.core.types import CameraFrame, Change, ChangeType, Confidence, Frame
from tests.test_c10_c12 import RUN, make_synthetic_run

CFG = load_config().vlm


# --------------------------------------------------------------------------------------------
# JSON parsing and the provider wrapper
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        '{"change_present": false, "confidence": 0.9}',
        '```json\n{"change_present": false, "confidence": 0.9}\n```',
        'Sure! Here it is:\n{"change_present": false, "confidence": 0.9}\nHope that helps.',
    ],
)
def test_parse_json_tolerates_fences_and_prose(text) -> None:
    assert llm.parse_json(text) == {"change_present": False, "confidence": 0.9}


@pytest.mark.parametrize("text", ["no json here", '{"a": 1,,}', "[1, 2]"])
def test_parse_json_rejects_invalid(text) -> None:
    with pytest.raises(ValueError):
        llm.parse_json(text)


def test_ask_json_caches_by_content(tmp_path, monkeypatch) -> None:
    calls = []

    def fake(cfg, prompt, schema, images):
        calls.append(prompt)
        return '{"change_present": true, "confidence": 0.7}'

    monkeypatch.setitem(llm.PROVIDERS, "anthropic", fake)
    img = llm.encode_jpeg(np.zeros((8, 8, 3), np.uint8))
    a = llm.ask_json(CFG, "prompt", c7.SCHEMA, [img], tmp_path)
    b = llm.ask_json(CFG, "prompt", c7.SCHEMA, [img], tmp_path)
    assert a == b == {"change_present": True, "confidence": 0.7} and len(calls) == 1
    llm.ask_json(CFG, "other prompt", c7.SCHEMA, [img], tmp_path)
    assert len(calls) == 2  # different prompt -> new call


def test_unknown_provider_is_unavailable(tmp_path) -> None:
    cfg = load_config(overrides=["vlm.provider=somebody"]).vlm
    with pytest.raises(llm.LLMUnavailable):
        llm.ask_json(cfg, "p", c7.SCHEMA, [], tmp_path)


def test_credentials_check(monkeypatch, tmp_path) -> None:
    for k in (
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_PROFILE",
        "ANTHROPIC_FEDERATION_RULE_ID",
    ):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    assert not llm.credentials_available("anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    assert llm.credentials_available("anthropic")
    assert not llm.credentials_available("other")


# --------------------------------------------------------------------------------------------
# Verdicts
# --------------------------------------------------------------------------------------------


def change(kind=ChangeType.REMOVED, label="jacket", conf=Confidence.UNVERIFIED) -> Change:
    return Change(
        "chg_001", kind, label, "A_000", None, np.zeros(3), None, None, None, None, None, conf
    )


def test_normalise_verdict() -> None:
    v = c7.normalise_verdict(
        {
            "same_object": "not_applicable",
            "change_present": False,
            "label": " clothes rail ",
            "short_description": "x",
            "confidence": 1.4,
        }
    )
    assert v["same_object"] is None and v["confidence"] == 1.0 and v["label"] == "clothes rail"
    assert c7.normalise_verdict({"same_object": "yes", "confidence": 0.5})["same_object"] is True


def test_decide_rejects_only_confident_no_change() -> None:
    c = change()
    c7.decide(
        c,
        c7.normalise_verdict({"change_present": False, "confidence": 0.9, "label": "jacket"}),
        0.8,
    )
    assert c.confidence is Confidence.REJECTED and c.vlm_verdict["decision"] == "rejected"

    c = change()
    c7.decide(c, c7.normalise_verdict({"change_present": False, "confidence": 0.5}), 0.8)
    assert c.confidence is Confidence.UNVERIFIED and c.vlm_verdict["decision"] == "kept"

    c = change(label="chair", conf=Confidence.CONFIRMED)
    c7.decide(
        c,
        c7.normalise_verdict(
            {"change_present": True, "confidence": 0.95, "label": "black office chair"}
        ),
        0.8,
    )
    assert c.confidence is Confidence.CONFIRMED
    assert c.vlm_verdict["refined_label"] == "black office chair"


def test_prompts_name_the_claim() -> None:
    assert "REMOVED" in c7.build_prompt(change()) and "'jacket'" in c7.build_prompt(change())
    replaced = c7.build_prompt(change(ChangeType.REPLACED, "mug->bottle"))
    assert "'mug'" in replaced and "'bottle'" in replaced


# --------------------------------------------------------------------------------------------
# Evidence views
# --------------------------------------------------------------------------------------------


def test_projected_box() -> None:
    K = np.array([[100.0, 0, 49.5], [0, 100.0, 49.5], [0, 0, 1]])
    cam = CameraFrame(Frame("B", 0, 0, 0.0, "", 0.0), K, np.eye(4), "")
    pts = np.array([[-0.1, -0.1, 2.0], [0.1, 0.1, 2.0]] * 10)
    x0, y0, x1, y1, frac = c7.projected_box(pts, cam, 100, 100)
    np.testing.assert_allclose([x0, y0, x1, y1], [44.5, 44.5, 54.5, 54.5])
    assert frac == 1.0
    assert c7.projected_box(pts * [1, 1, -1], cam, 100, 100) is None  # behind the camera


# --------------------------------------------------------------------------------------------
# Stage
# --------------------------------------------------------------------------------------------


@pytest.fixture
def synthetic(monkeypatch):
    make_synthetic_run(np.random.default_rng(0))
    img = np.full((50, 40, 3), 128, np.uint8)
    monkeypatch.setattr(
        c7, "evidence_pair", lambda run, ch, objects, recon, cfg: (img, img, ("before", "after"))
    )
    return RUN


def test_stage_without_credentials_passes_through(synthetic, monkeypatch) -> None:
    monkeypatch.setattr(llm, "credentials_available", lambda provider: False)
    changes = c7.verify_changes(synthetic)
    assert [c.confidence for c in changes] == [
        Confidence.CONFIRMED,
        Confidence.UNVERIFIED,
        Confidence.CONFIRMED,
    ]
    assert all(c.vlm_verdict is None for c in changes)
    assert cache.exists(synthetic, "viz/c7_evidence/chg_000.jpg")


def test_stage_rejects_false_candidate(synthetic, monkeypatch) -> None:
    monkeypatch.setattr(llm, "credentials_available", lambda provider: True)
    answers = {
        "chg_000": {"change_present": True, "confidence": 0.95},  # real removal
        "chg_001": {"change_present": False, "confidence": 0.9},
    }  # still there

    def fake_ask(cfg, prompt, schema, images, cache_dir):
        return {
            "same_object": "not_applicable",
            "label": "thing",
            "short_description": "s",
            **answers[current["id"]],
        }

    current = {}
    real_pair = c7.evidence_pair

    def pair(run, ch, objects, recon, cfg):
        current["id"] = ch.id
        return real_pair(run, ch, objects, recon, cfg)

    monkeypatch.setattr(c7, "evidence_pair", pair)
    monkeypatch.setattr(llm, "ask_json", fake_ask)
    changes = {c.id: c for c in c7.verify_changes(synthetic)}
    assert changes["chg_000"].confidence is Confidence.CONFIRMED
    assert changes["chg_001"].confidence is Confidence.REJECTED
    assert changes["chg_002"].vlm_verdict is None  # UNCHANGED is never sent
    saved = json.loads(cache.run_path(synthetic, "changes/changes_verified.json").read_text())
    assert saved[1]["vlm_verdict"]["decision"] == "rejected"


def test_stage_stops_calling_when_unavailable(synthetic, monkeypatch) -> None:
    monkeypatch.setattr(llm, "credentials_available", lambda provider: True)

    def unavailable(*args, **kwargs):
        raise llm.LLMUnavailable("bad key")

    monkeypatch.setattr(llm, "ask_json", unavailable)
    save_config(load_config(overrides=["viz.gif_frames=3"]), cache.run_path(RUN, "config.yaml"))
    changes = c7.verify_changes(synthetic)
    assert all(c.vlm_verdict is None for c in changes)
