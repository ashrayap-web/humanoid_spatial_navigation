"""Every core type must survive a JSON round trip with numpy arrays and enums restored."""

from __future__ import annotations

import numpy as np
import pytest

from changedet.core.types import (
    CameraFrame,
    Change,
    ChangeReport,
    ChangeType,
    Confidence,
    Detection2D,
    Frame,
    Object3D,
    Reconstruction,
)


def make_frame(session: str = "A", index: int = 0) -> Frame:
    return Frame(session, index, 120 + index, 2.0, f"frames/{session}/{index:06d}.jpg", 312.5)


def make_camera(session: str = "A", index: int = 0) -> CameraFrame:
    T = np.eye(4)
    T[:3, 3] = [1.0, 2.0, 0.5]
    K = np.array([[666.4, 0, 358.7], [0, 666.4, 478.6], [0, 0, 1]])
    return CameraFrame(make_frame(session, index), K, T, f"recon/{session}/depth/{index:06d}.npy")


def make_object(obj_id: str = "A_000") -> Object3D:
    rng = np.random.default_rng(0)
    clip = rng.normal(size=512).astype(np.float32)
    return Object3D(
        id=obj_id,
        session="A",
        label="bag",
        label_scores={"bag": 2.4, "backpack": 0.6},
        points_path=f"objects/A/{obj_id}.ply",
        centroid=np.array([1.0, 2.0, 0.6]),
        bbox_min=np.array([0.8, 1.8, 0.5]),
        bbox_max=np.array([1.2, 2.2, 0.7]),
        n_observations=7,
        clip_embedding=clip / np.linalg.norm(clip),
        dino_embedding=np.ones(768) / np.sqrt(768),
        best_view=("A", 12),
        best_crop_path="objects/crops/A_000.jpg",
    )


def make_change(change_type: ChangeType = ChangeType.MOVED) -> Change:
    moved = change_type is ChangeType.MOVED
    return Change(
        id="chg_000",
        type=change_type,
        label="chair",
        object_a="A_001",
        object_b="B_003",
        position_a=np.array([0.0, 1.0, 0.4]),
        position_b=np.array([1.0, 1.0, 0.4]),
        translation=np.array([1.0, 0.0, 0.0]) if moved else None,
        rotation_deg=85.0 if moved else None,
        T_a_to_b=np.eye(4) if moved else None,
        match_cost=0.31,
        confidence=Confidence.UNVERIFIED,
        visibility={"free_frac": 0.1, "cameras": [1, 4]},
    )


def assert_round_trip(obj) -> None:
    restored = type(obj).from_json(obj.to_json())
    assert type(restored) is type(obj)
    assert restored.to_dict() == obj.to_dict()


@pytest.mark.parametrize(
    "obj",
    [
        make_frame(),
        make_camera(),
        Detection2D("B", 3, "bag", 0.71, "detections/B/masks/000003_00.png", (10, 20, 110, 220)),
        make_object(),
        make_change(ChangeType.MOVED),
        make_change(ChangeType.REMOVED),
        Reconstruction(
            cameras={"A": [make_camera("A", 0), make_camera("A", 1)], "B": [make_camera("B", 0)]},
            cloud_paths={"A": "recon/cloud_A.ply", "B": "recon/cloud_B.ply"},
            background_cloud_path="recon/background.ply",
            floor_plane=np.array([0.0, 0.0, 1.0, 0.0]),
            scale_is_metric=True,
            alignment_rmse=0.012,
        ),
        ChangeReport(
            "demo",
            [make_change(), make_change(ChangeType.ADDED)],
            "One chair moved.",
            {"counts": {"moved": 1}, "warnings": []},
        ),
    ],
    ids=lambda o: type(o).__name__,
)
def test_round_trip(obj) -> None:
    assert_round_trip(obj)


def test_round_trip_restores_numpy_enums_and_tuples() -> None:
    change = Change.from_json(make_change().to_json())
    assert isinstance(change.type, ChangeType) and change.type is ChangeType.MOVED
    assert isinstance(change.confidence, Confidence)
    assert isinstance(change.T_a_to_b, np.ndarray) and change.T_a_to_b.shape == (4, 4)
    assert change.description is None

    obj = Object3D.from_json(make_object().to_json())
    assert obj.best_view == ("A", 12)
    assert isinstance(obj.clip_embedding, np.ndarray) and obj.clip_embedding.shape == (512,)

    recon = Reconstruction.from_dict(
        Reconstruction({"A": [make_camera()]}, {}, "bg.ply", np.zeros(4), True, 0.0).to_dict()
    )
    assert isinstance(recon.cameras["A"][0], CameraFrame)
    assert isinstance(recon.cameras["A"][0].frame, Frame)


def test_optional_arrays_stay_none() -> None:
    change = Change.from_json(make_change(ChangeType.REMOVED).to_json())
    assert change.translation is None and change.T_a_to_b is None
