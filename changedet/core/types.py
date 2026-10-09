"""Core data types shared by every stage (spec Section 5.2).

All types are dataclasses with ``to_dict`` / ``from_dict`` / ``to_json`` / ``from_json``.
Numpy arrays are stored as nested lists and restored as arrays using the field type hints.

File paths stored in these types (``image_path``, ``depth_path``, ...) are **relative to the run
directory** (``runs/<run>/``) so a run can be moved or committed as an example; resolve them with
``changedet.core.cache.run_path(run, relpath)``.
"""

from __future__ import annotations

import dataclasses
import functools
import json
import types
import typing
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

# --------------------------------------------------------------------------------------------
# JSON (de)serialisation helpers
# --------------------------------------------------------------------------------------------


def to_jsonable(obj: Any) -> Any:
    """Convert dataclasses, enums, numpy arrays/scalars and containers into plain JSON values."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    return obj


@functools.cache
def _type_hints(cls: type) -> dict[str, Any]:
    return typing.get_type_hints(cls)


def _decode(tp: Any, value: Any) -> Any:
    """Rebuild a value of type ``tp`` from its JSON form."""
    if value is None:
        return None
    origin = typing.get_origin(tp)
    if origin in (typing.Union, types.UnionType):
        args = [a for a in typing.get_args(tp) if a is not type(None)]
        if len(args) != 1:
            raise TypeError(f"Only Optional[X] unions are supported, got {tp}")
        return _decode(args[0], value)
    if tp is np.ndarray:
        return np.asarray(value)
    if dataclasses.is_dataclass(tp):
        return from_dict(tp, value)
    if isinstance(tp, type) and issubclass(tp, Enum):
        return tp(value)
    if origin is list:
        (item_tp,) = typing.get_args(tp)
        return [_decode(item_tp, v) for v in value]
    if origin is tuple:
        args = typing.get_args(tp)
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_decode(args[0], v) for v in value)
        return tuple(_decode(a, v) for a, v in zip(args, value, strict=True))
    if origin is dict:
        _, value_tp = typing.get_args(tp)
        return {k: _decode(value_tp, v) for k, v in value.items()}
    return value


def from_dict(cls: type, data: dict) -> Any:
    """Build dataclass ``cls`` from a dict produced by ``to_jsonable``."""
    hints = _type_hints(cls)
    kwargs = {
        f.name: _decode(hints[f.name], data[f.name])
        for f in dataclasses.fields(cls)
        if f.name in data
    }
    return cls(**kwargs)


class JsonMixin:
    """Adds JSON round-tripping to a dataclass."""

    def to_dict(self) -> dict:
        return to_jsonable(self)

    @classmethod
    def from_dict(cls, data: dict):
        return from_dict(cls, data)

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, text: str):
        return cls.from_dict(json.loads(text))


# --------------------------------------------------------------------------------------------
# Types
# --------------------------------------------------------------------------------------------


@dataclass
class Frame(JsonMixin):
    session: str  # "A" or "B"
    index: int  # index within the session's kept frames
    source_index: int  # index in the original recording (Record3D frame number / video frame)
    timestamp: float  # seconds from start of video
    image_path: str  # RGB image, relative to the run dir
    sharpness: float  # variance of Laplacian


@dataclass
class CameraFrame(JsonMixin):
    frame: Frame
    K: np.ndarray  # (3,3)
    T_world_cam: np.ndarray  # (4,4) camera-to-world, OpenCV camera axes
    depth_path: str  # .npy float32 depth in metres, same resolution as image
    confidence_path: str | None = None  # None for Record3D exports


@dataclass
class Reconstruction(JsonMixin):
    cameras: dict[str, list[CameraFrame]]  # keyed by session
    cloud_paths: dict[str, str]  # per-session fused point cloud (.ply)
    background_cloud_path: str  # merged static background (.ply)
    floor_plane: np.ndarray  # (4,) plane coefficients in world frame (~[0,0,1,0])
    scale_is_metric: bool
    alignment_rmse: float  # residual after background alignment (m)


@dataclass
class Detection2D(JsonMixin):
    session: str
    frame_index: int
    label: str
    score: float
    mask_path: str  # .png binary mask
    bbox_xyxy: tuple[int, int, int, int]


@dataclass
class Object3D(JsonMixin):
    id: str  # "A_003"
    session: str
    label: str  # majority-vote label
    label_scores: dict[str, float]
    points_path: str  # .ply of object points in world frame
    centroid: np.ndarray  # (3,)
    bbox_min: np.ndarray  # (3,) axis-aligned
    bbox_max: np.ndarray  # (3,)
    n_observations: int  # number of frames it was seen in
    clip_embedding: np.ndarray  # (D_clip,) L2-normalised
    dino_embedding: np.ndarray  # (D_dino,) L2-normalised
    best_view: tuple[str, int]  # (session, frame_index) with largest visible mask
    best_crop_path: str  # image crop for VLM / report


class ChangeType(str, Enum):
    ADDED = "added"
    REMOVED = "removed"
    MOVED = "moved"
    UNCHANGED = "unchanged"
    REPLACED = "replaced"  # removed + added at the same location (derived in C5)


class Confidence(str, Enum):
    CONFIRMED = "confirmed"  # supported by geometry (+ VLM if run)
    UNVERIFIED = "unverified"  # the other session never observed the relevant space
    REJECTED = "rejected"  # VLM or visibility check contradicted the candidate


@dataclass
class Change(JsonMixin):
    id: str  # "chg_000"
    type: ChangeType
    label: str
    object_a: str | None  # Object3D id in session A
    object_b: str | None  # Object3D id in session B
    position_a: np.ndarray | None  # centroid in A
    position_b: np.ndarray | None  # centroid in B
    translation: np.ndarray | None  # for MOVED: B - A (m)
    rotation_deg: float | None  # for MOVED: yaw change about z
    T_a_to_b: np.ndarray | None  # (4,4) rigid transform for MOVED
    match_cost: float | None
    confidence: Confidence
    visibility: dict | None = None  # filled by C6
    vlm_verdict: dict | None = None  # filled by C7
    description: str | None = None  # filled by C8
    match_margin: float | None = None  # C5: second-best cost minus match cost (small = ambiguous)


@dataclass
class ChangeReport(JsonMixin):
    run_name: str
    changes: list[Change]
    summary: str  # natural-language paragraph (C8)
    stats: dict  # counts per type/confidence, timings
