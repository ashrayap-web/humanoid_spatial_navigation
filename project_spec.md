# What Changed in the Room? — Project Specification

This file is the single source of truth for this project. Read it fully before writing any code.
The project is built **one component at a time**. The human will say which component to build next.
Only build that component. Do not implement future components early.

---

## 0. How to work on this project (instructions for Claude Code)

1. **Read this whole file first** at the start of every session.
2. **Work on one component only** — the one the human names (e.g. "build C3"). If something
   from a later component seems needed, write a minimal stub with a `TODO(Cx)` comment and say so.
3. **Respect the shared contracts** in Section 5 (`core/types.py`, file formats, coordinate
   conventions). If a contract must change, stop, explain why, and ask before changing it.
4. **Every component must:**
   - expose a single public entry function with the signature given in its section,
   - read its inputs from and write its outputs to the run cache (Section 6),
   - skip work if its outputs already exist, unless `force=True`,
   - log what it did (counts, timings, key stats) with the project logger,
   - write a debug visualization (rerun `.rrd` and/or PNG) so its output can be eyeballed,
   - have at least one test in `tests/` (synthetic data is fine and preferred where possible),
   - be runnable on its own via `python -m changedet.cli <component> --run <run_name>`.
5. **When a component is finished:** update the Status table (Section 1) and append a short
   entry to the Decisions Log (Section 12) describing any non-obvious choices.
6. **Verify third-party APIs** (MASt3R, VGGT, SAM 2, Grounding DINO, CLIP, DINOv2, rerun, Open3D)
   against their current READMEs before using them. Do not guess function signatures.
7. **Prefer simple, readable code** over clever code. Reviewers judge "simplicity and usability".
8. All thresholds and model choices go in `configs/default.yaml`, never hard-coded.

---

## 1. Status

| ID  | Component                               | Status      | Milestone |
|-----|-----------------------------------------|-------------|-----------|
| C0  | Project scaffold, types, config, cache  | done        | MVP       |
| C1  | Frame extraction & quality filtering    | done        | MVP       |
| C2  | Reconstruction & alignment (Record3D)   | done        | MVP       |
| C3  | 2D detection & segmentation             | done        | MVP       |
| C4  | 3D lifting & object fusion              | done        | MVP       |
| C5  | Cross-session matching & change classes | done (⚠️ see result) | MVP |
| C6  | Visibility reasoning                    | done        | Core      |
| C7  | VLM change verification                 | not started | Core      |
| C8  | Semantic description & report           | basic done (template); full (LLM) not started | MVP (basic) / Core (full) |
| C9  | Navigation impact analysis              | not started | Stretch   |
| C10 | Interactive visualization               | basic done (+ hero.png/gif); full not started | MVP (basic) / Core (full) |
| C11 | Evaluation against ground truth         | not started | Core      |
| C12 | End-to-end CLI, packaging, README       | done (sample-data link TBD) | MVP       |

**MVP** = meets every minimum requirement of the challenge. It must be working end-to-end
before any Core or Stretch work starts. Recommended order:
C0 → C1 → C2 → C3 → C4 → C5 → C8(basic) → C10(basic) → C12 → **tag `v0.1-mvp`** → C6 → C7 → C8(full) → C11 → C10(full) → C9.

---

## 2. The challenge (what we are building)

Build a system that takes **two phone videos of the same small indoor area** (a room),
recorded at different times from different viewpoints, and identifies **meaningful changes in 3D**:
moved furniture, new obstacles, removed objects.

Our recordings are made with the **Record3D** iPhone app (LiDAR), so each "video" is an export
folder of RGB frames + metric depth + ARKit camera poses (see Section 8.1). Plain `.mp4` input is a
secondary path (Section 4).

Minimum requirements from the brief:
1. Generate a **shared 3D representation** of the scene from both recordings.
2. **Identify and visualize** objects/regions that were **moved, added, or removed** in 3D.
3. **Describe** the detected changes using semantic labels or natural language.

The system must distinguish real changes from differences caused by **camera viewpoint,
occlusion, or incomplete observations**.

What the reviewers care about:
- Simplicity and usability
- Creativity in approach
- Quality of spatial change detection
- Clear, compelling presentation of results
- Coherence between detected changes, geometry, and semantics

Submission: public GitHub repo with a README (run instructions, example inputs/outputs,
design-choices note).

---

## 3. Our approach (one paragraph)

Bring both recordings into one metric, gravity-aligned coordinate frame (per-session LiDAR
depth + ARKit poses, then a gravity-constrained registration of session B onto A). Build an
**object-level 3D map** for each session (open-vocabulary 2D segmentation lifted into 3D and fused
across frames, each object carrying a semantic label and feature embeddings). **Match objects
across sessions** to classify them as added / removed / moved / unchanged, estimating a rigid
transform for moved objects. Use **visibility reasoning** (was this space actually observed in the
other session?) to separate real changes from occlusion and missing coverage, labelling
unverifiable claims as `unverified` instead of guessing. A **VLM verifies** each candidate with
before/after crops and a language model writes a **grounded natural-language report** using 3D
spatial relations. Finally, we show **what the changes mean for a robot** by diffing a floor
occupancy grid and re-planning a path. Everything is shown in an interactive **rerun** viewer.

Key differentiators to protect: (a) object-level rather than point-level diffing,
(b) three-way confidence (changed / unchanged / unverified), (c) navigation impact,
(d) honest quantitative evaluation including a no-change control.

---

## 4. Tech stack & environment

- **Python** 3.10+; dependency management with **uv** (`pyproject.toml`). Provide a Dockerfile in C12.
- **Package name:** `changedet`.
- **Core libs:** numpy, scipy, opencv-python, open3d, torch, pyyaml, tqdm, pydantic (or dataclasses).
- **Input / reconstruction:** **Record3D export (primary)** — RGB + LiDAR metric depth + ARKit poses,
  so no learned reconstruction is needed. For plain `.mp4` input (secondary, post-MVP): MASt3R or
  VGGT joint reconstruction; COLMAP as last-resort fallback.
- **EXR reading:** OpenCV cannot read the Record3D depth EXRs (ZIP-compressed single `R` channel);
  use the `OpenEXR` Python package.
- **Segmentation:** Grounding DINO + SAM 2 (open-vocabulary), both through Hugging Face
  `transformers` (≥ 5.x: `AutoModelForZeroShotObjectDetection`, `Sam2Model`) — no custom CUDA
  builds. Checkpoints `IDEA-Research/grounding-dino-base`, `facebook/sam2.1-hiera-large` cached in
  `checkpoints/hf`; loaders in `changedet/models.py`. Alternatives available in the same library:
  MM-Grounding-DINO, OWLv2, SAM 3 (text-prompted masks; gated weights).
- **Embeddings:** CLIP (semantic), DINOv2 (appearance).
- **VLM / LLM:** accessed through a thin provider-agnostic wrapper (`changedet/llm.py`); model name and
  API key come from config / environment variables. Must degrade gracefully (skip with a warning)
  if no key is set.
- **Visualization:** rerun (`rerun-sdk`); matplotlib for static PNGs.
- **Assumed hardware:** single NVIDIA L40S (46 GB VRAM), 12 CPU cores, 70 GB RAM, Ubuntu 22.04,
  on an NVIDIA Brev instance reached over SSH. Code should still run on a ≥24 GB GPU with lower frame limits.
  Every heavy model call must allow batch-size / frame-count limits in config.
- **Primary input:** Record3D export (RGB + metric depth + poses from iPhone LiDAR); C2 uses the given
  poses/depth directly and only has to align the two sessions to each other.

Large model checkpoints are downloaded into `checkpoints/` (git-ignored) by
`scripts/download_models.sh`.

---

## 5. Shared contracts (do not change without asking)

### 5.1 Conventions

- **Units:** metres, radians.
- **World frame:** right-handed, **z-up**, floor plane at **z = 0**, origin roughly at the floor
  centroid of the scene. Established in C2. Every later component works in this frame.
- **Camera poses:** stored as `T_world_cam` (4×4, camera-to-world), OpenCV camera convention
  (x right, y down, z forward).
- **Intrinsics:** 3×3 `K` per frame (allowed to differ per frame).
- **Session IDs:** always the strings `"A"` (before) and `"B"` (after).
- **Object IDs:** strings `"A_000"`, `"B_014"`, etc. Unique within a run.
- **Depth maps:** float32 metres, 0 = invalid, **same resolution as the RGB image** (Record3D depth
  is 192×256 and is upsampled in C2; NaNs, e.g. on the TV screen, become 0).
- **Record3D source convention** (converted away in C2, never used downstream): poses are
  `[qx, qy, qz, qw, tx, ty, tz]` camera-to-world, ARKit world (y-up, gravity-aligned, origin = first
  frame of *that* recording), OpenGL camera axes (x right, y up, z backward). OpenCV camera =
  `T_gl @ diag(1, -1, -1, 1)`. Quaternion order must be confirmed by a reprojection check in C2.
- **Point clouds on disk:** `.ply` (Open3D). **Arrays:** `.npz`. **Structured data:** `.json`.

### 5.2 Core types (`changedet/core/types.py`)

Implement as dataclasses (or pydantic models) with `to_json()` / `from_json()` helpers.

```python
@dataclass
class Frame:
    session: str  # "A" or "B"
    index: int  # index within the session's kept frames
    source_index: int  # index in the original recording (Record3D frame number / video frame)
    timestamp: float  # seconds from start of video
    image_path: str  # path to the RGB image in the cache
    sharpness: float  # variance of Laplacian


@dataclass
class CameraFrame:
    frame: Frame
    K: np.ndarray  # (3,3)
    T_world_cam: np.ndarray  # (4,4)
    depth_path: str  # .npy float32 depth in metres, same resolution as image
    confidence_path: str | None  # None for our Record3D exports (no confidence maps exported)


@dataclass
class Reconstruction:
    cameras: dict[str, list[CameraFrame]]  # keyed by session
    cloud_paths: dict[str, str]  # per-session fused point cloud (.ply)
    background_cloud_path: str  # merged static background (.ply)
    floor_plane: np.ndarray  # (4,) plane coefficients in world frame (should be ~[0,0,1,0])
    scale_is_metric: bool
    alignment_rmse: float  # residual after background alignment (m)


@dataclass
class Detection2D:
    session: str
    frame_index: int
    label: str
    score: float
    mask_path: str  # .png binary mask
    bbox_xyxy: tuple[int, int, int, int]


@dataclass
class Object3D:
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
class Change:
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
    visibility: dict | None  # filled by C6, see C6
    vlm_verdict: dict | None  # filled by C7, see C7
    description: str | None  # filled by C8
    match_margin: float | None  # C5: second-best cost - match cost (small = ambiguous)


@dataclass
class ChangeReport:
    run_name: str
    changes: list[Change]
    summary: str  # natural-language paragraph (C8)
    stats: dict  # counts per type/confidence, timings
```

### 5.3 Config

Single YAML file `configs/default.yaml`, loaded into a nested config object by
`changedet/core/config.py`. Each component reads its own section (`frames:`, `recon:`,
`detect:`, `fusion:`, `match:`, `visibility:`, `vlm:`, `report:`, `nav:`, `viz:`, `eval:`).
CLI flags can override any key with `--set section.key=value`.

---

## 6. Run cache & output layout

Each run has a name and a directory: `runs/<run_name>/`. Components read and write only here.

```
runs/<run_name>/
  config.yaml                 # resolved config snapshot
  inputs.json                 # {"A": {"path": ..., "kind": "record3d"|"video"}, "B": {...}}
  frames/{A,B}/000123.jpg     # C1
  frames/frames.json          # C1: list[Frame]
  recon/{A,B}/depth/*.npy     # C2
  recon/reconstruction.json   # C2: Reconstruction
  recon/cloud_A.ply, cloud_B.ply, background.ply
  detections/{A,B}/masks/*.png
  detections/detections.json  # C3: list[Detection2D]
  objects/objects_A.json, objects_B.json   # C4: list[Object3D]
  objects/{A,B}/*.ply
  objects/crops/*.jpg
  changes/changes_raw.json    # C5
  changes/visibility/*.npz    # C6
  changes/changes_verified.json  # C7
  report/report.md            # C8
  report/changes.json         # C8: final ChangeReport
  nav/                        # C9
  viz/*.rrd, viz/*.png        # debug + final visualizations
  eval/eval.json              # C11
  logs/run.log
```

Helpers in `changedet/core/cache.py`: `run_dir(run)`, `exists(run, relpath)`, `save_json`,
`load_json`, `save_ply`, `load_ply`, and a `@cached_stage("relpath")` decorator that skips a stage
when its outputs exist.

---

## 7. Repository layout

```
change-detect/
  CLAUDE.md                   # this file
  README.md                   # written in C12
  pyproject.toml
  Dockerfile
  configs/default.yaml
  scripts/download_models.sh
  changedet/
    __init__.py
    cli.py                    # `python -m changedet.cli <stage|all> --run NAME ...`
    llm.py                    # provider-agnostic VLM/LLM wrapper (C7/C8)
    core/
      types.py  config.py  cache.py  logging.py  geometry.py
    io/
      record3d.py             # Record3D export loader (C0)
    stages/
      c1_frames.py
      c2_reconstruct.py
      c3_detect.py
      c4_fuse.py
      c5_match.py
      c6_visibility.py
      c7_verify.py
      c8_describe.py
      c9_navigation.py
      c10_visualize.py
      c11_evaluate.py
  data/
    recordings/<pair>/{A,B}/  # Record3D exports (git-ignored; download link in README)
    samples/                  # small example data (or download links)
    ground_truth/<pair>.json
  tests/
  runs/                       # git-ignored
  checkpoints/                # git-ignored
```

`core/geometry.py` holds shared math: back-projection, projection, plane fitting, rigid
transforms, ICP wrappers, voxelization, 3D IoU. Add to it as components need it; keep it tested.

---

## 8. Data plan (human-recorded inputs)

### 8.1 What we have now

One pair, recorded with Record3D on an iPhone in a small bedroom (desk + office chair, white
chest of drawers with TV, bed, window, clothes rail). Currently in the repo root as `vid1/` (A)
and `vid2/` (B); they may be moved to `data/recordings/main/{A,B}` — code must take paths from
`inputs.json`, never assume a location.

| Session | Folder  | Frames | Duration | Notes |
|---------|---------|--------|----------|-------|
| A (before) | `vid1/` | 4385 | 73 s | two loops of the room, camera moves ~1.5 × 0.5 m |
| B (after)  | `vid2/` | 3992 | 66 s | similar coverage, different start pose → **different ARKit world frame** |

Export layout (both sessions):
```
<session>/
  rgb/0.jpg … N.jpg        # 720×960 portrait, already upright; numeric (not zero-padded) names → sort numerically
  depth/0.exr … N.exr      # 192×256 float32 metres, single channel "R", ZIP-compressed EXR, NaN = invalid
  metadata*.json           # file name varies ("metadata (1).json") → glob for metadata*.json
```
`metadata*.json` keys: `w, h` (RGB size), `dw, dh` (depth size), `fps` (60), `K` (3×3,
**column-major** flat list, at RGB resolution), `perFrameIntrinsicCoeffs` (`[fx, fy, cx, cy]` per
frame), `frameTimestamps` (s), `poses` (`[qx, qy, qz, qw, tx, ty, tz]` per frame, see 5.1),
`initPose`, `cameraType`. There are no confidence maps in these exports.

**Ground truth for `main`:** one change — the **black bag lying on the bed is removed** in B.
Incidental differences that must **not** be reported: the bed cover is rumpled differently and the
daylight/exposure differs slightly. These make `main` a small false-positive test as well.

```json
{
  "pair": "main",
  "changes": [
    {"type": "removed", "label": "bag", "note": "black bag lying on the bed, near the window end"}
  ],
  "expected_unverified": []
}
```

### 8.2 What this means for the plan

- With one real change, MOVED / ADDED / REPLACED are exercised **only by synthetic tests** until
  more pairs exist. Every component's synthetic test must therefore cover all change types.
- REMOVED is exactly the case visibility reasoning (C6) is built for (did B look at the bed and
  see it empty?), so C6 is still well exercised.
- C9 (navigation) will correctly report "no navigation impact" on `main` — the bag was on the bed,
  not the floor. That is a valid result but not a compelling demo.
- C11 on one change is a sanity check, not a benchmark; synthetic results + `nochange` carry the
  quantitative story.

### 8.3 Recordings still wanted (in priority order)

| Pair       | Purpose | Contents |
|------------|---------|----------|
| `nochange` | **Control / false-positive test** — most important for credibility | Re-record the same room with nothing changed, different walking path |
| `main2`    | Richer headline demo | Move the office chair, add a box on the floor in a walking route (gives C9 a real demo), rotate an object in place, swap a mug for a bottle |
| `hard`     | Robustness | A change hidden behind furniture in B; one moved object only partly visible |

Recording tips: walk a slightly different path in B than A but cover the same surfaces; keep
objects of interest within ~3 m (LiDAR range); pan slowly.

### Ground-truth format (`data/ground_truth/<pair>.json`)

As in the `main` example above. Allowed types: `moved`, `added`, `removed`, `replaced`
(`label` as `"mug->bottle"`). Optional: approximate positions can be added later by clicking in the
viewer (C11 supports matching by label only if positions are missing).

---

## 9. Components

Each component lists: **Purpose**, **Entry point**, **Inputs**, **Outputs**, **Method**,
**Config keys**, **Debug output**, **Tests**, **Acceptance criteria**, **Out of scope**.

---

### C0 — Project scaffold, types, config, cache, logging

**Purpose:** Everything later components rely on.

**Entry point:** none (library code) + `changedet/cli.py` skeleton.

**Deliverables:**
- `pyproject.toml` with uv, base dependencies, `ruff` + `pytest` dev deps.
- `core/types.py` exactly as Section 5.2, with JSON (de)serialization that handles numpy arrays.
- `core/config.py`: load YAML, merge `--set` overrides, snapshot into the run dir.
- `core/cache.py`: as Section 6, including the `@cached_stage` decorator and `force` handling.
- `core/logging.py`: console + `runs/<run>/logs/run.log`, timing context manager.
- `core/geometry.py`: initial helpers (backproject, project, transform points, fit plane RANSAC, voxel downsample).
- `cli.py`: subcommands `init` (create run, record input paths; a directory containing
  `metadata*.json` + `rgb/` + `depth/` is detected as `kind: "record3d"`, a file as `kind: "video"`),
  one subcommand per stage (stubs printing "not implemented"), and `all`.
- `changedet/io/record3d.py`: loader for a Record3D export — numeric frame listing, metadata
  parsing (incl. column-major `K`), EXR depth reading via `OpenEXR`. Pure I/O, no conversions.
- `configs/default.yaml` with sections for every component (placeholder values are fine).
- `.gitignore` (runs/, checkpoints/, *.rrd, __pycache__, vid*/, data/recordings/).

**Tests:** round-trip JSON for every type; config override; cache skip/force behaviour; geometry
helpers on toy data (project(backproject(x)) == x).

**Acceptance criteria:** `uv run pytest` passes; `python -m changedet.cli init --run demo --a vid1 --b vid2`
creates `runs/demo/` with `inputs.json` (both detected as `record3d`) and `config.yaml`; the
Record3D loader reads frame 0 of both sessions (RGB 960×720, depth 256×192).

**Out of scope:** any model code.

---

### C1 — Frame extraction & quality filtering

**Purpose:** Turn each recording (~4000 frames at 60 fps) into a manageable set of sharp,
well-spread keyframes.

**Entry point:** `extract_frames(run: str, force: bool = False) -> dict[str, list[Frame]]`

**Inputs:** `inputs.json` paths (Record3D export dirs; `.mp4` for `kind: "video"`).

**Outputs:** `frames/{A,B}/<index:06d>.jpg`, `frames/frames.json` = `{"A": [Frame...], "B": [...]}`
(each `Frame` records `source_index` so C2 can fetch the matching depth and pose).

**Method:**
1. Read frames: Record3D → `rgb/<i>.jpg` in numeric order, timestamps from metadata (already
   upright, no rotation needed). Video (post-MVP, currently a `TODO(C12)` stub) → OpenCV,
   respecting rotation metadata.
2. Score **every** frame's sharpness (variance of Laplacian on a half-resolution grayscale decode;
   threaded, ~1–2 s per session) and keep the **sharpest frame in each `1/frames.fps` window**
   (default 3 fps → ~200–220 candidates per session).
3. Drop frames whose sharpness is below `frames.blur_ratio` × the median of their
   `frames.blur_window` neighbouring candidates on each side. (Relative, not a global
   percentile: a global cut-off removed sharp but low-texture views such as the dark chair and
   plain walls — see Decisions log.)
4. Drop near-duplicates. Record3D: skip a frame if the camera moved less than
   `frames.min_trans_m` **and** rotated less than `frames.min_rot_deg` since the last kept frame
   (pose-based — more reliable than pixels). Video: downscaled grayscale difference < `frames.min_diff`.
5. Cap at `frames.max_per_session` (default 100) by uniform subsampling.
6. Resize so the long side is `frames.max_side` (default 1024; Record3D's 960 is copied
   byte-for-byte, no re-encode). If C1 resizes, C2 must scale intrinsics by the image size ratio.

**Config keys:** `fps`, `blur_ratio`, `blur_window`, `min_trans_m`, `min_rot_deg`, `min_diff`,
`max_per_session`, `max_side`.

**Debug output:** `viz/c1_contact_sheet_{A,B}.png` thumbnails grid with sharpness values.

**Tests:** synthetic frame sequence with some blurred frames → blurred ones dropped; synthetic
poses with a stationary segment → duplicates dropped.

**Acceptance criteria:** on `main`, 50–100 frames per session, no visibly blurry frames in the
contact sheet, the bag on the bed visible in several A frames and the empty bed in several B frames.

**Out of scope:** pose estimation.

---

### C2 — Joint reconstruction & alignment

**Purpose:** Produce cameras, depth, and point clouds for both sessions in **one shared, metric,
gravity-aligned world frame**. This is the foundation; spend effort here, but cap at the fallback.

**Entry point:** `reconstruct(run: str, force: bool = False) -> Reconstruction`

**Inputs:** C1 frames + the Record3D exports (`recon.method == "record3d"`, default), or
C1 frames only for the learned methods.

**Outputs:** `recon/reconstruction.json`, per-frame depth `.npy`, `cloud_A.ply`, `cloud_B.ply`,
`background.ply`.

**Method (`record3d`, primary):**
1. **Per-session cameras:** for each kept frame (via `Frame.source_index`) take `[fx, fy, cx, cy]`
   from `perFrameIntrinsicCoeffs` (scaled if C1 resized) and the pose; convert
   ARKit/OpenGL → OpenCV camera convention (Section 5.1).
   **Sanity check first:** back-project frame i's depth and reproject into a nearby frame j;
   median depth residual must be small (< 3 cm). This confirms quaternion order and axis
   conventions. Fail loudly if not.
2. **Depth:** read EXR, NaN/inf → 0, drop depth beyond `recon.max_depth` (default 4.0 m — LiDAR is
   unreliable further out), suppress "flying pixels" by zeroing pixels whose depth jumps to a valid
   4-neighbour by more than `recon.edge_rel_thresh` × their own depth, then upsample 192×256 → RGB resolution with **nearest-neighbour**
   (never bilinear — it invents depth across object edges). Save as `.npy`.
3. **Gravity & floor (per session):** ARKit worlds are already gravity-aligned with +y up; rotate
   each session to z-up. Floor = lowest height-histogram bin with ≥ `recon.floor_min_frac` × the
   fullest bin's count, refined by RANSAC on points near it; level so the floor is z = 0, normal +z
   (on `main` the fitted floor is within 0.15° of ARKit gravity). Gravity is now shared, so the remaining A↔B difference is only **yaw + x/y translation**
   (plus a tiny z residual).
4. **Per-session clouds:** back-project valid depth from kept frames, fuse, voxel downsample
   (`recon.voxel_size`, default 0.02 m), statistical outlier removal.
5. **Cross-session registration (B onto A):** each recording has its own ARKit origin, so this is
   required.
   a. Coarse (exhaustive, no initial guess needed): for every yaw in 0–360° in
      `recon.yaw_step_deg` steps, rasterise the top-down occupancy of wall/furniture points
      (`recon.coarse_z_range`, cell `recon.coarse_cell`) and find the best 2D shift by FFT
      cross-correlation with A's grid. Keep the `recon.coarse_top_k` best yaws (≥ 20° apart).
   b. Fine: Huber-weighted point-to-plane ICP that solves **only yaw + xyz**
      (`geometry.icp_yaw`, coarse-to-fine cut-offs 0.3 → 0.15 → `recon.icp_max_dist`) on clouds
      downsampled to `recon.icp_voxel`, from each candidate; keep the highest inlier fraction.
   c. Apply the transform to B's cameras and cloud. Record `alignment_rmse` (inlier RMSE at
      `recon.icp_max_dist`); warn if > 0.03 m or inlier fraction < 0.5.
   d. Centre the world on A's cloud (x, y); the floor stays at z = 0.
6. **Scale:** LiDAR depth is metric → `scale_is_metric = True`.
7. **Background cloud:** points from both sessions that agree (nearest neighbour across sessions
   within `recon.static_thresh`) → `background.ply`. Approximate near changes: parts of a removed
   object within `static_thresh` of a surface in the other session are kept (see Decisions log).
   Used by the viewer; object-level stages (C4–C6) do not rely on it.

**Method (learned, secondary, post-MVP — implement as `TODO` stub until needed):**
`mast3r` / `vggt` — joint reconstruction of both sessions' frames in one optimisation, keep
per-pixel confidence, then steps 3–7 (floor fit must also find gravity; scale from
`recon.scale_factor` with a warning). `separate` — reconstruct A and B separately, align with
FPFH + RANSAC + ICP.

**Config keys:** `method` (record3d | mast3r | vggt | separate), `voxel_size`, `max_depth`,
`edge_rel_thresh`, `reproj_tol`, `floor_bin`, `floor_min_frac`, `outlier_nb`, `outlier_std`,
`coarse_cell`, `coarse_z_range`, `yaw_step_deg`, `coarse_top_k`, `icp_voxel`, `icp_max_dist`,
`static_thresh`, `conf_thresh`, `scale_factor`, `max_frames_joint`.

**Debug output:** `viz/c2_recon.rrd` with both clouds (A blue, B orange), camera frustums for both
sessions, the floor plane. `viz/c2_topdown.png` — top-down overlay of A and B after alignment (the
quickest way to eyeball registration). Log `alignment_rmse` and the fraction of points in
`background.ply`.

**Tests:** synthetic: two copies of a random room-like scene (floor + walls + boxes) with a
known yaw + translation offset (incl. yaw > 90°) → registration recovers it within 1° / 2 cm;
floor alignment puts a tilted plane at z = 0 with +z normal; OpenGL→OpenCV pose conversion
round-trips through project/backproject on a toy example.

**Acceptance criteria:** on `main`, walls/floor/desk/drawers of A and B overlap visually in the
viewer and the top-down PNG; `alignment_rmse` < 0.03 m; floor at z ≈ 0; the bag is visible in
`cloud_A.ply` and absent from `cloud_B.ply`; the bag's upper part is absent from `background.ply`.

**Result on `main` (2026-10-09):** ✅ walls overlap as single lines in `c2_topdown.png`;
pose check residual 7.2 / 8.5 mm (A / B); B→A yaw 65.1°, `alignment_rmse` 2.25 cm, inlier
fraction 0.83, median cross-cloud NN distance 1.2 cm, B-depth→A-camera residual 1.6 cm; joint
floor tilt 0.13°, offset 3 mm; bag = 1.2k-point dark cluster on the bed in A only. ⚠️ ~half of
the background points in the bag footprint are above the bed (bag's lower part is within 5 cm of
B's rumpled duvet). Runtime ~30 s; `recon/` ≈ 550 MB (upsampled depth `.npy`).

**Out of scope:** object reasoning.

---

### C3 — 2D detection & segmentation

**Purpose:** Find object instances in every kept frame with open-vocabulary labels and masks.

**Entry point:** `detect(run: str, force: bool = False) -> list[Detection2D]`

**Inputs:** C1 frames.

**Outputs:** `detections/detections.json`, masks as PNGs
(`detections/{A,B}/masks/<frame_index:06d>_<k:02d>.png`, 0/255).

**Method:**
1. Grounding DINO prompted with `detect.vocabulary` **plus** `detect.structural` (so e.g. a
   window competes with "picture frame"). Each box is scored against **each phrase separately**
   (max sigmoid token probability within the phrase's token span); label = best phrase,
   score = its probability. (The library's phrase decoding glues adjacent phrases together,
   e.g. "chair office chair".)
2. SAM 2 with those boxes as prompts → instance masks (`multimask_output=False`).
3. Filter: score < `detect.min_score`; masks smaller than `detect.min_mask_px`; masks whose
   outline lies mostly on the image border (`border_fraction` > `detect.max_border_frac`).
4. Per-frame NMS on masks (IoU > `detect.mask_nms_iou`), across labels — duplicates such as
   desk/table/chest of drawers on the same furniture keep only the best-scoring one.
5. Structural classes are dropped; they belong to the background.
6. **Exclusive masks:** each pixel goes to the smallest mask covering it, so an object lying on a
   larger one (the bag on the bed) is a separate segment and does not leak into the bed in C4.
   Masks that shrink below `min_mask_px` are dropped.

**Alternative mode** (`detect.mode == "sam_auto"`): SAM 2 automatic masks, then label each mask
by CLIP similarity against the vocabulary. Implement only if the primary mode is poor.

**Config keys:** `mode`, `detector_model`, `segmenter_model`, `device`, `vocabulary`,
`structural`, `min_score`, `min_mask_px`, `max_border_frac`, `mask_nms_iou`, `batch_size`,
`debug_every`.

**Debug output:** `viz/c3_overlays_{A,B}/` — a sample of frames with coloured masks and labels.

**Tests:** pure helpers (phrase spans, per-phrase labelling, border fraction, NMS, exclusive
masks) and the per-frame filter with a fake segmenter; a smoke test runs the real models on
`vid1` frame 1340 and checks the bag and bed are found with disjoint masks (markers `models` +
`real_data`, skipped if checkpoints or the recording are missing — no room images are committed).

**Acceptance criteria:** on `main`, the bag is detected in several A frames; the bed, desk,
chair, drawers, TV and monitor are detected in both sessions; few spurious labels on the overlays.

**Result on `main` (2026-10-09):** ✅ A: 886 detections / 100 frames — **bag in 31 frames**, bed 36,
chest of drawers 140, monitor 127, chair 44, desk 31, … B: 747 / 100 — **bag 0**, bed 31, … Bag mask
disjoint from the bed in the overlays. Known label noise: the TV is mostly called "monitor"
(tv only 10/6), furniture occasionally split into two masks, a stray "box" on the clothes rail —
left to C4's label voting / 3D fusion. Runtime ~75 s for both sessions on the L40S.

**Out of scope:** any 3D.

---

### C4 — 3D lifting & object fusion

**Purpose:** Turn per-frame 2D detections into a clean **object map per session**: one `Object3D`
per physical object, with label, geometry and embeddings.

**Entry point:** `fuse_objects(run: str, force: bool = False) -> dict[str, list[Object3D]]`

**Inputs:** C2 cameras and depth, C3 detections.

**Outputs:** `objects/objects_{A,B}.json`, `objects/{A,B}/*.ply`, `objects/crops/*.jpg`.

**Method (per session independently):**
1. **Lift:** for each detection, drop "see-through" depth (masked pixels deeper than the
   **median** depth of a `fusion.ring_px` ring around the mask + `fusion.behind_margin`), erode
   the mask (`fusion.erode_px`), back-project every `fusion.pixel_stride`-th masked pixel into
   world points, keep the largest DBSCAN cluster.
   *Why the see-through rule:* LiDAR on the glossy TV records a full **mirror image of the room
   behind the wall** (~16% of each C2 cloud on `main`); without it the TV object was 2 m wide and
   absorbed its neighbours.
2. **Embed:** CLIP (`openai/clip-vit-large-patch14`, projected 768-d) and DINOv2
   (`facebook/dinov2-base`, CLS 768-d) of the crop with non-mask pixels greyed out.
3. **Associate incrementally** (frame by frame): match each new segment to existing objects using
   - geometric overlap: fraction of the segment's voxels shared with the object (voxel size
     `fusion.voxel_size`),
   - semantic similarity: cosine of CLIP embeddings,
   - merge if overlap > `fusion.overlap_thresh` **and** similarity > `fusion.sim_thresh`;
     otherwise create a new object.
4. **Accumulate:** union of points (voxel-downsampled), running mean of embeddings (re-normalised),
   label vote counts weighted by detection score, observation count, and the frame with the largest
   mask area as `best_view`.
5. **Post-process:**
   - drop objects with `n_observations < fusion.min_observations`,
   - merge pairs of objects whose 3D boxes overlap heavily (IoU > `merge_iou`, or
     > `overlap_thresh` of the smaller's voxels touch the larger) and whose labels or CLIP agree
     (fixes over-segmentation),
   - absorb an object into a larger one, **whatever the labels**, if it lies inside the larger's
     box and > `contain_frac` of its voxels are on the larger's surface (e.g. "jacket" detected on
     the people in a photo → part of the picture frame). An object lying *on* another (bag on
     bed) only touches it with its underside and is not absorbed,
   - drop objects below the floor or implausibly large.
6. Save a crop of the best view for each object (used by C7/C8 and the README).

**Config keys:** `clip_model`, `dino_model`, `embed_batch`, `crop_pad`, `erode_px`, `ring_px`,
`behind_margin`, `pixel_stride`, `min_segment_points`, `dbscan_eps`, `dbscan_min_points`,
`voxel_size`, `overlap_thresh`, `sim_thresh`, `min_observations`, `merge_iou`, `contain_frac`,
`max_object_extent`.

**Debug output:** `viz/c4_objects.rrd` — each session's objects as coloured point clouds with
3D boxes and labels, over the grey background; `viz/c4_objects_topdown.png` — A | B side by side.

**Tests:** synthetic segments (no models): two boxes seen from several noisy partial views →
exactly two objects with correct box centres (± 2 voxels); a bag lying on a bed stays separate;
duplicate tracks with different labels merge; a "jacket" on a photo is absorbed by the frame;
rare / sub-floor / oversized objects are dropped; see-through filter and DBSCAN lifting on toy
depth maps.

**Acceptance criteria:** on `main`, each real object of interest appears as **one** object per
session (no duplicates, no merges of distinct objects) — in particular the bag in A is a single
object **separate from the bed** it lies on; labels are sensible.

**Result on `main` (2026-10-09):** ✅ A: 779 segments → 65 tracks → **22 objects**; B: 701 → 71 →
**14**. Bag = `A_011` (seen in 31 frames, 0.60 × 0.56 × 0.23 m, centre z 0.66) separate from the
bed `A_009`; no bag in B. Bed, chest of drawers, desk pedestal, PC tower, bottles, chair, TV,
picture frames each appear once per session at consistent positions. ⚠️ Known issues for C5/C6:
TV still carries some reflection points (A 1.35 m wide); A-only small objects from areas B saw
less or labelled differently (clothes rail "jacket" ×2, extra picture frames on the collage,
"table" = bedside table seen 3×, small "monitor" by the window vs B's "keyboard"); labels differ
across sessions for some objects (desk monitor vs laptop, chest of drawers vs desk). Runtime ~55 s.

**Out of scope:** cross-session reasoning.

---

### C5 — Cross-session matching & change classification

**Purpose:** Compare the two object maps and produce candidate changes. **This completes the
core pipeline.**

**Entry point:** `match_and_classify(run: str, force: bool = False) -> list[Change]`

**Inputs:** C4 object maps, C2 reconstruction.

**Outputs:** `changes/changes_raw.json` (list of `Change`, confidence initially `CONFIRMED`).

**Method:**
1. **Cost matrix** between every A object *i* and B object *j*:
   ```
   cost = w_sem * (1 - cos(clip_i, clip_j))
        + w_app * (1 - cos(dino_i, dino_j))
        + w_geo * min(||c_i - c_j|| / match.max_move_dist, 1)
        + w_size * |log(vol_i / vol_j)|
        + w_label * [label_i != label_j]
   ```
   Note: geometry has a low weight so moved objects can still match by appearance.
   *Calibrated on `main`:* CLIP cosine is ~0.97–1.00 for the same object but still ~0.81
   (p95 0.90) for different objects, so `w_sem` = 3; DINOv2 separates far better (~0.9 vs ~0.15),
   `w_app` = 1; box volumes are noisy under partial views (`w_size` = 0.1) and labels differ for
   the same object across sessions (monitor/laptop), `w_label` = 0.3; `max_cost` = 1.0.
2. **Hungarian assignment** (scipy `linear_sum_assignment`) with gating: pairs with
   cost > `match.max_cost` are treated as unmatched.
3. **Classify:**
   - matched, displacement < `match.move_thresh_m` (default 0.15) and yaw change < `match.rot_thresh_deg` (default 20) → `UNCHANGED`,
   - matched otherwise → `MOVED`,
   - unmatched in A → `REMOVED`,
   - unmatched in B → `ADDED`.
4. **Rigid transform for every matched pair:** yaw + translation ICP (`geometry.icp_yaw`) with
   the smaller cloud as source (partial views fit inside the fuller one). Candidates: "stayed put"
   (identity init) and centroid translation × yaw inits `match.icp_yaw_inits_deg`; a move is only
   taken if its inlier fraction beats staying put by `match.stay_margin` (partial views and
   symmetric objects are not reported as moved). Store `T_a_to_b`, `translation` (= T(c_a) − c_a),
   `rotation_deg`. A rotated-in-place object therefore becomes `MOVED` with ~0 translation.
4b. **Same space, different granularity:** an unmatched object whose voxels overlap an object of
   the other session by ≥ `match.same_space_frac` **and** whose DINOv2 cosine with it is
   ≥ `match.same_space_min_dino` is `UNCHANGED` (e.g. two bottles in A fused into one in B). The
   appearance condition keeps genuine replacements (different object, same spot) for step 5.
5. **Replaced:** a `REMOVED` and an `ADDED` whose centroids are within `match.replace_dist`
   are merged into one `REPLACED` change (keep both object ids).
6. **Ambiguity note:** when two identical objects exist (two chairs), log the second-best cost
   margin for each match; the margin is stored in `Change.match_margin` for the report.
7. `changes_raw.json` contains **all** changes including UNCHANGED (used as landmarks in C8 and
   shown dimmed in C10), ordered moved, replaced, removed, added, unchanged.

**Config keys:** `w_sem`, `w_app`, `w_geo`, `w_size`, `w_label`, `max_cost`, `max_move_dist`,
`move_thresh_m`, `rot_thresh_deg`, `replace_dist`, `same_space_frac`, `same_space_min_dino`,
`voxel_size`, `ambiguous_margin`, `icp_voxel`, `icp_max_dist`, `icp_yaw_inits_deg`,
`stay_margin`.

**Debug output:** `viz/c5_changes.rrd` — background cloud in grey; A objects of REMOVED in red;
B objects of ADDED in green; MOVED as A (faded) + B (solid) with an arrow; UNCHANGED dim.
`viz/c5_changes_topdown.png` with the same colours. Also print a table of changes to the console.

**Tests:** synthetic object maps: one moved, one added, one removed, one rotated in place,
two identical objects swapped positions → correct types; translation and yaw recovered within
tolerance.

**Acceptance criteria:** on `main`, the bag appears as a REMOVED candidate and the bed (rumpled
cover) is matched as UNCHANGED; ≤ 1 other spurious candidate. On `nochange` (once recorded),
≤ 1 spurious change.

**Result on `main` (2026-10-09):** ⚠️ partly met. ✅ bag = REMOVED (`chg_000`, A_011); bed
UNCHANGED (stayed 0.02 m, 0°); all 13 Hungarian matches correct, no false MOVED; 2 granularity
leftovers (bottle, small monitor) correctly UNCHANGED. ❌ 7 other candidates (target ≤ 1): REMOVED
"jacket" ×2 (clothes rail), "monitor" (TV-reflection fragment, seen 4×), "picture frame" ×2
(collage), "table" (bedside, seen 3×); ADDED "pillow" (windowsill). All are objects one session
detected and the other did not. The other session's cloud lies within 3 cm of 79–100% of the
points of the jackets, frames and pillow (→ C6 should REJECT them as "still occupied"); the table
(4%) and monitor (34%) look under-observed (→ likely UNVERIFIED). The bag: 43%, mostly its
underside on the bed. Runtime ~14 s.

**Out of scope:** occlusion handling (C6), language (C8).

---

### C6 — Visibility reasoning

**Purpose:** Separate **real changes** from **missing observations**. A REMOVED claim is only
confirmed if session B actually looked at that space and saw it empty (and vice versa for ADDED).

**Entry point:** `assess_visibility(run: str, force: bool = False) -> list[Change]`

**Inputs:** C5 changes, C4 objects, C2 cameras + depth.

**Outputs:** updates `changes/changes_raw.json` → `changes/changes_vis.json`; per-change
`changes/visibility/<change_id>.npz`.

**Method (projection-based, vectorised — no explicit ray marching needed):**
For a change involving object X from session S, test X's points against the cameras of the
*other* session S':
1. For each point p and each camera in S', project p into the image. If it lands in the image and
   in front of the camera, compare the point's depth `d_p` with the observed depth `D(u,v)`:
   - `D(u,v) > d_p + margin` → the camera saw **past** p → p is **observed free** in S',
   - `|D(u,v) - d_p| <= margin` → p is **observed occupied** in S',
   - `D(u,v) < d_p - margin` → p is **occluded** in that view (unknown),
   - invalid depth / outside image → unknown.
   Depth under the other session's C3 masks of glossy objects (`visibility.unreliable_labels`:
   tv, monitor) is treated as invalid — LiDAR returns a mirror image there, which "sees past"
   anything behind the screen (on `main` this turned a TV-reflection fragment into a false
   CONFIRMED before the rule).
2. Aggregate per point over all S' cameras: free if free votes ≥ `visibility.vote_ratio` ×
   occupied votes (and ≥ 1), occupied symmetrically, else unknown. (The original "any free and no
   occupied" rule is `vote_ratio` = ∞; 3 tolerates a stray vote from depth noise at edges.)
3. Per change:
   - `free_frac`, `occupied_frac`, `unknown_frac` of the object's points,
   - REMOVED / ADDED: `CONFIRMED` if `free_frac >= vis.confirm_free_frac`; `REJECTED` if
     `occupied_frac >= vis.reject_occ_frac` (the object is actually still there — matching failed);
     else `UNVERIFIED`,
   - MOVED: check both the old location (should be free in B) and the new one (should be free
     in A); confidence is the weaker of the two (REJECTED < UNVERIFIED < CONFIRMED),
   - REPLACED: only the new object's location (free in A); the old spot is legitimately occupied
     by the new object in B,
   - UNCHANGED: not checked (stays CONFIRMED, `visibility` None).
4. `Change.visibility` = `{"old_location" | "new_location": {"seen_from", "n_points", "free_frac",
   "occupied_frac", "unknown_frac", "confidence", "free_cameras", "occupied_cameras"}}`, where the
   camera lists are up to `max_cameras` frame indices (of `seen_from`) with the most votes —
   C7 uses `free_cameras` as evidence views. `changes_vis.json` holds all changes.

**Config keys:** `margin` (default 0.05 m), `vote_ratio`, `confirm_free_frac`, `reject_occ_frac`,
`point_subsample`, `max_cameras`, `unreliable_labels`.

**Debug output:** `viz/c6_visibility.rrd` — per change, object points coloured by state
(green = observed free, red = observed occupied, grey = unknown) plus the cameras that saw it;
`viz/c6_visibility_topdown.png` with the fractions and verdict per change.

**Tests:** synthetic scene: object present in A; in B, case 1 a camera looks at the empty spot
(→ CONFIRMED), case 2 no camera covers it (→ UNVERIFIED), case 3 a wall occludes it (→ UNVERIFIED),
case 4 object still there (→ REJECTED).

**Acceptance criteria:** on `main`, the bag removal stays CONFIRMED (B's cameras saw the empty
bed surface), with `free_frac` logged. Until `hard` is recorded, the UNVERIFIED path is proven by
the synthetic tests; additionally, an artificial check on `main` — drop all B frames that see the
bed — must turn the bag change into UNVERIFIED.

**Result on `main` (2026-10-09):** ✅ bag CONFIRMED — free 0.84, occupied 0.06, unknown 0.10
(supported by B frames, ~33 views/point). Artificial check: without the 45 / 100 B frames that see
the bag's spot → unknown 1.00 → UNVERIFIED (also a `real_data` test). Spurious C5 candidates:
REJECTED 4 (picture frame ×2 occ 1.00, pillow occ 1.00, one jacket occ 0.50), UNVERIFIED 3 (jacket
free 0.34 / unknown 0.42; TV-reflection "monitor" unknown 0.87; bedside "table" unknown 0.55).
**The bag is the only CONFIRMED change.** Runtime ~3 s.

**Out of scope:** changing change types (only confidence is updated, except REJECTED).

---

### C7 — VLM change verification

**Purpose:** Use a vision-language model as a second opinion to reject false positives and
sharpen labels.

**Entry point:** `verify_changes(run: str, force: bool = False) -> list[Change]`

**Inputs:** C6 changes (or C5 if C6 not run), object crops, frames.

**Outputs:** `changes/changes_verified.json`.

**Method:**
1. For each non-UNCHANGED change, build an evidence image pair:
   - MOVED / REPLACED: best crop from A and best crop from B,
   - REMOVED: A's best crop + a B frame (from C6's supporting cameras) cropped to the projected
     location of the object,
   - ADDED: the reverse.
2. Ask the VLM (via `changedet/llm.py`) a fixed prompt requesting **JSON only**:
   `{"same_object": bool|null, "change_present": bool, "label": str, "short_description": str, "confidence": 0-1}`.
3. If `change_present` is false with confidence ≥ `vlm.reject_conf`, mark the change `REJECTED`.
   If the VLM label is more specific (e.g. "office chair" vs "chair"), store it as `refined_label`.
4. Cache every VLM response by a hash of (images, prompt) so reruns are free.
5. If no API key is configured, log a warning and pass changes through unchanged.

**Config keys:** `enabled`, `provider`, `model`, `reject_conf`, `max_calls`, `prompt_version`.

**Debug output:** `viz/c7_evidence/<change_id>.jpg` side-by-side evidence with the verdict
written on it (also used in the README).

**Tests:** mock the LLM wrapper; check JSON parsing (including fenced/invalid JSON) and
REJECTED logic.

**Acceptance criteria:** the bag removal on `main` survives; any spurious C5 candidates on `main`
(and on `nochange`, once recorded) are rejected.

**Out of scope:** writing the final report (C8).

---

### C8 — Semantic description & report

**Purpose:** Produce the final structured `ChangeReport` and a readable natural-language summary
grounded in 3D geometry.

**Entry point:** `describe(run: str, force: bool = False) -> ChangeReport`

**Inputs:** latest available change list (verified > visibility > raw), objects, reconstruction.

**Outputs:** `report/changes.json` (ChangeReport), `report/report.md`.

**Method:**
- **Basic (MVP):** template-based sentences per change using labels, positions and distances,
  e.g. `"chair moved 1.4 m and rotated 85°."`. No LLM required.
- **Full (Core):**
  1. Compute spatial relations for each change from the 3D map: nearest unchanged landmark
     objects (with direction: left/right/in front/behind relative to a fixed room frame, or
     "next to"/"on" based on bbox contact), distance to the nearest wall, on-floor vs on-surface.
  2. Turn these into a compact fact list per change (no raw coordinates beyond rounded metres).
  3. Ask the LLM to write one sentence per change **using only those facts**, plus a 2–3 sentence
     overall summary. Include confidence wording ("could not be verified because the area was
     not visible in the second recording").
  4. Fall back to the basic templates if the LLM is unavailable.
- `report.md` includes: summary, a table of changes (type, label, confidence, displacement),
  per-change sentences, and links to evidence images.

**Config keys:** `mode` (template | llm), `landmark_k`, `language_style`.

**Debug output:** the report itself.

**Tests:** template mode on synthetic changes produces expected strings; spatial relation helper
on toy boxes.

**Acceptance criteria:** every reported sentence is consistent with geometry (distances and
directions match the viewer); UNVERIFIED changes are clearly flagged; REJECTED changes are
omitted from the main summary but listed in an appendix.

**Basic (template) as built (2026-10-09):** location phrase from unchanged landmark objects of the
same session (`where()`: "on X" if the footprint is > 50% over X and the object starts above X's
base; else "[on the floor] next to / about d m from X"; "another X" for same-label landmarks).
Verb chosen by confidence ("has been removed" / "may have been removed" / "seemed to have been
removed") and the C6 evidence appended ("Confirmed: 84% of that space was seen empty in the
second recording."). Config adds `on_tol`, `next_to_dist`, `near_dist`.
**Result on `main`:** ✅ "1 change was confirmed: the bag that was on the bed has been removed."
3 unverified listed as "may have been removed … could not be verified", 4 rejected in the
appendix only.

---

### C9 — Navigation impact analysis (stretch)

**Purpose:** Show what the changes mean for a robot moving through the room.

**Entry point:** `navigation_impact(run: str, force: bool = False) -> dict`

**Inputs:** C2 clouds, C8 report (or latest changes).

**Outputs:** `nav/occupancy_{A,B}.npy`, `nav/paths.json`, `nav/nav_diff.png`, adds a
`navigation` section to `report/changes.json` and `report.md`.

**Method:**
1. For each session, take points with `nav.min_z < z < nav.robot_height` (excluding the floor),
   project onto a 2D grid (`nav.cell_size`, default 0.05 m), and inflate obstacles by
   `nav.robot_radius`.
2. Mark cells never observed (no floor seen) as unknown; treat unknown as untraversable by default.
3. Choose start/goal: from config if given; otherwise the two most distant free cells in the
   free space common to both sessions.
4. Run A* in both grids. Report path length change, whether the path is blocked, and which
   change(s) intersect the A-path in B's grid.
5. Add sentences like "The new box blocks the direct route between the door and the desk; the
   detour is 1.4 m longer."

**Config keys:** `cell_size`, `robot_height`, `robot_radius`, `min_z`, `start`, `goal`, `unknown_is_free`.

**Debug output:** `nav/nav_diff.png` — side-by-side top-down grids with both paths and changed
objects highlighted; also logged to the rerun viewer as a floor-plane image.

**Tests:** synthetic grid with an added obstacle → longer path and correct blocking change id.

**Acceptance criteria:** on `main`, correctly reports no navigation impact (the bag lay on the
bed, which is an obstacle in both sessions, so the free space is unchanged). On `main2` (once
recorded), the added floor box visibly changes the path. The synthetic test is the primary proof
until then.

---

### C10 — Interactive visualization

**Purpose:** The main way reviewers will experience the result. Must be clear at a glance.

**Entry point:** `visualize(run: str, open_viewer: bool = True) -> str` (returns `.rrd` path)

**Outputs:** `viz/final.rrd`, `viz/hero.png`, `viz/hero.gif` (for README).

**Basic (MVP):** one rerun recording with:
- background cloud (grey), camera trajectories of A (blue) and B (orange),
- change objects: REMOVED = red points + red box, ADDED = green, MOVED = faded old + solid new
  + arrow, REPLACED = purple, UNVERIFIED = dashed/grey box with label suffix "(unverified)",
- 3D text labels with the change label and short description.

**Full (Core):**
- timeline with two time points so the user can scrub between "before" and "after" scenes,
- an animated sequence interpolating moved objects from old to new pose,
- evidence crops logged as images linked to each change entity,
- navigation grid and paths (if C9 ran),
- a static top-down render `viz/hero.png` and a short orbit `viz/hero.gif` (Open3D offscreen
  render or rerun screenshot workflow) for the README.

**Config keys:** colours, point sizes, `gif_frames`, `top_down_resolution`.

**Tests:** builds an `.rrd` from synthetic data without errors.

**Acceptance criteria:** a person who has never seen the project can open the viewer and
understand what changed within ~10 seconds.

**Basic as built (2026-10-09):** `final.rrd` (2.3 MB on `main`) carries a default blueprint: 3D
view (left, 3/4) + report markdown + "before" crop of each confirmed change (right). Rejected
candidates are not drawn (they are in report.md). Also built from the Full list because the
README needs them: `hero.png` (top-down map + evidence crop + summary, matplotlib) and
`hero.gif` (matplotlib 3D orbit, background cut at `viz.gif_max_z` so the inside is visible).
`visualize(run, open_viewer=True, force=False)` — `force` added for CLI uniformity; the viewer is
spawned only if `$DISPLAY` is set, else the `rerun --web-viewer` command is printed.

---

### C11 — Evaluation against ground truth

**Purpose:** Honest quantitative results for the README.

**Entry point:** `evaluate(run: str, gt_path: str) -> dict`

**Outputs:** `eval/eval.json`, `eval/eval.md` (table).

**Method:**
1. Match predicted changes (excluding REJECTED) to ground-truth changes: same type (REPLACED may
   match REMOVED+ADDED pairs), compatible label (string match or CLIP text similarity ≥ threshold),
   and position within `eval.pos_tol` if GT positions exist.
2. Report precision, recall, F1 overall and per type; number of UNVERIFIED predictions and
   whether they correspond to `expected_unverified`.
3. For moved objects with GT displacement (optional), report translation error.
4. Script `scripts/eval_all.sh` runs all pairs and builds a combined table.

**Tests:** synthetic predictions vs GT with known TP/FP/FN counts.

**Acceptance criteria:** produces a table for every pair that has ground truth (currently only
`main`, plus `nochange` / `main2` / `hard` once recorded) and for the synthetic suite; the README
quotes it and states honestly how many real changes it is based on.

---

### C12 — End-to-end CLI, packaging, README

**Purpose:** Make the project trivially runnable and well presented.

**Deliverables:**
- `python -m changedet.cli all --a before.mp4 --b after.mp4 --run demo` runs C1→C10 (C11 if
  `--gt` given), skipping cached stages; `--from c4` and `--force` flags.
- `scripts/download_models.sh` fetching all checkpoints.
- Dockerfile (CUDA base image) that runs the full pipeline on the sample data.
- Example data in `data/samples/` (or a download script if too large) and **committed example
  outputs** (`report.md`, `changes.json`, `hero.png`, `hero.gif`, a small `.rrd`) under `examples/`.
- **README.md** structure:
  1. One-line description + hero GIF
  2. Quickstart (3 commands)
  3. Example output: report excerpt + evidence images
  4. How it works (pipeline diagram + one paragraph per stage)
  5. Design choices & trade-offs (object-level diffing, visibility reasoning, VLM as verifier
     rather than detector, joint reconstruction)
  6. Evaluation table (from C11) incl. the no-change control
  7. Failure modes & limitations (honest)
  8. What I would do with more time
  9. Recording tips for your own videos

**Acceptance criteria:** a fresh clone + the quickstart commands reproduces the example output.

**As built (2026-10-09):** `all` creates the run if `--a/--b` are given, runs every implemented
stage in order (C7, C9 skipped with a warning; C11 only with `--gt`, not implemented yet),
`--from cX` forces cX and later, `--open` opens the viewer. `export --run R --out DIR` copies
report, hero images, `.rrd` and linked crops (links rewritten) → `examples/demo`. Dockerfile on
`nvidia/cuda:12.8.1-runtime-ubuntu24.04` + uv 0.12.24, checkpoints fetched on first use into a
mounted `checkpoints/` (system libs open3d needs: libgl1, libegl1, libusb-1.0-0, libtbb12). **Verified:** a fresh `all` run takes 181 s natively and 177 s in Docker on the L40S, both reproducing the `demo` report exactly. `data/ground_truth/main.json` added. ⚠️ The recordings are not in git
(640 MB, private room): README has a placeholder for a download link.

---

## 10. Milestones

| Milestone | Components | Definition of done |
|-----------|------------|--------------------|
| **M1: Geometry** | C0–C2 | Both sessions aligned in one metric, z-up frame, visible in rerun |
| **M2: MVP** | C3–C5, C8 basic, C10 basic, C12 | End-to-end run on `main` produces changes, viewer, template report. Tag `v0.1-mvp` |
| **M3: Robustness** | C6, C7, C11 | `main` bag CONFIRMED with no spurious changes; UNVERIFIED proven (synthetic + frame-dropping check); `nochange` ≈ zero changes once recorded; eval table |
| **M4: Polish** | C8 full, C10 full, C9 | LLM report, timeline viewer, navigation demo, README complete |

---

## 11. Coding conventions

- Type hints everywhere; docstrings on public functions (one-line summary + args/returns).
- `ruff` formatting and lint clean.
- No notebooks in the core package (notebooks allowed in `notebooks/` for exploration only).
- Heavy models are loaded lazily and once per process (`functools.lru_cache` loader functions).
- GPU memory: wrap model inference in `torch.inference_mode()`; free models after each stage
  when running `all`.
- Deterministic where possible: fixed seeds for RANSAC/DBSCAN/ICP.
- Never silently swallow errors; if a stage degrades (e.g. no VLM key), log a clear warning and
  record it in `ChangeReport.stats["warnings"]`.

---

## 12. Decisions log

Append one entry per finished component: date, component, decision, reason.

| Date | Component | Decision | Reason |
|------|-----------|----------|--------|
| 2026-10-09 | Spec | Record3D (LiDAR depth + ARKit poses) is the primary input; MASt3R/VGGT demoted to a post-MVP path for plain `.mp4` | Our recordings are Record3D exports; metric depth and poses for free removes the riskiest stage |
| 2026-10-09 | Spec | C2 aligns sessions with a gravity-constrained (yaw + translation) registration | Each Record3D session has its own ARKit origin, but both are gravity-aligned, so only 4-DoF remain — more robust than 6-DoF |
| 2026-10-09 | Spec | Added `Frame.source_index`; `inputs.json` records `kind` per session | C2 needs to look up depth/pose for each kept frame; input kind drives C1/C2 behaviour |
| 2026-10-09 | Spec | Data plan reduced to one real pair (`main`: bag removed from bed); other change types covered by synthetic tests; `nochange` recording requested | Only two recordings exist so far |
| 2026-10-09 | C0 | Paths stored in types are **relative to the run dir**; resolve with `cache.run_path(run, rel)` | Runs stay relocatable, so example outputs can be committed under `examples/` (C12) |
| 2026-10-09 | C0 | Types are dataclasses with a generic, type-hint-driven JSON codec (`to_dict`/`from_dict`/`to_json`/`from_json`); fields filled by later stages (`Change.visibility`, `vlm_verdict`, `description`, `CameraFrame.confidence_path`) default to `None` | One codec for all types, no per-class boilerplate; no pydantic dependency |
| 2026-10-09 | C0 | Heavy ML deps (torch, SAM 2, CLIP, …) are **not** in `pyproject.toml` yet; each component adds what it needs | Keeps C0–C2 installable in seconds; avoids pinning CUDA wheels before they're used |
| 2026-10-09 | C0 | Python pinned to 3.12 (`requires-python >=3.10,<3.13`) | Newest version with open3d wheels |
| 2026-10-09 | C0 | `--set` overrides are parsed as YAML and must name an existing key | Typos fail loudly instead of being silently ignored |
| 2026-10-09 | C0 | `cached_stage` skips only when **all** listed outputs exist; JSON/PLY writes are atomic (tmp + rename) | A crashed stage never looks finished; stages must write their listed outputs last |
| 2026-10-09 | C0 | Runs dir can be redirected with `$CHANGEDET_RUNS_DIR` | Tests use a temp dir; never touch the real `runs/` |
| 2026-10-09 | C0 | Stage stubs in the CLI exit with code 1 ("not implemented") | `all` must not report success while stages are missing |
| 2026-10-09 | C0 | ruff excludes `*.md` | ruff ≥0.16 reformats Python blocks inside Markdown, which would rewrite this spec |
| 2026-10-09 | C1 | Sharpest frame per `1/fps` window instead of fixed-interval sampling | All ~4000 frames can be scored in ~1–2 s; raised the 10th-percentile sharpness of picked frames by ~15–20% on `main` |
| 2026-10-09 | C1 | Blur filter is **relative to temporal neighbours** (`blur_ratio`, `blur_window`), replacing `blur_percentile` | On `main`, the 20th-percentile rule rejected ~40 frames per session that were sharp but low-texture (dark jacket on chair, desk against plain wall), removing most desk-area coverage. The relative rule drops only isolated dips (1 and 8 frames) |
| 2026-10-09 | C1 | `frames.json` is keyed by session; frame files named by kept index | Matches the entry function's return type; `Detection2D.frame_index` refers to the same index |
| 2026-10-09 | C1 | Result on `main`: A 4385 → 220 windows → 219 → 104 → **100**, B 3992 → 200 → 192 → 108 → **100** (windows → blur → motion → cap) | You move slowly (median ~2–3° per 1/3 s), so the motion filter does most of the thinning |
| 2026-10-09 | C2 | Pose convention verified empirically: `[qx,qy,qz,qw]`, camera-to-world, OpenGL axes → OpenCV via `diag(1,-1,-1,1)`. Kept as a runtime check (`reproj_tol`) | Median reprojection 6–9 mm vs 7–18 cm for the other 3 order/flip combinations |
| 2026-10-09 | C2 | Coarse registration is an exhaustive yaw sweep with FFT cross-correlation of top-down occupancy grids (not 2D ICP / FPFH) | No initial guess, no local minima in translation, ~1 s; top-3 yaws are refined so mirror-symmetric rooms are handled |
| 2026-10-09 | C2 | Own 4-DoF ICP (`geometry.icp_yaw`, Huber point-to-plane) instead of Open3D 6-DoF + projection | Gravity is shared, so only yaw + xyz are unknown; fewer DoF = more robust; reusable for C5 moved-object transforms |
| 2026-10-09 | C2 | Clouds are fused from native 192×256 depth (intrinsics scaled with pixel-centre convention); only the saved `.npy` maps are upsampled (index-based nearest) | Upsampled depth adds no information but ~14× the points |
| 2026-10-09 | C2 | Flying-pixel filter is relative (`edge_rel_thresh` × depth) | A fixed metric jump would cut slanted floor far away and miss near edges |
| 2026-10-09 | C2 | Keep `static_thresh` = 0.05 m although the background is impure at the bag | Tightening to 0.03 / 0.025 m cut the bag's leakage only modestly but lost 6–10% of true static points; B's duvet is raised right where the bag was. Background is for the viewer, not for detection |
| 2026-10-09 | C3 | Grounding DINO + SAM 2 via Hugging Face `transformers` (torch 2.14 cu130, transformers 5.19) | One pip dependency for both models, no CUDA extension builds; checkpoints cached in `checkpoints/hf` |
| 2026-10-09 | C3 | Label = best single vocabulary phrase per box (own token-span scoring), not the library's decoded phrase | Library output merged neighbouring phrases ("monitor tv", "chair office chair") |
| 2026-10-09 | C3 | Structural classes are prompted, then dropped | Lets windows/doors absorb boxes that would otherwise be mislabelled as objects |
| 2026-10-09 | C3 | Masks made exclusive (smallest wins) | SAM 2's bed mask covered the bag; C4 needs the bag as its own segment |
| 2026-10-09 | C3 | Removed "office chair" from the vocabulary | Near-synonym of "chair" causes label flips between sessions; NMS + C4 voting handle the rest |
| 2026-10-09 | C0/C3 | `load_run_config` = run snapshot deep-merged **on top of current defaults** | Runs created before a component existed lacked its config section and crashed; the run's own values still win (re-`init` to adopt new defaults) |
| 2026-10-09 | C4 | See-through filter: drop masked depth deeper than the ring-around-the-mask **median** + 0.25 m | Removes the TV's mirror-image "room behind the wall"; p90 of the ring let it through when the bright window was beside the TV. A room-wall filter was tried and rejected: the mirror room puts ~16% of points behind the real wall, so a safe wall test could not accept it |
| 2026-10-09 | C4 | Overlap = fraction of a segment's voxels touching an object's voxels **dilated by one voxel** (int64 voxel keys + `np.isin`) | 1–2 cm noise between views otherwise splits objects; fast enough for ~800 segments per session |
| 2026-10-09 | C4 | Containment absorb ignores labels but requires box-inside + `contain_frac` 0.6 | Fixes detections of things *in* a picture; box-inside condition stopped a bottle beside the TV being absorbed |
| 2026-10-09 | C4 | CLIP ViT-L/14 + DINOv2-base via transformers; embeddings of grey-background masked crops; best-view crop saved unmasked with 15% context | Masked crops for identity, context crops for the VLM/report |
| 2026-10-09 | C5 | Added optional `Change.match_margin` (contract addition, default `None`) | Spec step 6 asks for the ambiguity margin to be stored in the change; no existing field fit |
| 2026-10-09 | C5 | Cost weights calibrated on `main` (`w_sem` 3, `w_app` 1, `w_geo` 0.5, `w_size` 0.1, `w_label` 0.3, `max_cost` 1.0) | CLIP has a high cosine baseline between different objects; DINOv2 is the discriminative term; volumes and labels are noisy across sessions |
| 2026-10-09 | C5 | ICP prefers "stayed put" unless a move fits ≥ 0.1 inlier fraction better; smaller cloud is the ICP source | Partial views and symmetric objects otherwise produce spurious small moves/rotations |
| 2026-10-09 | C5 | "Same space + similar appearance" rule for unmatched objects | C4 fuses at different granularity per session (2 bottles vs 1); without it these become spurious REMOVED. DINOv2 gate (≥ 0.5) keeps true replacements |
| 2026-10-09 | C5 | Did **not** add a cloud-occupancy test to C5 to hit the "≤ 1 spurious" target | That is exactly C6's job (free/occupied/unknown along camera rays); duplicating a cruder version here would blur the stage boundary |
| 2026-10-09 | C0 | Voxel-key helpers (`voxel_keys`, `dilate_keys`, `overlap_fraction`, `box_iou`) moved from C4 to `core/geometry.py` | Shared by C4 and C5 |
| 2026-10-09 | C6 | Built before C8/C10 (deviation from the recommended order) | Without it the MVP report/viewer would show 8 changes for 1 real one |
| 2026-10-09 | C6 | Ignore depth under C3 "tv"/"monitor" masks of the other session | Mirror-image depth through the TV screen counted as "seeing past" a reflection fragment → false CONFIRMED |
| 2026-10-09 | C6 | Per-point votes with `vote_ratio` 3 instead of "no view may disagree" | Same verdicts on `main`, but robust to a single noisy depth vote at object edges |
| 2026-10-09 | C6 | Fractions over all tested points; REJECTED checked before CONFIRMED | Matches the spec; points come from surfaces session S observed, so they are mostly visible from S' too |
| 2026-10-09 | C8 | Template sentences pick the verb by confidence and quote the C6 evidence | "The jacket … has been removed. Rejected: …" read as a contradiction; numbers make each sentence checkable |
| 2026-10-09 | C8 | "on X" test uses footprint overlap + "starts above X's base", not "bottom ≈ X's box top" | The bed's box top is set by pillows/headboard, so the bag on the mattress was "next to the bed" |
| 2026-10-09 | C10 | Rejected candidates hidden in the viewer; unverified drawn grey with "(unverified)" | 10-second readability; the report keeps the full list |
| 2026-10-09 | C10 | hero.gif rendered with matplotlib (grey background, z < 1.6 m, elevated orbit) | No display/offscreen GL needed; the first full-colour version hid the bag |
| 2026-10-09 | C12 | `all` skips unimplemented stages instead of failing; `export` subcommand for committed examples | MVP must run end-to-end while C7/C9/C11 are pending |

---

## 13. Open questions (resolve as we go)

- MASt3R vs VGGT — only relevant if/when plain `.mp4` input is supported (post-MVP).
- Coarse yaw search vs constrained FPFH + RANSAC for C2 registration — decide on `main` data.
- ~~Whether 192×256 LiDAR depth can separate the bag from the bed~~ — yes (C4, exclusive C3 masks).
- Glossy screens: LiDAR returns a mirror image of the room behind the TV. C4 filters it per mask;
  the C2 clouds (viewer, C6 visibility) still contain it. Option: zero depth under C3 "tv/monitor"
  masks before C6, or fit room walls robustly to this case.
- Vocabulary size vs false detections in C3.
- Whether to expose a small "ask about changes" query (CLIP text → object) in the viewer — only
  after M4 if time allows.
