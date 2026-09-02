# Agent instructions — DeepDrive

Persistent rules for coding agents working in this repository. For current milestone status and remaining work, read `PROJECT_STATUS.md` and `README.md`. Do not copy transient chat details into this file.

After finishing a phase (code + tests), **update `PROJECT_STATUS.md`** to match the tree and **update this file only where standing rules changed**. The student switches agents when the session cap hits; a stale status file is a bug.

## Product stance

DeepDrive is a student **driver-risk monitoring / assistance prototype**. Prefer techniques that match the subproblem (geometry, temporal rules, then neural nets). Not every module needs a trained model.

Always separate **observable visual behavior** from **internal states**:

- Allowed labels: prolonged eye closure, blink, yawn, looking away, phone present, seatbelt missing.
- Forbidden claims: diagnosing fatigue, knowing the driver is tired/stressed/angry, preventing accidents, 100% accuracy, or novelty.

**Do not implement emotion detection** (stress, anger, mood).

## Architecture

Build **one phase at a time**. Do not add YOLO, risk scoring, alerts, logging, reports, or Streamlit until the current detector is tested.

Intended pipeline:

```text
Camera → OpenCV frames → per-domain modules → risk engine → alerts → score / report
```

Package layout (add folders only when that phase starts):

```text
src/face/          landmarks, EAR, live OpenCV viewer
src/drowsiness/    EyeClosureTracker + DrowsinessEstimator (PERCLOS; not a diagnosis)
src/yawning/       MAR + YawnTracker (long mouth opening; not a fatigue diagnosis)
src/head_pose/
src/phone/
src/seatbelt/
src/risk/
src/alerts/
app/               dashboard (later)
tests/
```

Conventions:

- Keep **I/O and drawing** in the viewer (or later app). Keep **geometry and classifiers** in importable modules with no `cv2.imshow`.
- Put numeric thresholds in **config dataclasses**, not buried in drawing code. CLI flags may override config; they must not become the only source of truth.
- Prefer **wall-clock seconds** (`time.perf_counter()` or caller-supplied timestamps) over frame counts. Camera FPS is unstable (phone-as-webcam).
- Use **hysteresis** for noisy ratios (EAR: low = closed; MAR: high = open).
- **Missing face** must have an explicit policy (reset or pause in-progress events). Document and test it. For PERCLOS, **exclude no-face time** from both numerator and denominator (do not treat dropouts as open or closed).
- MediaPipe Face Mesh: **subject left/right** (the person’s own left eye), not the mirrored image’s left. EAR and MAR point order is Soukupová & Čech p1..p6. Use `(x, y)` only unless a later phase needs `z`. Shared formula lives in `src/face/aspect_ratio.py`.
- One face for the live prototype (`max_num_faces=1`) unless the user asks to change that.
- Windows capture: OpenCV `VideoCapture(index, cv2.CAP_DSHOW)`.
- Default **mirror** the preview; `--no-mirror` exists for debugging.
- No database or HTTP API unless the user asks. Future logs should start as local files (e.g. under `results/`), not a server.

Risk weights (when that phase exists) stay **configurable and experimental**. Do not present them as medically or statistically validated.

## Coding style

- Python 3.11, project `.venv` only (`.\.venv\Scripts\python.exe`).
- Modular packages under `src/`. Run modules as `python -m src.face.landmark_viewer` from the repo root.
- Type hints on public functions and dataclasses. Docstrings on important classes and functions, in language a viva can repeat.
- Meaningful names. Configuration types are frozen dataclasses with validation in `__post_init__` when invariants exist.
- `from __future__ import annotations` where the rest of the package already uses it.
- Handle camera and frame failures with clear `RuntimeError` messages; do not swallow errors silently.
- No giant monolithic scripts, no duplicated EAR/tracker logic in the viewer, no hard-coded dataset paths.
- Do not over-engineer. The team must understand every important line.

Example: geometry stays pure; the viewer only calls it.

```python
# Good: src/face/aspect_ratio.py
def six_point_aspect_ratio(points: Sequence[Point]) -> float: ...

# Bad: computing EAR/MAR inline inside the OpenCV loop with magic index numbers
```

## Framework and dependency conventions

Current stack: OpenCV, MediaPipe Face Mesh, NumPy **1.26.x**, stdlib `unittest`.

Pinned `requirements.txt` exists so MediaPipe/JAX/OpenCV do not drift to **NumPy 2**. Do not unpin or “upgrade everything” without checking Windows + Python 3.11 + MediaPipe compatibility.

Before adding PyTorch, Ultralytics, Streamlit, or other heavy deps: explain why, check compatibility, then add **pinned** versions.

Local GPU is a **GTX 1650 (4 GB)**. Prefer lightweight models (YOLO nano/small), modest resolution and batch size. Use Colab/Kaggle for large training jobs.

Do not assume CUDA is required for the live landmark/EAR path (CPU MediaPipe is acceptable).

## UI and design

The OpenCV window is a **lab/debug overlay**, not the final product UI. A polished dashboard (Streamlit) comes after core detectors work. Do not restyle or replace the viewer with a web app in the same change as a detector unless asked.

While the OpenCV viewer is the demo surface:

- Keep status text readable (EAR, MAR, eye state, blink count, drowsiness estimate, mouth/yawn). Use distinct BGR colours: prolonged/HIGH/YAWNING red, closed/MODERATE/mouth-OPEN amber, open/LOW/mouth-CLOSED green, EAR/MAR cyan, WARMUP grey, eyes green, mouth orange — stay consistent with the existing viewer.
- Quit with `q` while the video window is focused.
- Do not add audio alerts until the alerts phase.
- Overlay copy must stay observational (`PROLONGED CLOSURE`, `Drowsy est: LOW/MODERATE/HIGH`, `YAWNING`), never “YOU ARE DROWSY” or “YOU ARE EXHAUSTED” as a medical statement.

When Streamlit exists: same product language, configurable thresholds, show reasons next to any risk level.

## Testing

Run from the repository root, using the project interpreter:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Rules:

- Add focused tests next to the new logic (`tests/test_*.py`).
- Detector timing and geometry tests must be **deterministic** (synthetic EAR/MAR sequences, fake landmark objects). Do not require a webcam in CI/unittest.
- Cover hysteresis, duration gates, missing-face behaviour, and invalid config.
- Do not invent accuracy, mAP, or FPS numbers. Report only measurements the team actually ran.

Live check after viewer changes (agent cannot replace this): student runs `python -m src.face.landmark_viewer` (often `--camera-index 1` for phone-as-webcam).

## Collaboration and git

- Do not `git commit` or `git push` unless the user explicitly asks.
- Do not change git config, force-push, or skip hooks.
- Do not commit `.venv`, datasets, recordings, or secrets (see `.gitignore`).

## Must not change without asking

- Product scope: adding emotion detection, claiming diagnosis/accident prevention, or skipping the incremental roadmap.
- `requirements.txt` pins and major new dependencies.
- Default EAR / eye-closure thresholds, blink duration windows, drowsiness PERCLOS cutoffs, and yawn MAR/duration gates (CLI overrides for EAR/closure/MAR are fine; changing defaults needs a reason and user OK).
- MediaPipe left/right eye index mapping, EAR p1..p6 order, and MAR p1..p6 order (`MOUTH_MAR_INDICES`).
- Replacing EAR/MAR with a CNN, or replacing Face Mesh, without an explicit decision.
- Introducing a database, cloud backend, or auth.
- Rewriting the OpenCV viewer into Streamlit (or vice versa) as a surprise.
- Training on or downloading large datasets without license/size discussion.
- Repo visibility, collaborators, and GitHub settings (user does this unless they give a username and ask).
- Unrelated files outside the current phase.

## How to work with the students

Explain non-obvious choices before implementing them. If several approaches are valid, compare briefly and recommend one. Challenge weak designs. Keep the work feasible for a two-person semester project. Prefer official docs when APIs and versions matter.
