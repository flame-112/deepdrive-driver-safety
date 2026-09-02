# DeepDrive — Project Status Handoff

**Audience:** another AI coding agent continuing this work.  
**Date of handoff:** 23 August 2026  
**Do not treat this file as application code.** It is a snapshot of product intent, code state, and next work.

---

## 1. What the product is

**Product name:** DeepDrive  
**Formal title:** AI-Powered Driver Safety Assistance System Using Computer Vision and Deep Learning for Real-Time Driver Monitoring

A **semester-long BTech CSE/AIML** project by a **two-member student team**. It is meant to be a **working prototype** (live camera, real detections, alerts, session summary), not a Colab notebook that only trains a model.

**Purpose:** Monitor a driver from an RGB webcam (or phone-as-webcam), detect **observable** unsafe visual behaviors, raise configurable warnings, and produce a session-level safety summary.

**What it is not:** A medical device, a fatigue diagnosis, an accident-prevention guarantee, or an emotion/stress/anger detector. Faculty-facing language must stay “driver-risk monitoring / assistance prototype.”

---

## 2. Problem and objective (as committed with the team)

**Problem:** Fatigue, drowsiness, distraction, phone use, and missing seatbelts contribute to crashes; conventional systems do not continuously watch the driver from a cheap camera.

**Objective:** Real-time CV/DL monitoring that flags those **observable** behaviors, alerts in time, and produces an overall driver-safety assessment.

**Explicitly out of scope:** Emotion detection (stress/anger/etc.). Removed because it is unreliable from a normal webcam and is not needed for the safety story.

---

## 3. Team, hardware, and environment

| Item | Value |
|------|--------|
| Local folder | `D:\Deep learning Project` |
| Intended GitHub remote | https://github.com/flame-112/deepdrive-driver-safety |
| Owner GitHub | `flame-112` |
| Python | 3.11.9 (use **project `.venv` only**) |
| OS | Windows 10/11, PowerShell |
| Laptop | Intel i5-9300H, 16 GB RAM, **NVIDIA GTX 1650 4 GB**, SSD |
| Demo camera | Laptop webcam is poor. Primary demo path: **phone camera via Microsoft Phone Link (wireless)**. `--camera-index 1` is typical. |

**Git note:** This workspace still has **no `.git` directory** in the file listing. Reconcile git before commit/push. **Never push unless the user explicitly asks.** They previously said **do not push yet**.

Teammate GitHub collaborator: user was told to add them **manually** in GitHub Settings → Collaborators (Write). Do not spend tokens on `gh` unless they give a username **and** ask you to run it.

---

## 4. Tech stack

### In use now

- Python 3.11
- OpenCV (`opencv-contrib-python==4.10.0.84`) — capture, overlay, `CAP_DSHOW` on Windows
- MediaPipe Face Mesh (`mediapipe==0.10.21`) — 468 landmarks, `refine_landmarks=True`, `max_num_faces=1`
- NumPy 1.26.4 (pinned)
- unittest (stdlib)

### Planned later (not installed for those features yet)

- PyTorch (when a trained model is actually needed)
- Ultralytics YOLO (phone + seatbelt)
- Pandas, Matplotlib (logs, plots, evaluation)
- Streamlit (dashboard — **Phase 16**, not the current OpenCV window)
- Git/GitHub
- Colab/Kaggle GPU for heavy training; keep local models light for 4 GB VRAM

**Dependency rule:** Do not blindly bump versions. The pin set in `requirements.txt` exists to stop MediaPipe/JAX/OpenCV from pulling **NumPy 2.x**. Explain before adding packages.

---

## 5. Current implementation status (honest)

**Code for phases 1–7 is in the tree.** Phases 1–6 were student-tested live. **Phase 7 (yawning) was implemented in this session; live Phone Link confirmation is still on the student.**

There is **no database, no HTTP API, no Streamlit app, no YOLO, no audio alerts, no session report, no head-pose detector.** Yawn is **not** yet fed into the drowsiness estimator (that is later integration).

Runnable demo:

```powershell
.\.venv\Scripts\Activate.ps1
.\.venv\Scripts\python.exe -m src.face.landmark_viewer
# phone camera:
.\.venv\Scripts\python.exe -m src.face.landmark_viewer --camera-index 1
```

Quit: focus the OpenCV window and press `q`.

Tests:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Last known test result: **45 tests passed** after Phase 7.

---

## 6. Features completed (in code)

1. **Project setup** — README, `requirements.txt`, `.gitignore`, `src/`, `tests/`, venv.
2. **Webcam capture** — OpenCV `VideoCapture`, optional mirror (default on), `--camera-index`.
3. **Face + landmarks** — MediaPipe Face Mesh; full mesh in grey; **eyes green**, **mouth orange**.
4. **EAR** — left, right, average overlay (`src/face/ear.py` via shared `aspect_ratio.py`).
5. **Blink vs prolonged closure** — `EyeClosureTracker`; **OPEN / CLOSED / PROLONGED CLOSURE / NO FACE**; blink count; CLI overrides.
6. **Drowsiness estimate** — `DrowsinessEstimator`; **WARMUP / LOW / MODERATE / HIGH** + PERCLOS. Student confirmed live. **Not a diagnosis.**
7. **Yawning** — `YawnTracker` + MAR; **CLOSED / OPEN / YAWNING**; yawn counted once after **1.5 s** of open mouth. Overlay + CLI `--open-mar`, `--closed-mar`, `--yawn-seconds`. **Awaiting student live test.**

**UI:** OpenCV is a **lab/debug viewer**, not the faculty-facing final UI. Streamlit is Phase 16.

Student confirmations through Phase 6: proper blinks count over Phone Link; half-blinks should not; prolonged closure works; drowsiness estimate works.

---

## 7. Features partially implemented

- Mouth contour landmarks plus dedicated MAR points (`MOUTH_MAR_INDICES`).
- Yawn does **not** raise the drowsiness estimate yet (keep separate until integration).
- README mentions `docs/`; folder may be empty. Standing rules: `AGENTS.md`.
- GitHub collaborator: process explained, not verified here.
- Blink min/max durations not CLI-exposed. PERCLOS cutoffs not CLI-exposed.

---

## 8. Features remaining

Do **not** jump ahead.

| Phase | Feature | Status |
|-------|---------|--------|
| 7 | Yawning | **In code; needs live confirm** |
| 8 | Head pose / looking away | **Next after yawn is live-OK** |
| 9 | YOLO phone detection | Remaining |
| 10 | YOLO seatbelt detection | Remaining |
| 11 | Integrate detectors | Remaining |
| 12 | Configurable risk engine (LOW / MODERATE / HIGH + reasons) | Remaining |
| 13 | Audio/visual alerts | Remaining |
| 14 | Safety score + event logging | Remaining |
| 15 | Trip/session report | Remaining |
| 16 | Streamlit dashboard | Remaining |
| 17 | Evaluation + docs | Remaining |

---

## 9. Architecture (as built)

```text
landmark_viewer.py
    ├─ MediaPipe FaceMesh
    ├─ face/aspect_ratio.py      # shared six-point ratio
    ├─ face/config.py            # EAR + MAR index tuples
    ├─ face/ear.py
    ├─ drowsiness/eye_closure.py
    ├─ drowsiness/estimator.py   # PERCLOS; eye-only for now
    ├─ yawning/mar.py
    └─ yawning/yawn.py
```

- I/O and drawing stay in the viewer. Geometry/trackers have no `cv2.imshow`.
- Tests do not need a camera.

Do not create until needed: `src/head_pose/`, `src/phone/`, `src/seatbelt/`, `src/risk/`, `src/alerts/`, `app/`.

---

## 10. Important design decisions

1. Observable labels vs internal states (including yawns).
2. Incremental phases. No YOLO/Streamlit in a detector change.
3. Shared six-point ratio for EAR and MAR:  
   `(||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)` on `(x, y)` only.
4. MAR indices `MOUTH_MAR_INDICES = (61, 13, 0, 291, 17, 14)`  
   p1,p4 corners; p2,p6 inner lips 13–14; p3,p5 outer 0–17.
5. Yawn hysteresis is **inverted vs eyes**: open if MAR **> 0.60**, closed again if MAR **< 0.45**. Count **once** when open duration ≥ **1.5 s**. Talking/smiling can raise MAR; duration is the discriminator.
6. Missing face resets in-progress yawn **without** incrementing the count.
7. Do not feed yawn into PERCLOS/drowsiness until the user asks or Phase 11.
8. Time in seconds, not frames. Phone Link FPS is unstable.
9. Pinned NumPy 1.26.4 / MediaPipe 0.10.21.
10. Token hygiene + **keep PROJECT_STATUS.md in sync** after every phase (user switches agents at cap).
11. OpenCV ≠ final UI.

Eye-closure / PERCLOS defaults unchanged (see section 17).

---

## 11. Known bugs and current issues

**Not bugs (confirmed):** half-blinks; wireless can miss tiny blinks; full blinks count; estimate staying MODERATE ~10 s after a prolonged close.

**Caveats:**

- **Phase 7 not live-confirmed yet.** MAR defaults (0.60 / 0.45 / 1.5 s) may need `--open-mar` on Phone Link.
- Continuous talking might hold MAR in the hysteresis gap and look like OPEN; it should not become YAWNING unless it lasts 1.5 s.
- No `.git` in this workspace listing.
- Viewer raises if camera/frame fails. Single face only. No invented metrics.

---

## 12. Files that matter

| Path | Role |
|------|------|
| `README.md` | Student-facing setup |
| `AGENTS.md` | Persistent agent rules (update when rules change) |
| `PROJECT_STATUS.md` | This handoff |
| `src/face/aspect_ratio.py` | Shared six-point formula |
| `src/face/config.py` | Indices including `MOUTH_MAR_INDICES` |
| `src/face/ear.py` | EAR wrappers |
| `src/face/landmark_viewer.py` | Only runtime entrypoint |
| `src/drowsiness/*` | Closure + PERCLOS estimate |
| `src/yawning/config.py` | `YawnConfig` |
| `src/yawning/mar.py` | MAR |
| `src/yawning/yawn.py` | `YawnTracker` |
| `tests/test_mar.py` | MAR geometry |
| `tests/test_yawn.py` | Hysteresis, duration, no-face reset |

CLI: `--camera-index`, `--no-mirror`, `--closed-ear`, `--open-ear`, `--prolonged-seconds`, `--open-mar`, `--closed-mar`, `--yawn-seconds`.

---

## 13. Database / API

**None.** Do not add a database for head pose.

---

## 14. How to work with this user

- Explain before large implementation. Challenge weak ideas.
- **Do not implement emotion detection.**
- **Do not commit or push** unless asked.
- After a phase: tests + update `PROJECT_STATUS.md` (and `AGENTS.md` if rules changed) + wait for live confirm / next go.
- User asked to **always keep status + agents files current** so another agent can continue at the usage cap.

---

## 15. What happened immediately before this handoff

1. Phases 1–6 done and live-tested (EAR, blinks, prolonged closure, drowsiness estimate).
2. User asked to refresh status/agents (Phase 6 had been missing from an old handoff).
3. User: proceed, and **keep updating status + agents alongside**.
4. **This session implemented Phase 7 yawning** and refreshed these docs.

---

## 16. Exact next steps for the next agent

1. If tests were not run or failed, run `.\.venv\Scripts\python.exe -m unittest discover -s tests -v` and fix.
2. If the student has **not** live-tested yawns: tell them how (open mouth 1.5+ s; talking should not count). Wait. If MAR never rises, help tune `--open-mar`.
3. If they confirmed yawns: **Phase 8 head pose / looking away** only. SolvePnP or MediaPipe pose from face landmarks; yaw/pitch thresholds; overlay LOOKING AWAY after a duration. Unittests with synthetic angles. No YOLO.
4. Do not start Streamlit, phone YOLO, or audio.
5. Update `PROJECT_STATUS.md` again when Phase 8 lands.

---

## 17. Default numbers

**EyeClosureConfig:** closed 0.21, open 0.24, blink 0.08–0.45 s, prolonged 1.0 s  

**DrowsinessConfig:** window 30 s, min observation 5 s, PERCLOS moderate 0.08 / high 0.20, recent prolonged 10 s  

**YawnConfig:** open MAR 0.60, closed MAR 0.45, min yawn 1.5 s  

```powershell
.\.venv\Scripts\python.exe -m src.face.landmark_viewer --camera-index 1 --closed-ear 0.20 --open-ear 0.23
.\.venv\Scripts\python.exe -m src.face.landmark_viewer --camera-index 1 --open-mar 0.55 --yawn-seconds 1.2
```
