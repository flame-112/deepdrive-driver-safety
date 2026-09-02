# DeepDrive: AI-Powered Driver Safety Assistance System

DeepDrive is a student-built computer-vision prototype for real-time driver-risk monitoring. It will identify observable safety indicators—such as prolonged eye closure, yawning, looking away, phone presence, and seatbelt compliance—and turn them into configurable alerts and a session-level safety summary.

It is an assistance and monitoring prototype. It does not diagnose internal mental states or claim to prevent accidents.

## Current milestone

Live driver-face monitoring currently includes:

- webcam capture with OpenCV;
- MediaPipe Face Mesh landmarks (eyes in green, mouth in orange);
- left, right, and averaged Eye Aspect Ratio (EAR);
- blink counting vs prolonged eye closure from EAR over time;
- a drowsiness *estimate* (LOW / MODERATE / HIGH) from prolonged closure and a simplified PERCLOS;
- Mouth Aspect Ratio (MAR) and yawn detection (long mouth opening).

A blink is a short closure that ends when the eyes open again. Prolonged closure is shown if the eyes stay closed for about one second. The drowsiness line is an estimate from those observables, not a medical diagnosis. A **yawn** is a mouth opening that lasts about **1.5 s**; talking or a quick smile should not count. Head-pose, phone, and seatbelt detection are not implemented yet.

The OpenCV window is a development viewer for checking detectors. It is not the final product interface. A Streamlit dashboard (camera preview, scores, session report) is planned after the core modules work.

## Setup (Windows PowerShell)

Python 3.11 and a project-local virtual environment have been prepared. Activate it from the project folder:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If PowerShell prevents activation, run the interpreter directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run the webcam landmark viewer

```powershell
.\.venv\Scripts\python.exe -m src.face.landmark_viewer
```

Press `q` while the video window is focused to close it. For an external camera, try `--camera-index 1`.

The window shows EAR, MAR, eye state, blink count, drowsiness estimate, and **Mouth: CLOSED / OPEN / YAWNING**. Hold a real yawn (mouth open ~1.5 s) to increment the yawn count. Talking briefly should stay OPEN or CLOSED, not YAWNING.

Starting thresholds (tune on your demo camera):

- closed EAR below **0.21**, open again above **0.24** (hysteresis);
- blink if a closure lasts **0.08–0.45 s**;
- prolonged closure after **1.0 s** still shut;
- mouth opening starts above MAR **0.60**, ends below **0.45**;
- yawn if that opening lasts **1.5 s**.

Phone-as-webcam example if natural blinks are missed or false:

```powershell
.\.venv\Scripts\python.exe -m src.face.landmark_viewer --camera-index 1 --closed-ear 0.20 --open-ear 0.23
```

If yawns never trigger, try `--open-mar 0.55 --yawn-seconds 1.2`.

## Run tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

## Project layout

```text
src/face/              Landmarks, EAR, live viewer
src/drowsiness/        Blink vs prolonged closure, drowsiness estimate
src/yawning/           MAR and yawn tracker
tests/                 Automated checks
docs/                  Design notes and project documentation
```

Later modules: head pose, object detection, risk assessment, alerts, logging, reporting, and dashboard.
