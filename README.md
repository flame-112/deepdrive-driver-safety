# DeepDrive: AI-Powered Driver Safety Assistance System

DeepDrive is a student-built computer-vision prototype for real-time driver-risk monitoring. It will identify observable safety indicators—such as prolonged eye closure, yawning, looking away, phone presence, and seatbelt compliance—and turn them into configurable alerts and a session-level safety summary.

It is an assistance and monitoring prototype. It does not diagnose internal mental states or claim to prevent accidents.

## First milestone

The current milestone is deliberately limited to a reliable live video foundation:

- open a webcam with OpenCV;
- detect a face and obtain MediaPipe Face Mesh landmarks;
- highlight eye and mouth landmarks in real time.

No drowsiness, yawn, distraction, phone, or seatbelt decision is made yet.

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

## Project layout

```text
src/face/              Face-landmark module (current milestone)
tests/                 Automated checks
docs/                  Design notes and project documentation
```

Later modules will be added only after this milestone is tested: drowsiness, yawning, head pose, object detection, risk assessment, alerts, logging, reporting, and dashboard.
