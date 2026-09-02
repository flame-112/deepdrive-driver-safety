"""Display webcam frames with MediaPipe facial landmarks.

Run from the repository root:
    .\\.venv\\Scripts\\python.exe -m src.face.landmark_viewer
"""

from __future__ import annotations

import argparse
import time

import cv2
import mediapipe as mp

from src.drowsiness.config import (
    DEFAULT_DROWSINESS_CONFIG,
    DEFAULT_EYE_CLOSURE_CONFIG,
    DrowsinessConfig,
    EyeClosureConfig,
)
from src.drowsiness.estimator import (
    LEVEL_HIGH,
    LEVEL_LOW,
    LEVEL_MODERATE,
    DrowsinessEstimate,
    DrowsinessEstimator,
)
from src.drowsiness.eye_closure import EyeClosureSnapshot, EyeClosureTracker, LABEL_NO_FACE
from src.face.config import (
    LEFT_EYE_EAR_INDICES,
    LEFT_EYE_INDICES,
    MOUTH_INDICES,
    MOUTH_MAR_INDICES,
    RIGHT_EYE_EAR_INDICES,
    RIGHT_EYE_INDICES,
)
from src.face.ear import ear_from_landmarks
from src.yawning.config import DEFAULT_YAWN_CONFIG, YawnConfig
from src.yawning.mar import mar_from_landmarks
from src.yawning.yawn import YawnSnapshot, YawnTracker

WINDOW_NAME = "DeepDrive | eyes + yawn estimate (press q to quit)"


def parse_arguments() -> argparse.Namespace:
    """Parse camera options without mixing configuration into detection logic."""
    parser = argparse.ArgumentParser(description="Visualize live facial landmarks.")
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="OpenCV camera index to use (default: 0).",
    )
    parser.add_argument(
        "--no-mirror",
        action="store_true",
        help="Keep the camera image unmirrored.",
    )
    parser.add_argument(
        "--closed-ear",
        type=float,
        default=DEFAULT_EYE_CLOSURE_CONFIG.closed_ear_threshold,
        help="EAR below this value starts a closure (default: 0.21).",
    )
    parser.add_argument(
        "--open-ear",
        type=float,
        default=DEFAULT_EYE_CLOSURE_CONFIG.open_ear_threshold,
        help="EAR above this value ends a closure (default: 0.24).",
    )
    parser.add_argument(
        "--prolonged-seconds",
        type=float,
        default=DEFAULT_EYE_CLOSURE_CONFIG.prolonged_closure_duration_s,
        help="Seconds of closure before PROLONGED CLOSURE (default: 1.0).",
    )
    parser.add_argument(
        "--open-mar",
        type=float,
        default=DEFAULT_YAWN_CONFIG.open_mar_threshold,
        help="MAR above this value starts a mouth-open interval (default: 0.60).",
    )
    parser.add_argument(
        "--closed-mar",
        type=float,
        default=DEFAULT_YAWN_CONFIG.closed_mar_threshold,
        help="MAR below this value ends a mouth-open interval (default: 0.45).",
    )
    parser.add_argument(
        "--yawn-seconds",
        type=float,
        default=DEFAULT_YAWN_CONFIG.min_yawn_duration_s,
        help="Seconds of mouth opening before YAWNING (default: 1.5).",
    )
    return parser.parse_args()


def draw_landmark_region(
    frame: cv2.Mat,
    landmarks: list,
    indices: tuple[int, ...],
    colour: tuple[int, int, int],
    radius: int = 2,
) -> None:
    """Draw a selected facial-landmark region on an OpenCV BGR frame."""
    height, width = frame.shape[:2]
    for index in indices:
        landmark = landmarks[index]
        point = (int(landmark.x * width), int(landmark.y * height))
        cv2.circle(frame, point, radius, colour, thickness=-1)


def draw_face_landmarks(frame: cv2.Mat, landmarks: list) -> None:
    """Draw the complete face mesh plus clearly coloured eyes and mouth."""
    height, width = frame.shape[:2]
    for landmark in landmarks:
        point = (int(landmark.x * width), int(landmark.y * height))
        cv2.circle(frame, point, 1, (160, 160, 160), thickness=-1)

    draw_landmark_region(frame, landmarks, LEFT_EYE_INDICES, (0, 255, 0))
    draw_landmark_region(frame, landmarks, RIGHT_EYE_INDICES, (0, 255, 0))
    draw_landmark_region(frame, landmarks, MOUTH_INDICES, (0, 140, 255))
    draw_landmark_region(frame, landmarks, MOUTH_MAR_INDICES, (0, 140, 255), radius=3)


def _label_colour(snapshot: EyeClosureSnapshot) -> tuple[int, int, int]:
    if snapshot.prolonged_closure:
        return (0, 0, 255)
    if snapshot.eyes_closed:
        return (0, 200, 255)
    if snapshot.label == LABEL_NO_FACE:
        return (0, 0, 255)
    return (0, 200, 0)


def _eye_status_text(snapshot: EyeClosureSnapshot) -> str:
    if snapshot.eyes_closed:
        return (
            f"Eyes: {snapshot.label} {snapshot.closure_duration_s:.2f}s | "
            f"Blinks: {snapshot.blink_count}"
        )
    return f"Eyes: {snapshot.label} | Blinks: {snapshot.blink_count}"


def _drowsiness_colour(estimate: DrowsinessEstimate) -> tuple[int, int, int]:
    if estimate.level == LEVEL_HIGH:
        return (0, 0, 255)
    if estimate.level == LEVEL_MODERATE:
        return (0, 165, 255)
    if estimate.level == LEVEL_LOW:
        return (0, 200, 0)
    return (180, 180, 180)


def _drowsiness_status_text(estimate: DrowsinessEstimate) -> str:
    reason = estimate.reasons[0] if estimate.reasons else ""
    return (
        f"Drowsy est: {estimate.level} | PERCLOS {estimate.perclos:.0%} | {reason}"
    )


def _mouth_status_text(snapshot: YawnSnapshot) -> str:
    if snapshot.mouth_open:
        return (
            f"Mouth: {snapshot.label} {snapshot.open_duration_s:.2f}s | "
            f"Yawns: {snapshot.yawn_count}"
        )
    return f"Mouth: {snapshot.label} | Yawns: {snapshot.yawn_count}"


def _mouth_colour(snapshot: YawnSnapshot) -> tuple[int, int, int]:
    if snapshot.yawning:
        return (0, 0, 255)
    if snapshot.mouth_open:
        return (0, 200, 255)
    if snapshot.label == "NO FACE":
        return (0, 0, 255)
    return (0, 200, 0)


def run_landmark_viewer(
    camera_index: int,
    mirror: bool,
    eye_closure_config: EyeClosureConfig,
    drowsiness_config: DrowsinessConfig,
    yawn_config: YawnConfig,
) -> None:
    """Capture webcam frames and show eyes, drowsiness estimate, and yawns."""
    camera = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not camera.isOpened():
        raise RuntimeError(
            f"Could not open camera index {camera_index}. "
            "Check that the camera is connected and not used by another app."
        )

    tracker = EyeClosureTracker(eye_closure_config)
    estimator = DrowsinessEstimator(drowsiness_config)
    yawn_tracker = YawnTracker(yawn_config)
    face_mesh = mp.solutions.face_mesh
    try:
        with face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as detector:
            while True:
                success, frame = camera.read()
                if not success:
                    raise RuntimeError("The camera did not return a video frame.")

                if mirror:
                    frame = cv2.flip(frame, 1)

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = detector.process(rgb_frame)
                now = time.perf_counter()

                if result.multi_face_landmarks:
                    landmarks = result.multi_face_landmarks[0].landmark
                    draw_face_landmarks(frame, landmarks)
                    left_ear = ear_from_landmarks(landmarks, LEFT_EYE_EAR_INDICES)
                    right_ear = ear_from_landmarks(landmarks, RIGHT_EYE_EAR_INDICES)
                    average_ear = (left_ear + right_ear) / 2.0
                    snapshot = tracker.update(average_ear, now)
                    mar = mar_from_landmarks(landmarks, MOUTH_MAR_INDICES)
                    yawn_snapshot = yawn_tracker.update(mar, now)
                    ear_status = (
                        f"L EAR: {left_ear:.3f}   R EAR: {right_ear:.3f}   "
                        f"Avg EAR: {average_ear:.3f}   MAR: {mar:.3f}"
                    )
                    face_present = True
                else:
                    snapshot = tracker.update(None, now)
                    yawn_snapshot = yawn_tracker.update(None, now)
                    ear_status = "L EAR: --   R EAR: --   Avg EAR: --   MAR: --"
                    face_present = False

                estimate = estimator.update(
                    timestamp_s=now,
                    face_present=face_present,
                    eyes_closed=snapshot.eyes_closed,
                    prolonged_closure=snapshot.prolonged_closure,
                )
                eye_status = _eye_status_text(snapshot)
                drowsy_status = _drowsiness_status_text(estimate)
                mouth_status = _mouth_status_text(yawn_snapshot)
                status_colour = _label_colour(snapshot)
                drowsy_colour = _drowsiness_colour(estimate)
                mouth_colour = _mouth_colour(yawn_snapshot)

                cv2.putText(
                    frame,
                    eye_status,
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    status_colour,
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    ear_status,
                    (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.60,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    drowsy_status,
                    (20, 105),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    drowsy_colour,
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    mouth_status,
                    (20, 140),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    mouth_colour,
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow(WINDOW_NAME, frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        camera.release()
        cv2.destroyAllWindows()


def main() -> None:
    """Run the live landmark, EAR, and eye-closure viewer."""
    arguments = parse_arguments()
    eye_closure_config = EyeClosureConfig(
        closed_ear_threshold=arguments.closed_ear,
        open_ear_threshold=arguments.open_ear,
        min_blink_duration_s=DEFAULT_EYE_CLOSURE_CONFIG.min_blink_duration_s,
        max_blink_duration_s=DEFAULT_EYE_CLOSURE_CONFIG.max_blink_duration_s,
        prolonged_closure_duration_s=arguments.prolonged_seconds,
    )
    yawn_config = YawnConfig(
        open_mar_threshold=arguments.open_mar,
        closed_mar_threshold=arguments.closed_mar,
        min_yawn_duration_s=arguments.yawn_seconds,
    )
    run_landmark_viewer(
        arguments.camera_index,
        mirror=not arguments.no_mirror,
        eye_closure_config=eye_closure_config,
        drowsiness_config=DEFAULT_DROWSINESS_CONFIG,
        yawn_config=yawn_config,
    )


if __name__ == "__main__":
    main()
