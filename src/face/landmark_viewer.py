"""Display webcam frames with MediaPipe facial landmarks.

Run from the repository root:
    .\\.venv\\Scripts\\python.exe -m src.face.landmark_viewer
"""

from __future__ import annotations

import argparse

import cv2
import mediapipe as mp

from src.face.config import LEFT_EYE_INDICES, MOUTH_INDICES, RIGHT_EYE_INDICES

WINDOW_NAME = "DeepDrive | Face landmarks (press q to quit)"


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


def run_landmark_viewer(camera_index: int, mirror: bool) -> None:
    """Capture webcam frames and show facial landmarks until the user quits."""
    camera = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not camera.isOpened():
        raise RuntimeError(
            f"Could not open camera index {camera_index}. "
            "Check that the camera is connected and not used by another app."
        )

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

                if result.multi_face_landmarks:
                    draw_face_landmarks(frame, result.multi_face_landmarks[0].landmark)
                    status = "Face detected | Eyes: green | Mouth: orange"
                    status_colour = (0, 200, 0)
                else:
                    status = "No face detected"
                    status_colour = (0, 0, 255)

                cv2.putText(
                    frame,
                    status,
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    status_colour,
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
    """Run the first DeepDrive implementation milestone."""
    arguments = parse_arguments()
    run_landmark_viewer(arguments.camera_index, mirror=not arguments.no_mirror)


if __name__ == "__main__":
    main()
