"""Eye Aspect Ratio (EAR) from six facial-landmark points.

EAR estimates how open an eye appears from geometry only. It is not a
drowsiness diagnosis. Typical published blink thresholds are often near 0.2
with dlib landmarks; MediaPipe values can differ, so this module only
computes the ratio.
"""

from __future__ import annotations

from typing import Sequence

from src.face.aspect_ratio import six_point_aspect_ratio

Point = tuple[float, float]


def eye_aspect_ratio(points: Sequence[Point]) -> float:
    """Return EAR for one eye from six (x, y) points in p1..p6 order."""
    return six_point_aspect_ratio(points)


def ear_from_landmarks(landmarks: Sequence, indices: Sequence[int]) -> float:
    """Compute EAR from MediaPipe-style landmarks using p1..p6 indices."""
    points = tuple((landmarks[index].x, landmarks[index].y) for index in indices)
    return eye_aspect_ratio(points)
