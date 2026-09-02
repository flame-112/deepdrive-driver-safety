"""Mouth Aspect Ratio (MAR) from six lip landmarks.

MAR uses the same six-point formula as EAR. A high value means the mouth
looks more open. It is not by itself a yawn.
"""

from __future__ import annotations

from typing import Sequence

from src.face.aspect_ratio import six_point_aspect_ratio

Point = tuple[float, float]


def mouth_aspect_ratio(points: Sequence[Point]) -> float:
    """Return MAR from six (x, y) points in p1..p6 order."""
    return six_point_aspect_ratio(points)


def mar_from_landmarks(landmarks: Sequence, indices: Sequence[int]) -> float:
    """Compute MAR from MediaPipe-style landmarks using p1..p6 indices."""
    points = tuple((landmarks[index].x, landmarks[index].y) for index in indices)
    return mouth_aspect_ratio(points)
