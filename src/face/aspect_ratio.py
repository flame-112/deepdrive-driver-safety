"""Six-point aspect ratio shared by EAR and MAR.

EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
MAR uses the same formula on mouth corners and lips.
"""

from __future__ import annotations

import math
from typing import Sequence

Point = tuple[float, float]

POINT_COUNT = 6
_MIN_HORIZONTAL_DISTANCE = 1e-6


def six_point_aspect_ratio(points: Sequence[Point]) -> float:
    """Return the Soukupová-style ratio for six (x, y) points in p1..p6 order.

    Returns 0.0 if the horizontal distance is numerically zero.
    """
    if len(points) != POINT_COUNT:
        raise ValueError(f"Aspect ratio requires exactly {POINT_COUNT} points, got {len(points)}.")

    p1, p2, p3, p4, p5, p6 = points
    vertical_1 = math.hypot(p2[0] - p6[0], p2[1] - p6[1])
    vertical_2 = math.hypot(p3[0] - p5[0], p3[1] - p5[1])
    horizontal = math.hypot(p1[0] - p4[0], p1[1] - p4[1])
    if horizontal < _MIN_HORIZONTAL_DISTANCE:
        return 0.0
    return (vertical_1 + vertical_2) / (2.0 * horizontal)
