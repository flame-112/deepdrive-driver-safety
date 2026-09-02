"""Unit tests for Mouth Aspect Ratio geometry."""

import unittest
from types import SimpleNamespace

from src.face.config import MOUTH_MAR_INDICES
from src.yawning.mar import mar_from_landmarks, mouth_aspect_ratio


class MouthAspectRatioTests(unittest.TestCase):
    def test_known_open_mouth_matches_formula(self) -> None:
        points = ((0.0, 0.0), (1.0, 1.0), (2.0, 1.0), (3.0, 0.0), (2.0, -1.0), (1.0, -1.0))
        self.assertAlmostEqual(mouth_aspect_ratio(points), 4.0 / 6.0)

    def test_closed_mouth_is_smaller_than_open_mouth(self) -> None:
        open_mouth = ((0.0, 0.0), (1.0, 1.0), (2.0, 1.0), (3.0, 0.0), (2.0, -1.0), (1.0, -1.0))
        closed_mouth = (
            (0.0, 0.0),
            (1.0, 0.05),
            (2.0, 0.05),
            (3.0, 0.0),
            (2.0, -0.05),
            (1.0, -0.05),
        )
        self.assertLess(mouth_aspect_ratio(closed_mouth), mouth_aspect_ratio(open_mouth))

    def test_wrong_point_count_raises(self) -> None:
        with self.assertRaises(ValueError):
            mouth_aspect_ratio(((0.0, 0.0), (1.0, 1.0)))

    def test_mar_from_landmarks_uses_named_indices(self) -> None:
        landmarks = [SimpleNamespace(x=0.0, y=0.0) for _ in range(468)]
        p1, p2, p3, p4, p5, p6 = MOUTH_MAR_INDICES
        landmarks[p1] = SimpleNamespace(x=0.0, y=0.0)
        landmarks[p2] = SimpleNamespace(x=1.0, y=1.0)
        landmarks[p3] = SimpleNamespace(x=2.0, y=1.0)
        landmarks[p4] = SimpleNamespace(x=3.0, y=0.0)
        landmarks[p5] = SimpleNamespace(x=2.0, y=-1.0)
        landmarks[p6] = SimpleNamespace(x=1.0, y=-1.0)
        self.assertAlmostEqual(mar_from_landmarks(landmarks, MOUTH_MAR_INDICES), 4.0 / 6.0)


if __name__ == "__main__":
    unittest.main()
