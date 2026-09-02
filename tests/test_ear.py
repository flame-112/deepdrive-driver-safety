"""Unit tests for Eye Aspect Ratio geometry."""

import math
import unittest
from types import SimpleNamespace

from src.face.config import LEFT_EYE_EAR_INDICES, RIGHT_EYE_EAR_INDICES
from src.face.ear import ear_from_landmarks, eye_aspect_ratio


class EyeAspectRatioTests(unittest.TestCase):
    """EAR must follow the six-point formula and stay numerically stable."""

    def test_known_open_eye_matches_formula(self) -> None:
        # p1=(0,0), p2=(1,1), p3=(2,1), p4=(3,0), p5=(2,-1), p6=(1,-1)
        # verticals = 2 and 2; width = 3; EAR = 4 / 6
        points = ((0.0, 0.0), (1.0, 1.0), (2.0, 1.0), (3.0, 0.0), (2.0, -1.0), (1.0, -1.0))
        self.assertAlmostEqual(eye_aspect_ratio(points), 4.0 / 6.0)

    def test_nearly_closed_eye_is_much_smaller_than_open_eye(self) -> None:
        open_eye = ((0.0, 0.0), (1.0, 1.0), (2.0, 1.0), (3.0, 0.0), (2.0, -1.0), (1.0, -1.0))
        closed_eye = (
            (0.0, 0.0),
            (1.0, 0.05),
            (2.0, 0.05),
            (3.0, 0.0),
            (2.0, -0.05),
            (1.0, -0.05),
        )
        self.assertLess(eye_aspect_ratio(closed_eye), 0.1)
        self.assertGreater(eye_aspect_ratio(open_eye), 0.5)
        self.assertLess(eye_aspect_ratio(closed_eye), eye_aspect_ratio(open_eye))

    def test_scale_invariance(self) -> None:
        small = ((0.0, 0.0), (1.0, 1.0), (2.0, 1.0), (3.0, 0.0), (2.0, -1.0), (1.0, -1.0))
        large = tuple((x * 10.0, y * 10.0) for x, y in small)
        self.assertAlmostEqual(eye_aspect_ratio(small), eye_aspect_ratio(large))

    def test_zero_width_eye_returns_zero(self) -> None:
        collapsed = ((0.0, 0.0), (0.0, 1.0), (0.0, 1.0), (0.0, 0.0), (0.0, -1.0), (0.0, -1.0))
        self.assertEqual(eye_aspect_ratio(collapsed), 0.0)

    def test_wrong_point_count_raises(self) -> None:
        with self.assertRaises(ValueError):
            eye_aspect_ratio(((0.0, 0.0), (1.0, 1.0)))

    def test_ear_from_landmarks_uses_named_indices(self) -> None:
        landmarks = [SimpleNamespace(x=0.0, y=0.0) for _ in range(468)]
        p1, p2, p3, p4, p5, p6 = LEFT_EYE_EAR_INDICES
        landmarks[p1] = SimpleNamespace(x=0.0, y=0.0)
        landmarks[p2] = SimpleNamespace(x=1.0, y=1.0)
        landmarks[p3] = SimpleNamespace(x=2.0, y=1.0)
        landmarks[p4] = SimpleNamespace(x=3.0, y=0.0)
        landmarks[p5] = SimpleNamespace(x=2.0, y=-1.0)
        landmarks[p6] = SimpleNamespace(x=1.0, y=-1.0)
        self.assertAlmostEqual(ear_from_landmarks(landmarks, LEFT_EYE_EAR_INDICES), 4.0 / 6.0)

    def test_left_and_right_ear_index_sets_are_six_distinct_points(self) -> None:
        self.assertEqual(len(LEFT_EYE_EAR_INDICES), 6)
        self.assertEqual(len(RIGHT_EYE_EAR_INDICES), 6)
        self.assertEqual(len(set(LEFT_EYE_EAR_INDICES)), 6)
        self.assertEqual(len(set(RIGHT_EYE_EAR_INDICES)), 6)
        self.assertFalse(set(LEFT_EYE_EAR_INDICES) & set(RIGHT_EYE_EAR_INDICES))


class EyeAspectRatioFormulaSanityTests(unittest.TestCase):
    def test_manual_hypot_matches_implementation(self) -> None:
        points = ((0.2, 0.4), (0.3, 0.5), (0.5, 0.52), (0.7, 0.41), (0.5, 0.31), (0.3, 0.29))
        p1, p2, p3, p4, p5, p6 = points
        expected = (
            math.hypot(p2[0] - p6[0], p2[1] - p6[1])
            + math.hypot(p3[0] - p5[0], p3[1] - p5[1])
        ) / (2.0 * math.hypot(p1[0] - p4[0], p1[1] - p4[1]))
        self.assertAlmostEqual(eye_aspect_ratio(points), expected)


if __name__ == "__main__":
    unittest.main()
