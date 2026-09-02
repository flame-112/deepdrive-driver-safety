"""Fast checks for Face Mesh landmark-region configuration."""

import unittest

from src.face.config import (
    FACE_MESH_LANDMARK_COUNT,
    LEFT_EYE_EAR_INDICES,
    LEFT_EYE_INDICES,
    MOUTH_INDICES,
    MOUTH_MAR_INDICES,
    RIGHT_EYE_EAR_INDICES,
    RIGHT_EYE_INDICES,
)


class LandmarkRegionTests(unittest.TestCase):
    """Ensure selected landmark identifiers remain valid Face Mesh indices."""

    def test_every_landmark_index_is_in_range(self) -> None:
        for index in LEFT_EYE_INDICES + RIGHT_EYE_INDICES + MOUTH_INDICES + MOUTH_MAR_INDICES:
            self.assertGreaterEqual(index, 0)
            self.assertLess(index, FACE_MESH_LANDMARK_COUNT)

    def test_eye_regions_are_distinct(self) -> None:
        self.assertFalse(set(LEFT_EYE_INDICES) & set(RIGHT_EYE_INDICES))

    def test_ear_indices_match_the_visualized_eye_points(self) -> None:
        self.assertEqual(set(LEFT_EYE_EAR_INDICES), set(LEFT_EYE_INDICES))
        self.assertEqual(set(RIGHT_EYE_EAR_INDICES), set(RIGHT_EYE_INDICES))

    def test_mar_uses_six_distinct_mouth_points(self) -> None:
        self.assertEqual(len(MOUTH_MAR_INDICES), 6)
        self.assertEqual(len(set(MOUTH_MAR_INDICES)), 6)


if __name__ == "__main__":
    unittest.main()
