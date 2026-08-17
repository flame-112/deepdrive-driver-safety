"""Fast checks for Face Mesh landmark-region configuration."""

import unittest

from src.face.config import (
    FACE_MESH_LANDMARK_COUNT,
    LEFT_EYE_INDICES,
    MOUTH_INDICES,
    RIGHT_EYE_INDICES,
)


class LandmarkRegionTests(unittest.TestCase):
    """Ensure selected landmark identifiers remain valid Face Mesh indices."""

    def test_every_landmark_index_is_in_range(self) -> None:
        for index in LEFT_EYE_INDICES + RIGHT_EYE_INDICES + MOUTH_INDICES:
            self.assertGreaterEqual(index, 0)
            self.assertLess(index, FACE_MESH_LANDMARK_COUNT)

    def test_eye_regions_are_distinct(self) -> None:
        self.assertFalse(set(LEFT_EYE_INDICES) & set(RIGHT_EYE_INDICES))


if __name__ == "__main__":
    unittest.main()
