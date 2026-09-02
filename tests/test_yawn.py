"""Deterministic tests for yawn timing from MAR."""

import unittest

from src.yawning.config import YawnConfig
from src.yawning.yawn import LABEL_CLOSED, LABEL_NO_FACE, LABEL_OPEN, LABEL_YAWNING, YawnTracker


def _tracker() -> YawnTracker:
    return YawnTracker(
        YawnConfig(
            open_mar_threshold=0.60,
            closed_mar_threshold=0.45,
            min_yawn_duration_s=1.5,
        )
    )


class YawnConfigTests(unittest.TestCase):
    def test_rejects_inverted_hysteresis(self) -> None:
        with self.assertRaises(ValueError):
            YawnConfig(open_mar_threshold=0.40, closed_mar_threshold=0.50)


class YawnTrackerTests(unittest.TestCase):
    def test_closed_mouth_stays_closed(self) -> None:
        tracker = _tracker()
        snapshot = tracker.update(0.30, 0.0)
        self.assertEqual(snapshot.label, LABEL_CLOSED)
        self.assertEqual(snapshot.yawn_count, 0)

    def test_brief_open_is_not_a_yawn(self) -> None:
        tracker = _tracker()
        tracker.update(0.30, 0.00)
        tracker.update(0.80, 0.10)
        still_open = tracker.update(0.80, 1.00)
        self.assertEqual(still_open.label, LABEL_OPEN)
        self.assertFalse(still_open.yawning)
        closed = tracker.update(0.30, 1.10)
        self.assertEqual(closed.label, LABEL_CLOSED)
        self.assertEqual(closed.yawn_count, 0)

    def test_long_open_counts_one_yawn(self) -> None:
        tracker = _tracker()
        tracker.update(0.30, 0.00)
        tracker.update(0.80, 0.10)
        early = tracker.update(0.80, 1.40)
        self.assertEqual(early.label, LABEL_OPEN)
        yawning = tracker.update(0.80, 1.70)
        self.assertEqual(yawning.label, LABEL_YAWNING)
        self.assertTrue(yawning.yawn_just_started)
        self.assertEqual(yawning.yawn_count, 1)
        still = tracker.update(0.80, 2.00)
        self.assertTrue(still.yawning)
        self.assertFalse(still.yawn_just_started)
        self.assertEqual(still.yawn_count, 1)

    def test_hysteresis_keeps_open_in_the_gap(self) -> None:
        tracker = _tracker()
        tracker.update(0.80, 0.00)
        between = tracker.update(0.50, 0.40)
        self.assertTrue(between.mouth_open)
        self.assertEqual(between.label, LABEL_OPEN)

    def test_hysteresis_keeps_closed_in_the_gap(self) -> None:
        tracker = _tracker()
        tracker.update(0.30, 0.00)
        between = tracker.update(0.50, 0.20)
        self.assertEqual(between.label, LABEL_CLOSED)
        self.assertFalse(between.mouth_open)

    def test_missing_face_resets_an_in_progress_opening(self) -> None:
        tracker = _tracker()
        tracker.update(0.80, 0.00)
        lost = tracker.update(None, 2.00)
        self.assertEqual(lost.label, LABEL_NO_FACE)
        self.assertEqual(lost.yawn_count, 0)
        closed = tracker.update(0.30, 2.10)
        self.assertEqual(closed.label, LABEL_CLOSED)
        self.assertEqual(closed.yawn_count, 0)

    def test_duration_uses_time_not_frame_count(self) -> None:
        tracker = _tracker()
        tracker.update(0.80, 0.0)
        snapshot = tracker.update(0.80, 1.6)
        self.assertEqual(snapshot.label, LABEL_YAWNING)
        self.assertAlmostEqual(snapshot.open_duration_s, 1.6)


if __name__ == "__main__":
    unittest.main()
