"""Deterministic tests for blink vs prolonged-closure timing."""

import unittest

from src.drowsiness.config import EyeClosureConfig
from src.drowsiness.eye_closure import (
    LABEL_CLOSED,
    LABEL_NO_FACE,
    LABEL_OPEN,
    LABEL_PROLONGED,
    EyeClosureTracker,
)


def _tracker() -> EyeClosureTracker:
    return EyeClosureTracker(
        EyeClosureConfig(
            closed_ear_threshold=0.21,
            open_ear_threshold=0.24,
            min_blink_duration_s=0.08,
            max_blink_duration_s=0.45,
            prolonged_closure_duration_s=1.0,
        )
    )


class EyeClosureConfigTests(unittest.TestCase):
    def test_rejects_inverted_hysteresis(self) -> None:
        with self.assertRaises(ValueError):
            EyeClosureConfig(closed_ear_threshold=0.30, open_ear_threshold=0.20)

    def test_rejects_prolonged_shorter_than_a_blink(self) -> None:
        with self.assertRaises(ValueError):
            EyeClosureConfig(max_blink_duration_s=0.80, prolonged_closure_duration_s=0.50)


class EyeClosureTrackerTests(unittest.TestCase):
    def test_open_eyes_stay_open(self) -> None:
        tracker = _tracker()
        snapshot = tracker.update(0.30, 0.0)
        self.assertEqual(snapshot.label, LABEL_OPEN)
        self.assertFalse(snapshot.eyes_closed)
        self.assertEqual(snapshot.blink_count, 0)

    def test_short_closure_then_open_counts_as_one_blink(self) -> None:
        tracker = _tracker()
        tracker.update(0.30, 0.00)
        tracker.update(0.10, 0.10)
        closed = tracker.update(0.10, 0.30)
        self.assertEqual(closed.label, LABEL_CLOSED)
        opened = tracker.update(0.30, 0.35)
        self.assertEqual(opened.label, LABEL_OPEN)
        self.assertTrue(opened.blink_just_completed)
        self.assertEqual(opened.blink_count, 1)

    def test_tiny_closure_is_treated_as_noise_not_a_blink(self) -> None:
        tracker = _tracker()
        tracker.update(0.30, 0.00)
        tracker.update(0.10, 0.10)
        opened = tracker.update(0.30, 0.14)
        self.assertFalse(opened.blink_just_completed)
        self.assertEqual(opened.blink_count, 0)

    def test_slow_closure_is_not_a_blink_and_not_yet_prolonged(self) -> None:
        tracker = _tracker()
        tracker.update(0.30, 0.00)
        tracker.update(0.10, 0.10)
        still_closed = tracker.update(0.10, 0.80)
        self.assertEqual(still_closed.label, LABEL_CLOSED)
        opened = tracker.update(0.30, 0.85)
        self.assertFalse(opened.blink_just_completed)
        self.assertEqual(opened.blink_count, 0)
        self.assertFalse(opened.prolonged_closure)

    def test_one_second_closure_is_prolonged_while_eyes_remain_shut(self) -> None:
        tracker = _tracker()
        tracker.update(0.30, 0.00)
        tracker.update(0.10, 0.20)
        early = tracker.update(0.10, 0.90)
        self.assertEqual(early.label, LABEL_CLOSED)
        self.assertFalse(early.prolonged_closure)
        later = tracker.update(0.10, 1.25)
        self.assertEqual(later.label, LABEL_PROLONGED)
        self.assertTrue(later.prolonged_closure)
        self.assertGreaterEqual(later.closure_duration_s, 1.0)

    def test_prolonged_flag_clears_after_eyes_open(self) -> None:
        tracker = _tracker()
        tracker.update(0.30, 0.00)
        tracker.update(0.10, 0.10)
        tracker.update(0.10, 1.20)
        opened = tracker.update(0.30, 1.30)
        self.assertEqual(opened.label, LABEL_OPEN)
        self.assertFalse(opened.prolonged_closure)
        self.assertEqual(opened.blink_count, 0)

    def test_hysteresis_keeps_closed_in_the_gap(self) -> None:
        tracker = _tracker()
        tracker.update(0.30, 0.00)
        tracker.update(0.10, 0.10)
        between = tracker.update(0.22, 0.30)
        self.assertTrue(between.eyes_closed)
        self.assertEqual(between.label, LABEL_CLOSED)

    def test_hysteresis_keeps_open_in_the_gap(self) -> None:
        tracker = _tracker()
        tracker.update(0.30, 0.00)
        between = tracker.update(0.22, 0.10)
        self.assertEqual(between.label, LABEL_OPEN)
        self.assertFalse(between.eyes_closed)

    def test_missing_face_resets_an_in_progress_closure(self) -> None:
        tracker = _tracker()
        tracker.update(0.30, 0.00)
        tracker.update(0.10, 0.10)
        lost = tracker.update(None, 1.50)
        self.assertEqual(lost.label, LABEL_NO_FACE)
        reopened = tracker.update(0.30, 1.60)
        self.assertEqual(reopened.label, LABEL_OPEN)
        self.assertEqual(reopened.blink_count, 0)

    def test_duration_uses_time_not_frame_count(self) -> None:
        tracker = _tracker()
        tracker.update(0.30, 0.0)
        tracker.update(0.10, 5.0)
        snapshot = tracker.update(0.10, 6.1)
        self.assertEqual(snapshot.label, LABEL_PROLONGED)
        self.assertAlmostEqual(snapshot.closure_duration_s, 1.1)


if __name__ == "__main__":
    unittest.main()
