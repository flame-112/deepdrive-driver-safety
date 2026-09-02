"""Deterministic tests for the rolling-window drowsiness estimate."""

import unittest

from src.drowsiness.config import DrowsinessConfig
from src.drowsiness.estimator import (
    LEVEL_HIGH,
    LEVEL_LOW,
    LEVEL_MODERATE,
    LEVEL_WARMUP,
    DrowsinessEstimator,
)


def _estimator() -> DrowsinessEstimator:
    return DrowsinessEstimator(
        DrowsinessConfig(
            window_s=30.0,
            min_observation_s=5.0,
            perclos_moderate=0.08,
            perclos_high=0.20,
            recent_prolonged_s=10.0,
        )
    )


def _run_open(estimator: DrowsinessEstimator, start_s: float, end_s: float, step_s: float = 0.2) -> None:
    timestamp = start_s
    while timestamp <= end_s:
        estimator.update(
            timestamp_s=timestamp,
            face_present=True,
            eyes_closed=False,
            prolonged_closure=False,
        )
        timestamp += step_s


class DrowsinessConfigTests(unittest.TestCase):
    def test_rejects_inverted_perclos_cutoffs(self) -> None:
        with self.assertRaises(ValueError):
            DrowsinessConfig(perclos_moderate=0.30, perclos_high=0.10)


class DrowsinessEstimatorTests(unittest.TestCase):
    def test_warmup_before_enough_face_time(self) -> None:
        estimator = _estimator()
        estimate = estimator.update(0.0, True, False, False)
        estimate = estimator.update(2.0, True, False, False)
        self.assertEqual(estimate.level, LEVEL_WARMUP)

    def test_open_eyes_become_low_after_warmup(self) -> None:
        estimator = _estimator()
        _run_open(estimator, 0.0, 6.0)
        estimate = estimator.update(6.2, True, False, False)
        self.assertEqual(estimate.level, LEVEL_LOW)
        self.assertLess(estimate.perclos, 0.08)

    def test_current_prolonged_closure_is_high_immediately(self) -> None:
        estimator = _estimator()
        estimate = estimator.update(0.0, True, True, True)
        self.assertEqual(estimate.level, LEVEL_HIGH)
        self.assertIn("prolonged eye closure now", estimate.reasons)

    def test_recent_prolonged_event_is_moderate_after_eyes_open(self) -> None:
        estimator = _estimator()
        _run_open(estimator, 0.0, 6.0)
        estimator.update(6.2, True, True, True)
        estimate = estimator.update(6.5, True, False, False)
        self.assertEqual(estimate.level, LEVEL_MODERATE)
        self.assertTrue(any("prolonged" in reason for reason in estimate.reasons))

    def test_old_prolonged_event_does_not_keep_moderate(self) -> None:
        estimator = _estimator()
        estimator.update(0.0, True, True, True)
        _run_open(estimator, 0.2, 16.0)
        estimate = estimator.update(16.2, True, False, False)
        self.assertEqual(estimate.level, LEVEL_LOW)

    def test_high_closed_fraction_is_high_without_current_prolonged(self) -> None:
        estimator = _estimator()
        timestamp = 0.0
        while timestamp <= 6.0:
            closed = timestamp >= 1.0
            estimator.update(timestamp, True, closed, False)
            timestamp += 0.2
        estimate = estimator.update(6.2, True, True, False)
        self.assertGreaterEqual(estimate.perclos, 0.20)
        self.assertEqual(estimate.level, LEVEL_HIGH)

    def test_missing_face_time_is_excluded_from_perclos(self) -> None:
        estimator = _estimator()
        _run_open(estimator, 0.0, 6.0)
        estimator.update(6.5, False, False, False)
        estimator.update(20.0, False, False, False)
        estimate = estimator.update(21.0, True, False, False)
        self.assertLess(estimate.perclos, 0.08)
        self.assertEqual(estimate.level, LEVEL_LOW)

    def test_duration_uses_time_not_frame_count(self) -> None:
        estimator = _estimator()
        estimator.update(0.0, True, False, False)
        estimator.update(5.0, True, True, False)
        estimate = estimator.update(10.0, True, True, False)
        self.assertGreater(estimate.perclos, 0.40)
        self.assertLess(estimate.perclos, 0.70)


if __name__ == "__main__":
    unittest.main()
