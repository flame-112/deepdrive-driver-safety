"""Estimate drowsiness from observable eye-closure history.

This is a prototype risk-style estimate, not a medical diagnosis of fatigue.
It combines a simplified PERCLOS (closed-time / face-present time) with
whether a prolonged closure is happening now or happened recently.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from src.drowsiness.config import DrowsinessConfig

LEVEL_WARMUP = "WARMUP"
LEVEL_LOW = "LOW"
LEVEL_MODERATE = "MODERATE"
LEVEL_HIGH = "HIGH"


@dataclass(frozen=True)
class _Sample:
    timestamp_s: float
    face_present: bool
    eyes_closed: bool
    prolonged_closure: bool


@dataclass(frozen=True)
class DrowsinessEstimate:
    """One frame of drowsiness estimate for display and later logging."""

    level: str
    perclos: float
    observation_s: float
    reasons: tuple[str, ...]


class DrowsinessEstimator:
    """Rolling-window estimator driven by caller-supplied timestamps."""

    def __init__(self, config: DrowsinessConfig | None = None) -> None:
        self.config = config or DrowsinessConfig()
        self._samples: deque[_Sample] = deque()

    def update(
        self,
        timestamp_s: float,
        face_present: bool,
        eyes_closed: bool,
        prolonged_closure: bool,
    ) -> DrowsinessEstimate:
        """Record this frame and return the current estimate."""
        self._samples.append(
            _Sample(
                timestamp_s=timestamp_s,
                face_present=face_present,
                eyes_closed=eyes_closed and face_present,
                prolonged_closure=prolonged_closure and face_present,
            )
        )
        self._prune(timestamp_s)
        closed_s, observation_s, last_prolonged_at = self._accumulate(timestamp_s)
        perclos = (closed_s / observation_s) if observation_s > 0.0 else 0.0
        currently_prolonged = face_present and prolonged_closure
        recent_prolonged = last_prolonged_at is not None and (
            timestamp_s - last_prolonged_at
        ) <= self.config.recent_prolonged_s

        if currently_prolonged:
            return DrowsinessEstimate(
                level=LEVEL_HIGH,
                perclos=perclos,
                observation_s=observation_s,
                reasons=("prolonged eye closure now",),
            )

        if observation_s < self.config.min_observation_s:
            return DrowsinessEstimate(
                level=LEVEL_WARMUP,
                perclos=perclos,
                observation_s=observation_s,
                reasons=("collecting eye-closure history",),
            )

        reasons: list[str] = []
        if perclos >= self.config.perclos_high:
            reasons.append(f"PERCLOS {perclos:.0%} in last {self.config.window_s:.0f}s")
            return DrowsinessEstimate(
                level=LEVEL_HIGH,
                perclos=perclos,
                observation_s=observation_s,
                reasons=tuple(reasons),
            )

        if perclos >= self.config.perclos_moderate:
            reasons.append(f"PERCLOS {perclos:.0%} in last {self.config.window_s:.0f}s")
        if recent_prolonged:
            reasons.append("prolonged closure recently")

        if reasons:
            return DrowsinessEstimate(
                level=LEVEL_MODERATE,
                perclos=perclos,
                observation_s=observation_s,
                reasons=tuple(reasons),
            )

        return DrowsinessEstimate(
            level=LEVEL_LOW,
            perclos=perclos,
            observation_s=observation_s,
            reasons=("eyes mostly open",),
        )

    def _prune(self, timestamp_s: float) -> None:
        cutoff = timestamp_s - self.config.window_s
        while self._samples and self._samples[0].timestamp_s < cutoff:
            self._samples.popleft()

    def _accumulate(self, now_s: float) -> tuple[float, float, float | None]:
        """Time-weight samples so variable FPS does not distort PERCLOS."""
        if not self._samples:
            return 0.0, 0.0, None

        closed_s = 0.0
        observation_s = 0.0
        last_prolonged_at: float | None = None
        previous: _Sample | None = None
        for sample in self._samples:
            if previous is not None:
                closed_s, observation_s, last_prolonged_at = self._add_interval(
                    previous,
                    sample.timestamp_s - previous.timestamp_s,
                    closed_s,
                    observation_s,
                    last_prolonged_at,
                    sample.timestamp_s,
                )
            previous = sample

        if previous is not None:
            closed_s, observation_s, last_prolonged_at = self._add_interval(
                previous,
                now_s - previous.timestamp_s,
                closed_s,
                observation_s,
                last_prolonged_at,
                now_s,
            )
        return closed_s, observation_s, last_prolonged_at

    def _add_interval(
        self,
        sample: _Sample,
        duration_s: float,
        closed_s: float,
        observation_s: float,
        last_prolonged_at: float | None,
        interval_end_s: float,
    ) -> tuple[float, float, float | None]:
        if duration_s <= 0.0 or not sample.face_present:
            return closed_s, observation_s, last_prolonged_at
        observation_s += duration_s
        if sample.eyes_closed:
            closed_s += duration_s
        if sample.prolonged_closure:
            last_prolonged_at = interval_end_s
        return closed_s, observation_s, last_prolonged_at
