"""Experimental thresholds for blink vs prolonged eye closure.

These numbers are starting points for a webcam prototype, not medically or
statistically validated cutoffs. Tune them on the same camera used for demos
(phone-as-webcam often needs a slightly different EAR threshold than a laptop).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EyeClosureConfig:
    """Hysteresis and duration gates for classifying a closure event.

    EAR below ``closed_ear_threshold`` starts a closure. EAR must rise above
    ``open_ear_threshold`` before the eyes count as open again. Duration is
    measured in seconds so behaviour does not silently change with FPS.
    """

    closed_ear_threshold: float = 0.21
    open_ear_threshold: float = 0.24
    min_blink_duration_s: float = 0.08
    max_blink_duration_s: float = 0.45
    prolonged_closure_duration_s: float = 1.0

    def __post_init__(self) -> None:
        if self.open_ear_threshold < self.closed_ear_threshold:
            raise ValueError("open_ear_threshold must be >= closed_ear_threshold.")
        if self.min_blink_duration_s < 0:
            raise ValueError("min_blink_duration_s must be non-negative.")
        if self.max_blink_duration_s < self.min_blink_duration_s:
            raise ValueError("max_blink_duration_s must be >= min_blink_duration_s.")
        if self.prolonged_closure_duration_s <= self.max_blink_duration_s:
            raise ValueError(
                "prolonged_closure_duration_s must be greater than max_blink_duration_s."
            )


DEFAULT_EYE_CLOSURE_CONFIG = EyeClosureConfig()


@dataclass(frozen=True)
class DrowsinessConfig:
    """Experimental mapping from eye-closure history to a drowsiness estimate.

    PERCLOS here means the fraction of *observed* (face-present) time that the
    eyes were closed. The cutoffs are demo starting points, not clinical rules.
    """

    window_s: float = 30.0
    min_observation_s: float = 5.0
    perclos_moderate: float = 0.08
    perclos_high: float = 0.20
    recent_prolonged_s: float = 10.0

    def __post_init__(self) -> None:
        if self.window_s <= 0:
            raise ValueError("window_s must be positive.")
        if self.min_observation_s < 0 or self.min_observation_s > self.window_s:
            raise ValueError("min_observation_s must be between 0 and window_s.")
        if not 0.0 <= self.perclos_moderate < self.perclos_high <= 1.0:
            raise ValueError("Need 0 <= perclos_moderate < perclos_high <= 1.")
        if self.recent_prolonged_s < 0 or self.recent_prolonged_s > self.window_s:
            raise ValueError("recent_prolonged_s must be between 0 and window_s.")


DEFAULT_DROWSINESS_CONFIG = DrowsinessConfig()
