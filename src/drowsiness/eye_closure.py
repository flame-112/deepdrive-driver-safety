"""Track average EAR over time to separate blinks from prolonged closure.

A blink is recorded only after the eyes open again and the closed interval
was short. Prolonged closure is flagged while the eyes stay below threshold
longer than the configured duration. Neither label is a fatigue diagnosis.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.drowsiness.config import EyeClosureConfig

LABEL_NO_FACE = "NO FACE"
LABEL_OPEN = "OPEN"
LABEL_CLOSED = "CLOSED"
LABEL_PROLONGED = "PROLONGED CLOSURE"


@dataclass(frozen=True)
class EyeClosureSnapshot:
    """Per-frame result of the eye-closure tracker."""

    label: str
    eyes_closed: bool
    prolonged_closure: bool
    blink_just_completed: bool
    blink_count: int
    closure_duration_s: float


class EyeClosureTracker:
    """Stateful classifier driven by caller-supplied timestamps."""

    def __init__(self, config: EyeClosureConfig | None = None) -> None:
        self.config = config or EyeClosureConfig()
        self._eyes_closed = False
        self._closed_started_at: float | None = None
        self._blink_count = 0

    def update(self, average_ear: float | None, timestamp_s: float) -> EyeClosureSnapshot:
        """Update using averaged EAR, or ``None`` when no face is present."""
        if average_ear is None:
            self._clear_open_closure()
            return self._snapshot(
                label=LABEL_NO_FACE,
                eyes_closed=False,
                prolonged_closure=False,
                blink_just_completed=False,
                closure_duration_s=0.0,
            )

        if not self._eyes_closed:
            if average_ear < self.config.closed_ear_threshold:
                self._eyes_closed = True
                self._closed_started_at = timestamp_s
                return self._closed_snapshot(timestamp_s, blink_just_completed=False)
            return self._snapshot(
                label=LABEL_OPEN,
                eyes_closed=False,
                prolonged_closure=False,
                blink_just_completed=False,
                closure_duration_s=0.0,
            )

        if average_ear > self.config.open_ear_threshold:
            duration = self._closure_duration(timestamp_s)
            blink_just_completed = (
                self.config.min_blink_duration_s <= duration <= self.config.max_blink_duration_s
            )
            if blink_just_completed:
                self._blink_count += 1
            self._clear_open_closure()
            return self._snapshot(
                label=LABEL_OPEN,
                eyes_closed=False,
                prolonged_closure=False,
                blink_just_completed=blink_just_completed,
                closure_duration_s=0.0,
            )

        return self._closed_snapshot(timestamp_s, blink_just_completed=False)

    def _clear_open_closure(self) -> None:
        self._eyes_closed = False
        self._closed_started_at = None

    def _closure_duration(self, timestamp_s: float) -> float:
        if self._closed_started_at is None:
            return 0.0
        return max(0.0, timestamp_s - self._closed_started_at)

    def _closed_snapshot(self, timestamp_s: float, blink_just_completed: bool) -> EyeClosureSnapshot:
        duration = self._closure_duration(timestamp_s)
        prolonged = duration >= self.config.prolonged_closure_duration_s
        return self._snapshot(
            label=LABEL_PROLONGED if prolonged else LABEL_CLOSED,
            eyes_closed=True,
            prolonged_closure=prolonged,
            blink_just_completed=blink_just_completed,
            closure_duration_s=duration,
        )

    def _snapshot(
        self,
        label: str,
        eyes_closed: bool,
        prolonged_closure: bool,
        blink_just_completed: bool,
        closure_duration_s: float,
    ) -> EyeClosureSnapshot:
        return EyeClosureSnapshot(
            label=label,
            eyes_closed=eyes_closed,
            prolonged_closure=prolonged_closure,
            blink_just_completed=blink_just_completed,
            blink_count=self._blink_count,
            closure_duration_s=closure_duration_s,
        )
