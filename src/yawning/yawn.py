"""Track MAR over time to flag yawns as prolonged mouth opening.

A yawn is recorded once when the mouth has stayed open long enough. Brief
speech or a smile should not count. This is an observable label, not a
diagnosis that the driver is exhausted.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.yawning.config import YawnConfig

LABEL_NO_FACE = "NO FACE"
LABEL_CLOSED = "CLOSED"
LABEL_OPEN = "OPEN"
LABEL_YAWNING = "YAWNING"


@dataclass(frozen=True)
class YawnSnapshot:
    """Per-frame result of the yawn tracker."""

    label: str
    mouth_open: bool
    yawning: bool
    yawn_just_started: bool
    yawn_count: int
    open_duration_s: float


class YawnTracker:
    """Stateful classifier driven by caller-supplied timestamps."""

    def __init__(self, config: YawnConfig | None = None) -> None:
        self.config = config or YawnConfig()
        self._mouth_open = False
        self._open_started_at: float | None = None
        self._yawn_counted_this_opening = False
        self._yawn_count = 0

    def update(self, mar: float | None, timestamp_s: float) -> YawnSnapshot:
        """Update using MAR, or ``None`` when no face is present."""
        if mar is None:
            self._clear_opening()
            return self._snapshot(
                label=LABEL_NO_FACE,
                mouth_open=False,
                yawning=False,
                yawn_just_started=False,
                open_duration_s=0.0,
            )

        if not self._mouth_open:
            if mar > self.config.open_mar_threshold:
                self._mouth_open = True
                self._open_started_at = timestamp_s
                self._yawn_counted_this_opening = False
                return self._open_snapshot(timestamp_s, yawn_just_started=False)
            return self._snapshot(
                label=LABEL_CLOSED,
                mouth_open=False,
                yawning=False,
                yawn_just_started=False,
                open_duration_s=0.0,
            )

        if mar < self.config.closed_mar_threshold:
            self._clear_opening()
            return self._snapshot(
                label=LABEL_CLOSED,
                mouth_open=False,
                yawning=False,
                yawn_just_started=False,
                open_duration_s=0.0,
            )

        return self._open_snapshot(timestamp_s, yawn_just_started=False)

    def _clear_opening(self) -> None:
        self._mouth_open = False
        self._open_started_at = None
        self._yawn_counted_this_opening = False

    def _open_duration(self, timestamp_s: float) -> float:
        if self._open_started_at is None:
            return 0.0
        return max(0.0, timestamp_s - self._open_started_at)

    def _open_snapshot(self, timestamp_s: float, yawn_just_started: bool) -> YawnSnapshot:
        duration = self._open_duration(timestamp_s)
        yawning = duration >= self.config.min_yawn_duration_s
        if yawning and not self._yawn_counted_this_opening:
            self._yawn_count += 1
            self._yawn_counted_this_opening = True
            yawn_just_started = True
        return self._snapshot(
            label=LABEL_YAWNING if yawning else LABEL_OPEN,
            mouth_open=True,
            yawning=yawning,
            yawn_just_started=yawn_just_started,
            open_duration_s=duration,
        )

    def _snapshot(
        self,
        label: str,
        mouth_open: bool,
        yawning: bool,
        yawn_just_started: bool,
        open_duration_s: float,
    ) -> YawnSnapshot:
        return YawnSnapshot(
            label=label,
            mouth_open=mouth_open,
            yawning=yawning,
            yawn_just_started=yawn_just_started,
            yawn_count=self._yawn_count,
            open_duration_s=open_duration_s,
        )
