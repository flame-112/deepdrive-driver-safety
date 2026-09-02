"""Experimental thresholds for yawn detection from Mouth Aspect Ratio.

Talking and smiling can also raise MAR. Duration is what separates a brief
mouth opening from a yawn candidate. These cutoffs are demo starting points.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class YawnConfig:
    """Hysteresis and duration gates for classifying a yawn.

    MAR *above* ``open_mar_threshold`` starts a mouth-open interval. MAR must
    fall *below* ``closed_mar_threshold`` before the mouth counts as closed
    again. A yawn is counted once the open interval lasts long enough.
    """

    open_mar_threshold: float = 0.60
    closed_mar_threshold: float = 0.45
    min_yawn_duration_s: float = 1.5

    def __post_init__(self) -> None:
        if self.open_mar_threshold <= self.closed_mar_threshold:
            raise ValueError("open_mar_threshold must be greater than closed_mar_threshold.")
        if self.min_yawn_duration_s <= 0:
            raise ValueError("min_yawn_duration_s must be positive.")


DEFAULT_YAWN_CONFIG = YawnConfig()
