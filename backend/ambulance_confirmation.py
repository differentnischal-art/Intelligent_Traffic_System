"""
Temporal confirmation for ambulance detections.

The detector may produce a high-confidence ambulance candidate for a single
frame. This tracker separates raw sightings from confirmed emergency priority
so one noisy frame cannot preempt the active signal.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import ceil
from typing import Iterable

from backend.config import (
    AMBULANCE_CONF,
    AMBULANCE_CONFIRMATION_RATIO,
    AMBULANCE_CONFIRMATION_SECONDS,
    AMBULANCE_MAX_CONFIRMATION_MISSES,
    AMBULANCE_MIN_CONFIRMATION_HITS,
)


NO_AMBULANCE = "NO_AMBULANCE"
POSSIBLE_AMBULANCE = "POSSIBLE_AMBULANCE"
CONFIRMED_AMBULANCE = "CONFIRMED_AMBULANCE"


@dataclass(frozen=True)
class AmbulanceConfirmation:
    state: str
    seen: bool
    confirmed: bool
    hit_count: int
    sample_count: int
    required_samples: int
    hit_ratio: float
    latest_confidence: float
    avg_confidence: float
    misses_since_detection: int


class AmbulanceConfirmationTracker:
    def __init__(
        self,
        fps: float,
        detection_interval: int,
        confirmation_seconds: float = AMBULANCE_CONFIRMATION_SECONDS,
        confirmation_ratio: float = AMBULANCE_CONFIRMATION_RATIO,
        min_hits: int = AMBULANCE_MIN_CONFIRMATION_HITS,
        max_confirmation_misses: int = AMBULANCE_MAX_CONFIRMATION_MISSES,
        min_confidence: float = AMBULANCE_CONF,
    ) -> None:
        samples_per_second = max(1.0, float(fps) / max(1, int(detection_interval)))
        self.required_samples = max(
            1,
            int(min_hits),
            int(ceil(samples_per_second * float(confirmation_seconds))),
        )
        self.confirmation_ratio = max(0.0, min(1.0, float(confirmation_ratio)))
        self.min_hits = max(1, int(min_hits))
        self.max_confirmation_misses = max(0, int(max_confirmation_misses))
        self.min_confidence = max(0.0, float(min_confidence))
        self._history = deque(maxlen=self.required_samples)
        self._misses_since_detection = self.required_samples

    def update(self, detected: bool, detections: Iterable[dict]) -> AmbulanceConfirmation:
        latest_confidence = _max_confidence(detections)
        is_reliable_hit = bool(detected) and latest_confidence >= self.min_confidence

        self._history.append((is_reliable_hit, latest_confidence if is_reliable_hit else 0.0))
        if is_reliable_hit:
            self._misses_since_detection = 0
        else:
            self._misses_since_detection += 1

        return self.snapshot()

    def snapshot(self) -> AmbulanceConfirmation:
        sample_count = len(self._history)
        hits = [confidence for hit, confidence in self._history if hit]
        hit_count = len(hits)
        hit_ratio = hit_count / sample_count if sample_count else 0.0
        avg_confidence = sum(hits) / hit_count if hit_count else 0.0
        latest_confidence = self._history[-1][1] if self._history else 0.0

        confirmed = (
            sample_count >= self.required_samples
            and hit_count >= self.min_hits
            and hit_ratio >= self.confirmation_ratio
            and avg_confidence >= self.min_confidence
            and self._misses_since_detection <= self.max_confirmation_misses
        )

        if confirmed:
            state = CONFIRMED_AMBULANCE
        elif hit_count > 0:
            state = POSSIBLE_AMBULANCE
        else:
            state = NO_AMBULANCE

        return AmbulanceConfirmation(
            state=state,
            seen=hit_count > 0,
            confirmed=confirmed,
            hit_count=hit_count,
            sample_count=sample_count,
            required_samples=self.required_samples,
            hit_ratio=round(hit_ratio, 3),
            latest_confidence=round(latest_confidence, 3),
            avg_confidence=round(avg_confidence, 3),
            misses_since_detection=self._misses_since_detection,
        )


def _max_confidence(detections: Iterable[dict]) -> float:
    max_confidence = 0.0
    for detection in detections:
        try:
            max_confidence = max(max_confidence, float(detection.get("conf", 0) or 0))
        except (AttributeError, TypeError, ValueError):
            continue
    return max_confidence
