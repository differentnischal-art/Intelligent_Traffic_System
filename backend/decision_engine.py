"""
Priority rules for the intelligent traffic controller.

Detection code should only publish lane facts. This module turns those facts
into the next lane decision: ambulance priority first, otherwise density.
"""

from dataclasses import dataclass
from typing import Iterable, Optional


RED = "RED"
YELLOW = "YELLOW"
GREEN = "GREEN"

YELLOW_SECONDS = 3
AMBULANCE_MIN_GREEN_SECONDS = 10
EMERGENCY_CLEARANCE_SECONDS = 4
CONTROL_TICK_SECONDS = 0.2

MIN_GREEN_SECONDS = 10
MAX_GREEN_SECONDS = 70
BASE_GREEN_SECONDS = 5
GREEN_SECONDS_PER_WEIGHTED_DENSITY = 2.0


@dataclass(frozen=True)
class LaneSnapshot:
    lane_id: str
    density: float
    ambulance: bool
    active: bool = True
    weighted_density: Optional[float] = None

    @classmethod
    def from_state(cls, lane_id: str, state: dict) -> "LaneSnapshot":
        density = _safe_density(state.get("stable_density", state.get("density", 0)))
        ambulance = bool(state.get("ambulance", False))
        if state.get("ambulance_state") is not None:
            ambulance = state.get("ambulance_state") == "CONFIRMED_AMBULANCE"
        elif state.get("ambulance_stable") is not None:
            ambulance = bool(state.get("ambulance_stable", False))
        return cls(
            lane_id=lane_id,
            density=density,
            ambulance=ambulance,
            active=bool(state.get("active", False)),
            weighted_density=_safe_density(state.get("weighted_density", density)),
        )


@dataclass(frozen=True)
class SignalDecision:
    lane_id: str
    green_seconds: int
    reason: str
    density: float
    ambulance_priority: bool = False
    weighted_density: Optional[float] = None
    selection_reason: Optional[str] = None


def dynamic_green_time(density: float) -> int:
    """
    Map smoothed weighted density to a bounded real-time green duration.

    This intentionally uses one continuous linear scale instead of hard
    density bands, so a small density change only creates a small timer change.
    """
    density = _safe_density(density)
    green_seconds = BASE_GREEN_SECONDS + (density * GREEN_SECONDS_PER_WEIGHTED_DENSITY)
    return _clamp_seconds(green_seconds, MIN_GREEN_SECONDS, MAX_GREEN_SECONDS)


class TrafficDecisionEngine:
    """
    Chooses the next lane only at controller decision boundaries.

    The most recently served lane is skipped for one normal density cycle when
    other active lanes are available. That prevents greedy reselection without
    turning the controller into a fixed rotation. Ambulance priority still wins.
    """

    def __init__(self) -> None:
        self._last_served_at: dict[str, float] = {}
        self._cooldown_lane_id: Optional[str] = None
        self._served_round: set[str] = set()

    def choose_next(
        self,
        lanes: Iterable[LaneSnapshot],
        now: float,
    ) -> Optional[SignalDecision]:
        active_lanes = [lane for lane in lanes if lane.active]
        if not active_lanes:
            return None

        ambulance_lanes = [lane for lane in active_lanes if lane.ambulance]
        if ambulance_lanes:
            chosen = max(ambulance_lanes, key=lambda lane: self._priority_key(lane, now))
            return SignalDecision(
                lane_id=chosen.lane_id,
                green_seconds=AMBULANCE_MIN_GREEN_SECONDS,
                reason="ambulance_priority",
                density=chosen.density,
                ambulance_priority=True,
                weighted_density=chosen.weighted_density,
                selection_reason="ambulance_priority",
            )

        density_candidates = self._apply_fair_rotation(active_lanes)
        density_candidates = self._apply_density_cooldown(density_candidates)
        chosen = max(density_candidates, key=lambda lane: self._priority_key(lane, now))
        selection_reason = self._density_selection_reason(active_lanes, chosen, now)
        return SignalDecision(
            lane_id=chosen.lane_id,
            green_seconds=dynamic_green_time(chosen.density),
            reason="highest_density",
            density=chosen.density,
            weighted_density=chosen.weighted_density,
            selection_reason=selection_reason,
        )

    def mark_served(self, lane_id: str, now: float) -> None:
        self._last_served_at[lane_id] = now
        self._cooldown_lane_id = lane_id
        self._served_round.add(lane_id)

    def forget_missing(self, lane_ids: Iterable[str]) -> None:
        present = set(lane_ids)
        for lane_id in list(self._last_served_at):
            if lane_id not in present:
                del self._last_served_at[lane_id]
        self._served_round.intersection_update(present)
        if self._cooldown_lane_id not in present:
            self._cooldown_lane_id = None

    def reset(self) -> None:
        self._last_served_at.clear()
        self._cooldown_lane_id = None
        self._served_round.clear()

    def _apply_fair_rotation(self, lanes: list[LaneSnapshot]) -> list[LaneSnapshot]:
        if len(lanes) <= 1:
            return lanes

        active_ids = {lane.lane_id for lane in lanes}
        if active_ids and active_ids.issubset(self._served_round):
            self._served_round.clear()

        unserved = [lane for lane in lanes if lane.lane_id not in self._served_round]
        return unserved or lanes

    def _apply_density_cooldown(self, lanes: list[LaneSnapshot]) -> list[LaneSnapshot]:
        if not self._cooldown_lane_id or len(lanes) <= 1:
            return lanes

        candidates = [lane for lane in lanes if lane.lane_id != self._cooldown_lane_id]
        return candidates or lanes

    def _priority_key(self, lane: LaneSnapshot, now: float) -> tuple[float, float, int]:
        last_served = self._last_served_at.get(lane.lane_id)
        waited_seconds = 1_000_000_000.0 if last_served is None else max(0.0, now - last_served)
        return (lane.density, waited_seconds, -_lane_number(lane.lane_id))

    def _density_selection_reason(
        self,
        active_lanes: list[LaneSnapshot],
        chosen: LaneSnapshot,
        now: float,
    ) -> str:
        highest_density_lane = max(active_lanes, key=lambda lane: self._priority_key(lane, now))
        if highest_density_lane.lane_id == chosen.lane_id:
            return "highest_density"
        return "fairness"


def _clamp_seconds(value: float, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, int(round(value))))


def _lane_number(lane_id: str) -> int:
    suffix = lane_id.rsplit("_", 1)[-1]
    try:
        return int(suffix)
    except (TypeError, ValueError):
        return 10_000


def _safe_density(value) -> float:
    try:
        return max(0.0, float(value or 0))
    except (TypeError, ValueError):
        return 0.0
