"""
Time-based smart traffic signal state machine.

Lane workers publish density, ambulance, and frame data. This controller owns
the signal phase, countdown timer, ambulance preemption, and JSON-facing state.
"""

import math
import threading
import time
from dataclasses import dataclass
from typing import Optional

from backend.ambulance_confirmation import CONFIRMED_AMBULANCE, POSSIBLE_AMBULANCE
from backend.decision_engine import (
    AMBULANCE_MIN_GREEN_SECONDS,
    CONTROL_TICK_SECONDS,
    EMERGENCY_CLEARANCE_SECONDS,
    GREEN,
    RED,
    YELLOW,
    YELLOW_SECONDS,
    LaneSnapshot,
    SignalDecision,
    TrafficDecisionEngine,
    dynamic_green_time,
)


NORMAL_TRAFFIC = "NORMAL_TRAFFIC"
EMERGENCY_CLEARANCE_MODE = "EMERGENCY_CLEARANCE_MODE"
AMBULANCE_PRIORITY_ACTIVE = "AMBULANCE_PRIORITY_ACTIVE"


@dataclass(frozen=True)
class DecisionSnapshot:
    selected_lane_id: str
    stable_density: float
    weighted_density: Optional[float]
    assigned_green_time: int
    decision_timestamp: str
    decision_timestamp_epoch: float
    decision_reason: str
    decision_reason_code: str

    @classmethod
    def from_decision(cls, decision: SignalDecision, now: float) -> "DecisionSnapshot":
        reason_code = decision.selection_reason or decision.reason
        return cls(
            selected_lane_id=decision.lane_id,
            stable_density=_snapshot_number(decision.density),
            weighted_density=_snapshot_number(
                decision.weighted_density
                if decision.weighted_density is not None
                else decision.density
            ),
            assigned_green_time=int(decision.green_seconds),
            decision_timestamp=time.strftime("%H:%M:%S", time.localtime(now)),
            decision_timestamp_epoch=now,
            decision_reason=_decision_reason_label(reason_code),
            decision_reason_code=reason_code,
        )

    def as_dict(self) -> dict:
        return {
            "selected_lane_id": self.selected_lane_id,
            "lane_id": self.selected_lane_id,
            "decision_density": self.stable_density,
            "stable_density": self.stable_density,
            "weighted_density": self.weighted_density,
            "assigned_green_time": self.assigned_green_time,
            "assigned_green_time_seconds": self.assigned_green_time,
            "decision_timestamp": self.decision_timestamp,
            "decision_timestamp_epoch": self.decision_timestamp_epoch,
            "decision_reason": self.decision_reason,
            "decision_reason_code": self.decision_reason_code,
        }


@dataclass
class SignalPhase:
    lane_id: Optional[str] = None
    state: str = RED
    started_at: float = 0.0
    ends_at: float = 0.0
    duration: int = 0
    reason: str = "idle"
    ambulance_lock_until: float = 0.0
    decision_snapshot: Optional[DecisionSnapshot] = None
    pending_ambulance_decision: Optional[SignalDecision] = None


class SmartSignalController:
    def __init__(
        self,
        shared_state: dict,
        state_lock: threading.Lock,
        decision_engine: Optional[TrafficDecisionEngine] = None,
        tick_seconds: float = CONTROL_TICK_SECONDS,
    ) -> None:
        self.shared_state = shared_state
        self.state_lock = state_lock
        self.decision_engine = decision_engine or TrafficDecisionEngine()
        self.tick_seconds = tick_seconds
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._phase = SignalPhase()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="smart-signal-controller",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2)
        self._thread = None

    def reset(self) -> None:
        with self.state_lock:
            self._phase = SignalPhase()
            self.decision_engine.reset()
            self._set_all_red_locked("idle")

    def current_decision_snapshot(self) -> Optional[dict]:
        snapshot = self._phase.decision_snapshot
        return snapshot.as_dict() if snapshot else None

    def current_emergency_status(self) -> dict:
        return self._emergency_status_locked(time.time())

    def step(self, now: Optional[float] = None) -> None:
        """Run one deterministic control step; useful for tests and simulations."""
        with self.state_lock:
            self._step_locked(time.time() if now is None else now)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            self.step()
            time.sleep(self.tick_seconds)

    def _snapshots_locked(self) -> list[LaneSnapshot]:
        return [
            LaneSnapshot.from_state(lane_id, lane)
            for lane_id, lane in self.shared_state.items()
            if isinstance(lane, dict)
        ]

    def _set_all_red_locked(self, reason: str) -> None:
        for lane_id, lane in self.shared_state.items():
            if not isinstance(lane, dict):
                continue
            lane.update({
                "lane_id": lane_id,
                "signal": RED,
                "signal_state": RED,
                "timer": 0,
                "green_time": 0,
                "active_phase": RED,
                "controller_reason": reason,
                "phase_duration": 0,
                "decision_snapshot": None,
                "emergency_state": NORMAL_TRAFFIC,
                "emergency_mode": False,
                "emergency_lane_id": None,
                "emergency_message": "",
            })

    def _publish_phase_locked(self, now: float) -> None:
        remaining = _seconds_remaining(self._phase.ends_at, now)
        emergency_status = self._emergency_status_locked(now)
        decision_snapshot = (
            self._phase.decision_snapshot.as_dict()
            if self._phase.decision_snapshot
            else None
        )

        for lane_id, lane in self.shared_state.items():
            if not isinstance(lane, dict):
                continue

            active = bool(lane.get("active", False))
            is_current = active and lane_id == self._phase.lane_id
            signal = self._phase.state if is_current else RED
            reason = self._phase.reason if is_current else ("inactive" if not active else "not_selected")

            lane.update({
                "lane_id": lane_id,
                "signal": signal,
                "signal_state": signal,
                "timer": remaining if is_current else 0,
                "green_time": self._phase.duration if is_current and signal == GREEN else 0,
                "active_phase": self._phase.state if is_current else RED,
                "controller_reason": reason,
                "phase_duration": self._phase.duration if is_current else 0,
                "decision_snapshot": dict(decision_snapshot)
                if is_current and decision_snapshot
                else None,
                "emergency_state": emergency_status["state"],
                "emergency_mode": emergency_status["active"],
                "emergency_lane_id": emergency_status["lane_id"],
                "emergency_message": emergency_status["message"],
            })

    def _start_green_locked(self, decision: SignalDecision, now: float) -> None:
        duration = int(decision.green_seconds)
        decision_snapshot = DecisionSnapshot.from_decision(decision, now)
        self._phase = SignalPhase(
            lane_id=decision.lane_id,
            state=GREEN,
            started_at=now,
            ends_at=now + duration,
            duration=duration,
            reason=decision.reason,
            ambulance_lock_until=now + AMBULANCE_MIN_GREEN_SECONDS
            if decision.ambulance_priority
            else 0.0,
            decision_snapshot=decision_snapshot,
        )
        self.decision_engine.mark_served(decision.lane_id, now)
        self._publish_phase_locked(now)

    def _start_yellow_locked(self, now: float) -> None:
        if not self._phase.lane_id:
            self._phase = SignalPhase()
            self._set_all_red_locked("idle")
            return

        self._phase = SignalPhase(
            lane_id=self._phase.lane_id,
            state=YELLOW,
            started_at=now,
            ends_at=now + YELLOW_SECONDS,
            duration=YELLOW_SECONDS,
            reason="transition",
            decision_snapshot=self._phase.decision_snapshot,
            pending_ambulance_decision=self._phase.pending_ambulance_decision,
        )
        self._publish_phase_locked(now)

    def _start_emergency_clearance_locked(
        self,
        ambulance_decision: SignalDecision,
        now: float,
    ) -> None:
        self._phase = SignalPhase(
            lane_id=self._phase.lane_id,
            state=GREEN,
            started_at=now,
            ends_at=now + EMERGENCY_CLEARANCE_SECONDS,
            duration=EMERGENCY_CLEARANCE_SECONDS,
            reason="emergency_clearance",
            decision_snapshot=self._phase.decision_snapshot,
            pending_ambulance_decision=ambulance_decision,
        )
        self._publish_phase_locked(now)

    def _step_locked(self, now: float) -> None:
        snapshots = self._snapshots_locked()
        self.decision_engine.forget_missing(lane.lane_id for lane in snapshots)

        active_lanes = [lane for lane in snapshots if lane.active]
        if not active_lanes:
            self._phase = SignalPhase()
            self._set_all_red_locked("idle")
            return

        current_lane_active = any(
            lane.lane_id == self._phase.lane_id and lane.active for lane in active_lanes
        )

        if current_lane_active:
            if now < self._phase.ends_at:
                ambulance_decision = self._ambulance_preemption(active_lanes, now)
                if ambulance_decision:
                    self._start_emergency_clearance_locked(ambulance_decision, now)
                    return

                self._publish_phase_locked(now)
                return

            if self._phase.state == GREEN:
                self._start_yellow_locked(now)
                return

        pending_decision = self._phase.pending_ambulance_decision
        if pending_decision:
            if _lane_is_active(pending_decision.lane_id, active_lanes):
                self._start_green_locked(pending_decision, now)
                return

            self._phase = SignalPhase()

        decision = self.decision_engine.choose_next(active_lanes, now)
        if decision:
            self._start_green_locked(decision, now)
            return

        self._phase = SignalPhase()
        self._set_all_red_locked("idle")

    def _ambulance_preemption(
        self,
        active_lanes: list[LaneSnapshot],
        now: float,
    ) -> Optional[SignalDecision]:
        if (
            self._phase.state != GREEN
            or self._phase.reason == "ambulance_priority"
            or self._phase.reason == "emergency_clearance"
            or self._ambulance_lock_active(now, True)
        ):
            return None

        ambulance_lanes = [
            lane
            for lane in active_lanes
            if lane.ambulance and lane.lane_id != self._phase.lane_id
        ]
        if not ambulance_lanes:
            return None

        return self.decision_engine.choose_next(ambulance_lanes, now)

    def _ambulance_lock_active(self, now: float, current_lane_active: bool) -> bool:
        return (
            current_lane_active
            and self._phase.state == GREEN
            and self._phase.reason == "ambulance_priority"
            and now < self._phase.ambulance_lock_until
        )

    def _emergency_status_locked(self, now: float) -> dict:
        pending_decision = self._phase.pending_ambulance_decision
        if pending_decision:
            return _emergency_status(
                EMERGENCY_CLEARANCE_MODE,
                pending_decision.lane_id,
                "Emergency Vehicle Detected - Preparing Intersection Clearance...",
                active=True,
                remaining=_seconds_remaining(self._phase.ends_at, now),
            )

        if self._phase.reason == "ambulance_priority":
            return _emergency_status(
                AMBULANCE_PRIORITY_ACTIVE,
                self._phase.lane_id,
                "Ambulance Priority Active",
                active=True,
                remaining=_seconds_remaining(self._phase.ends_at, now),
            )

        if (
            self._phase.state == YELLOW
            and self._phase.decision_snapshot
            and self._phase.decision_snapshot.decision_reason_code == "ambulance_priority"
        ):
            return _emergency_status(
                AMBULANCE_PRIORITY_ACTIVE,
                self._phase.lane_id,
                "Ambulance Priority Active",
                active=True,
                remaining=_seconds_remaining(self._phase.ends_at, now),
            )

        confirmed_lane_id = self._lane_with_ambulance_state(CONFIRMED_AMBULANCE)
        if confirmed_lane_id:
            return _emergency_status(
                CONFIRMED_AMBULANCE,
                confirmed_lane_id,
                "Emergency Vehicle Detected",
                active=True,
                remaining=0,
            )

        possible_lane_id = self._lane_with_ambulance_state(POSSIBLE_AMBULANCE)
        if possible_lane_id:
            return _emergency_status(
                POSSIBLE_AMBULANCE,
                possible_lane_id,
                "",
                active=False,
                remaining=0,
            )

        return _emergency_status(NORMAL_TRAFFIC, None, "", active=False, remaining=0)

    def _lane_with_ambulance_state(self, ambulance_state: str) -> Optional[str]:
        for lane_id, lane in self.shared_state.items():
            if not isinstance(lane, dict) or not lane.get("active", False):
                continue
            if lane.get("ambulance_state") == ambulance_state:
                return lane_id
            if ambulance_state == CONFIRMED_AMBULANCE and lane.get("ambulance"):
                return lane_id
        return None


def _seconds_remaining(ends_at: float, now: float) -> int:
    return max(0, int(math.ceil(ends_at - now)))


def _lane_is_active(lane_id: str, active_lanes: list[LaneSnapshot]) -> bool:
    return any(lane.lane_id == lane_id and lane.active for lane in active_lanes)


def _emergency_status(
    state: str,
    lane_id: Optional[str],
    message: str,
    active: bool,
    remaining: int,
) -> dict:
    return {
        "state": state,
        "active": active,
        "lane_id": lane_id,
        "message": message,
        "remaining_seconds": remaining,
    }


def _decision_reason_label(reason_code: str) -> str:
    return {
        "highest_density": "Highest Density",
        "ambulance_priority": "Ambulance Priority",
        "emergency_clearance": "Emergency Clearance",
        "fairness": "Fairness",
    }.get(reason_code, str(reason_code).replace("_", " ").title())


def _snapshot_number(value) -> Optional[float]:
    if value is None:
        return None

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    if number.is_integer():
        return int(number)
    return round(number, 2)


__all__ = [
    "SmartSignalController",
    "DecisionSnapshot",
    "SignalPhase",
    "RED",
    "YELLOW",
    "GREEN",
    "YELLOW_SECONDS",
    "AMBULANCE_MIN_GREEN_SECONDS",
    "EMERGENCY_CLEARANCE_SECONDS",
    "NORMAL_TRAFFIC",
    "POSSIBLE_AMBULANCE",
    "CONFIRMED_AMBULANCE",
    "EMERGENCY_CLEARANCE_MODE",
    "AMBULANCE_PRIORITY_ACTIVE",
    "dynamic_green_time",
]
