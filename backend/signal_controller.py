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

from backend.decision_engine import (
    AMBULANCE_MIN_GREEN_SECONDS,
    CONTROL_TICK_SECONDS,
    GREEN,
    RED,
    YELLOW,
    YELLOW_SECONDS,
    LaneSnapshot,
    SignalDecision,
    TrafficDecisionEngine,
    dynamic_green_time,
)


@dataclass
class SignalPhase:
    lane_id: Optional[str] = None
    state: str = RED
    started_at: float = 0.0
    ends_at: float = 0.0
    duration: int = 0
    reason: str = "idle"
    ambulance_lock_until: float = 0.0


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
            })

    def _publish_phase_locked(self, now: float) -> None:
        remaining = _seconds_remaining(self._phase.ends_at, now)

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
            })

    def _start_green_locked(self, decision: SignalDecision, now: float) -> None:
        duration = int(decision.green_seconds)
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
                    self._start_green_locked(ambulance_decision, now)
                    return

                self._publish_phase_locked(now)
                return

            if self._phase.state == GREEN:
                self._start_yellow_locked(now)
                return

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


def _seconds_remaining(ends_at: float, now: float) -> int:
    return max(0, int(math.ceil(ends_at - now)))


__all__ = [
    "SmartSignalController",
    "SignalPhase",
    "RED",
    "YELLOW",
    "GREEN",
    "YELLOW_SECONDS",
    "AMBULANCE_MIN_GREEN_SECONDS",
    "dynamic_green_time",
]
