import threading
import unittest

from backend.ambulance_confirmation import (
    CONFIRMED_AMBULANCE,
    POSSIBLE_AMBULANCE,
    AmbulanceConfirmationTracker,
)
from backend.config import AMBULANCE_CONF, VEHICLE_DENSITY_WEIGHTS
from backend.decision_engine import LaneSnapshot, TrafficDecisionEngine, dynamic_green_time
from backend.detector import detect_ambulance
from backend.signal_controller import (
    AMBULANCE_MIN_GREEN_SECONDS,
    AMBULANCE_PRIORITY_ACTIVE,
    EMERGENCY_CLEARANCE_MODE,
    EMERGENCY_CLEARANCE_SECONDS,
    GREEN,
    RED,
    YELLOW,
    YELLOW_SECONDS,
    SmartSignalController,
)


def lane_state(density=0, ambulance=False, active=True):
    return {
        "lane_id": "",
        "density": density,
        "count": density,
        "ambulance": ambulance,
        "active": active,
        "signal": RED,
        "signal_state": RED,
        "timer": 0,
        "green_time": 0,
        "decision_snapshot": None,
    }


class _FakeScalar:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class _FakeBox:
    def __init__(self, cls_id, confidence):
        self.cls = _FakeScalar(cls_id)
        self.conf = _FakeScalar(confidence)
        self.xyxy = [_FakeXyxy([1, 2, 30, 40])]


class _FakeXyxy:
    def __init__(self, values):
        self.values = values

    def tolist(self):
        return self.values


class _FakePrediction:
    def __init__(self, boxes):
        self.boxes = boxes
        self.names = {0: "car", 1: "ambulance"}


class DecisionEngineTests(unittest.TestCase):
    def test_dynamic_green_time_scales_smoothly(self):
        self.assertEqual(dynamic_green_time(0), 10)
        self.assertEqual(dynamic_green_time(5), 15)
        self.assertEqual(dynamic_green_time(6), 17)
        self.assertEqual(dynamic_green_time(7), 19)
        self.assertEqual(dynamic_green_time(8), 21)
        self.assertEqual(dynamic_green_time(40), 70)
        self.assertEqual(dynamic_green_time(100), 70)

    def test_vehicle_density_weights_match_traffic_load(self):
        self.assertEqual(VEHICLE_DENSITY_WEIGHTS["bike"], 1)
        self.assertEqual(VEHICLE_DENSITY_WEIGHTS["motorcycle"], 1)
        self.assertEqual(VEHICLE_DENSITY_WEIGHTS["car"], 2)
        self.assertEqual(VEHICLE_DENSITY_WEIGHTS["bus"], 5)
        self.assertEqual(VEHICLE_DENSITY_WEIGHTS["truck"], 5)

    def test_dynamic_green_time_is_deterministic_for_same_density(self):
        values = [dynamic_green_time(5) for _ in range(5)]
        self.assertEqual(values, [15, 15, 15, 15, 15])

    def test_lane_snapshot_prefers_stable_density(self):
        snapshot = LaneSnapshot.from_state(
            "lane_1",
            {
                "density": 12,
                "stable_density": 5,
                "ambulance": False,
                "active": True,
            },
        )

        self.assertEqual(snapshot.density, 5)

    def test_lane_snapshot_ignores_possible_ambulance_state(self):
        snapshot = LaneSnapshot.from_state(
            "lane_1",
            {
                "density": 12,
                "ambulance": True,
                "ambulance_seen": True,
                "ambulance_stable": False,
                "ambulance_state": POSSIBLE_AMBULANCE,
                "active": True,
            },
        )

        self.assertFalse(snapshot.ambulance)

    def test_ambulance_priority_beats_density(self):
        engine = TrafficDecisionEngine()
        decision = engine.choose_next(
            [
                LaneSnapshot("lane_1", density=35, ambulance=False),
                LaneSnapshot("lane_2", density=2, ambulance=True),
            ],
            now=100.0,
        )

        self.assertEqual(decision.lane_id, "lane_2")
        self.assertTrue(decision.ambulance_priority)
        self.assertEqual(decision.green_seconds, AMBULANCE_MIN_GREEN_SECONDS)


class AmbulanceConfirmationTests(unittest.TestCase):
    def test_detector_filters_low_confidence_and_non_ambulance_classes(self):
        low_confidence = max(0.0, AMBULANCE_CONF - 0.01)
        high_confidence = max(AMBULANCE_CONF, 0.95)

        seen, detections = detect_ambulance([
            _FakePrediction([
                _FakeBox(0, 0.99),
                _FakeBox(1, low_confidence),
                _FakeBox(1, high_confidence),
            ])
        ])

        self.assertTrue(seen)
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0]["label"], "ambulance")

    def test_single_frame_detection_stays_possible_only(self):
        tracker = AmbulanceConfirmationTracker(
            fps=10,
            detection_interval=1,
            confirmation_seconds=1.0,
            confirmation_ratio=0.70,
            min_hits=3,
        )

        state = tracker.update(True, [{"conf": 0.95}])
        self.assertEqual(state.state, POSSIBLE_AMBULANCE)
        self.assertFalse(state.confirmed)

        for _ in range(tracker.required_samples):
            state = tracker.update(False, [])

        self.assertFalse(state.confirmed)

    def test_majority_window_confirms_persistent_detection(self):
        tracker = AmbulanceConfirmationTracker(
            fps=6,
            detection_interval=1,
            confirmation_seconds=1.0,
            confirmation_ratio=0.70,
            min_hits=3,
        )

        state = None
        for detected in [True, True, True, False, True, True]:
            detections = [{"conf": 0.88}] if detected else []
            state = tracker.update(detected, detections)

        self.assertEqual(state.state, CONFIRMED_AMBULANCE)
        self.assertTrue(state.confirmed)


class SmartSignalControllerTests(unittest.TestCase):
    def setUp(self):
        self.state = {
            "lane_1": lane_state(density=4),
            "lane_2": lane_state(density=22),
            "lane_3": lane_state(density=8),
            "lane_4": lane_state(density=1),
        }
        for lane_id, lane in self.state.items():
            lane["lane_id"] = lane_id
        self.controller = SmartSignalController(self.state, threading.Lock())

    def test_selects_highest_density_and_transitions_to_yellow(self):
        now = 100.0
        duration = dynamic_green_time(22)

        self.controller.step(now)

        self.assertEqual(self.state["lane_2"]["signal"], GREEN)
        self.assertEqual(self.state["lane_2"]["timer"], duration)
        self.assertEqual(self.state["lane_1"]["signal"], RED)
        self.assertEqual(self.state["lane_3"]["signal"], RED)
        self.assertEqual(self.state["lane_4"]["signal"], RED)

        self.controller.step(now + duration)

        self.assertEqual(self.state["lane_2"]["signal"], YELLOW)
        self.assertEqual(self.state["lane_2"]["timer"], YELLOW_SECONDS)
        self.assertEqual(self.state["lane_1"]["signal"], RED)

    def test_density_changes_do_not_reselect_during_active_cycle(self):
        now = 150.0
        self.controller.step(now)

        self.state["lane_1"]["density"] = 100
        self.controller.step(now + 1)

        self.assertEqual(self.state["lane_2"]["signal"], GREEN)
        self.assertEqual(self.state["lane_1"]["signal"], RED)

    def test_possible_ambulance_does_not_preempt_active_green(self):
        now = 155.0
        self.controller.step(now)
        self.assertEqual(self.state["lane_2"]["signal"], GREEN)

        self.state["lane_3"]["ambulance"] = True
        self.state["lane_3"]["ambulance_seen"] = True
        self.state["lane_3"]["ambulance_stable"] = False
        self.state["lane_3"]["ambulance_state"] = POSSIBLE_AMBULANCE
        self.controller.step(now + 1)

        self.assertEqual(self.state["lane_2"]["signal"], GREEN)
        self.assertEqual(self.state["lane_3"]["signal"], RED)

        self.state["lane_3"]["ambulance_stable"] = True
        self.state["lane_3"]["ambulance_state"] = CONFIRMED_AMBULANCE
        self.controller.step(now + 2)

        self.assertEqual(self.state["lane_2"]["signal"], GREEN)
        self.assertEqual(self.state["lane_2"]["controller_reason"], "emergency_clearance")
        self.assertEqual(self.state["lane_2"]["timer"], EMERGENCY_CLEARANCE_SECONDS)
        self.assertEqual(self.state["lane_2"]["emergency_state"], EMERGENCY_CLEARANCE_MODE)
        self.assertEqual(self.state["lane_3"]["signal"], RED)

        self.controller.step(now + 2 + EMERGENCY_CLEARANCE_SECONDS)
        self.assertEqual(self.state["lane_2"]["signal"], YELLOW)
        self.assertEqual(self.state["lane_2"]["timer"], YELLOW_SECONDS)

        self.controller.step(now + 2 + EMERGENCY_CLEARANCE_SECONDS + YELLOW_SECONDS)
        self.assertEqual(self.state["lane_3"]["signal"], GREEN)
        self.assertEqual(self.state["lane_3"]["emergency_state"], AMBULANCE_PRIORITY_ACTIVE)

    def test_decision_snapshot_locks_density_for_green_and_yellow(self):
        now = 160.0
        duration = dynamic_green_time(22)
        self.state["lane_2"]["stable_density"] = 22
        self.state["lane_2"]["weighted_density"] = 22

        self.controller.step(now)

        snapshot = dict(self.state["lane_2"]["decision_snapshot"])
        self.assertEqual(snapshot["selected_lane_id"], "lane_2")
        self.assertEqual(snapshot["decision_density"], 22)
        self.assertEqual(snapshot["weighted_density"], 22)
        self.assertEqual(snapshot["assigned_green_time"], duration)
        self.assertEqual(snapshot["decision_reason_code"], "highest_density")
        self.assertEqual(snapshot["decision_reason"], "Highest Density")
        self.assertIn("decision_timestamp", snapshot)

        self.state["lane_2"]["density"] = 1
        self.state["lane_2"]["stable_density"] = 1
        self.state["lane_2"]["weighted_density"] = 1
        self.controller.step(now + 1)

        self.assertEqual(self.state["lane_2"]["timer"], duration - 1)
        self.assertEqual(self.state["lane_2"]["decision_snapshot"], snapshot)

        self.controller.step(now + duration)

        self.assertEqual(self.state["lane_2"]["signal"], YELLOW)
        self.assertEqual(self.state["lane_2"]["decision_snapshot"], snapshot)

    def test_decision_snapshot_marks_fairness_selection(self):
        now = 170.0
        first_duration = dynamic_green_time(22)

        self.controller.step(now)
        self.controller.step(now + first_duration)
        self.controller.step(now + first_duration + YELLOW_SECONDS)

        snapshot = self.state["lane_3"]["decision_snapshot"]
        self.assertEqual(self.state["lane_3"]["signal"], GREEN)
        self.assertEqual(snapshot["selected_lane_id"], "lane_3")
        self.assertEqual(snapshot["decision_reason_code"], "fairness")
        self.assertEqual(snapshot["decision_reason"], "Fairness")

    def test_recent_lane_is_skipped_for_next_density_cycle(self):
        now = 175.0
        first_duration = dynamic_green_time(22)

        self.controller.step(now)
        self.assertEqual(self.state["lane_2"]["signal"], GREEN)

        self.controller.step(now + first_duration)
        self.assertEqual(self.state["lane_2"]["signal"], YELLOW)

        self.controller.step(now + first_duration + YELLOW_SECONDS)

        self.assertEqual(self.state["lane_3"]["signal"], GREEN)
        self.assertEqual(self.state["lane_2"]["signal"], RED)

    def test_density_rotation_serves_all_active_lanes_before_repeat(self):
        now = 180.0
        self.state["lane_2"]["density"] = 100
        self.state["lane_3"]["density"] = 80

        self.controller.step(now)
        self.assertEqual(self.state["lane_2"]["signal"], GREEN)

        now += dynamic_green_time(100)
        self.controller.step(now)
        now += YELLOW_SECONDS
        self.controller.step(now)
        self.assertEqual(self.state["lane_3"]["signal"], GREEN)

        now += dynamic_green_time(80)
        self.controller.step(now)
        now += YELLOW_SECONDS
        self.controller.step(now)
        self.assertEqual(self.state["lane_1"]["signal"], GREEN)

        now += dynamic_green_time(4)
        self.controller.step(now)
        now += YELLOW_SECONDS
        self.controller.step(now)
        self.assertEqual(self.state["lane_4"]["signal"], GREEN)

        now += dynamic_green_time(1)
        self.controller.step(now)
        now += YELLOW_SECONDS
        self.controller.step(now)
        self.assertEqual(self.state["lane_2"]["signal"], GREEN)

    def test_density_changes_do_not_reselect_during_yellow(self):
        now = 190.0
        first_duration = dynamic_green_time(22)

        self.controller.step(now)
        self.controller.step(now + first_duration)

        self.state["lane_1"]["density"] = 100
        self.controller.step(now + first_duration + 1)

        self.assertEqual(self.state["lane_2"]["signal"], YELLOW)
        self.assertEqual(self.state["lane_1"]["signal"], RED)

    def test_ambulance_priority_does_not_reset_each_tick(self):
        now = 195.0
        self.state["lane_3"]["ambulance"] = True

        self.controller.step(now)
        self.controller.step(now + 1)

        self.assertEqual(self.state["lane_3"]["signal"], GREEN)
        self.assertEqual(self.state["lane_3"]["timer"], AMBULANCE_MIN_GREEN_SECONDS - 1)

    def test_ambulance_preempts_and_holds_minimum_green(self):
        now = 200.0
        self.controller.step(now)
        self.assertEqual(self.state["lane_2"]["signal"], GREEN)

        self.state["lane_3"]["ambulance"] = True
        self.controller.step(now + 1)

        self.assertEqual(self.state["lane_2"]["signal"], GREEN)
        self.assertEqual(self.state["lane_2"]["controller_reason"], "emergency_clearance")
        self.assertEqual(self.state["lane_2"]["timer"], EMERGENCY_CLEARANCE_SECONDS)
        self.assertEqual(self.state["lane_3"]["signal"], RED)

        self.state["lane_4"]["ambulance"] = True
        self.state["lane_4"]["density"] = 99
        self.controller.step(now + 2)

        self.assertEqual(self.state["lane_2"]["signal"], GREEN)
        self.assertEqual(self.state["lane_3"]["signal"], RED)
        self.assertEqual(self.state["lane_4"]["signal"], RED)

        self.controller.step(now + 1 + EMERGENCY_CLEARANCE_SECONDS)
        self.assertEqual(self.state["lane_2"]["signal"], YELLOW)
        self.assertEqual(self.state["lane_3"]["signal"], RED)

        self.controller.step(now + 1 + EMERGENCY_CLEARANCE_SECONDS + YELLOW_SECONDS)
        self.assertEqual(self.state["lane_3"]["signal"], GREEN)
        self.assertEqual(self.state["lane_3"]["timer"], AMBULANCE_MIN_GREEN_SECONDS)
        self.assertEqual(self.state["lane_2"]["signal"], RED)

        self.assertEqual(self.state["lane_3"]["signal"], GREEN)
        self.assertEqual(self.state["lane_4"]["signal"], RED)

        self.state["lane_3"]["ambulance"] = False
        self.state["lane_4"]["ambulance"] = False
        self.controller.step(
            now
            + 1
            + EMERGENCY_CLEARANCE_SECONDS
            + YELLOW_SECONDS
            + AMBULANCE_MIN_GREEN_SECONDS
        )

        self.assertEqual(self.state["lane_3"]["signal"], YELLOW)
        self.assertEqual(self.state["lane_3"]["emergency_state"], AMBULANCE_PRIORITY_ACTIVE)

        self.controller.step(
            now
            + 1
            + EMERGENCY_CLEARANCE_SECONDS
            + YELLOW_SECONDS
            + AMBULANCE_MIN_GREEN_SECONDS
            + YELLOW_SECONDS
        )
        self.assertNotEqual(self.state["lane_3"].get("emergency_state"), AMBULANCE_PRIORITY_ACTIVE)


if __name__ == "__main__":
    unittest.main()
