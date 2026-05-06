import threading
import unittest

from backend.config import VEHICLE_DENSITY_WEIGHTS
from backend.decision_engine import LaneSnapshot, TrafficDecisionEngine, dynamic_green_time
from backend.signal_controller import (
    AMBULANCE_MIN_GREEN_SECONDS,
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
    }


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

        self.assertEqual(self.state["lane_3"]["signal"], GREEN)
        self.assertEqual(self.state["lane_3"]["timer"], AMBULANCE_MIN_GREEN_SECONDS)
        self.assertEqual(self.state["lane_2"]["signal"], RED)

        self.state["lane_4"]["ambulance"] = True
        self.state["lane_4"]["density"] = 99
        self.controller.step(now + 5)

        self.assertEqual(self.state["lane_3"]["signal"], GREEN)
        self.assertEqual(self.state["lane_4"]["signal"], RED)

        self.state["lane_3"]["ambulance"] = False
        self.state["lane_4"]["ambulance"] = False
        self.controller.step(now + 1 + AMBULANCE_MIN_GREEN_SECONDS)

        self.assertEqual(self.state["lane_3"]["signal"], YELLOW)


if __name__ == "__main__":
    unittest.main()
