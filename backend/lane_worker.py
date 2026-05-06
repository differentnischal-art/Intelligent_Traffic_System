"""
Lane video worker used by backend.server.

Each uploaded video is treated as exactly one lane. Detection runs on the full
frame, updates one lane state, and exits when the video ends.
"""

from __future__ import annotations

from collections import deque
import threading
import time

import cv2

from backend.detector import (
    annotate_frame,
    count_vehicles,
    detect_ambulance,
    encode_frame,
)


AMBULANCE_STABLE_SECONDS = 9
MOVING_AVG_FRAMES = 10
DETECTION_EVERY_N_FRAMES = 3
PLAYBACK_SPEED_MULTIPLIER = 1.08
MODEL_INFERENCE_WAIT_SECONDS = 0.04
WORKER_ERROR_BACKOFF_SECONDS = 0.05


def _finish_lane(lane_id, shared_state, state_lock, stop_event, error=None):
    with state_lock:
        if lane_id in shared_state:
            shared_state[lane_id].update({
                "signal": "RED",
                "signal_state": "RED",
                "timer": 0,
                "green_time": 0,
                "ambulance": False,
                "ambulance_seen": False,
                "ambulance_stable": False,
                "frame": "",
                "active": False,
                "error": error,
                "controller_reason": "inactive",
            })
    stop_event.set()


def _video_fps(cap):
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1 or fps != fps:
        return 30
    return fps


def _predict(model, model_lock, frame):
    if model_lock is None:
        return model(frame, verbose=False)
    acquired = model_lock.acquire(timeout=MODEL_INFERENCE_WAIT_SECONDS)
    if not acquired:
        return None
    try:
        return model(frame, verbose=False)
    finally:
        model_lock.release()


def _stable_density(counts):
    """Smooth weighted density over recent detection samples."""
    if not counts:
        return 0
    return int(round(sum(counts) / len(counts)))


def _pace_frame(next_frame_at, frame_interval, stop_event):
    next_frame_at += frame_interval
    sleep_for = next_frame_at - time.perf_counter()
    if sleep_for > 0 and stop_event.wait(sleep_for):
        return next_frame_at, True
    if sleep_for < -frame_interval:
        next_frame_at = time.perf_counter()
    return next_frame_at, False


def _mark_lane_error(lane_id, shared_state, state_lock, error):
    with state_lock:
        if lane_id in shared_state:
            shared_state[lane_id].update({
                "lane_id": lane_id,
                "active": True,
                "error": error,
            })


def run_lane(
    lane_id,
    video_path,
    vehicle_model,
    ambulance_model,
    shared_state,
    state_lock,
    stop_event,
    vehicle_model_lock: threading.Lock | None = None,
    ambulance_model_lock: threading.Lock | None = None,
):
    print(f"[{lane_id}] Worker started -> {video_path}")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        _finish_lane(lane_id, shared_state, state_lock, stop_event, "Cannot open video")
        print(f"[{lane_id}] Cannot open video")
        return

    fps = _video_fps(cap)
    frame_interval = 1.0 / (fps * PLAYBACK_SPEED_MULTIPLIER)
    detection_interval = max(1, DETECTION_EVERY_N_FRAMES)
    ambulance_required_frames = max(1, int((fps / detection_interval) * AMBULANCE_STABLE_SECONDS))
    ambulance_detected_frames = 0
    density_window = deque(maxlen=MOVING_AVG_FRAMES)
    frame_index = 0
    next_frame_at = time.perf_counter()
    vehicle_count = 0
    weighted_density = 0
    density = 0
    vehicle_detections = []
    ambulance_seen = False
    ambulance_stable = False
    ambulance_detections = []

    try:
        while not stop_event.is_set():
            ok, frame = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_index = 0
                ok, frame = cap.read()
                if not ok:
                    _finish_lane(lane_id, shared_state, state_lock, stop_event, "Cannot read video frame")
                    print(f"[{lane_id}] Cannot read video frame")
                    break

            if frame is None or frame.size == 0:
                _finish_lane(lane_id, shared_state, state_lock, stop_event, "Invalid frame")
                print(f"[{lane_id}] Invalid OpenCV frame")
                break

            try:
                should_detect = frame_index % detection_interval == 0
                if should_detect:
                    vehicle_result = _predict(vehicle_model, vehicle_model_lock, frame)
                    ambulance_result = _predict(ambulance_model, ambulance_model_lock, frame)

                    if vehicle_result is not None:
                        vehicle_count, weighted_density, vehicle_detections = count_vehicles(vehicle_result)
                        density_window.append(weighted_density)
                        density = _stable_density(density_window)

                    if ambulance_result is not None:
                        ambulance_seen, ambulance_detections = detect_ambulance(ambulance_result)
                        if ambulance_seen:
                            ambulance_detected_frames += 1
                        else:
                            ambulance_detected_frames = 0

                        ambulance_stable = ambulance_detected_frames >= ambulance_required_frames

                lane_snapshot = {"signal": "RED", "green_time": 0, "timer": 0}
                with state_lock:
                    if lane_id in shared_state:
                        shared_state[lane_id].update({
                            "lane_id": lane_id,
                            "density": density,
                            "stable_density": density,
                            "weighted_density": density,
                            "raw_weighted_density": weighted_density,
                            "count": vehicle_count,
                            "ambulance": ambulance_seen,
                            "ambulance_seen": ambulance_seen,
                            "ambulance_stable": ambulance_stable,
                            "ambulance_streak": ambulance_detected_frames,
                            "active": True,
                            "error": None,
                        })
                        lane_snapshot = dict(shared_state[lane_id])

                signal = lane_snapshot.get("signal", "RED")
                remaining = lane_snapshot.get("timer", 0)

                annotated = annotate_frame(
                    frame.copy(),
                    vehicle_count=vehicle_count,
                    vehicle_detections=vehicle_detections,
                    ambulance_detections=ambulance_detections,
                    signal=signal,
                    density=density,
                    timer=remaining,
                    ambulance_stable=ambulance_stable,
                    ambulance_seen=ambulance_seen,
                    ambulance_streak=ambulance_detected_frames,
                    ambulance_required_frames=ambulance_required_frames,
                )
                encoded_frame = encode_frame(annotated)

                with state_lock:
                    if lane_id in shared_state:
                        shared_state[lane_id].update({
                            "frame": encoded_frame,
                            "active": True,
                            "error": None,
                        })
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                print(f"[{lane_id}] Frame processing error: {error}")
                _mark_lane_error(lane_id, shared_state, state_lock, error)
                try:
                    encoded_frame = encode_frame(frame)
                    with state_lock:
                        if lane_id in shared_state:
                            shared_state[lane_id].update({
                                "frame": encoded_frame,
                                "active": True,
                            })
                except Exception:
                    pass
                if stop_event.wait(WORKER_ERROR_BACKOFF_SECONDS):
                    break

            frame_index += 1
            next_frame_at, should_stop = _pace_frame(next_frame_at, frame_interval, stop_event)
            if should_stop:
                break
    finally:
        cap.release()
        _finish_lane(lane_id, shared_state, state_lock, stop_event)
        print(f"[{lane_id}] Worker stopped")
