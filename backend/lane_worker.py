"""
Lane video worker used by backend.server.

Each uploaded video is treated as exactly one lane. Detection runs on the full
frame, updates one lane state, and loops the source until the lane is stopped.
"""

from __future__ import annotations

from collections import deque
import logging
import os
from pathlib import Path
import threading
import time

import cv2

from backend.ambulance_confirmation import (
    AmbulanceConfirmationTracker,
    NO_AMBULANCE,
)
from backend.detector import (
    annotate_frame,
    count_vehicles,
    detect_ambulance,
    encode_frame,
)


logger = logging.getLogger(__name__)


def _is_render() -> bool:
    return os.getenv("RENDER", "").lower() == "true" or bool(os.getenv("RENDER_SERVICE_ID"))


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


IS_RENDER = _is_render()

MOVING_AVG_FRAMES = 10
DETECTION_EVERY_N_FRAMES = max(
    1,
    _env_int("ITMS_DETECTION_EVERY_N_FRAMES", 5 if IS_RENDER else 3),
)
PLAYBACK_SPEED_MULTIPLIER = max(
    0.1,
    _env_float("ITMS_PLAYBACK_SPEED_MULTIPLIER", 1.0 if IS_RENDER else 1.08),
)
MODEL_INFERENCE_WAIT_SECONDS = max(
    0.0,
    _env_float("ITMS_MODEL_INFERENCE_WAIT_SECONDS", 0.04),
)
WORKER_ERROR_BACKOFF_SECONDS = max(
    0.01,
    _env_float("ITMS_WORKER_ERROR_BACKOFF_SECONDS", 0.05),
)
CAPTURE_REOPEN_BACKOFF_SECONDS = max(
    0.25,
    _env_float("ITMS_CAPTURE_REOPEN_BACKOFF_SECONDS", 1.0 if IS_RENDER else 0.5),
)
MAX_STREAM_FPS = max(
    0.0,
    _env_float("ITMS_MAX_STREAM_FPS", 12.0 if IS_RENDER else 0.0),
)
MODEL_IMGSZ = max(
    0,
    _env_int("ITMS_MODEL_IMGSZ", 480 if IS_RENDER else 640),
)


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
                "ambulance_confirmed": False,
                "ambulance_state": NO_AMBULANCE,
                "ambulance_streak": 0,
                "ambulance_required_frames": 0,
                "ambulance_confidence": 0.0,
                "ambulance_avg_confidence": 0.0,
                "ambulance_hit_ratio": 0.0,
                "frame": "",
                "active": False,
                "error": error,
                "controller_reason": "inactive",
                "decision_snapshot": None,
                "emergency_state": "NORMAL_TRAFFIC",
                "emergency_mode": False,
                "emergency_lane_id": None,
                "emergency_message": "",
            })
    stop_event.set()


def _video_fps(cap):
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1 or fps != fps:
        return 30
    return fps


def _frame_interval(cap) -> float:
    playback_fps = _video_fps(cap) * PLAYBACK_SPEED_MULTIPLIER
    if MAX_STREAM_FPS > 0:
        playback_fps = min(playback_fps, MAX_STREAM_FPS)
    return 1.0 / max(1.0, playback_fps)


def _capture_backend_name(cap) -> str:
    try:
        return cap.getBackendName()
    except Exception:
        return "unknown"


def _file_size(path: Path) -> int | None:
    try:
        return path.stat().st_size
    except OSError:
        return None


def _resolve_video_path(video_path) -> Path:
    path = Path(video_path).expanduser()
    if not path.is_absolute():
        path = path.resolve()
    return path.resolve()


def _log_capture_details(lane_id: str, cap) -> None:
    logger.info(
        "[%s] VideoCapture details backend=%s fps=%.2f frames=%.0f width=%.0f height=%.0f",
        lane_id,
        _capture_backend_name(cap),
        cap.get(cv2.CAP_PROP_FPS),
        cap.get(cv2.CAP_PROP_FRAME_COUNT),
        cap.get(cv2.CAP_PROP_FRAME_WIDTH),
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
    )


def _open_video_capture(lane_id: str, video_path):
    path = _resolve_video_path(video_path)
    exists = path.exists()
    logger.info(
        "[%s] Loading video path=%s exists=%s size_bytes=%s",
        lane_id,
        path,
        exists,
        _file_size(path),
    )

    if not exists:
        logger.error(
            "[%s] Video path does not exist. Check Linux case-sensitive spelling: %s",
            lane_id,
            path,
        )
        return None, path

    attempts = [("default", None)]
    if hasattr(cv2, "CAP_FFMPEG"):
        attempts.append(("ffmpeg", cv2.CAP_FFMPEG))

    for backend_name, backend in attempts:
        cap = (
            cv2.VideoCapture(str(path))
            if backend is None
            else cv2.VideoCapture(str(path), backend)
        )
        opened = cap.isOpened()
        logger.info(
            "[%s] cv2.VideoCapture backend_attempt=%s isOpened=%s backend=%s",
            lane_id,
            backend_name,
            opened,
            _capture_backend_name(cap) if opened else "unavailable",
        )
        if opened:
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
            _log_capture_details(lane_id, cap)
            return cap, path
        cap.release()

    logger.error(
        "[%s] OpenCV could not open video. This often means a missing file, bad case, "
        "unsupported codec/container, or incomplete upload: %s",
        lane_id,
        path,
    )
    return None, path


def _valid_frame(frame) -> bool:
    return frame is not None and getattr(frame, "size", 0) > 0


def _read_after_rewind(cap):
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        cap.set(cv2.CAP_PROP_POS_MSEC, 0)
    except Exception:
        pass
    ok, frame = cap.read()
    return ok and _valid_frame(frame), frame


def _restart_capture(lane_id: str, cap, video_path, reason: str):
    if cap is not None and cap.isOpened():
        logger.warning(
            "[%s] Frame read failed (%s). Restarting stream at pos=%.0f total=%.0f",
            lane_id,
            reason,
            cap.get(cv2.CAP_PROP_POS_FRAMES),
            cap.get(cv2.CAP_PROP_FRAME_COUNT),
        )
        ok, frame = _read_after_rewind(cap)
        if ok:
            logger.info("[%s] Stream restarted by rewinding to frame 0", lane_id)
            return cap, frame, True

        logger.warning("[%s] Rewind failed; releasing and reopening capture", lane_id)
        cap.release()
    elif cap is not None:
        cap.release()

    cap, _ = _open_video_capture(lane_id, video_path)
    if cap is None:
        return None, None, False

    ok, frame = cap.read()
    if ok and _valid_frame(frame):
        logger.info("[%s] Stream restarted with a fresh VideoCapture", lane_id)
        return cap, frame, True

    logger.error("[%s] Reopened video but first frame could not be read", lane_id)
    cap.release()
    return None, None, False


def _predict(model, model_lock, frame):
    if model_lock is None:
        kwargs = {"verbose": False}
        if MODEL_IMGSZ > 0:
            kwargs["imgsz"] = MODEL_IMGSZ
        return model(frame, **kwargs)

    acquired = model_lock.acquire(timeout=MODEL_INFERENCE_WAIT_SECONDS)
    if not acquired:
        return None
    try:
        kwargs = {"verbose": False}
        if MODEL_IMGSZ > 0:
            kwargs["imgsz"] = MODEL_IMGSZ
        return model(frame, **kwargs)
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
    logger.info("[%s] Worker started video_path=%s", lane_id, video_path)
    cap = None

    try:
        while not stop_event.is_set() and cap is None:
            cap, resolved_path = _open_video_capture(lane_id, video_path)
            if cap is None:
                error = f"Cannot open video: {resolved_path}"
                _mark_lane_error(lane_id, shared_state, state_lock, error)
                logger.error(
                    "[%s] %s; retrying in %.2fs",
                    lane_id,
                    error,
                    CAPTURE_REOPEN_BACKOFF_SECONDS,
                )
                if stop_event.wait(CAPTURE_REOPEN_BACKOFF_SECONDS):
                    break

        if cap is None:
            return

        frame_interval = _frame_interval(cap)
        detection_interval = max(1, DETECTION_EVERY_N_FRAMES)
        ambulance_tracker = AmbulanceConfirmationTracker(_video_fps(cap), detection_interval)
        ambulance_confirmation = ambulance_tracker.snapshot()
        density_window = deque(maxlen=MOVING_AVG_FRAMES)
        frame_index = 0
        next_frame_at = time.perf_counter()
        vehicle_count = 0
        weighted_density = 0
        density = 0
        vehicle_detections = []
        ambulance_seen = False
        ambulance_stable = False
        ambulance_state = NO_AMBULANCE
        ambulance_detections = []

        logger.info(
            "[%s] Worker ready fps=%.2f detection_every=%s model_imgsz=%s max_stream_fps=%.1f",
            lane_id,
            _video_fps(cap),
            detection_interval,
            MODEL_IMGSZ,
            MAX_STREAM_FPS,
        )

        while not stop_event.is_set():
            if cap is None or not cap.isOpened():
                cap, resolved_path = _open_video_capture(lane_id, video_path)
                if cap is None:
                    error = f"Cannot reopen video: {resolved_path}"
                    _mark_lane_error(lane_id, shared_state, state_lock, error)
                    if stop_event.wait(CAPTURE_REOPEN_BACKOFF_SECONDS):
                        break
                    continue
                frame_interval = _frame_interval(cap)
                next_frame_at = time.perf_counter()

            ok, frame = cap.read()
            if not ok or not _valid_frame(frame):
                reason = "ret=false" if not ok else "empty-frame"
                cap, frame, restarted = _restart_capture(lane_id, cap, video_path, reason)
                frame_index = 0
                if not restarted:
                    error = f"Cannot read video frame; retrying ({reason})"
                    _mark_lane_error(lane_id, shared_state, state_lock, error)
                    if stop_event.wait(CAPTURE_REOPEN_BACKOFF_SECONDS):
                        break
                    continue

                _mark_lane_error(lane_id, shared_state, state_lock, None)
                frame_interval = _frame_interval(cap)
                next_frame_at = time.perf_counter()

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
                        raw_ambulance_seen, ambulance_detections = detect_ambulance(ambulance_result)
                        ambulance_confirmation = ambulance_tracker.update(
                            raw_ambulance_seen,
                            ambulance_detections,
                        )
                        ambulance_seen = raw_ambulance_seen
                        ambulance_stable = ambulance_confirmation.confirmed
                        ambulance_state = ambulance_confirmation.state

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
                            "ambulance": ambulance_stable,
                            "ambulance_seen": ambulance_seen,
                            "ambulance_stable": ambulance_stable,
                            "ambulance_confirmed": ambulance_stable,
                            "ambulance_state": ambulance_state,
                            "ambulance_streak": ambulance_confirmation.hit_count,
                            "ambulance_required_frames": ambulance_confirmation.required_samples,
                            "ambulance_confidence": ambulance_confirmation.latest_confidence,
                            "ambulance_avg_confidence": ambulance_confirmation.avg_confidence,
                            "ambulance_hit_ratio": ambulance_confirmation.hit_ratio,
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
                    ambulance_streak=ambulance_confirmation.hit_count,
                    ambulance_required_frames=ambulance_confirmation.required_samples,
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
                logger.exception("[%s] Frame processing error: %s", lane_id, error)
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
                    logger.exception("[%s] Failed to encode fallback frame", lane_id)
                if stop_event.wait(WORKER_ERROR_BACKOFF_SECONDS):
                    break

            frame_index += 1
            next_frame_at, should_stop = _pace_frame(next_frame_at, frame_interval, stop_event)
            if should_stop:
                break
    finally:
        if cap is not None:
            cap.release()
        _finish_lane(lane_id, shared_state, state_lock, stop_event)
        logger.info("[%s] Worker stopped", lane_id)
