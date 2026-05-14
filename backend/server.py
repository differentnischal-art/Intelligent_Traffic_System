"""
backend/server.py
FastAPI app factory + all API routes.
"""

import logging
import os
import shutil
import threading
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO

from backend.ambulance_confirmation import NO_AMBULANCE
from backend.config import (
    IS_RENDER, UPLOAD_DIR, TEMPLATES_DIR, STATIC_DIR,
    VEHICLE_MODEL_PATH, AMBULANCE_MODEL_PATH,
)
from backend.lane_worker import run_lane
from backend.signal_controller import SmartSignalController


logging.basicConfig(
    level=os.getenv("ITMS_LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ── Global runtime objects (one instance per process) ────────────────────────
_v_model:     Optional[YOLO]       = None
_a_model:     Optional[YOLO]       = None
_shared_state: dict                = {}          # lane_id → state dict
_state_lock:   threading.Lock      = threading.Lock()
_v_model_lock: threading.Lock      = threading.Lock()
_a_model_lock: threading.Lock      = threading.Lock()
_stop_events:  dict                = {}          # lane_id → threading.Event
_threads:      list                = []
_signal_controller: Optional[SmartSignalController] = None


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _configure_runtime_limits() -> None:
    """Keep OpenCV/PyTorch thread fan-out small for multi-lane cloud runs."""
    cv_threads = max(1, _env_int("ITMS_OPENCV_THREADS", 1 if IS_RENDER else 2))
    torch_threads = max(1, _env_int("ITMS_TORCH_THREADS", 1 if IS_RENDER else 2))
    os.environ.setdefault("OMP_NUM_THREADS", str(torch_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(torch_threads))

    try:
        cv2.setNumThreads(cv_threads)
        logger.info("OpenCV threads set to %s", cv_threads)
    except Exception:
        logger.exception("Could not configure OpenCV thread count")

    try:
        import torch

        torch.set_num_threads(torch_threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        logger.info("Torch threads set to %s", torch_threads)
    except Exception:
        logger.exception("Could not configure Torch thread count")


def _path_size(path: Path) -> int | None:
    try:
        return path.stat().st_size
    except OSError:
        return None


def _safe_upload_suffix(filename: str | None) -> str:
    suffix = Path(filename or "").suffix.lower()
    if suffix and len(suffix) <= 12 and suffix[1:].replace(".", "").isalnum():
        return suffix
    return ".mp4"


def _stop_existing_lane_worker(lane_id: str) -> None:
    existing = _stop_events.pop(lane_id, None)
    if existing:
        logger.info("[%s] Stopping existing worker before replacing video", lane_id)
        existing.set()

    worker_name = f"worker-{lane_id}"
    remaining_threads = []
    for thread in _threads:
        if thread.name == worker_name:
            thread.join(timeout=2)
        if thread.is_alive():
            remaining_threads.append(thread)
    _threads[:] = remaining_threads


def _default_lane_state() -> dict:
    return {
        "lane_id":    "",
        "signal":     "RED",
        "signal_state": "RED",
        "density":    0,
        "count":      0,
        "green_time": 0,
        "timer":      0,
        "ambulance":  False,
        "ambulance_seen": False,
        "ambulance_stable": False,
        "ambulance_confirmed": False,
        "ambulance_state": NO_AMBULANCE,
        "ambulance_streak": 0,
        "ambulance_required_frames": 0,
        "ambulance_confidence": 0.0,
        "ambulance_avg_confidence": 0.0,
        "ambulance_hit_ratio": 0.0,
        "frame":      "",
        "active":     False,
        "error":      None,
        "active_phase": "RED",
        "phase_duration": 0,
        "controller_reason": "idle",
        "decision_snapshot": None,
        "emergency_state": "NORMAL_TRAFFIC",
        "emergency_mode": False,
        "emergency_lane_id": None,
        "emergency_message": "",
    }


def _public_lane_state(lane_id: str, lane: dict) -> dict:
    state = {**lane, "lane_id": lane.get("lane_id") or lane_id}
    return {key: value for key, value in state.items() if key != "frame"}


def _default_emergency_status() -> dict:
    return {
        "state": "NORMAL_TRAFFIC",
        "active": False,
        "lane_id": None,
        "message": "",
        "remaining_seconds": 0,
    }


def _prepare_model_for_threads(model: YOLO, model_lock: threading.Lock, label: str) -> None:
    """
    Force Ultralytics' lazy setup to happen during startup, not inside lane
    worker threads. A lock is still used during inference because YOLO model
    instances mutate internal predictor state.
    """
    dummy = np.zeros((64, 64, 3), dtype=np.uint8)
    with model_lock:
        model(dummy, verbose=False)
    logger.info("%s model ready for threaded inference", label)


def _load_yolo_model(model_path, label: str, model_lock: threading.Lock) -> YOLO:
    path = Path(model_path).resolve()
    logger.info(
        "Loading %s model path=%s exists=%s size_bytes=%s",
        label,
        path,
        path.exists(),
        _path_size(path),
    )
    if not path.exists():
        raise FileNotFoundError(
            f"{label} model file not found at {path}. Check GitHub tracking, .gitignore, "
            "Git LFS, and Linux case-sensitive spelling."
        )

    try:
        model = YOLO(str(path))
        _prepare_model_for_threads(model, model_lock, label)
        logger.info("%s model loaded successfully from %s", label, path)
        return model
    except Exception:
        logger.exception("%s model failed to load from %s", label, path)
        raise


def create_app() -> FastAPI:
    _configure_runtime_limits()

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
    STATIC_DIR.mkdir(parents=True, exist_ok=True)

    app = FastAPI(title="ITMS API", version="2.0")

    # Allow all origins (for local dev)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Serve static files (CSS / JS)
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    # ── Startup: load models ──────────────────────────────────────────────────
    @app.on_event("startup")
    def _load_models():
        global _v_model, _a_model, _a_model_lock
        _v_model = _load_yolo_model(VEHICLE_MODEL_PATH, "Vehicle", _v_model_lock)

        amb_path = Path(AMBULANCE_MODEL_PATH)
        try:
            _a_model = _load_yolo_model(amb_path, "Ambulance", _a_model_lock)
        except FileNotFoundError as exc:
            logger.warning("%s Using vehicle model as ambulance fallback.", exc)
            _a_model = _v_model
            _a_model_lock = _v_model_lock
        except Exception:
            logger.exception("Ambulance model failed. Using vehicle model as fallback.")
            _a_model = _v_model
            _a_model_lock = _v_model_lock

    # ── Shutdown: stop all workers ────────────────────────────────────────────
    @app.on_event("shutdown")
    def _stop_all():
        global _signal_controller
        if _signal_controller:
            _signal_controller.stop()
            _signal_controller = None
        for ev in _stop_events.values():
            ev.set()
        for t in _threads:
            t.join(timeout=2)

    # ─────────────────────────────────────────────────────────────────────────
    # HTML ROUTES
    # ─────────────────────────────────────────────────────────────────────────

    @app.get("/", response_class=HTMLResponse)
    def index():
        """Setup page — user selects lanes and uploads videos."""
        page = TEMPLATES_DIR / "index.html"
        if not page.exists():
            raise HTTPException(404, "index.html not found in frontend/templates/")
        return page.read_text(encoding="utf-8")

    @app.get("/dashboard", response_class=HTMLResponse)
    def dashboard():
        """Live dashboard page."""
        with _state_lock:
            running = bool(_shared_state)
        if not running:
            # Nothing started yet — redirect to setup
            return HTMLResponse(
                '<meta http-equiv="refresh" content="0;url=/">',
                status_code=302,
            )
        page = TEMPLATES_DIR / "dashboard.html"
        if not page.exists():
            raise HTTPException(404, "dashboard.html not found in frontend/templates/")
        return page.read_text(encoding="utf-8")

    # ─────────────────────────────────────────────────────────────────────────
    # API ROUTES
    # ─────────────────────────────────────────────────────────────────────────

    @app.post("/api/start")
    def start_system(lane_count: int = Form(...)):
        """
        Step 1: Tell backend how many lanes will be monitored.
        Clears any previous session and prepares fresh state.
        """
        global _shared_state, _stop_events, _threads, _signal_controller

        # Stop existing workers cleanly
        for ev in _stop_events.values():
            ev.set()
        for t in _threads:
            t.join(timeout=2)
        _threads.clear()
        _stop_events.clear()
        if _signal_controller:
            _signal_controller.stop()
            _signal_controller = None

        with _state_lock:
            _shared_state = {
                f"lane_{i+1}": {**_default_lane_state(), "lane_id": f"lane_{i+1}"}
                for i in range(lane_count)
            }

        _signal_controller = SmartSignalController(_shared_state, _state_lock)
        _signal_controller.start()

        return {"status": "ready", "lane_count": lane_count}


    @app.post("/api/upload/{lane_id}")
    async def upload_video(lane_id: str, file: UploadFile = File(...)):
        """
        Step 2: Upload video for a specific lane_id (e.g. lane_1).
        Saves file and starts a worker thread for that lane.
        """
        with _state_lock:
            if lane_id not in _shared_state:
                raise HTTPException(400, f"Unknown lane_id '{lane_id}'. Call /api/start first.")
        if _v_model is None or _a_model is None:
            raise HTTPException(503, "Models are still loading. Try again shortly.")

        _stop_existing_lane_worker(lane_id)

        # Save uploaded file. Preserve the container extension when possible so
        # OpenCV/FFmpeg do not have to guess a WebM/MOV/AVI file named .mp4.
        dest = (UPLOAD_DIR / f"{lane_id}{_safe_upload_suffix(file.filename)}").resolve()
        with open(dest, "wb") as f:
            shutil.copyfileobj(file.file, f)
        size = _path_size(dest) or 0
        logger.info(
            "[%s] Video saved path=%s original_filename=%s size_bytes=%s",
            lane_id,
            dest,
            file.filename,
            size,
        )
        if size <= 0:
            raise HTTPException(400, f"Uploaded video for {lane_id} is empty.")

        # Create stop event for this lane
        stop_ev = threading.Event()
        _stop_events[lane_id] = stop_ev

        # Start worker thread
        t = threading.Thread(
            target=run_lane,
            args=(lane_id, str(dest), _v_model, _a_model,
                  _shared_state, _state_lock, stop_ev,
                  _v_model_lock, _a_model_lock),
            daemon=True,
            name=f"worker-{lane_id}",
        )
        t.start()
        _threads.append(t)

        return {"status": "ok", "lane_id": lane_id, "file": dest.name}


    @app.post("/api/setup")
    async def setup_system(
        lane_count: int = Form(...),
        lane_1: Optional[UploadFile] = File(None),
        lane_2: Optional[UploadFile] = File(None),
        lane_3: Optional[UploadFile] = File(None),
        lane_4: Optional[UploadFile] = File(None),
        lane_5: Optional[UploadFile] = File(None),
        lane_6: Optional[UploadFile] = File(None),
        lane_7: Optional[UploadFile] = File(None),
        lane_8: Optional[UploadFile] = File(None),
    ):
        """
        Backward-compatible setup endpoint for the existing setup page.
        It prepares lane state and starts one worker per uploaded lane video.
        """
        files = [lane_1, lane_2, lane_3, lane_4, lane_5, lane_6, lane_7, lane_8]
        start_system(lane_count)

        uploaded = []
        for idx in range(lane_count):
            file = files[idx]
            if file is None:
                raise HTTPException(400, f"Missing video for lane_{idx + 1}")
            result = await upload_video(f"lane_{idx + 1}", file)
            uploaded.append(result["lane_id"])

        return {"status": "ok", "lane_count": lane_count, "lanes": uploaded}


    @app.get("/api/state")
    def get_state():
        """
        Returns all lane states (excluding heavy frame data).
        Frontend polls this every ~400ms to update signals/counts.
        """
        with _state_lock:
            running = any(v.get("active", False) for v in _shared_state.values())
            lanes = {
                k: _public_lane_state(k, v)
                for k, v in _shared_state.items()
            }
            decision_snapshot = (
                _signal_controller.current_decision_snapshot()
                if _signal_controller
                else None
            )
            emergency_status = (
                _signal_controller.current_emergency_status()
                if _signal_controller
                else _default_emergency_status()
            )
            return {
                "running":    running,
                "lane_count": len(_shared_state),
                "lanes": lanes,
                "lane_states": list(lanes.values()),
                "decision_snapshot": decision_snapshot,
                "emergency_status": emergency_status,
            }


    @app.get("/api/frame/{lane_id}")
    def get_frame(lane_id: str):
        """
        Returns base64 JPEG frame for a specific lane.
        Frontend polls this every ~120ms per lane for live video.
        """
        with _state_lock:
            lane = _shared_state.get(lane_id, {})
            frame = lane.get("frame", "")
            active = lane.get("active", False)
        return {
            "frame":  frame,
            "active": active,
        }


    @app.get("/api/summary")
    def get_summary():
        """Aggregated summary across all lanes — total vehicles, mode, etc."""
        with _state_lock:
            lanes = dict(_shared_state)
            decision_snapshot = (
                _signal_controller.current_decision_snapshot()
                if _signal_controller
                else None
            )
            emergency_status = (
                _signal_controller.current_emergency_status()
                if _signal_controller
                else _default_emergency_status()
            )

        total    = sum(v.get("count", 0) for v in lanes.values())
        greens   = sum(1 for v in lanes.values() if v.get("signal") == "GREEN")
        amb_cnt  = sum(1 for v in lanes.values() if v.get("ambulance"))
        busiest  = max(lanes, key=lambda k: lanes[k].get("count", 0), default="—")
        any_amb  = any(v.get("ambulance") for v in lanes.values())
        emergency_state = emergency_status.get("state")

        return {
            "total_vehicles": total,
            "green_lanes":    greens,
            "ambulances":     amb_cnt,
            "busiest_lane":   busiest,
            "mode":           (
                "EMERGENCY CLEARANCE"
                if emergency_state == "EMERGENCY_CLEARANCE_MODE"
                else (
                    "AMBULANCE OVERRIDE"
                    if emergency_state == "AMBULANCE_PRIORITY_ACTIVE" or any_amb
                    else "DENSITY BASED"
                )
            ),
            "current_green_lane": next(
                (lane_id for lane_id, lane in lanes.items() if lane.get("signal") == "GREEN"),
                None,
            ),
            "decision_snapshot": decision_snapshot,
            "emergency_status": emergency_status,
        }


    @app.post("/api/stop")
    def stop_system():
        """Stop all lane workers and clear state."""
        global _signal_controller
        if _signal_controller:
            _signal_controller.stop()
            _signal_controller = None
        for ev in _stop_events.values():
            ev.set()
        for t in _threads:
            t.join(timeout=2)
        _stop_events.clear()
        _threads.clear()
        with _state_lock:
            _shared_state.clear()
        return {"status": "stopped"}


    @app.delete("/api/lane/{lane_id}")
    def stop_lane(lane_id: str):
        """Stop a single lane worker."""
        if lane_id in _stop_events:
            _stop_events[lane_id].set()
            with _state_lock:
                if lane_id in _shared_state:
                    del _shared_state[lane_id]
            return {"status": "stopped", "lane_id": lane_id}
        raise HTTPException(404, f"Lane '{lane_id}' not found")

    return app


app = create_app()
