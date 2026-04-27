"""
backend/server.py
FastAPI app factory + all API routes.
"""

import shutil
import threading
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO

from backend.config import (
    UPLOAD_DIR, TEMPLATES_DIR, STATIC_DIR,
    VEHICLE_MODEL_PATH, AMBULANCE_MODEL_PATH,
)
from backend.lane_worker import run_lane


# ── Global runtime objects (one instance per process) ────────────────────────
_v_model:     Optional[YOLO]       = None
_a_model:     Optional[YOLO]       = None
_shared_state: dict                = {}          # lane_id → state dict
_state_lock:   threading.Lock      = threading.Lock()
_stop_events:  dict                = {}          # lane_id → threading.Event
_threads:      list                = []


def _default_lane_state() -> dict:
    return {
        "signal":     "RED",
        "density":    0,
        "count":      0,
        "green_time": 0,
        "timer":      0,
        "ambulance":  False,
        "frame":      "",
        "active":     False,
        "error":      None,
    }


def create_app() -> FastAPI:
    UPLOAD_DIR.mkdir(exist_ok=True)

    app = FastAPI(title="ITMS API", version="2.0")

    # Allow all origins (for local dev)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Serve static files (CSS / JS)
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    # ── Startup: load models ──────────────────────────────────────────────────
    @app.on_event("startup")
    def _load_models():
        global _v_model, _a_model
        print("Loading vehicle model (yolov8n.pt)...")
        _v_model = YOLO(VEHICLE_MODEL_PATH)

        print(f"Loading ambulance model ({AMBULANCE_MODEL_PATH})...")
        amb_path = Path(AMBULANCE_MODEL_PATH)
        if amb_path.exists():
            _a_model = YOLO(str(amb_path))
            print("Ambulance model loaded.")
        else:
            print(f"WARNING: {amb_path} not found — using vehicle model as fallback.")
            _a_model = _v_model

    # ── Shutdown: stop all workers ────────────────────────────────────────────
    @app.on_event("shutdown")
    def _stop_all():
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
        global _shared_state, _stop_events, _threads

        # Stop existing workers cleanly
        for ev in _stop_events.values():
            ev.set()
        for t in _threads:
            t.join(timeout=2)
        _threads.clear()
        _stop_events.clear()

        with _state_lock:
            _shared_state = {
                f"lane_{i+1}": _default_lane_state()
                for i in range(lane_count)
            }

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

        # Save uploaded file
        dest = UPLOAD_DIR / f"{lane_id}.mp4"
        with open(dest, "wb") as f:
            shutil.copyfileobj(file.file, f)
        print(f"[{lane_id}] Video saved → {dest}")

        # Create stop event for this lane
        stop_ev = threading.Event()
        _stop_events[lane_id] = stop_ev

        # Start worker thread
        t = threading.Thread(
            target=run_lane,
            args=(lane_id, str(dest), _v_model, _a_model,
                  _shared_state, _state_lock, stop_ev),
            daemon=True,
            name=f"worker-{lane_id}",
        )
        t.start()
        _threads.append(t)

        return {"status": "ok", "lane_id": lane_id, "file": dest.name}


    @app.get("/api/state")
    def get_state():
        """
        Returns all lane states (excluding heavy frame data).
        Frontend polls this every ~400ms to update signals/counts.
        """
        with _state_lock:
            return {
                "running":    bool(_shared_state),
                "lane_count": len(_shared_state),
                "lanes": {
                    k: {kk: vv for kk, vv in v.items() if kk != "frame"}
                    for k, v in _shared_state.items()
                },
            }


    @app.get("/api/frame/{lane_id}")
    def get_frame(lane_id: str):
        """
        Returns base64 JPEG frame for a specific lane.
        Frontend polls this every ~120ms per lane for live video.
        """
        with _state_lock:
            lane = _shared_state.get(lane_id, {})
        return {
            "frame":  lane.get("frame", ""),
            "active": lane.get("active", False),
        }


    @app.get("/api/summary")
    def get_summary():
        """Aggregated summary across all lanes — total vehicles, mode, etc."""
        with _state_lock:
            lanes = dict(_shared_state)

        total    = sum(v["count"]     for v in lanes.values())
        greens   = sum(1 for v in lanes.values() if v["signal"] == "GREEN")
        amb_cnt  = sum(1 for v in lanes.values() if v["ambulance"])
        busiest  = max(lanes, key=lambda k: lanes[k]["count"], default="—")
        any_amb  = any(v["ambulance"] for v in lanes.values())

        return {
            "total_vehicles": total,
            "green_lanes":    greens,
            "ambulances":     amb_cnt,
            "busiest_lane":   busiest,
            "mode":           "AMBULANCE OVERRIDE" if any_amb else "DENSITY BASED",
        }


    @app.post("/api/stop")
    def stop_system():
        """Stop all lane workers and clear state."""
        for ev in _stop_events.values():
            ev.set()
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