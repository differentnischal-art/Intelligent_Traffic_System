"""Project configuration shared by the backend runtime."""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def _is_render() -> bool:
    return os.getenv("RENDER", "").lower() == "true" or bool(os.getenv("RENDER_SERVICE_ID"))


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _path_from_env(name: str, default: Path) -> Path:
    raw_value = os.getenv(name)
    path = Path(raw_value) if raw_value else default
    if not path.is_absolute():
        path = BASE_DIR / path
    return path.resolve()


IS_RENDER = _is_render()

UPLOAD_DIR = BASE_DIR / "uploads"
FRONTEND_DIR = BASE_DIR / "frontend"
TEMPLATES_DIR = FRONTEND_DIR / "templates"
STATIC_DIR = FRONTEND_DIR / "static"

VEHICLE_MODEL_PATH = _path_from_env("ITMS_VEHICLE_MODEL_PATH", BASE_DIR / "yolov8n.pt")
AMBULANCE_MODEL_PATH = _path_from_env("ITMS_AMBULANCE_MODEL_PATH", BASE_DIR / "models" / "best1.pt")

VEHICLE_CONF = 0.40
AMBULANCE_CONF = 0.65
AMBULANCE_CONFIRMATION_SECONDS = 2.5
AMBULANCE_CONFIRMATION_RATIO = 0.70
AMBULANCE_MIN_CONFIRMATION_HITS = 4
AMBULANCE_MAX_CONFIRMATION_MISSES = 1

VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

VEHICLE_DENSITY_WEIGHTS = {
    "bike": 1,
    "bicycle": 1,
    "motorcycle": 1,
    "car": 2,
    "bus": 5,
    "truck": 5,
}

MAX_VEHICLES = 15
MIN_GREEN_TIME = 10
MAX_GREEN_TIME = 30
AMBULANCE_GREEN_TIME = 60

FRAME_ENCODE_WIDTH = _env_int("ITMS_FRAME_ENCODE_WIDTH", 520 if IS_RENDER else 620)
FRAME_JPEG_QUALITY = _env_int("ITMS_FRAME_JPEG_QUALITY", 65 if IS_RENDER else 72)

SUB_LANES_PER_FEED = 2
ROAD_TOP_FRACTION = 0.35
ZEBRA_FALLBACK = 0.72
