"""
backend/config.py
All tunable constants in one place.
Change AMBULANCE_MODEL_PATH to your actual best.pt path.
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

UPLOAD_DIR = BASE_DIR / "uploads"
FRONTEND_DIR = BASE_DIR / "frontend"
TEMPLATES_DIR = FRONTEND_DIR / "templates"
STATIC_DIR = FRONTEND_DIR / "static"

VEHICLE_MODEL_PATH = "yolov8n.pt"
AMBULANCE_MODEL_PATH = str(BASE_DIR / "models" / "best1.pt")

VEHICLE_CONF = 0.40
AMBULANCE_CONF = 0.40

VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

MAX_VEHICLES = 15
MIN_GREEN_TIME = 10
MAX_GREEN_TIME = 30
AMBULANCE_GREEN_TIME = 60

FRAME_ENCODE_WIDTH = 620
FRAME_JPEG_QUALITY = 72

SUB_LANES_PER_FEED = 2
ROAD_TOP_FRACTION = 0.35
ZEBRA_FALLBACK = 0.72