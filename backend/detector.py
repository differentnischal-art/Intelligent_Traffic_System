"""
Full-frame detection helpers for one uploaded video = one traffic lane.
"""

import base64

import cv2

from backend.config import (
    AMBULANCE_CONF,
    FRAME_ENCODE_WIDTH,
    FRAME_JPEG_QUALITY,
    VEHICLE_CLASSES,
    VEHICLE_CONF,
    VEHICLE_DENSITY_WEIGHTS,
)


def vehicle_density_weight(label):
    return VEHICLE_DENSITY_WEIGHTS.get(label, 1)


def count_vehicles(result):
    """
    Count vehicles across the entire frame.
    Returns a raw count, weighted traffic density, and bounding box details.
    """
    count = 0
    weighted_density = 0
    detections = []
    boxes = getattr(result[0], "boxes", None)
    if boxes is None:
        return count, weighted_density, detections

    for box in boxes:
        cls_id = int(box.cls.item())
        conf = float(box.conf.item())
        if cls_id not in VEHICLE_CLASSES or conf < VEHICLE_CONF:
            continue

        x1, y1, x2, y2 = box.xyxy[0].tolist()
        label = VEHICLE_CLASSES[cls_id]
        count += 1
        weighted_density += vehicle_density_weight(label)
        detections.append({
            "label": label,
            "conf": round(conf, 2),
            "box": [int(x1), int(y1), int(x2), int(y2)],
        })

    return count, weighted_density, detections


def detect_ambulance(result):
    """
    Detect ambulance candidates anywhere in the full frame.
    Temporal stability is handled by lane_worker.py.
    """
    detections = []
    prediction = result[0]
    boxes = getattr(prediction, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return False, detections

    names = getattr(prediction, "names", {}) or {}
    for box in boxes:
        cls_id = int(box.cls.item())
        if names:
            label = _class_name(names, cls_id)
            if "ambulance" not in label:
                continue

        conf = float(box.conf.item())
        if conf < AMBULANCE_CONF:
            continue

        x1, y1, x2, y2 = box.xyxy[0].tolist()
        detections.append({
            "label": "ambulance",
            "conf": round(conf, 2),
            "box": [int(x1), int(y1), int(x2), int(y2)],
        })

    return bool(detections), detections


def _class_name(names, cls_id: int) -> str:
    try:
        if isinstance(names, dict):
            return str(names.get(cls_id, "")).lower()
        return str(names[cls_id]).lower()
    except (IndexError, KeyError, TypeError):
        return ""


def annotate_frame(
    frame,
    vehicle_count,
    vehicle_detections,
    ambulance_detections,
    signal,
    density,
    timer,
    ambulance_stable,
    ambulance_seen,
    ambulance_streak,
    ambulance_required_frames,
):
    """
    Draw full-frame detections and one clean lane status overlay.
    """
    h, w = frame.shape[:2]

    for det in vehicle_detections:
        x1, y1, x2, y2 = det["box"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (160, 160, 160), 2)
        cv2.putText(
            frame,
            f"{det['label']} {det['conf']}",
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )

    for det in ambulance_detections:
        x1, y1, x2, y2 = det["box"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 255), 3)
        cv2.putText(
            frame,
            f"ambulance {det['conf']}",
            (x1, max(22, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (0, 240, 255),
            2,
            cv2.LINE_AA,
        )

    signal_color = (45, 235, 80) if signal == "GREEN" else ((0, 220, 255) if signal == "YELLOW" else (45, 45, 235))
    mode = "AMBULANCE PRIORITY" if ambulance_stable else "DENSITY BASED"
    pending = max(0, ambulance_required_frames - ambulance_streak) if ambulance_seen and not ambulance_stable else 0

    cv2.rectangle(frame, (0, 0), (w, 82), (10, 12, 18), -1)
    cv2.rectangle(frame, (0, 0), (w, 82), signal_color, 2)
    cv2.putText(
        frame,
        f"Signal: {signal} | Vehicles: {vehicle_count} | Density: {density} | Timer: {timer}s",
        (14, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        signal_color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Ambulance: {'STABLE' if ambulance_stable else ('DETECTED' if ambulance_seen else 'NO')} | Mode: {mode}",
        (14, 54),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.56,
        (0, 220, 255) if ambulance_seen else (185, 185, 185),
        1,
        cv2.LINE_AA,
    )

    if pending:
        cv2.putText(
            frame,
            f"Priority in {pending} frames",
            (w - 230, 54),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 220, 255),
            1,
            cv2.LINE_AA,
        )

    return frame


def encode_frame(frame):
    if frame is None or frame.size == 0:
        return ""

    h, w = frame.shape[:2]
    if h <= 0 or w <= 0:
        return ""

    resized = cv2.resize(frame, (FRAME_ENCODE_WIDTH, int(h * FRAME_ENCODE_WIDTH / w)))
    ok, buffer = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, FRAME_JPEG_QUALITY])
    if not ok:
        return ""

    return base64.b64encode(buffer).decode("ascii")
