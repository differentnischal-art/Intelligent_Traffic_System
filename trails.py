"""
backend/detector.py
Pure detection functions — no FastAPI, no threads.
Called by lane_worker.py for each frame.
"""

import cv2
import numpy as np
from collections import defaultdict

from backend.config import (
    VEHICLE_CLASSES, VEHICLE_CONF, AMBULANCE_CONF,
    MAX_VEHICLES, MIN_GREEN_TIME, MAX_GREEN_TIME, AMBULANCE_GREEN_TIME,
    SUB_LANES_PER_FEED, ROAD_TOP_FRACTION, ZEBRA_FALLBACK,
)


# ── Lane geometry ─────────────────────────────────────────────────────────────

def build_sub_lanes(frame: np.ndarray) -> dict:
    """
    Auto-divide bottom portion of frame into equal vertical strips.
    No manual coordinate input needed — works for any resolution.
    Returns: {"sub_1": (x1,y1,x2,y2), "sub_2": ...}
    """
    h, w   = frame.shape[:2]
    top    = int(h * ROAD_TOP_FRACTION)
    lw     = w // SUB_LANES_PER_FEED
    return {
        f"sub_{i+1}": (
            i * lw,
            top,
            (i+1)*lw if i < SUB_LANES_PER_FEED-1 else w,
            h
        )
        for i in range(SUB_LANES_PER_FEED)
    }


def find_zebra_y(frame: np.ndarray) -> int:
    """
    Detect zebra crossing y-coordinate automatically using Hough lines.
    Falls back to ZEBRA_FALLBACK fraction if nothing found.
    """
    h, w   = frame.shape[:2]
    roi    = frame[int(h * 0.55):h, :]
    gray   = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges  = cv2.Canny(gray, 50, 150)
    lines  = cv2.HoughLinesP(
        edges, 1, np.pi/180,
        threshold=80, minLineLength=w//4, maxLineGap=20
    )
    if lines is not None:
        ys = []
        for ln in lines:
            x1, y1, x2, y2 = ln[0]
            angle = abs(np.arctan2(y2-y1, x2-x1) * 180 / np.pi)
            if angle < 12:
                ys.append(y1 + int(h * 0.55))
        if ys:
            return int(np.median(ys))
    return int(h * ZEBRA_FALLBACK)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _in_region(cx: float, cy: float, coords: tuple) -> bool:
    x1, y1, x2, y2 = coords
    return x1 <= cx <= x2 and y1 <= cy <= y2


# ── Vehicle counting ──────────────────────────────────────────────────────────

def count_vehicles(v_results, sub_lanes: dict) -> tuple:
    """
    Count vehicles in each sub-lane from YOLO detection results.
    Returns:
        counts  = {"sub_1": 5, "sub_2": 3}
        details = {"sub_1": [{"cls","box","conf"}, ...]}
    """
    counts  = defaultdict(int)
    details = defaultdict(list)

    for box in v_results[0].boxes:
        cls_id = int(box.cls.item())
        if cls_id not in VEHICLE_CLASSES:
            continue
        conf = float(box.conf.item())
        if conf < VEHICLE_CONF:
            continue

        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cx, cy = (x1+x2)/2, (y1+y2)/2

        for name, coords in sub_lanes.items():
            if _in_region(cx, cy, coords):
                counts[name] += 1
                details[name].append({
                    "cls":  VEHICLE_CLASSES[cls_id],
                    "box":  [int(x1), int(y1), int(x2), int(y2)],
                    "conf": round(conf, 2),
                })

    return dict(counts), dict(details)


# ── Ambulance detection ───────────────────────────────────────────────────────

def detect_ambulance(a_results, sub_lanes: dict, zebra_y: int) -> tuple:
    """
    Detect ambulance per sub-lane.
    Returns:
        amb_lanes = {"sub_1": True/False, ...}
        crossed   = True if ambulance bottom edge passed zebra line
    """
    amb_lanes = defaultdict(bool)
    crossed   = False

    boxes = a_results[0].boxes
    if boxes is None or len(boxes) == 0:
        return dict(amb_lanes), crossed

    for box in boxes:
        if float(box.conf.item()) < AMBULANCE_CONF:
            continue
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cx, cy = (x1+x2)/2, (y1+y2)/2

        for name, coords in sub_lanes.items():
            if _in_region(cx, cy, coords):
                amb_lanes[name] = True

        if y2 > zebra_y:
            crossed = True

    return dict(amb_lanes), crossed


# ── Signal decision ───────────────────────────────────────────────────────────

def _green_time(count: int) -> tuple:
    """Returns (green_seconds, density_percent) for a vehicle count."""
    density = min(count / MAX_VEHICLES, 1.0)
    gtime   = round(MIN_GREEN_TIME + density * (MAX_GREEN_TIME - MIN_GREEN_TIME))
    return gtime, round(density * 100)


def decide_signals(counts: dict, amb_lanes: dict, sub_lanes: dict) -> dict:
    """
    Core traffic signal decision.

    AMBULANCE MODE  → ambulance lane gets GREEN for 60s, all others RED
    DENSITY MODE    → busiest lane gets GREEN, others RED

    Returns: {sub_lane: {signal, time, density, reason}}
    """
    signals     = {}
    amb_present = any(amb_lanes.values())

    if amb_present:
        for lane in sub_lanes:
            if amb_lanes.get(lane, False):
                signals[lane] = {
                    "signal":  "GREEN",
                    "time":    AMBULANCE_GREEN_TIME,
                    "density": 100,
                    "reason":  "AMBULANCE PRIORITY",
                }
            else:
                signals[lane] = {
                    "signal":  "RED",
                    "time":    0,
                    "density": 0,
                    "reason":  "AMBULANCE OVERRIDE",
                }
    else:
        max_count = max(counts.values()) if counts else 0
        for lane in sub_lanes:
            count   = counts.get(lane, 0)
            gtime, d = _green_time(count)
            signals[lane] = {
                "signal":  "GREEN" if count == max_count and count > 0 else "RED",
                "time":    gtime,
                "density": d,
                "reason":  f"{d}% density",
            }

    return signals


# ── Frame annotation ──────────────────────────────────────────────────────────

def annotate_frame(frame, sub_lanes, signals, counts, details, amb_lanes, zebra_y):
    """Draw lane borders, signal info, vehicle boxes, and zebra line on frame."""
    h, w    = frame.shape[:2]
    overlay = frame.copy()

    # Zebra crossing line
    cv2.line(frame, (0, zebra_y), (w, zebra_y), (255, 220, 50), 2)

    for name, (x1, y1, x2, y2) in sub_lanes.items():
        sig    = signals.get(name, {})
        signal = sig.get("signal", "RED")
        count  = counts.get(name, 0)
        is_amb = amb_lanes.get(name, False)

        # Lane color
        color = (0,220,255) if is_amb else (
                (50,255,80) if signal=="GREEN" else (50,50,255))

        # Transparent lane fill
        cv2.rectangle(overlay, (x1,y1),(x2,y2), color, -1)
        cv2.addWeighted(overlay, 0.07, frame, 0.93, 0, frame)

        # Lane border
        cv2.rectangle(frame, (x1,y1),(x2,y2), color, 2)

        # Info box
        bx, by = x1+6, y1+6
        cv2.rectangle(frame,(bx,by),(bx+224,by+84),(12,12,18),-1)
        cv2.rectangle(frame,(bx,by),(bx+224,by+84),color,1)
        cv2.putText(frame, name.upper(),
                    (bx+8,by+20), cv2.FONT_HERSHEY_SIMPLEX,0.5,(210,210,210),1)
        cv2.putText(frame, f"Signal : {signal}",
                    (bx+8,by+40), cv2.FONT_HERSHEY_SIMPLEX,0.52,color,2)
        cv2.putText(frame, f"Vehicles: {count}  Density: {sig.get('density',0)}%",
                    (bx+8,by+60), cv2.FONT_HERSHEY_SIMPLEX,0.38,(160,160,160),1)
        cv2.putText(frame, sig.get("reason",""),
                    (bx+8,by+76), cv2.FONT_HERSHEY_SIMPLEX,0.36,
                    (0,220,255) if is_amb else (120,120,120),1)

        # Vehicle boxes
        for v in details.get(name, []):
            vx1,vy1,vx2,vy2 = v["box"]
            cv2.rectangle(frame,(vx1,vy1),(vx2,vy2),(150,150,150),1)
            cv2.putText(frame,f"{v['cls']} {v['conf']}",
                        (vx1,vy1-4),cv2.FONT_HERSHEY_SIMPLEX,0.3,(150,150,150),1)

    # Top status bar
    any_amb = any(amb_lanes.values())
    mc   = (0,200,255) if any_amb else (50,255,80)
    mode = "AMBULANCE OVERRIDE" if any_amb else "DENSITY BASED"
    cv2.rectangle(frame,(0,0),(w,28),(10,10,14),-1)
    cv2.putText(frame,
                f"ITMS  |  {mode}  |  Total: {sum(counts.values())}",
                (10,18),cv2.FONT_HERSHEY_SIMPLEX,0.5,mc,1)

    return frame