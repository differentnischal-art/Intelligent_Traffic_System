import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import os

# ─────────────────────────────────────────────────────────
# CONFIG — only change these two paths
# ─────────────────────────────────────────────────────────
VIDEO_SOURCE      = r"C:\Projects\Intelligent_Traffic_Signal\Imgvideofortesting\trafficv.mp4"
VEHICLE_MODEL     = "yolov8n.pt"
AMBULANCE_MODEL   = r"C:\Projects\Intelligent_Traffic_Signal\runs\segment\ambulance_v24\weights\best1.pt"
OUTPUT_PATH       = r"/output/traffic_output.mp4"

NUM_LANES         = 4
MAX_VEHICLES      = 15
MIN_GREEN_TIME    = 10
MAX_GREEN_TIME    = 30
AMBULANCE_CONF    = 0.4
VEHICLE_CONF      = 0.4

VEHICLE_CLASSES   = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# ─────────────────────────────────────────────────────────
# AUTO LANE DETECTION
# Divides bottom 60% of frame into equal lane regions
# Works for any video resolution automatically
# ─────────────────────────────────────────────────────────

def auto_detect_lanes(frame, num_lanes=4):
    h, w = frame.shape[:2]
    road_top    = int(h * 0.40)
    road_bottom = h
    lane_width  = w // num_lanes
    lanes = {}
    for i in range(num_lanes):
        x1 = i * lane_width
        x2 = (i + 1) * lane_width if i < num_lanes - 1 else w
        lanes[f"lane_{i+1}"] = (x1, road_top, x2, road_bottom)
    return lanes


def auto_detect_zebra_line(frame):
    h, w = frame.shape[:2]
    roi   = frame[int(h * 0.6):h, :]
    gray  = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180,
                             threshold=100,
                             minLineLength=w // 3,
                             maxLineGap=20)
    if lines is not None:
        horizontal = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            if angle < 10:
                horizontal.append(y1 + int(h * 0.6))
        if horizontal:
            return int(np.median(horizontal))
    return int(h * 0.75)


# ─────────────────────────────────────────────────────────
# VEHICLE COUNTING
# ─────────────────────────────────────────────────────────

def is_in_lane(cx, cy, coords):
    x1, y1, x2, y2 = coords
    return x1 <= cx <= x2 and y1 <= cy <= y2


def count_vehicles_per_lane(vehicle_results, lanes):
    counts  = defaultdict(int)
    details = defaultdict(list)
    for box in vehicle_results[0].boxes:
        cls_id = int(box.cls.item())
        if cls_id not in VEHICLE_CLASSES:
            continue
        conf = float(box.conf.item())
        if conf < VEHICLE_CONF:
            continue
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        for lane_name, coords in lanes.items():
            if is_in_lane(cx, cy, coords):
                counts[lane_name] += 1
                details[lane_name].append({
                    "class": VEHICLE_CLASSES[cls_id],
                    "box":   (int(x1), int(y1), int(x2), int(y2)),
                    "conf":  conf
                })
    return counts, details


# ─────────────────────────────────────────────────────────
# AMBULANCE DETECTION
# ─────────────────────────────────────────────────────────

def detect_ambulance_per_lane(amb_results, lanes, zebra_y):
    ambulance_lanes   = defaultdict(bool)
    ambulance_crossed = False
    boxes = amb_results[0].boxes
    if boxes is None or len(boxes) == 0:
        return ambulance_lanes, ambulance_crossed
    for box in boxes:
        conf = float(box.conf.item())
        if conf < AMBULANCE_CONF:
            continue
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        for lane_name, coords in lanes.items():
            if is_in_lane(cx, cy, coords):
                ambulance_lanes[lane_name] = True
        if y2 > zebra_y:
            ambulance_crossed = True
    return ambulance_lanes, ambulance_crossed


# ─────────────────────────────────────────────────────────
# SIGNAL DECISION LOGIC
# ─────────────────────────────────────────────────────────

def calculate_green_time(count):
    density    = min(count / MAX_VEHICLES, 1.0)
    green_time = MIN_GREEN_TIME + density * (MAX_GREEN_TIME - MIN_GREEN_TIME)
    return round(green_time), round(density * 100)


def decide_signals(vehicle_counts, ambulance_lanes, lanes):
    signals           = {}
    ambulance_present = any(ambulance_lanes.values())

    if ambulance_present:
        for lane_name in lanes:
            if ambulance_lanes.get(lane_name, False):
                signals[lane_name] = {"signal": "GREEN", "time": 999,
                                      "reason": "AMBULANCE", "density": 100}
            else:
                signals[lane_name] = {"signal": "RED",   "time": 0,
                                      "reason": "AMB OVERRIDE", "density": 0}
    else:
        max_count = max(vehicle_counts.values()) if vehicle_counts else 0
        for lane_name in lanes:
            count = vehicle_counts.get(lane_name, 0)
            green_time, density_pct = calculate_green_time(count)
            if count == max_count and count > 0:
                signals[lane_name] = {"signal": "GREEN", "time": green_time,
                                      "reason": f"density {density_pct}%",
                                      "density": density_pct}
            else:
                signals[lane_name] = {"signal": "RED",   "time": green_time,
                                      "reason": f"density {density_pct}%",
                                      "density": density_pct}
    return signals


# ─────────────────────────────────────────────────────────
# DRAW OVERLAY
# ─────────────────────────────────────────────────────────

def draw_overlay(frame, lanes, signals, vehicle_counts,
                 vehicle_details, ambulance_lanes, zebra_y):
    h, w   = frame.shape[:2]
    overlay = frame.copy()

    # Zebra crossing
    cv2.line(frame, (0, zebra_y), (w, zebra_y), (255, 255, 0), 2)
    cv2.putText(frame, "Zebra Crossing", (10, zebra_y - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    for lane_name, (x1, y1, x2, y2) in lanes.items():
        info    = signals.get(lane_name, {})
        signal  = info.get("signal", "RED")
        reason  = info.get("reason", "")
        count   = vehicle_counts.get(lane_name, 0)
        is_amb  = ambulance_lanes.get(lane_name, False)
        color   = (0, 255, 255) if is_amb else (
                   (0, 255, 0) if signal == "GREEN" else (0, 0, 255))

        # Transparent fill
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(overlay, 0.08, frame, 0.92, 0, frame)

        # Lane border
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Info panel
        cv2.rectangle(frame, (x1+4, y1+4), (x1+200, y1+95), (20, 20, 20), -1)
        cv2.rectangle(frame, (x1+4, y1+4), (x1+200, y1+95), color, 1)
        cv2.putText(frame, lane_name.upper(),       (x1+10, y1+22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)
        cv2.putText(frame, f"Signal : {signal}",    (x1+10, y1+42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        cv2.putText(frame, f"Vehicles: {count}",    (x1+10, y1+62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1)
        cv2.putText(frame, reason,                   (x1+10, y1+80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4,  (180,180,0), 1)

        if is_amb:
            cv2.putText(frame, "AMBULANCE!",
                        (x1+6, y2-12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,255), 2)

        # Draw individual vehicle boxes
        for v in vehicle_details.get(lane_name, []):
            vx1, vy1, vx2, vy2 = v["box"]
            cv2.rectangle(frame, (vx1, vy1), (vx2, vy2), (180,180,180), 1)
            cv2.putText(frame, v["class"], (vx1, vy1-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180,180,180), 1)

    # Top status bar
    total   = sum(vehicle_counts.values())
    amb_on  = any(ambulance_lanes.values())
    mode    = "AMBULANCE OVERRIDE" if amb_on else "DENSITY BASED"
    mcolor  = (0,255,255) if amb_on else (0,255,0)
    cv2.rectangle(frame, (0,0), (w,35), (20,20,20), -1)
    cv2.putText(frame,
                f"Intelligent Traffic System  |  Vehicles: {total}  |  {mode}",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, mcolor, 1)
    return frame


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────

def main():
    print("Loading models...")
    vehicle_model   = YOLO(VEHICLE_MODEL)
    ambulance_model = YOLO(AMBULANCE_MODEL)

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"Cannot open: {VIDEO_SOURCE}")
        return

    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    ret, first_frame = cap.read()
    if not ret:
        print("Cannot read video")
        return

    # AUTO-DETECT lanes and zebra from first frame
    print("Auto-detecting lanes and zebra crossing...")
    lanes   = auto_detect_lanes(first_frame, NUM_LANES)
    zebra_y = auto_detect_zebra_line(first_frame)

    print(f"Lanes detected: {len(lanes)}")
    for name, coords in lanes.items():
        print(f"  {name}: {coords}")
    print(f"Zebra crossing y={zebra_y}")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    out = cv2.VideoWriter(OUTPUT_PATH,
                          cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    ambulance_active = False
    frame_count      = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    print("Running... Press Q to quit\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        v_res = vehicle_model(frame,   verbose=False)
        a_res = ambulance_model(frame, verbose=False, task="segment")

        vehicle_counts, vehicle_details = count_vehicles_per_lane(v_res, lanes)
        ambulance_lanes, crossed = detect_ambulance_per_lane(a_res, lanes, zebra_y)

        if crossed and ambulance_active:
            print(f"[Frame {frame_count}] Ambulance crossed zebra — normal mode")
            ambulance_active = False
            ambulance_lanes  = defaultdict(bool)

        if any(ambulance_lanes.values()):
            ambulance_active = True

        signals = decide_signals(vehicle_counts, ambulance_lanes, lanes)

        if frame_count % 30 == 0:
            print(f"[Frame {frame_count}]")
            for lane, info in signals.items():
                print(f"  {lane}: {info['signal']:5s} | "
                      f"{vehicle_counts.get(lane,0):2d} vehicles | {info['reason']}")

        frame = draw_overlay(frame, lanes, signals,
                             vehicle_counts, vehicle_details,
                             ambulance_lanes, zebra_y)
        out.write(frame)
        cv2.imshow("Intelligent Traffic System", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()