import cv2
import os

video_path = r"C:\Projects\Intelligent_Traffic_Signal\Imgvideofortesting\Road_vechile.mp4"

cap = cv2.VideoCapture(video_path)
os.makedirs("frames", exist_ok=True)

count = 0
saved = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if count % 19 == 0:   # 👈 change this value
        cv2.imwrite(f"frames/frame_{saved}.jpg", frame)
        saved += 1

    count += 1

cap.release()