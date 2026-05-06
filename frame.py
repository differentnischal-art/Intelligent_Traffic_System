import cv2
import os

video_path = ""  # your video file
output_folder = "frames"  # folder to save frames

# create folder if not exists
os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(video_path)

frame_count = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break  # video ended

    frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
    cv2.imwrite(frame_filename, frame)

    frame_count += 1

cap.release()
print("Frames extracted:", frame_count)