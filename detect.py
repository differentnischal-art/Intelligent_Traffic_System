from ultralytics import YOLO

model = YOLO(r"C:\Projects\Intelligent_Traffic_Signal\best1.pt")

results = model(r"C:\Projects\Intelligent_Traffic_Signal\Imgvideofortesting\traffic1.jpg" ,
    save=True,
    project=r"C:\Projects\Intelligent_Traffic_Signal\output",
    name="predict",
    conf=0.5
)
for result in results:
    boxes = result.boxes
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        conf = box.conf[0]
        cls = box.cls[0]

        print("Coordinates:", float(x1), float(y1), float(x2), float(y2))
        print("Confidence:", float(conf))
        print("Class ID:", int(cls))
        print("------")
