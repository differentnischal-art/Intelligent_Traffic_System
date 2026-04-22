from ultralytics import YOLO

def main():
    model = YOLO("yolov8n-seg.pt")

    model.train(
        data="ambulance_dataset/Ambulance labelling.v2i.yolov8/data.yaml",
        epochs=100,
        imgsz=640,
        batch=8,
        name="ambulance_v1",
        val=False,
        workers=0
    )

if __name__ == "__main__":
    main()