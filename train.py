from ultralytics import YOLO

def main():
    model = YOLO("yolov8n-seg.pt")
    model.train(
        data         = r"C:\Projects\Intelligent_Traffic_Signal\dataset\Ambulance labelling.v2i.yolov8\data.yaml",
        epochs       = 100,
        imgsz        = 640,
        batch        = 16       ,
        name         = "ambulance_v2",
        patience     = 20,
        val          = False,        # ← skip val during training
        workers      = 0,
        overlap_mask = False,
        device       = 0,
        rect         = False,        # ← add this
    )

    # Validate separately after training completes
    print("\nTraining done. Running validation...")
    trained = YOLO("runs/segment/ambulance_v2/weights/best.pt")
    metrics = trained.val(
        data         = "dataset/Ambulance labelling.v2i.yolov8/data.yaml",
        workers      = 0,
        overlap_mask = False,
        split        = "val",
    )
    print(f"mAP50     : {metrics.box.map50:.3f}")
    print(f"mAP50-95  : {metrics.box.map:.3f}")
    print(f"Precision : {metrics.box.mp:.3f}")
    print(f"Recall    : {metrics.box.mr:.3f}")

if __name__ == "__main__":
    main()