from ultralytics import YOLO

def main():
    model = YOLO("yolov8m.pt")

    model.train(
        data="dirt_dataset_sampled.yaml",
        epochs=20,
        imgsz=512,
        batch=16,
        workers=8,
        project="runs",
        name="dirt_teacher",
        pretrained=True,
        patience=10,
        device=0,
        amp=True,
        cache=False
    )

if __name__ == "__main__":
    main()