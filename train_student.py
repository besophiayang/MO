from ultralytics import YOLO


def main():
    model = YOLO("yolov8s.pt")

    model.train(
        data="dirt_dataset_distilled.yaml",
        epochs=20,
        imgsz=640,
        batch=16,
        device=0,
        workers=0,
        project="runs",
        name="dirt_student_final",
        pretrained=True,
        amp=True,
        cache=False,
        patience=10
    )


if __name__ == "__main__":
    main()