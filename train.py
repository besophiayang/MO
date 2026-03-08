from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="dirt_dataset.yaml",
    epochs=50,
    imgsz=640,
    batch=8,
    workers=2,
    project="runs",
    name="dirt_yolo"
)