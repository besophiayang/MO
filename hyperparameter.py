from ultralytics import YOLO

EXPERIMENTS = [
    {"model": "yolov8n.pt", "imgsz": 512, "batch": 16, "name": "student_n_512"},
    {"model": "yolov8n.pt", "imgsz": 640, "batch": 16, "name": "student_n_640"},
    {"model": "yolov8s.pt", "imgsz": 512, "batch": 16, "name": "student_s_512"},
    {"model": "yolov8s.pt", "imgsz": 640, "batch": 16, "name": "student_s_640"},
    {"model": "yolov8m.pt", "imgsz": 512, "batch": 16, "name": "student_m_512"},
]

def main():
    for exp in EXPERIMENTS:

        print("Starting experiment:", exp["name"])

        model = YOLO(exp["model"])

        model.train(
            data="dirt_dataset_distilled.yaml",
            epochs=8,
            imgsz=exp["imgsz"],
            batch=exp["batch"],
            workers=0,
            device=0,
            amp=True,
            cache=False,
            pretrained=True,
            patience=5,
            project="runs_tuning",
            name=exp["name"]
        )

if __name__ == "__main__":
    main() 