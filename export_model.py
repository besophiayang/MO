from ultralytics import YOLO


def main():
    model = YOLO("runs/detect/runs/dirt_student_final/weights/best.pt")
    model.export(format="ncnn", imgsz=320)


if __name__ == "__main__":
    main()