import os
import shutil
from ultralytics import YOLO


def main():
    teacher_path = "runs/detect/runs/dirt_teacher2/weights/best.pt"

    source_dir = "extra_images"
    out_img_dir = "dataset_pseudo/images/train"
    out_lbl_dir = "dataset_pseudo/labels/train"

    conf_thresh = 0.60

    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    model = YOLO(teacher_path)

    image_files = [
        f for f in os.listdir(source_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    print(f"Found {len(image_files)} images in {source_dir}")

    for img_name in image_files:
        img_path = os.path.join(source_dir, img_name)

        results = model.predict(
            source=img_path,
            conf=conf_thresh,
            verbose=False,
            device=0
        )

        result = results[0]
        boxes = result.boxes

        shutil.copy2(img_path, os.path.join(out_img_dir, img_name))

        label_name = os.path.splitext(img_name)[0] + ".txt"
        label_path = os.path.join(out_lbl_dir, label_name)

        with open(label_path, "w") as f:
            if boxes is not None and len(boxes) > 0:
                xywhn = boxes.xywhn.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                clses = boxes.cls.cpu().numpy()

                for row, conf, cls_id in zip(xywhn, confs, clses):
                    if conf < conf_thresh:
                        continue

                    x_center, y_center, w, h = row
                    cls_id = int(cls_id)

                    f.write(f"{cls_id} {x_center} {y_center} {w} {h}\n")

    print("Pseudo-label generation complete.")
    print(f"Images saved to: {out_img_dir}")
    print(f"Labels saved to: {out_lbl_dir}")


if __name__ == "__main__":
    main()