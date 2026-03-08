import os
import shutil
from PIL import Image

SOURCE_DIR = r"C:\Users\besop\Desktop\DirtData"

TRAIN_IMG_DIR = os.path.join("dataset", "images", "train")
VAL_IMG_DIR = os.path.join("dataset", "images", "val")
TRAIN_LABEL_DIR = os.path.join("dataset", "labels", "train")
VAL_LABEL_DIR = os.path.join("dataset", "labels", "val")

TRAIN_ANN_FILE = os.path.join(SOURCE_DIR, "bbox_training.txt")
VAL_ANN_FILE = os.path.join(SOURCE_DIR, "bbox_val.txt")

os.makedirs(TRAIN_IMG_DIR, exist_ok=True)
os.makedirs(VAL_IMG_DIR, exist_ok=True)
os.makedirs(TRAIN_LABEL_DIR, exist_ok=True)
os.makedirs(VAL_LABEL_DIR, exist_ok=True)


def get_image_size(image_path):
    with Image.open(image_path) as img:
        return img.size 


def convert_bbox_xyxy_to_yolo(x1, y1, x2, y2, img_w, img_h):
    x1 = max(0, min(x1, img_w))
    x2 = max(0, min(x2, img_w))
    y1 = max(0, min(y1, img_h))
    y2 = max(0, min(y2, img_h))

    x_left = min(x1, x2)
    x_right = max(x1, x2)
    y_top = min(y1, y2)
    y_bottom = max(y1, y2)

    box_w = x_right - x_left
    box_h = y_bottom - y_top

    if box_w <= 0 or box_h <= 0:
        return None

    x_center = (x_left + x_right) / 2.0 / img_w
    y_center = (y_top + y_bottom) / 2.0 / img_h
    w_norm = box_w / img_w
    h_norm = box_h / img_h

    return x_center, y_center, w_norm, h_norm


def convert_split(annotation_file, out_image_dir, out_label_dir):
    annotations = {}

    with open(annotation_file, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 6:
                print("skip bc the line is malformed: {line}")
                continue

            image_name = parts[0]
            x1 = float(parts[1])
            y1 = float(parts[2])
            x2 = float(parts[3])
            y2 = float(parts[4])
            class_name = parts[5]

            if class_name != "dirt":
                continue

            if image_name not in annotations:
                annotations[image_name] = []

            annotations[image_name].append((x1, y1, x2, y2))

    print(f"Found {len(annotations)} dirt images in {annotation_file}")

    for i, (image_name, boxes) in enumerate(annotations.items(), start=1):
        src_image_name = image_name if image_name.lower().endswith(".png") else image_name + ".png"
        src_image_path = os.path.join(SOURCE_DIR, src_image_name)
        dst_image_path = os.path.join(out_image_dir, src_image_name)

        if not os.path.exists(src_image_path):
            continue

        if not os.path.exists(dst_image_path):
            shutil.copy2(src_image_path, dst_image_path)

        img_w, img_h = get_image_size(src_image_path)

        label_name = os.path.splitext(src_image_name)[0] + ".txt"
        out_label_path = os.path.join(out_label_dir, label_name)

        valid_count = 0
        with open(out_label_path, "w", encoding="utf-8") as f:
            for x1, y1, x2, y2 in boxes:
                yolo_box = convert_bbox_xyxy_to_yolo(x1, y1, x2, y2, img_w, img_h)
                if yolo_box is None:
                    continue

                x_center, y_center, w_norm, h_norm = yolo_box
                f.write(f"0 {x_center} {y_center} {w_norm} {h_norm}\n")
                valid_count += 1

        if valid_count == 0 and os.path.exists(out_label_path):
            os.remove(out_label_path)


def main():
    convert_split(TRAIN_ANN_FILE, TRAIN_IMG_DIR, TRAIN_LABEL_DIR)
    convert_split(VAL_ANN_FILE, VAL_IMG_DIR, VAL_LABEL_DIR)
    print("we are done")


if __name__ == "__main__":
    main()