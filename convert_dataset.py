import os
import shutil
from PIL import Image

SOURCE_DIR = r"C:\Users\besop\Desktop\DirtData\Data"

TRAIN_IMG_DIR = os.path.join("dataset", "images", "train")
VAL_IMG_DIR = os.path.join("dataset", "images", "val")
TRAIN_LABEL_DIR = os.path.join("dataset", "labels", "train")
VAL_LABEL_DIR = os.path.join("dataset", "labels", "val")

os.makedirs(TRAIN_IMG_DIR, exist_ok=True)
os.makedirs(VAL_IMG_DIR, exist_ok=True)
os.makedirs(TRAIN_LABEL_DIR, exist_ok=True)
os.makedirs(VAL_LABEL_DIR, exist_ok=True)

TRAIN_ANN_FILE = os.path.join(SOURCE_DIR, "bbox_training.txt")
VAL_ANN_FILE = os.path.join(SOURCE_DIR, "bbox_val.txt")


def parse_annotation_line(parts):
    if len(parts) < 5:
        return None

    image_name = parts[0]
    x = float(parts[1])
    y = float(parts[2])
    w = float(parts[3])
    h = float(parts[4])
    return image_name, x, y, w, h


def get_image_size(image_path):
    with Image.open(image_path) as img:
        return img.size 


def convert_split(annotation_file, out_image_dir, out_label_dir):
    annotations = {}

    with open(annotation_file, "r") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            if "," in line:
                parts = [p.strip() for p in line.split(",")]
            else:
                parts = line.split()

            parsed = parse_annotation_line(parts)
            if parsed is None:
                print(f"skipping this line bc its malformed: {line}")
                continue

            image_name, x, y, w, h = parsed

            if image_name not in annotations:
                annotations[image_name] = []

            annotations[image_name].append((x, y, w, h))

    for i, (image_name, boxes) in enumerate(annotations.items(), start=1):
        src_image_name = image_name if image_name.lower().endswith(".png") else image_name + ".png"
        src_image_path = os.path.join(SOURCE_DIR, src_image_name)
        dst_image_path = os.path.join(out_image_dir, src_image_name)

        if not os.path.exists(src_image_path):
            print(f"u are missing the image: {src_image_path}")
            continue

        if not os.path.exists(dst_image_path):
            shutil.copy2(src_image_path, dst_image_path)

        img_w, img_h = get_image_size(src_image_path)

        label_name = os.path.splitext(src_image_name)[0] + ".txt"
        out_label_path = os.path.join(out_label_dir, label_name)

        with open(out_label_path, "w") as f:
            for x, y, w, h in boxes:
                x_center = (x + w / 2) / img_w
                y_center = (y + h / 2) / img_h
                w_norm = w / img_w
                h_norm = h / img_h
                f.write(f"0 {x_center} {y_center} {w_norm} {h_norm}\n")

        if i % 500 == 0:
            print(f"processed {i} images...")


def main():
    convert_split(TRAIN_ANN_FILE, TRAIN_IMG_DIR, TRAIN_LABEL_DIR)
    convert_split(VAL_ANN_FILE, VAL_IMG_DIR, VAL_LABEL_DIR)
    print("done")


if __name__ == "__main__":
    main()