import os
import random
import shutil

random.seed(42)

TRAIN_IMAGE_DIR = "dataset/images/train"
TRAIN_LABEL_DIR = "dataset/labels/train"
VAL_IMAGE_DIR = "dataset/images/val"
VAL_LABEL_DIR = "dataset/labels/val"

OUT_TRAIN_IMAGE_DIR = "dataset_sampled/images/train"
OUT_TRAIN_LABEL_DIR = "dataset_sampled/labels/train"
OUT_VAL_IMAGE_DIR = "dataset_sampled/images/val"
OUT_VAL_LABEL_DIR = "dataset_sampled/labels/val"

TRAIN_SAMPLES = 2000
VAL_SAMPLES = 200

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def sample_split(src_img_dir, src_lbl_dir, out_img_dir, out_lbl_dir, n_samples):
    ensure_dir(out_img_dir)
    ensure_dir(out_lbl_dir)

    images = [
        f for f in os.listdir(src_img_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    n_samples = min(n_samples, len(images))
    chosen = random.sample(images, n_samples)

    for img_name in chosen:
        base, _ = os.path.splitext(img_name)
        label_name = base + ".txt"

        shutil.copy2(
            os.path.join(src_img_dir, img_name),
            os.path.join(out_img_dir, img_name)
        )

        src_label_path = os.path.join(src_lbl_dir, label_name)
        dst_label_path = os.path.join(out_lbl_dir, label_name)

        if os.path.exists(src_label_path):
            shutil.copy2(src_label_path, dst_label_path)
        else:
            open(dst_label_path, "w").close()

    print(f"copied {n_samples} images to {out_img_dir}")

sample_split(
    TRAIN_IMAGE_DIR, TRAIN_LABEL_DIR,
    OUT_TRAIN_IMAGE_DIR, OUT_TRAIN_LABEL_DIR,
    TRAIN_SAMPLES
)

sample_split(
    VAL_IMAGE_DIR, VAL_LABEL_DIR,
    OUT_VAL_IMAGE_DIR, OUT_VAL_LABEL_DIR,
    VAL_SAMPLES
)