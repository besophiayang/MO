import os
import random
import shutil

random.seed(42)

image_dir = os.path.join("dataset", "images", "train")
label_dir = os.path.join("dataset", "labels", "train")

val_image_dir = os.path.join("dataset", "images", "val")
val_label_dir = os.path.join("dataset", "labels", "val")

os.makedirs(val_image_dir, exist_ok=True)
os.makedirs(val_label_dir, exist_ok=True)

images = [f for f in os.listdir(image_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
random.shuffle(images)

val_count = int(0.1 * len(images))
val_images = images[:val_count]

for img_name in val_images:
    label_name = os.path.splitext(img_name)[0] + ".txt"

    shutil.move(os.path.join(image_dir, img_name), os.path.join(val_image_dir, img_name))

    src_label = os.path.join(label_dir, label_name)
    dst_label = os.path.join(val_label_dir, label_name)

    if os.path.exists(src_label):
        shutil.move(src_label, dst_label)