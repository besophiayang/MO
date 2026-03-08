import os
import shutil


def copy_split(src_root, dst_root, split):
    src_img = os.path.join(src_root, "images", split)
    src_lbl = os.path.join(src_root, "labels", split)
    dst_img = os.path.join(dst_root, "images", split)
    dst_lbl = os.path.join(dst_root, "labels", split)

    os.makedirs(dst_img, exist_ok=True)
    os.makedirs(dst_lbl, exist_ok=True)

    if os.path.exists(src_img):
        for f in os.listdir(src_img):
            shutil.copy2(os.path.join(src_img, f), os.path.join(dst_img, f))

    if os.path.exists(src_lbl):
        for f in os.listdir(src_lbl):
            shutil.copy2(os.path.join(src_lbl, f), os.path.join(dst_lbl, f))


def main():
    real_root = "dataset_sampled"
    pseudo_root = "dataset_pseudo"
    merged_root = "dataset_distilled"

    copy_split(real_root, merged_root, "train")
    copy_split(real_root, merged_root, "val")
    copy_split(pseudo_root, merged_root, "train")

    print("Merged dataset created at dataset_distilled/")


if __name__ == "__main__":
    main()