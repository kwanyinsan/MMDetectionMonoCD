import os
import json
import shutil
from PIL import Image

# ==========================
# Define KITTI Root Path
# ==========================
# Modify this single line to set the path to the KITTI dataset
kitti_root = "/path/to/your/kitti/"

# ==========================
# Paths (constructed relative to KITTI root)
# ==========================
train_list_path = os.path.join(kitti_root, "testing", "ImageSets", "train.txt")  # each line: image basename without extension
val_list_path = os.path.join(kitti_root, "testing", "ImageSets", "val.txt")      # same format as train.txt
image_dir = os.path.join(kitti_root, "training", "image_2")                      # folder containing all images (e.g., 000000.png, 000001.png...)
label_dir = os.path.join(kitti_root, "training", "label_2")                      # folder containing all KITTI .txt label files
output_dir = "mm_data"                                                            # root output folder

# ==========================
# Category Mapping (only 'car')
# ==========================
categories = [{"id": 0, "name": "Car"}]
category_name_to_id = {"Car": 0}

# ==========================
# Create output directories
# ==========================
annotations_dir = os.path.join(output_dir, "annotations")
train_images_out = os.path.join(output_dir, "train")
val_images_out = os.path.join(output_dir, "val")
for d in [annotations_dir, train_images_out, val_images_out]:
    os.makedirs(d, exist_ok=True)

# ==========================
# Helper to process each split
# ==========================
def process_split(list_path, split_name):
    # Read image IDs
    with open(list_path, 'r') as f:
        image_ids = [line.strip() for line in f if line.strip()]

    images = []
    annotations = []
    ann_id = 0

    for img_id in image_ids:
        file_name = f"{img_id}.png"  # change extension if needed
        img_path = os.path.join(image_dir, file_name)
        label_path = os.path.join(label_dir, f"{img_id}.txt")

        # Load image to get dimensions
        img = Image.open(img_path)
        width, height = img.size
        img.close()

        # Image entry
        image_entry = {
            "file_name": file_name,
            "height": height,
            "width": width,
            "id": int(img_id)
        }
        images.append(image_entry)

        # Copy image to train/val folder
        dest_folder = train_images_out if split_name == "train" else val_images_out
        shutil.copy(img_path, os.path.join(dest_folder, file_name))

        # Parse label file and convert car annotations
        if os.path.exists(label_path):
            with open(label_path, 'r') as lf:
                for line in lf:
                    parts = line.strip().split()
                    cls = parts[0]
                    if cls != "Car":
                        continue
                    x1, y1, x2, y2 = map(float, parts[4:8])
                    w = x2 - x1
                    h = y2 - y1
                    annotation = {
                        "id": ann_id,
                        "image_id": int(img_id),
                        "category_id": category_name_to_id[cls],
                        "bbox": [x1, y1, w, h],
                        "area": w * h,
                        "iscrowd": 0
                    }
                    annotations.append(annotation)
                    ann_id += 1

    # Write COCO JSON
    out_json = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }
    out_path = os.path.join(annotations_dir, f"instances_{split_name}.json")
    with open(out_path, 'w') as out_file:
        json.dump(out_json, out_file, indent=4)
    print(f"Wrote {out_path} with {len(images)} images and {len(annotations)} annotations.")

# ==========================
# Run conversion for both splits
# ==========================
process_split(train_list_path, "train")
process_split(val_list_path, "val")

print("Dataset conversion to COCO format completed!")