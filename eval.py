import os
from data.datasets.evaluation.kitti_object_eval_python.evaluate import evaluate

# ─── Configuration ────────────────────────────────────────────────────────────
KITTI_ROOT   = "/path/to/your/kitti/"
GT_LABEL_DIR = os.path.join(KITTI_ROOT, "training", "label_2")
DT_LABEL_DIR = "result"   # your detection .txt folder
SPLIT_FILE   = os.path.join(KITTI_ROOT, "training", "ImageSets", "val.txt")
CLASS_IDX    = 0          # 0=Car,1=Pedestrian,2=Cyclist
METRIC       = "R40"      # "R40" for 40‐point AP, or "R11"
# ──────────────────────────────────────────────────────────────────────────────

def main():
    result_str, result_dict = evaluate(
        label_path       = GT_LABEL_DIR,
        result_path      = DT_LABEL_DIR,
        label_split_file = SPLIT_FILE,
        current_class    = CLASS_IDX,
        metric           = METRIC,
    )
    print(f"=== KITTI 3D AP@{METRIC} for class {CLASS_IDX} ===")
    print(result_str)
    print("Detailed results:")
    for key, val in result_dict.items():
        print(f"{key}: {val}")

if __name__ == "__main__":
    main()