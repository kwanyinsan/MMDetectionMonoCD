import argparse
import cv2
import numpy as np
from data.datasets.kitti_utils import (
    Calibration,      # [data/datasets/kitti_utils.py#L329]
    read_label,       # [data/datasets/kitti_utils.py#L142]
    compute_box_3d,   # [data/datasets/kitti_utils.py#L666]
    draw_projected_box3d  # [data/datasets/kitti_utils.py#L804]
)

def main(image_path, depth_txt, label_txt, calib_txt, out_img, out_txt):
    # 1) load inputs
    img = cv2.imread(image_path)
    depth_map = np.loadtxt(depth_txt)            # shape: H×W in meters
    calib = Calibration(calib_txt)               # KITTI calib parser

    # 2) parse objects
    objs = read_label(label_txt)                 # list of Object3d

    # accum corners for TXT dump
    all_corners3d = []

    for obj in objs:
        # 3) pick a depth per‐object (here at 2D‐bbox center)
        u = int((obj.xmin + obj.xmax) / 2)
        v = int((obj.ymin + obj.ymax) / 2)
        d = float(depth_map[v, u])

        # 4) project uv+d → 3D center (optional, not used below)
        # center3d = calib.project_image_to_rect(np.array([[u, v, d]]))[0]

        # 5) compute the 8 corners in rect‐camera coords
        corners2d, corners3d = compute_box_3d(obj, calib.P)

        print(f"corners2d: {corners2d}, corners3d: {corners3d}")
        if corners2d is None:
            continue

        # store for TXT
        all_corners3d.append(corners3d)

        # 6) draw 3D box on image
        img = draw_projected_box3d(
            img,
            corners2d.astype(np.int32),
            color=(0,255,0),
        )

    # 7) save visualization
    cv2.imwrite(out_img, img)

    # 8) dump corners to TXT
    with open(out_txt, 'w') as f:
        for corners3d in all_corners3d:
            # corners3d: (8,3)
            for x, y, z in corners3d.reshape(-1, 3):
                f.write(f"{x:.3f} {y:.3f} {z:.3f}\n")
            f.write("\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",   required=True,  help="path to RGB image")
    parser.add_argument("--depth",   required=True,  help="path to depth TXT")
    parser.add_argument("--label",   required=True,  help="path to KITTI .txt label")
    parser.add_argument("--calib",   required=True,  help="path to KITTI calib file")
    parser.add_argument("--out_img", default="out_with_3d.png", help="where to save visualized image")
    parser.add_argument("--out_txt", default="corners3d.txt",   help="where to save 3D corners")
    args = parser.parse_args()
    main(
        args.image,
        args.depth,
        args.label,
        args.calib,
        args.out_img,
        args.out_txt,
    )