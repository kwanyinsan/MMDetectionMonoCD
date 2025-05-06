import os
import cv2
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# === Configurable paths ===
KITTI_ROOT   = "/path/to/your/kitti/"            # your KITTI root
VIS_DIR      = "output/visualization"            # depth_*.txt, boxes3d_*.png
BOX2D_DIR    = "boxes2d"                         # 2D bbox txt files
IMGSET_FILE  = os.path.join(KITTI_ROOT, "training", "ImageSets", "val.txt")

# KITTI subfolders
IMG_DIR    = os.path.join(KITTI_ROOT, "training", "image_2")
CALIB_DIR  = os.path.join(KITTI_ROOT, "training", "calib")

# where to put all outputs
RESULT_DIR = os.path.join(os.getcwd(), "result")
os.makedirs(RESULT_DIR, exist_ok=True)

# sanity-check existence of folders
for d in (IMG_DIR, CALIB_DIR, VIS_DIR, BOX2D_DIR):
    if not os.path.isdir(d):
        raise FileNotFoundError(f"Directory not found: {d}")
if not os.path.isfile(IMGSET_FILE):
    raise FileNotFoundError(f"File not found: {IMGSET_FILE}")

# === Helper functions ===
def load_calibration(calib_path):
    with open(calib_path, 'r') as f:
        for line in f:
            if line.startswith('P2:'):
                vals = list(map(float, line.split()[1:]))
                return np.array(vals).reshape(3,4)
    raise ValueError(f"P2 not found in {calib_path}")

def compute_3d_box(h, w, l, x, y, z, ry):
    x_c = [ l/2,  l/2, -l/2, -l/2,  l/2,  l/2, -l/2, -l/2]
    y_c = [   0,    0,    0,    0,   -h,   -h,   -h,   -h]
    z_c = [ w/2, -w/2, -w/2,  w/2,  w/2, -w/2, -w/2,  w/2]
    corners = np.vstack((x_c, y_c, z_c))
    R = np.array([
        [ np.cos(ry), 0, np.sin(ry)],
        [          0, 1,          0],
        [-np.sin(ry), 0, np.cos(ry)]
    ])
    return (R @ corners + np.array([[x],[y],[z]])).T

def project_to_image(pts3d, P):
    pts_hom = np.hstack((pts3d, np.ones((pts3d.shape[0],1))))
    proj    = (P @ pts_hom.T)
    proj[:2] /= proj[2:3]
    return proj[:2].T

def draw_3d_box(img, pts2d, color=(0,255,0), thickness=2):
    pts = pts2d.astype(int)
    edges = [
        (0,1),(1,2),(2,3),(3,0),
        (4,5),(5,6),(6,7),(7,4),
        (0,4),(1,5),(2,6),(3,7)
    ]
    for i,j in edges:
        cv2.line(img, tuple(pts[i]), tuple(pts[j]), color, thickness)

def objective_hwl(hwl, label, P2):
    h, w, l = hwl
    x, y, z = label["location"]
    ry       = label["rotation_y"]
    corners3d = compute_3d_box(h, w, l, x, y, z, ry)
    corners2d = project_to_image(corners3d, P2)
    xmin_p, ymin_p = corners2d.min(axis=0)
    xmax_p, ymax_p = corners2d.max(axis=0)
    bx1, by1, bx2, by2 = label["bbox"]
    dx = ((bx1+bx2)/2) - ((xmin_p+xmax_p)/2)
    dy = ((by1+by2)/2) - ((ymin_p+ymax_p)/2)
    dw = (bx2-bx1)     - (xmax_p-xmin_p)
    dh = (by2-by1)     - (ymax_p-ymin_p)
    return dx*dx + dy*dy + dw*dw + dh*dh

def estimate_3d_boxes(img, depth, P2, boxes2d):
    K_inv = np.linalg.inv(P2[:,:3])
    labels = []
    for box in boxes2d:
        u = (box[0]+box[2])/2; v = (box[1]+box[3])/2
        ui, vi = int(round(u)), int(round(v))
        if not (0 <= vi < depth.shape[0] and 0 <= ui < depth.shape[1]):
            continue
        Z = depth[vi, ui]
        if Z <= 0:
            continue
        X, Y, Z = Z * (K_inv @ np.array([u, v, 1.0]))
        h0, w0, l0 = 1.52, 1.63, 3.88
        Y += h0/2
        ry    = np.arctan2(X, Z)
        alpha = ((ry - np.arctan2(X, Z) + np.pi) % (2*np.pi)) - np.pi

        lbl = {
            "type":       "Car",
            "truncation": 0.0,
            "occlusion":  0,
            "alpha":      alpha,
            "bbox":       box,
            "location":   [X, Y, Z],
            "rotation_y": ry,
            "dimensions": [h0, w0, l0]
        }
        res = minimize(objective_hwl, [h0,w0,l0], args=(lbl,P2), bounds=[(1,3),(1,3),(2,6)])
        lbl["dimensions"] = res.x.tolist()
        corners3d        = compute_3d_box(*lbl["dimensions"], *lbl["location"], lbl["rotation_y"])
        lbl["corners_2d"] = project_to_image(corners3d, P2)
        labels.append(lbl)
    return labels

# === Main loop over val set ===
with open(IMGSET_FILE, 'r') as f:
    img_ids = [l.strip() for l in f if l.strip()]

for img_id in img_ids:
    # build paths
    image_path   = os.path.join(IMG_DIR,    f"{img_id}.png")
    calib_path   = os.path.join(CALIB_DIR,  f"{img_id}.txt")
    boxes2d_path = os.path.join(BOX2D_DIR,  f"{img_id}.txt")
    depth_path   = os.path.join(VIS_DIR,    f"depth_{img_id}.txt")
    monocd_path  = os.path.join(VIS_DIR,    f"boxes3d_{img_id}.png")

    # load original
    img = cv2.imread(image_path)
    if img is None:
        print(f"[{img_id}] missing image, skipping")
        continue

    # check 2D file
    lines = []
    if os.path.isfile(boxes2d_path):
        with open(boxes2d_path,'r') as f:
            lines = [l for l in f if l.strip()]
    empty_2d = (len(lines)==0)

    # 1) 2D output
    out2d = os.path.join(RESULT_DIR, f"2d_{img_id}.png")
    if empty_2d:
        cv2.imwrite(out2d, img)
    else:
        boxes2d = np.loadtxt(boxes2d_path)
        if boxes2d.ndim==0: boxes2d=boxes2d.reshape(0,4)
        elif boxes2d.ndim==1: boxes2d=boxes2d[np.newaxis,:]
        img2d = img.copy()
        for x1,y1,x2,y2 in boxes2d:
            cv2.rectangle(img2d,(int(x1),int(y1)),(int(x2),int(y2)),(255,0,0),2)
        cv2.imwrite(out2d,img2d)

    # 2) 3D output
    out3d = os.path.join(RESULT_DIR, f"3d_{img_id}.png")
    can3d = not empty_2d and os.path.isfile(depth_path)
    if not can3d:
        cv2.imwrite(out3d, img)
    else:
        depth = np.loadtxt(depth_path)
        P2    = load_calibration(calib_path)
        boxes2d = np.loadtxt(boxes2d_path)
        if boxes2d.ndim==0: boxes2d=boxes2d.reshape(0,4)
        elif boxes2d.ndim==1: boxes2d=boxes2d[np.newaxis,:]
        labels = estimate_3d_boxes(img, depth, P2, boxes2d)
        img3d  = img.copy()
        for lbl in labels:
            draw_3d_box(img3d, np.array(lbl["corners_2d"]))
        cv2.imwrite(out3d, img3d)

    # 3) comparison
    cmp_out = os.path.join(RESULT_DIR, f"compare_{img_id}.png")
    mono = cv2.imread(monocd_path) if os.path.isfile(monocd_path) else img
    mmd  = cv2.imread(out3d)
    mono = cv2.cvtColor(mono,cv2.COLOR_BGR2RGB)
    mmd  = cv2.cvtColor(mmd, cv2.COLOR_BGR2RGB)
    fig, (a1,a2)=plt.subplots(1,2,figsize=(12,6))
    a1.imshow(mono);a1.set_title("MonoCD");a1.axis("off")
    a2.imshow(mmd);a2.set_title("MMDetection");a2.axis("off")
    plt.tight_layout();plt.savefig(cmp_out,bbox_inches="tight");plt.close(fig)

    # 4) KITTI‐style label file (always create)
    label_out = os.path.join(RESULT_DIR, f"{img_id}.txt")
    with open(label_out,'w') as f:
        if not empty_2d:
            for lbl in labels:
                x1,y1,x2,y2=lbl["bbox"]
                h,w,l   =lbl["dimensions"]
                X,Y,Z   =lbl["location"]
                ry,alp  =lbl["rotation_y"],lbl["alpha"]
                f.write(
                    f"{lbl['type']} {lbl['truncation']:.2f} {lbl['occlusion']} "
                    f"{alp:.2f} {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f} "
                    f"{h:.2f} {w:.2f} {l:.2f} "
                    f"{X:.2f} {Y:.2f} {Z:.2f} {ry:.2f}\n"
                )

    print(f"[{img_id}] → 2D:{os.path.basename(out2d)}, 3D:{os.path.basename(out3d)}, cmp:{os.path.basename(cmp_out)}, label:{os.path.basename(label_out)}")
