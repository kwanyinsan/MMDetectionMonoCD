import os
import pickle
import argparse
import numpy as np

def dump_bboxes(pkl_file: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    # 1) load the pickle, which is a list of dicts
    with open(pkl_file, 'rb') as f:
        results = pickle.load(f)

    # 2) iterate over each image_data dict
    for data in results:
        # extract the image id for filename
        img_id = data.get('img_id')
        if img_id is None:
            continue

        # extract the bbox tensor / array
        pred = data.get('pred_instances', {})
        bboxes = pred.get('bboxes')
        if bboxes is None:
            continue

        # convert torch.Tensor → numpy
        if hasattr(bboxes, 'cpu'):
            bboxes = bboxes.cpu().numpy()
        # ensure numpy array
        bboxes = np.asarray(bboxes)
        # take only the first 4 cols
        bboxes = bboxes[:, :4]

        scores = pred.get('scores')
        if hasattr(scores, 'cpu'):
            scores = scores.cpu().numpy()
        scores = np.asarray(scores)
        mask = scores > 0.45
        bboxes = bboxes[mask]

        # 3) write out one line per box
        fname = f"{int(img_id):06d}.txt"
        with open(os.path.join(out_dir, fname), 'w') as fw:
            for x1, y1, x2, y2 in bboxes:
                # one decimal place as in your example
                fw.write(f"{x1:.1f} {y1:.1f} {x2:.1f} {y2:.1f}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Dump bbox coords from a mmdet-style .pkl to per-image .txt files"
    )
    parser.add_argument("--pkl", required=True,
                        help="path to input pickle, e.g. work_dirs/.../results.pkl")
    parser.add_argument("--out-dir", required=True,
                        help="directory to save 6‑digit .txt files")
    args = parser.parse_args()
    dump_bboxes(args.pkl, args.out_dir)