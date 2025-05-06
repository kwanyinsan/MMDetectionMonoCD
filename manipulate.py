import os
import pickle
import numpy as np

def dump_bboxes(pkl_file: str):
    # fixed output directory
    out_dir = "boxes2d"
    os.makedirs(out_dir, exist_ok=True)

    # load the pickle, expected to be a list of dicts
    with open(pkl_file, 'rb') as f:
        results = pickle.load(f)

    # iterate over each image_data dict
    for data in results:
        img_id = data.get('img_id')
        if img_id is None:
            continue

        pred = data.get('pred_instances', {})
        bboxes = pred.get('bboxes')
        if bboxes is None:
            continue

        # convert torch.Tensor to numpy if needed
        if hasattr(bboxes, 'cpu'):
            bboxes = bboxes.cpu().numpy()
        bboxes = np.asarray(bboxes)
        bboxes = bboxes[:, :4]  # take first 4 columns

        scores = pred.get('scores', None)
        if scores is not None and hasattr(scores, 'cpu'):
            scores = scores.cpu().numpy()
        scores = np.asarray(scores) if scores is not None else None
        if scores is not None:
            mask = scores > 0.45
            bboxes = bboxes[mask]

        # write one line per box to boxes_2d/{img_id:06d}.txt
        fname = f"{int(img_id):06d}.txt"
        with open(os.path.join(out_dir, fname), 'w') as fw:
            for x1, y1, x2, y2 in bboxes:
                fw.write(f"{x1:.1f} {y1:.1f} {x2:.1f} {y2:.1f}\n")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python manipulate.py <path_to_results.pkl>")
        sys.exit(1)
    pkl_path = sys.argv[1]
    dump_bboxes(pkl_path)
