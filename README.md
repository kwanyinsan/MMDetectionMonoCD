# MMDetectionMonoCD

## Credits And Read First

This FYP project builds on two official open-source projects:

- [MonoCD: Monocular 3D Object Detection with Complementary Depths](https://github.com/dragonfly606/MonoCD)
- [MMDetection: OpenMMLab Detection Toolbox and Benchmark](https://github.com/open-mmlab/mmdetection)

Please read the official MonoCD and MMDetection repositories first, especially
their installation, dataset preparation, training, and evaluation instructions.
This project also uses the KITTI object detection dataset; the KITTI dataset
download link and expected dataset preparation are provided in the official
MonoCD repository. This repository documents my FYP integration work on top of
them: using MonoCD for monocular depth / 3D cues and MMDetection for a
self-trained 2D car detector, then combining both outputs through a late-fusion
post-processing pipeline.

This repository is an FYP prototype for fusing monocular depth estimation from
MonoCD with a self-trained 2D object detector based on MMDetection.

The current implementation is a late-fusion pipeline. MonoCD and MMDetection are
run as two separate models, then their outputs are combined in post-processing:

- MonoCD predicts monocular 3D detection outputs and exports a depth map.
- MMDetection predicts 2D car bounding boxes.
- `result.py` combines MMDetection 2D boxes with MonoCD depth and KITTI camera
  calibration to estimate 3D boxes.

## Pipeline Overview

The pipeline has two independent model branches and one final fusion stage.

**MonoCD branch**

```text
KITTI images + calibration
    -> run MonoCD inference
    -> export depth maps to output/visualization/depth_*.txt
    -> export MonoCD 3D visualizations to output/visualization/boxes3d_*.png
```

**MMDetection branch**

```text
KITTI images + KITTI labels
    -> convert KITTI labels to COCO format with mmdet/mm_tools/convert.py
    -> train the MMDetection CenterNet 2D car detector
    -> run MMDetection inference and save results.pkl
    -> extract 2D boxes with manipulate.py
    -> save boxes to boxes2d/*.txt
```

**Fusion stage**

```text
MonoCD depth maps
+ MMDetection 2D boxes
+ KITTI camera calibration
    -> run result.py
    -> estimate approximate 3D boxes
    -> save result/*.png and KITTI-style result/*.txt files
```

In short:

```text
MonoCD gives depth and 3D cues.
MMDetection gives 2D object boxes.
result.py combines both outputs with KITTI calibration to produce fused 3D boxes.
```

## Stage 1: Prepare KITTI

Update the dataset paths before running the pipeline.

Files that currently contain placeholder paths:

- `config/paths_catalog.py`
- `mmdet/mm_tools/convert.py`
- `result.py`

The expected KITTI layout is:

```text
KITTI/
  training/
    image_2/
    label_2/
    calib/
    planes/
    ImageSets/
      train.txt
      val.txt
  testing/
    image_2/
    calib/
```

## Stage 2: Run MonoCD

MonoCD is the monocular 3D detector and depth-estimation branch. The main config
is:

```text
runs/monocd.yaml
```

It uses a DLA/DCNv2 backbone and predicts:

- 2D bounding box dimensions
- 3D center offset
- 3D dimensions
- orientation
- direct depth
- keypoint-based depth
- compensated depth
- depth uncertainty
- horizon / ground-plane cues

Train MonoCD:

```bash
python tools/plain_train_net.py --config runs/monocd.yaml --batch_size 8 --num_work 8 --output output
```

Evaluate or visualize MonoCD with a checkpoint:

```bash
python tools/plain_train_net.py --config runs/monocd.yaml --eval --ckpt path/to/monocd_checkpoint.pth --output output --vis
```

The visualization path is important because the current fusion pipeline reads
the exported depth files from:

```text
output/visualization/depth_000000.txt
output/visualization/boxes3d_000000.png
output/visualization/heatmap_000000.png
output/visualization/bev_000000.png
```

## Stage 3: Train The MMDetection 2D Detector

The MMDetection branch is a one-class `Car` detector. Its config is:

```text
mmdet/resnet.py
```

It uses CenterNet with a ResNet-18 backbone and a COCO-style dataset converted
from KITTI.

Convert KITTI labels to COCO format:

```bash
python mmdet/mm_tools/convert.py
```

This creates:

```text
mm_data/
  train/
  val/
  annotations/
    instances_train.json
    instances_val.json
```

Train the 2D detector:

```bash
python mmdet/mm_tools/train.py mmdet/resnet.py --work-dir work_dirs/centernet_resnet18_car
```

Run inference and dump predictions:

```bash
python mmdet/mm_tools/test.py mmdet/resnet.py path/to/mmdet_checkpoint.pth --out results.pkl
```

## Stage 4: Export MMDetection 2D Boxes

`manipulate.py` reads the MMDetection pickle output and writes plain text 2D
bounding boxes.

```bash
python manipulate.py results.pkl
```

Output:

```text
boxes2d/
  000000.txt
  000001.txt
  ...
```

Each line stores:

```text
x1 y1 x2 y2
```

The script currently keeps boxes with score greater than `0.45`.

## Stage 5: Fuse 2D Boxes With MonoCD Depth

Run:

```bash
python result.py
```

For each validation image, `result.py`:

1. Loads the original KITTI image.
2. Loads the MMDetection 2D boxes from `boxes2d/*.txt`.
3. Loads the MonoCD depth map from `output/visualization/depth_*.txt`.
4. Loads KITTI camera calibration from `training/calib/*.txt`.
5. Samples the depth value at the center of each 2D box.
6. Back-projects the 2D center point into 3D using the camera matrix.
7. Initializes a car-sized 3D box.
8. Optimizes the 3D box dimensions so its projection better matches the 2D box.
9. Draws the estimated 3D box and writes a KITTI-style label file.

Output:

```text
result/
  2d_000000.png
  3d_000000.png
  compare_000000.png
  000000.txt
```

The comparison image shows:

- left: MonoCD original 3D visualization
- right: MMDetection + MonoCD-depth fused result

## What Is Being Fused

The fusion currently happens after model inference:

```text
MMDetection 2D box + MonoCD depth map + KITTI calibration -> estimated 3D box
```

This is not end-to-end feature fusion. MMDetection does not feed features or
boxes into MonoCD during training. Instead, the final 3D estimate is created by
post-processing the outputs of both models.

## Main Files

| File | Purpose |
| --- | --- |
| `runs/monocd.yaml` | Main MonoCD training and inference config |
| `model/detector.py` | MonoCD detector wrapper |
| `model/head/detector_infer.py` | MonoCD post-processing and depth decoding |
| `engine/visualize_infer.py` | Saves depth maps and visualization outputs |
| `mmdet/resnet.py` | MMDetection CenterNet-ResNet18 config |
| `mmdet/mm_tools/convert.py` | Converts KITTI annotations to COCO format |
| `mmdet/mm_tools/train.py` | MMDetection training entry point |
| `mmdet/mm_tools/test.py` | MMDetection testing and pickle export |
| `manipulate.py` | Converts MMDetection pickle predictions to 2D box text files |
| `result.py` | Final 2D-depth-calibration fusion script |

## Current Limitation

The fused 3D box is an approximate geometric reconstruction. It depends heavily
on the 2D box quality and on the sampled depth value at the box center. For a
stronger version, the fusion could use robust depth statistics inside each 2D
box, preserve MMDetection confidence scores, estimate orientation more carefully,
and evaluate the fused labels with KITTI 3D AP.
