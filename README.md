# XFeat ONNX Export for the TruckImager Stitching Pipeline

This is Optocycle's fork of [XFeat: Accelerated Features for Lightweight Image Matching](https://github.com/verlab/accelerated_features) (CVPR 2024). We use it to export the trained XFeat weights to ONNX. The exported models are the keypoint detector and descriptor in the improved **TruckImager image stitching pipeline**.

The upstream PyTorch code (model, training, evaluation, demos) is unchanged. The Optocycle additions are the export script and the MLflow/Triton publishing in [deployment/](deployment/).

## What gets exported

[deployment/export_xfeat_onnx.py](deployment/export_xfeat_onnx.py) exports the whole sparse XFeat detector as one graph: CNN, NMS, top-k selection and descriptor sampling. The pipeline needs no PyTorch and no post-processing beyond filtering and rescaling.

| | Name | Shape | Notes |
|---|---|---|---|
| Input | `image` | `(B, 1, H, W)` | float32 grayscale in `[0, 1]`. `H` and `W` must be multiples of 32. Batch, height and width are dynamic. |
| Input | `threshold` | `(1,)` | float32 keypoint detection threshold (XFeat default `0.05`). It is a run-time input, so you can tune it without re-exporting. |
| Output | `keypoints` | `(B, K, 2)` | `(x, y)` pixel coordinates in the network input image, sorted by descending score. |
| Output | `scores` | `(B, K)` | Padding rows have score `-1`. |
| Output | `descriptors` | `(B, K, 64)` | L2-normalised. |

`K` is fixed at export time (`--k-max`). The output always has exactly `K` rows. If the image has fewer than `K` keypoints above the threshold, the remaining rows are padding. The export needs `H*W >= K`.

Per frame, on the consumer side:

1. Resize the frame and convert it to grayscale. Both happen outside the model.
2. Run the model.
3. Keep rows with `scores > 0`.
4. Rescale the keypoints from network-input pixels to original-frame pixels.

The export settings (`k_max`, `nms_kernel`, opset, torch version, git commit) are stored in the ONNX metadata properties, so you can see how a `.onnx` file was produced.

## Tiled mode vs. full-image mode

Export one model per mode. `K` is baked into the graph, and a smaller `K` makes the model cheaper.

| Mode | Input | Suggested `--k-max` |
|---|---|---|
| **Tiled**: the image is split into tiles, and features are extracted per tile | tile | `128` |
| **Full image**: features are extracted from the whole frame | whole frame | `1024` |

```bash
# Tiled mode
poetry run python deployment/export_xfeat_onnx.py --k-max 128  --out deployment/xfeat_tile_128.onnx

# Full-image mode
poetry run python deployment/export_xfeat_onnx.py --k-max 1024 --out deployment/xfeat_full_1024.onnx
```

The values above are starting points. Pick `K` to match the number of keypoints your stitching stage can use. You can use fewer than `K` at run time by slicing the (sorted) outputs. You cannot get more without re-exporting.

## Setup

Requires Python 3.12 and [Poetry](https://python-poetry.org/). The environment installs CPU PyTorch 2.3.1, ONNX, ONNX Runtime and MLflow.

```bash
git clone <this repository>
cd accelerated_features
poetry install
```

The trained weights are in [weights/xfeat.pt](weights/xfeat.pt) and are the default input.

## Export options

```bash
poetry run python deployment/export_xfeat_onnx.py --help
```

| Flag | Default | Description |
|---|---|---|
| `--weights` | `weights/xfeat.pt` | Trained XFeat weights. |
| `--out` | `deployment/xfeat.onnx` | Output file. |
| `--k-max` | `4096` | Rows per output (`K`). See the table above. |
| `--nms-kernel` | `5` | NMS max-pool window. XFeat uses 5. |
| `--opset` | `17` | ONNX opset. `GridSample` needs 16 or higher. |
| `--threshold` | `0.05` | Threshold used by the built-in checks only. |
| `--dummy-hw H W` | `640 576` | Resolution used for tracing. The exported graph still accepts any size that is a multiple of 32. |
| `--check-hw H W` | `512 608` | A second resolution for the checks, to verify the dynamic axes. |
| `--video PATH` `--frame N` | none | Optionally check on a real frame (for example a TruckImager video) as well as the synthetic images. |

Set `--k-max` to the value you will deploy. Note that `--k-max` is also the top-k of the PyTorch reference used in the checks. Use `--video` with a frame from your own footage when you want a meaningful check.

## Verification

Every export runs two comparisons against the reference `XFeat.detectAndCompute` on noise, flat and (optionally) real images at two resolutions:

1. The PyTorch wrapper against the reference.
2. The exported model run in ONNX Runtime against the reference.

For each image the script prints how many keypoints both versions found, how many are unique to each, and the maximum score and descriptor difference on the shared keypoints. The graph is written by the export step even if these numbers look off, so read the output before using a model.

Keep in mind that ties and near-threshold peaks can make a few keypoints differ between top-k and the reference's sort. Scores and descriptors on shared keypoints should agree to float precision.

Exported `*.onnx` and `*.onnx.data` files are git-ignored and are not versioned. Re-create them from the weights with the commands above, and use the metadata to trace where a file came from.

## Using the model

```python
import numpy as np
import onnxruntime as ort

session = ort.InferenceSession("deployment/xfeat_tile_128.onnx", providers=["CPUExecutionProvider"])

image = np.random.rand(1, 1, 256, 256).astype(np.float32)   # gray, [0, 1], H and W multiples of 32
threshold = np.array([0.05], dtype=np.float32)

keypoints, scores, descriptors = session.run(None, {"image": image, "threshold": threshold})

keep = scores[0] > 0
kpts, desc = keypoints[0][keep], descriptors[0][keep]        # (N, 2), (N, 64)
```

The resulting keypoints and descriptors can be matched with any matcher, for example a mutual nearest neighbour search on the descriptors.

## Publishing to MLflow and Triton

The models are published the same way as [optocycle/RAFT-Stereo](https://github.com/optocycle/RAFT-Stereo). Each one is logged to MLflow as a Triton model with the `triton` flavor ([deployment/triton_flavor.py](deployment/triton_flavor.py)), and then deployed to Triton with the MLflow Triton plugin from `oc_ml/inference-server`.

Copy [.env.example](.env.example) to `.env` and fill in your MLflow credentials. Then export, check and publish in one run:

```bash
# Tiled mode: 4x4 tiles of 616x462 at scale 0.39 -> 256x192 network input per tile
poetry run python deployment/export_xfeat_onnx.py --k-max 128 --dummy-hw 192 256 \
    --out deployment/xfeat_tile_128.onnx --publish --triton-hw 192 256

# Full-image mode: any size that is a multiple of 32
poetry run python deployment/export_xfeat_onnx.py --k-max 1024 \
    --out deployment/xfeat_full_1024.onnx --publish
```

To publish a file that was already exported, run `poetry run python deployment/publish_xfeat_triton.py <file.onnx> --triton-hw 192 256`. Add `--dry-run DIR` to write only the Triton model directory, without logging anything.

| Flag | Default | Description |
|---|---|---|
| `--triton-hw H W` | `-1 -1` | Input height and width in `config.pbtxt`. `-1 -1` accepts any multiple of 32. For tiled mode, set it to the network size of one tile. |
| `--experiment` | `xfeat` | MLflow experiment. It is created if it does not exist. |
| `--registered-model-name` | none | Also registers the model under this name. Without it, register the run's `models` artifact in the MLflow UI. |

Each run logs `models/model/config.pbtxt` and `models/model/1/model.onnx`. It also logs the ONNX metadata (`k_max`, `nms_kernel`, opset, torch version, commit) and the Triton input size as run parameters. `K` in `config.pbtxt` comes from the ONNX metadata, so the config always matches the graph.

After registering the model, deploy it from `oc_ml/inference-server`:

```bash
poetry run mlflow deployments create -t triton --flavor triton --name xfeat_tile_k128 -m models:/<registered name>/<version>
```

The Triton config template is [deployment/xfeat.pbtxt](deployment/xfeat.pbtxt). Two of its settings differ from RAFT:

- It has **no `name`**. Triton then takes the model name from its directory, which is the deployment `--name`, so the two cannot get out of sync.
- It uses **`max_batch_size: 0`**. `threshold` has shape `(1,)` for the whole batch, so Triton cannot add a batch dimension to it. The batch dimension is therefore part of the dims (`image: [-1, 1, H, W]`). In tiled mode, send all tiles of a frame in one request, exactly as with ONNX Runtime. Triton does not merge separate requests into one batch.

The Triton model has the same inputs and outputs as the ONNX file (see [What gets exported](#what-gets-exported)). A gRPC call for one frame's tiles looks like this:

```python
import numpy as np
import tritonclient.grpc as tritonclient

client = tritonclient.InferenceServerClient(url="triton-inference.oc-ml.svc:8001")
tiles = np.random.rand(16, 1, 192, 256).astype(np.float32)  # (n_tiles, 1, H, W), gray in [0, 1]
inputs = [tritonclient.InferInput("image", tiles.shape, "FP32"), tritonclient.InferInput("threshold", [1], "FP32")]
inputs[0].set_data_from_numpy(tiles)
inputs[1].set_data_from_numpy(np.array([0.05], np.float32))
res = client.infer("xfeat_tile_k128", inputs)
keypoints, scores = res.as_numpy("keypoints"), res.as_numpy("scores")  # (16, 128, 2), (16, 128)
```

## Using the PyTorch model

The upstream API still works, for prototyping and for comparison with the ONNX output:

```python
import torch
from modules.xfeat import XFeat

xfeat = XFeat()
output = xfeat.detectAndCompute(torch.randn(1, 1, 480, 640), top_k=1024)[0]
```

Upstream evaluation (MegaDepth-1500, ScanNet-1500), training, the real-time demo and the XFeat + LighterGlue matcher are still in the repository. They are documented in the [original XFeat README](#original-xfeat-readme) below. Those scripts have their own dependencies (see [requirements.txt](requirements.txt)) that the Poetry environment does not install.

## Attribution and license

XFeat was created by Guilherme Potje, Felipe Cadar, Andre Araujo, Renato Martins and Erickson R. Nascimento (VeRLab, UFMG). Please cite the paper if you use it:

```bibtex
@INPROCEEDINGS{potje2024cvpr,
  author={Potje, Guilherme and Cadar, Felipe and Araujo, André and Martins, Renato and Nascimento, Erickson R.},
  booktitle={2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  title={XFeat: Accelerated Features for Lightweight Image Matching},
  year={2024},
  pages={2682-2691},
  doi={10.1109/CVPR52733.2024.00259}}
```

Licensed under Apache 2.0, see [LICENSE](LICENSE).

---

# Original XFeat README

Everything below is the unmodified upstream README from [verlab/accelerated_features](https://github.com/verlab/accelerated_features).

---

## XFeat: Accelerated Features for Lightweight Image Matching
[Guilherme Potje](https://guipotje.github.io/) · [Felipe Cadar](https://eucadar.com/) · [Andre Araujo](https://andrefaraujo.github.io/) · [Renato Martins](https://renatojmsdh.github.io/) · [Erickson R. Nascimento](https://homepages.dcc.ufmg.br/~erickson/)

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/verlab/accelerated_features/blob/main/notebooks/xfeat_matching.ipynb)  
[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm-dark.svg)](https://huggingface.co/spaces/qubvel-hf/xfeat)

### [[ArXiv]](https://arxiv.org/abs/2404.19174) | [[Project Page]](https://www.verlab.dcc.ufmg.br/descriptors/xfeat_cvpr24/) |  [[CVPR'24 Paper]](https://openaccess.thecvf.com/content/CVPR2024/html/Potje_XFeat_Accelerated_Features_for_Lightweight_Image_Matching_CVPR_2024_paper.html)

- Training code is now available -> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/verlab/accelerated_features/blob/main/notebooks/XFeat_training_example.ipynb)
- 🎉 **New!** XFeat + LighterGlue (smaller version of LightGlue) available! 🚀 [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/verlab/accelerated_features/blob/main/notebooks/xfeat%2Blg_torch_hub.ipynb)

<div align="center" style="display: flex; justify-content: center; align-items: center; flex-direction: column;">
  <div style="display: flex; justify-content: space-around; width: 100%;">
    <img src='./figs/xfeat.gif' width="400"/>
    <img src='./figs/sift.gif' width="400"/>
  </div>
  
  Real-time XFeat demonstration (left) compared to SIFT (right) on a textureless scene. SIFT cannot handle fast camera movements, while XFeat provides robust matches under adverse conditions, while being faster than SIFT on CPU.
  
</div>

**TL;DR**: Really fast learned keypoint detector and descriptor. Supports sparse and semi-dense matching.

Just wanna quickly try on your images? Check this out: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/verlab/accelerated_features/blob/main/notebooks/xfeat_torch_hub.ipynb) [![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm-dark.svg)](https://huggingface.co/spaces/qubvel-hf/xfeat)

## Table of Contents
- [Introduction](#introduction) <img align="right" src='./figs/xfeat_quali.jpg' width=360 />
- [Installation](#installation)
- [Usage](#usage)
  - [Inference](#inference)
  - [Training](#training)
  - [Evaluation](#evaluation)
- [Real-time demo app](#real-time-demo)
- [XFeat+LightGlue](#xfeat-with-lightglue)
- [Contribute](#contributing)
- [Citation](#citation)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Introduction
This repository contains the official implementation of the paper: *[XFeat: Accelerated Features for Lightweight Image Matching](https://arxiv.org/abs/2404.19174)*, to be presented at CVPR 2024.

**Motivation.** Why another keypoint detector and descriptor among dozens of existing ones? We noticed that the current trend in the literature focuses on accuracy but often neglects compute efficiency, especially when deploying these solutions in the real-world. For applications in mobile robotics and augmented reality, it is critical that models can run on hardware-constrained computers. To this end, XFeat was designed as an agnostic solution focusing on both accuracy and efficiency in an image matching pipeline.

**Capabilities.**
- Real-time sparse inference on CPU for VGA images (tested on laptop with an i5 CPU and vanilla pytorch);
- Simple architecture components which facilitates deployment on embedded devices (jetson, raspberry pi, custom AI chips, etc..);
- Supports both sparse and semi-dense matching of local features;
- Compact descriptors (64D);
- Performance comparable to known deep local features such as SuperPoint while being significantly faster and more lightweight. Also, XFeat exhibits much better robustness to viewpoint and illumination changes than classic local features as ORB and SIFT;
- Supports batched inference if you want ridiculously fast feature extraction. On VGA sparse setting, we achieved about 1,400 FPS using an RTX 4090.
- For single batch inference on GPU (VGA), one can easily achieve over 150 FPS while leaving lots of room on the GPU for other concurrent tasks.

##

**Paper Abstract.** We introduce a lightweight and accurate architecture for resource-efficient visual correspondence. Our method, dubbed XFeat (Accelerated Features), revisits fundamental design choices in convolutional neural networks for detecting, extracting, and matching local features. Our new model satisfies a critical need for fast and robust algorithms suitable to resource-limited devices. In particular, accurate image matching requires sufficiently large image resolutions -- for this reason, we keep the resolution as large as possible while limiting the number of channels in the network. Besides, our model is designed to offer the choice of matching at the sparse or semi-dense levels, each of which may be more suitable for different downstream applications, such as visual navigation and augmented reality. Our model is the first to offer semi-dense matching efficiently, leveraging a novel match refinement module that relies on coarse local descriptors. XFeat is versatile and hardware-independent, surpassing current deep learning-based local features in speed (up to 5x faster) with comparable or better accuracy, proven in pose estimation and visual localization. We showcase it running in real-time on an inexpensive laptop CPU without specialized hardware optimizations.

**Overview of XFeat's achitecture.**
XFeat extracts a keypoint heatmap $\mathbf{K}$, a compact 64-D dense descriptor map $\mathbf{F}$, and a reliability heatmap $\mathbf{R}$. It achieves unparalleled speed via early downsampling and shallow convolutions, followed by deeper convolutions in later encoders for robustness. Contrary to typical methods, it separates keypoint detection into a distinct branch, using $1 \times 1$ convolutions on an $8 \times 8$ tensor-block-transformed image for fast processing, being one of the few current learned methods that decouples detection & description and can be processed independently.

<img align="center" src="./figs/xfeat_arq.png" width=1000 />


## Timing Analyses on CPU.

We show that both detection branch & match refinement module costs are small and bring significant advantages in accuracy (please check the ablation section in the paper).

<img align="center" src="./figs/timings.png" width=840 />


Furthermore, XFeat performs effectively in both indoor and outdoor scenes, achieving an excellent compute-accuracy trade-off as demonstrated below. Note that in the paper, the teaser figure has a VGA resolution on the x-axis and 1,200 pixels on the y-axis. Below, we present an updated figure for improved clarity, maintaining the same x-y axis resolution.

<img align="center" src="./figs/speed_accuracy.png" width=840 />


## Installation
XFeat has minimal dependencies, only relying on torch. Also, XFeat does not need a GPU for real-time sparse inference (vanilla pytorch w/o any special optimization), unless you run it on high-res images. If you want to run the real-time matching demo, you will also need OpenCV.
We recommend using conda, but you can use any virtualenv of your choice.
If you use conda, just create a new env with:
```bash
git clone https://github.com/verlab/accelerated_features.git
cd accelerated_features

#Create conda env
conda create -n xfeat python=3.8
conda activate xfeat
```

Then, install [pytorch (>=1.10)](https://pytorch.org/get-started/previous-versions/) and then the rest of depencencies in case you want to run the demos:
```bash

#CPU only, for GPU check in pytorch website the most suitable version to your gpu.
pip install torch==1.10.1+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html
# CPU only for MacOS
# pip install torch==1.10.1 -f https://download.pytorch.org/whl/cpu/torch_stable.html

#Install dependencies for the demo
pip install opencv-contrib-python tqdm
```

## Usage

For your convenience, we provide ready to use notebooks for some examples.

|            **Description**     |  **Notebook**                     |
|--------------------------------|-------------------------------|
| Minimal example | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/verlab/accelerated_features/blob/main/notebooks/minimal_example.ipynb) |
| Matching & registration example | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/verlab/accelerated_features/blob/main/notebooks/xfeat_matching.ipynb) |
| Torch hub example | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/verlab/accelerated_features/blob/main/notebooks/xfeat_torch_hub.ipynb) |
| Training example (synthetic) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/verlab/accelerated_features/blob/main/notebooks/XFeat_training_example.ipynb) |
| XFeat + LightGlue | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/verlab/accelerated_features/blob/main/notebooks/xfeat%2Blg_torch_hub.ipynb) |


### Inference
To run XFeat on an image, three lines of code is enough:
```python
from modules.xfeat import XFeat

xfeat = XFeat()

#Simple inference with batch sz = 1
output = xfeat.detectAndCompute(torch.randn(1,3,480,640), top_k = 4096)[0]
```
Or you can use this [script](./minimal_example.py) in the root folder:
```bash
python3 minimal_example.py
```

If you already have pytorch, simply use torch hub if you like it:
```python
import torch

xfeat = torch.hub.load('verlab/accelerated_features', 'XFeat', pretrained = True, top_k = 4096)

#Simple inference with batch sz = 1
output = xfeat.detectAndCompute(torch.randn(1,3,480,640), top_k = 4096)[0]
```

### Training
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/verlab/accelerated_features/blob/main/notebooks/XFeat_training_example.ipynb)

To train XFeat as described in the paper, you will need MegaDepth & COCO_20k subset of COCO2017 dataset.
You can obtain the full COCO2017 train data at https://cocodataset.org/.
However, we [make available](https://drive.google.com/file/d/1ijYsPq7dtLQSl-oEsUOGH1fAy21YLc7H/view?usp=drive_link) a subset of COCO for convenience. We simply selected a subset of 20k images according to image resolution. Please check COCO [terms of use](https://cocodataset.org/#termsofuse) before using the data.

To reproduce the training setup from the paper, please follow the steps:
1. Download [COCO_20k](https://drive.google.com/file/d/1ijYsPq7dtLQSl-oEsUOGH1fAy21YLc7H/view?usp=drive_link) containing a subset of COCO2017;
2. Download MegaDepth dataset. You can follow [LoFTR instructions](https://github.com/zju3dv/LoFTR/blob/master/docs/TRAINING.md#download-datasets), we use the same standard as LoFTR. Then put the megadepth indices inside the MegaDepth root folder following the standard below:
```bash
{megadepth_root_path}/train_data/megadepth_indices #indices
{megadepth_root_path}/MegaDepth_v1 #images & depth maps & poses
```
3. Finally you can call training
```bash
python3 -m modules.training.train --training_type xfeat_default  --megadepth_root_path <path_to>/MegaDepth --synthetic_root_path <path_to>/coco_20k --ckpt_save_path /path/to/ckpts
```

### Evaluation
----
**MegaDepth-1500**

Please note that due to the stochastic nature of RANSAC and major code refactoring, you may observe slightly different AuC results; however, they should be very close to those reported in the paper.

To evaluate on the MegaDepth dataset, you need to first get the dataset:
```bash
python3 -m modules.dataset.download --megadepth-1500 --download_dir </path/to/desired/folder>
```
Then, you call the mega1500 eval script, you can choose between `xfeat, xfeat-star and alike`. It should take about a minute to run the benchmark:
```bash
python3 -m modules.eval.megadepth1500 --dataset-dir </data/Mega1500> --matcher xfeat --ransac-thr 2.5
```
---
**ScanNet-1500**

To evaluate on the ScanNet eval dataset, you need to first get the dataset:
```bash
python3 -m modules.dataset.download --scannet-1500 --download_dir </path/to/desired/folder>
```

Then, you can call the scannet1500 eval script, it should take a couple of minutes:
```bash
python3 -m modules.eval.scannet1500 --scannet_path </data/ScanNet1500> --output </data/ScanNet1500/output> && python3 -m modules.eval.scannet1500 --scannet_path </data/ScanNet1500> --output </data/ScanNet1500/output> --show
```

---

## Real-time Demo
To demonstrate the capabilities of XFeat, we provide a real-time matching demo with Homography registration. Currently, you can experiment with XFeat, ORB and SIFT. You will need a working webcam. To run the demo and show the possible input flags, please run:
```bash
python3 realtime_demo.py -h
```

Don't forget to press 's' to set a desired reference image. Notice that the demo only works correctly for planar scenes and rotation-only motion, because we're using a homography model.

If you want to run the demo with XFeat, please run:
```bash
python3 realtime_demo.py --method XFeat
```

Or test with SIFT or ORB:
```bash
python3 realtime_demo.py --method SIFT
python3 realtime_demo.py --method ORB
```

## XFeat with LightGlue
We have trained a lighter version of LightGlue (LighterGlue). It has fewer parameters and is approximately three times faster than the original LightGlue. Special thanks to the developers of the [GlueFactory](https://github.com/cvg/glue-factory) library, which enabled us to train this version of LightGlue with XFeat.
Below, we compare the original SP + LG using the [GlueFactory](https://github.com/cvg/glue-factory) evaluation script on MegaDepth-1500.
Please follow the example to test on your own images:  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/verlab/accelerated_features/blob/main/notebooks/xfeat%2Blg_torch_hub.ipynb)

Metrics (AUC @ 5 / 10 / 20)
| Setup           | Max Dimension | Keypoints | XFeat + LighterGlue           | SuperPoint + LightGlue (Official) |
|-----------------|---------------|-----------|-------------------------------|-----------------------------------|
| **Fast**  | 640           | 1300      | 0.444 / 0.610 / 0.746       | 0.469 / 0.633 / 0.762          |
| **Accurate** | 1024          | 4096      | 0.564 / 0.710 / 0.819       | 0.591 / 0.738 / 0.841            |

## Contributing
Contributions to XFeat are welcome! 
Currently, it would be nice to have an export script to efficient deployment engines such as TensorRT and ONNX. Also, it would be cool to train other lightweight learned matchers on top of XFeat local features.

## Citation
If you find this code useful for your research, please cite the paper:

```bibtex
@INPROCEEDINGS{potje2024cvpr,
  author={Potje, Guilherme and Cadar, Felipe and Araujo, André and Martins, Renato and Nascimento, Erickson R.},
  booktitle={2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)}, 
  title={XFeat: Accelerated Features for Lightweight Image Matching}, 
  year={2024},
  pages={2682-2691},
  keywords={Visualization;Accuracy;Image matching;Pose estimation;Feature extraction;Hardware;Real-time systems;Image matching;Local features;Lightweight;Fast},
  doi={10.1109/CVPR52733.2024.00259}}
```

## License
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

## Acknowledgements
- We thank the agencies CAPES, CNPq, and Google for funding different parts of this work.
- We thank the developers of Kornia for the [kornia library](https://github.com/kornia/kornia)!

**VeRLab:** Laboratory of Computer Vison and Robotics https://www.verlab.dcc.ufmg.br
<br>
<img align="left" width="auto" height="50" src="./figs/ufmg.png">
<img align="right" width="auto" height="50" src="./figs/verlab.png">
<br/>
