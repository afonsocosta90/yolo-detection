# Object Detection with YOLOv8

A learning project for real-time object detection and tracking with [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics), OpenCV, and [cvzone](https://github.com/cvzone/cvzone). Runs on AMD/Intel GPUs via [torch-directml](https://learn.microsoft.com/en-us/windows/ai/directml/pytorch-windows), with CUDA and CPU fallbacks.

## Project structure

```
object-detection/
├── README.md
├── requirements.txt
├── config.py              # all tunable settings — paths, device, display, class
├── lib/
│   └── directml.py        # device selection + DirectML workarounds (shared)
├── apps/                  # runnable applications
│   ├── yolo_image.py      # single-image inference + display
│   ├── yolo_webcam.py     # live detection on webcam or video
│   └── car_counter.py     # detection + ByteTrack tracking + unique counting
├── tools/
│   └── download_video.py  # fetch sample footage from Kaggle
├── data/
│   ├── images/            # sample input images
│   └── class-names.txt    # COCO label list
├── videos/                # input videos (gitignored — too large)
├── yolo-weights/          # YOLO weights (gitignored — too large)
└── runs/                  # YOLO prediction outputs (gitignored)
```

## What's where

| Concern | Location |
| --- | --- |
| Settings (paths, device, thresholds) | [config.py](config.py) |
| DirectML patches & GPU adapter selection | [lib/directml.py](lib/directml.py) |
| Single-image demo | [apps/yolo_image.py](apps/yolo_image.py) |
| Webcam / video live detection | [apps/yolo_webcam.py](apps/yolo_webcam.py) |
| Car counting with tracking | [apps/car_counter.py](apps/car_counter.py) |
| Kaggle dataset downloader | [tools/download_video.py](tools/download_video.py) |

All runtime configuration is in **[config.py](config.py)**. The apps don't redefine settings — they import them.

## Requirements

- Python 3.10+
- Windows 10/11 for DirectML; Linux/macOS works with CUDA or CPU
- Dependencies in [requirements.txt](requirements.txt)
- A YOLOv8 weights file (e.g. `yolov8l.pt`) in `yolo-weights/`

## Setup

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1          # PowerShell
# .\venv\Scripts\activate.bat        # cmd
# source venv/bin/activate           # bash / zsh

pip install -r requirements.txt
pip install torch torch-directml     # only for AMD/Intel GPUs on Windows
```

Download weights from [Ultralytics releases](https://github.com/ultralytics/assets/releases) into `yolo-weights/`. The default is `yolov8l.pt`; change `MODEL_PATH` in [config.py](config.py) to use a different size.

## Usage

Run apps as modules from the project root so imports resolve correctly:

```powershell
python -m apps.yolo_image          # single-image demo
python -m apps.yolo_webcam         # live webcam / video
python -m apps.car_counter         # car tracking + counting
```

### Controls

- `Space` — pause / resume
- `q` — quit

## Configuration

Edit [config.py](config.py) — every script reads from it.

| Setting | Purpose |
| --- | --- |
| `INPUT_SOURCE` | `"camera"` or `"video"` |
| `CAMERA_INDEX` / `CAMERA_WIDTH` / `CAMERA_HEIGHT` | Webcam capture |
| `IMAGE_PATH` / `VIDEO_PATH` | Input file paths |
| `MODEL_PATH` | Path to YOLO weights |
| `DEVICE_TYPE` | `"directml"`, `"cuda"`, or `"cpu"` |
| `TRACKED_CLASS` | Label to count (used by `car_counter`) |
| `CONFIDENCE_THRESHOLD` | Minimum confidence for drawn boxes |
| `BBOX_TEXT_SCALE` / `BBOX_TEXT_THICKNESS` | Label rendering |

## DirectML notes

DirectML has two rough edges that [lib/directml.py](lib/directml.py) patches at import time:

1. `torch.inference_mode` is aliased to `torch.no_grad` because DirectML can't set `version_counter` on inference-mode tensors, and Ultralytics wraps `predict()` in `@smart_inference_mode`.
2. NMS predictions are moved to CPU just for the `non_max_suppression` call. Inference itself stays on the GPU; this only works around `torch.cat` + bool indexing failures inside NMS.

`get_device("directml")` also auto-selects a discrete GPU (RX/RTX/Arc) over an integrated one when both are present, so the model doesn't accidentally run on the iGPU.

## Verifying GPU usage

[apps/car_counter.py](apps/car_counter.py) prints two diagnostic lines on the first frame:

```
Model param device: privateuseone:0
Single-frame inference: 45.2 ms
```

- `privateuseone:0` confirms the model lives on a DirectML device. `cpu` means it didn't move.
- Inference time on a discrete GPU should be 30–80 ms for YOLOv8l. If it's 500 ms+, you're likely on the iGPU.

Cross-check in **Task Manager → Performance → GPU** that the discrete GPU's Compute graph is active while running.
