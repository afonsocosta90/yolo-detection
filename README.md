# Object Detection with YOLOv8

A learning project exploring real-time object detection and tracking with [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics), OpenCV, and [cvzone](https://github.com/cvzone/cvzone). Runs on AMD/Intel GPUs via [torch-directml](https://learn.microsoft.com/en-us/windows/ai/directml/pytorch-windows), with CUDA and CPU fallbacks.

## Scripts

| File | What it does |
| --- | --- |
| [1_yolo_basics.py](1_yolo_basics.py) | Runs YOLOv8l on a single image and displays the annotated result. |
| [2_yolo_webcam.py](2_yolo_webcam.py) | Live detection on a webcam feed or video file, with custom bounding-box rendering via cvzone. |
| [3_car_counter.py](3_car_counter.py) | Tracks and counts unique cars across a video using YOLOv8's built-in ByteTrack integration. |
| [video.py](video.py) | Helper for pulling sample driving footage from Kaggle via `kagglehub`. |

## Requirements

- Python 3.10+
- Windows (DirectML path); Linux/macOS with CUDA or CPU also works
- Dependencies in [requirements.txt](requirements.txt)
- A YOLOv8 weights file (e.g. `yolov8l.pt`) placed in a `yolo-weights/` directory next to the project

## Setup

```bash
python -m venv yolo_env
yolo_env\Scripts\activate          # PowerShell / cmd
# source yolo_env/bin/activate     # bash / zsh

pip install -r requirements.txt
pip install torch torch-directml   # for AMD/Intel GPUs on Windows
```

Download the weights from the [Ultralytics releases](https://github.com/ultralytics/assets/releases) and place them in `../yolo-weights/` relative to the scripts (or edit `MODEL_PATH` in each file).

## Usage

```bash
python 1_yolo_basics.py           # single image inference
python 2_yolo_webcam.py           # live webcam / video detection
python 3_car_counter.py           # track + count cars in a video
```

Inside each script, the top `CONFIGURATION` block controls:

- `INPUT_SOURCE` — `"camera"` or `"video"`
- `CAMERA_INDEX`, `CAMERA_WIDTH`, `CAMERA_HEIGHT`
- `VIDEO_PATH` — absolute path to a video file
- `DEVICE_TYPE` — `"directml"`, `"cuda"`, or `"cpu"`
- `TRACKED_CLASS` (car counter only) — any label from [class-names.txt](class-names.txt)

### Controls

- `Space` — pause / resume
- `q` — quit

## DirectML notes

DirectML has a couple of rough edges that the scripts patch at import time:

1. `torch.inference_mode` is aliased to `torch.no_grad` because DirectML can't set `version_counter` on inference-mode tensors, and Ultralytics wraps `predict()` in `@smart_inference_mode`.
2. `ultralytics.utils.nms.non_max_suppression` is wrapped to move predictions to CPU for NMS only — inference itself still runs on the GPU. This works around DirectML failures on `torch.cat` + boolean indexing inside NMS.

## Project structure

```
object-detection/
├── 1_yolo_basics.py
├── 2_yolo_webcam.py
├── 3_car_counter.py
├── video.py
├── class-names.txt
├── requirements.txt
├── images/           # sample input images
├── videos/           # input videos (gitignored)
├── runs/             # YOLO prediction outputs (gitignored)
├── yolo-weights/     # model weights (gitignored)
└── yolo_env/         # virtualenv (gitignored)
```
