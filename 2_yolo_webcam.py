import torch
# DirectML can't set version_counter on inference-mode tensors.
# Ultralytics wraps predict() in @smart_inference_mode, so alias it to no_grad.
torch.inference_mode = torch.no_grad

import cv2
import cvzone
import math
import torch_directml

# DirectML fails on some ops inside NMS (torch.cat + bool indexing).
# Move predictions to CPU just for NMS; inference itself stays on GPU.
import ultralytics.utils.nms as _nms_mod
_orig_nms = _nms_mod.non_max_suppression
def _cpu_nms(prediction, *args, **kwargs):
    if isinstance(prediction, (list, tuple)):
        prediction = type(prediction)(
            p.detach().cpu() if torch.is_tensor(p) else p for p in prediction
        )
    elif torch.is_tensor(prediction):
        prediction = prediction.detach().cpu()
    return _orig_nms(prediction, *args, **kwargs)
_nms_mod.non_max_suppression = _cpu_nms

from ultralytics import YOLO

# ─────────────────────────────────────────
#                 CONFIGURATION
# ─────────────────────────────────────────

# Input source: "camera" or "video"
INPUT_SOURCE = "camera"

# Camera settings (used if INPUT_SOURCE = "camera")
CAMERA_INDEX  = 1
CAMERA_WIDTH  = 1280
CAMERA_HEIGHT = 720

# Video path (used if INPUT_SOURCE = "video")
VIDEO_PATH = r"C:\Users\afons\Documents\SwProjects\object-detection\videos\4.mp4"

# Model settings
MODEL_PATH       = "../yolo-weights/yolov8l.pt"
CLASS_NAMES_FILE = "class-names.txt"

# Device: "directml" (AMD/Intel GPU), "cuda" (Nvidia GPU), or "cpu"
DEVICE_TYPE = "directml"

# Detection display settings
BBOX_TEXT_SCALE     = 0.7
BBOX_TEXT_THICKNESS = 1

# ─────────────────────────────────────────
#                 DEVICE SETUP
# ─────────────────────────────────────────

if DEVICE_TYPE == "directml":
    device = torch_directml.device()
elif DEVICE_TYPE == "cuda":
    device = "cuda"
else:
    device = "cpu"

print(f"Using device: {device}")

# ─────────────────────────────────────────
#                 INPUT SETUP
# ─────────────────────────────────────────

if INPUT_SOURCE == "camera":
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    cap.set(3, CAMERA_WIDTH)
    cap.set(4, CAMERA_HEIGHT)
    if not cap.isOpened():
        print(f"Camera index {CAMERA_INDEX} not found, falling back to index 0...")
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

elif INPUT_SOURCE == "video":
    cap = cv2.VideoCapture(VIDEO_PATH)

else:
    raise ValueError(f"Unknown INPUT_SOURCE '{INPUT_SOURCE}'. Use 'camera' or 'video'.")

if not cap.isOpened():
    raise FileNotFoundError(f"Could not open input source: '{VIDEO_PATH if INPUT_SOURCE == 'video' else f'camera index {CAMERA_INDEX}'}'")
else:
    print(f"Input opened successfully: {VIDEO_PATH if INPUT_SOURCE == 'video' else f'Camera index {CAMERA_INDEX}'}")

# ─────────────────────────────────────────
#                 MODEL SETUP
# ─────────────────────────────────────────

with open(CLASS_NAMES_FILE, "r") as f:
    class_names = f.read().splitlines()

model = YOLO(MODEL_PATH)

print(f"Loaded {len(class_names)} classes from '{CLASS_NAMES_FILE}'")
print(f"Model ready: {MODEL_PATH}")

# ─────────────────────────────────────────
#              DETECTION LOOP
# ─────────────────────────────────────────

is_paused = False

while True:
    # Only read and process if not paused
    if not is_paused:
        success, frame = cap.read()
        if not success:
            break

        results = model(frame, stream=True, device=device)

        for r in results:
            for box in r.boxes:

                # Bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                _, _, w, h     = box.xywh[0]
                bbox           = (x1, y1, int(w), int(h))

                # Confidence score
                confidence = math.ceil(box.conf[0] * 100) / 100

                # Class label
                label = class_names[int(box.cls[0])]

                # Draw corner rectangle and label
                cvzone.cornerRect(frame, bbox)
                cvzone.putTextRect(
                    frame,
                    f"{label}: {confidence}",
                    (max(0, x1), max(35, y1)),
                    scale=BBOX_TEXT_SCALE,
                    thickness=BBOX_TEXT_THICKNESS,
                )

        cv2.imshow("YOLOv8 Detection", frame)

    # Logic for handling keypresses
    # If paused, wait indefinitely (0). If playing, wait 1ms.
    key = cv2.waitKey(0 if is_paused else 1) & 0xFF

    # Spacebar (32) to Toggle Pause
    if key == ord(" "):
        is_paused = not is_paused
        status = "PAUSED" if is_paused else "PLAYING"
        print(f"Video Status: {status}")

    # Press 'q' to quit
    if key == ord("q"):
        break

# ─────────────────────────────────────────
#                 CLEANUP
# ─────────────────────────────────────────

cap.release()
cv2.destroyAllWindows()