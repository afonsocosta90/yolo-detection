"""Live YOLOv8 detection on a webcam feed or video file."""
import lib.directml  # noqa: F401  -- side-effect import: must precede ultralytics

import math
import cv2
import cvzone
from ultralytics import YOLO

import config

device = lib.directml.get_device(config.DEVICE_TYPE)
print(f"Using device: {device}")

# ─────────────────────────────────────────
# Input setup
# ─────────────────────────────────────────
if config.INPUT_SOURCE == "camera":
    cap = cv2.VideoCapture(config.CAMERA_INDEX, cv2.CAP_DSHOW)
    cap.set(3, config.CAMERA_WIDTH)
    cap.set(4, config.CAMERA_HEIGHT)
    if not cap.isOpened():
        print(f"Camera index {config.CAMERA_INDEX} not found, falling back to index 0...")
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
elif config.INPUT_SOURCE == "video":
    cap = cv2.VideoCapture(str(config.VIDEO_PATH))
else:
    raise ValueError(f"Unknown INPUT_SOURCE '{config.INPUT_SOURCE}'. Use 'camera' or 'video'.")

if not cap.isOpened():
    src = config.VIDEO_PATH if config.INPUT_SOURCE == "video" else f"camera index {config.CAMERA_INDEX}"
    raise FileNotFoundError(f"Could not open input source: {src}")
print(f"Input opened successfully: {config.VIDEO_PATH if config.INPUT_SOURCE == 'video' else f'Camera index {config.CAMERA_INDEX}'}")

# ─────────────────────────────────────────
# Model setup
# ─────────────────────────────────────────
with open(config.CLASS_NAMES_FILE, "r") as f:
    class_names = f.read().splitlines()

model = YOLO(str(config.MODEL_PATH))
print(f"Loaded {len(class_names)} classes from '{config.CLASS_NAMES_FILE}'")
print(f"Model ready: {config.MODEL_PATH}")

# ─────────────────────────────────────────
# Detection loop
# ─────────────────────────────────────────
is_paused = False

while True:
    if not is_paused:
        success, frame = cap.read()
        if not success:
            break

        results = model(frame, stream=True, device=device)

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                _, _, w, h     = box.xywh[0]
                bbox           = (x1, y1, int(w), int(h))
                confidence     = math.ceil(box.conf[0] * 100) / 100
                label          = class_names[int(box.cls[0])]

                cvzone.cornerRect(frame, bbox)
                cvzone.putTextRect(
                    frame,
                    f"{label}: {confidence}",
                    (max(0, x1), max(35, y1)),
                    scale=config.BBOX_TEXT_SCALE,
                    thickness=config.BBOX_TEXT_THICKNESS,
                )

        cv2.imshow("YOLOv8 Detection", frame)

    key = cv2.waitKey(0 if is_paused else 1) & 0xFF
    if key == ord(" "):
        is_paused = not is_paused
        print(f"Video Status: {'PAUSED' if is_paused else 'PLAYING'}")
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
