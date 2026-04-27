"""Track and count unique objects (cars by default) across a video."""
import lib.directml  # noqa: F401  -- side-effect import: must precede ultralytics

import math
import time
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
# Detection + tracking loop
# ─────────────────────────────────────────
is_paused = False
seen_ids = set()
diagnostics_logged = False

while True:
    if not is_paused:
        success, frame = cap.read()
        if not success:
            break

        results = model.track(frame, stream=True, device=device, persist=True, tracker="bytetrack.yaml")

        if not diagnostics_logged:
            print("Model param device:", next(model.model.parameters()).device)
            t = time.perf_counter()
            _ = model.predict(frame, device=device, verbose=False)
            print(f"Single-frame inference: {(time.perf_counter() - t) * 1000:.1f} ms")
            diagnostics_logged = True

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                _, _, w, h     = box.xywh[0]
                bbox           = (x1, y1, int(w), int(h))
                confidence     = math.ceil(box.conf[0] * 100) / 100
                label          = class_names[int(box.cls[0])]

                if label != config.TRACKED_CLASS:
                    continue
                if box.id is None:
                    continue

                track_id = int(box.id[0])
                seen_ids.add(track_id)

                if confidence > config.CONFIDENCE_THRESHOLD:
                    cvzone.cornerRect(frame, bbox)
                    cvzone.putTextRect(
                        frame,
                        f"{label} #{track_id}: {confidence}",
                        (max(0, x1), max(35, y1)),
                        scale=config.BBOX_TEXT_SCALE,
                        thickness=config.BBOX_TEXT_THICKNESS,
                    )

        cvzone.putTextRect(frame, f"Total {config.TRACKED_CLASS}s seen: {len(seen_ids)}", (20, 40))
        cv2.imshow("YOLOv8 Detection", frame)

    key = cv2.waitKey(0 if is_paused else 1) & 0xFF
    if key == ord(" "):
        is_paused = not is_paused
        print(f"Video Status: {'PAUSED' if is_paused else 'PLAYING'}")
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
