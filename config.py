"""Central configuration for all apps. Edit values here, not inside scripts."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# ─────────────────────────────────────────
# Input source
# ─────────────────────────────────────────
# "camera" or "video"
INPUT_SOURCE = "video"

# Camera (used when INPUT_SOURCE == "camera")
CAMERA_INDEX  = 1
CAMERA_WIDTH  = 1280
CAMERA_HEIGHT = 720

# Image / video paths
IMAGE_PATH = PROJECT_ROOT / "data" / "images" / "4.jpg"
VIDEO_PATH = PROJECT_ROOT / "videos" / "5.mp4"

# ─────────────────────────────────────────
# Model
# ─────────────────────────────────────────
MODEL_PATH       = PROJECT_ROOT / "yolo-weights" / "yolov8l.pt"
CLASS_NAMES_FILE = PROJECT_ROOT / "data" / "class-names.txt"

# ─────────────────────────────────────────
# Compute device
# ─────────────────────────────────────────
# "directml" (AMD/Intel GPU on Windows), "cuda" (Nvidia), or "cpu"
DEVICE_TYPE = "directml"

# ─────────────────────────────────────────
# Display
# ─────────────────────────────────────────
BBOX_TEXT_SCALE     = 0.7
BBOX_TEXT_THICKNESS = 1

# ─────────────────────────────────────────
# Tracking / counting
# ─────────────────────────────────────────
# Any label from data/class-names.txt
TRACKED_CLASS = "car"
# Detections below this confidence are ignored when drawing
CONFIDENCE_THRESHOLD = 0.3
