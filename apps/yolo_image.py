"""Run YOLOv8 on a single image and display the annotated result."""
import lib.directml  # noqa: F401  -- side-effect import: must precede ultralytics

import cv2
from ultralytics import YOLO

import config

model = YOLO(str(config.MODEL_PATH))
results = model.predict(str(config.IMAGE_PATH))
annotated = results[0].plot()

cv2.imshow("YOLOv8 Detection", annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()
