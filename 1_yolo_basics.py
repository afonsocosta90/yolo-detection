from ultralytics import YOLO
import cv2

# Load the YOLOv8 large model from the specified weights file
model = YOLO("../yolo-weights/yolov8l.pt")

# Run inference on the image (show=False by default, so no auto-display)
results = model.predict("images/4.jpg")

# Render bounding boxes, labels, and confidence scores onto the image
# .plot() returns a NumPy array (BGR format) ready for OpenCV
annotated = results[0].plot()

# Plot the annotated frame and display it
cv2.imshow("YOLOv8l Detection", annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()
