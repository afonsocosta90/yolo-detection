"""Download sample driving footage from Kaggle."""
import kagglehub

path = kagglehub.dataset_download("robikscube/driving-video-with-object-tracking")
print(f"Downloaded to: {path}")
