import sys
from pathlib import Path

# Get project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "ultralytics"))

from ultralytics import YOLO

# Number of tasks in your work (detection + segmentation tasks)
number = 3

# Load the trained model
# Option 1: Use downloaded pre-trained model - update the path to your downloaded model
# model = YOLO('/path/to/your/downloaded/best.pt')

# Option 2: Use your trained model
model_path = PROJECT_ROOT / "ultralytics/runs/multi/yolopm/weights/best.pt"
model = YOLO(str(model_path))

# Predict on images
# Update 'source' to your image folder or video file path
model.predict(
    source='/path/to/your/images',  # Change this to your image folder
    imgsz=(384,672),    # Keep this size for best results with provided pre-trained model
    device=[0],         # Adjust GPU device ID, e.g., [0], or 'cpu'
    name='prediction_results',
    save=True,
    conf=0.25,
    iou=0.45,
    show_labels=False
)
