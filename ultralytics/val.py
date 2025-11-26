import sys
from pathlib import Path

# Get project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "ultralytics"))

from ultralytics import YOLO

# Load the trained model
# Option 1: Use downloaded pre-trained model
# model = YOLO('/path/to/your/downloaded/best.pt')

# Option 2: Use your trained model
model_path = PROJECT_ROOT / "ultralytics/runs/multi/yolopm/weights/best.pt"
model = YOLO(str(model_path))

# Validate the model
data_path = PROJECT_ROOT / "ultralytics/datasets/bdd-multi.yaml"
metrics = model.val(
    data=str(data_path),
    device=[0],         # Adjust GPU device ID, e.g., [0], or 'cpu'
    task='multi',
    name='val',
    iou=0.6,
    conf=0.001,
    imgsz=(640,640),
    classes=[2,3,4,9,10,11],
    combine_class=[2,3,4,9],
    single_cls=True
)

# Optional: Print metrics for each task
# number = 3  # Number of tasks in your work
# for i in range(number):
#     print(f'This is for task {i}')
#     print(metrics[i].box.map)    # map50-95
#     print(metrics[i].box.map50)  # map50
#     print(metrics[i].box.map75)  # map75
#     print(metrics[i].box.maps)   # a list contains map50-95 of each category