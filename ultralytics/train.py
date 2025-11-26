import sys
import os
from pathlib import Path

# Get project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "ultralytics"))

from ultralytics import YOLO

# Load a model
# Adjust the model path to your actual model file
model_path = PROJECT_ROOT / "ultralytics/models/v8/yolov8-bdd-v4-one-dropout-individual-n.yaml"
model = YOLO(str(model_path), task='multi')  # build a new model from YAML
# model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
# model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights

# Train the model
# Adjust the data path and training parameters according to your needs
data_path = PROJECT_ROOT / "ultralytics/datasets/bdd-multi.yaml"
model.train(
    data=str(data_path),
    batch=12,           # Adjust batch size based on your GPU memory
    epochs=300,
    imgsz=(640,640),
    device=[0],         # Adjust GPU device IDs, e.g., [0], [0,1,2], or 'cpu'
    name='yolopm',
    val=True,
    task='multi',
    classes=[2,3,4,9,10,11],
    combine_class=[2,3,4,9],
    single_cls=True
)
