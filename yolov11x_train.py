import os
import shutil
from pathlib import Path
import subprocess

ANNOTATIONS_DIR = "annotations"
IMAGES_DIR = "uploads/source"
DATASET_DIR = "dataset"
TRAIN_DIR = os.path.join(DATASET_DIR, "images/train")
LABEL_DIR = os.path.join(DATASET_DIR, "labels/train")
DATA_YAML_PATH = os.path.join(DATASET_DIR, "data.yaml")
MODEL_ARCH = "yolov11x.pt"
CLASS_NAMES = ["Bin", "Cover"]

shutil.rmtree(DATASET_DIR, ignore_errors=True)
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(LABEL_DIR, exist_ok=True)

for txt_file in os.listdir(ANNOTATIONS_DIR):
    if txt_file.endswith(".txt"):
        name = Path(txt_file).stem
        label_path = os.path.join(ANNOTATIONS_DIR, txt_file)
        image_path = os.path.join(IMAGES_DIR, f"{name}.jpg")
        if not os.path.exists(image_path):
            image_path = os.path.join(IMAGES_DIR, f"{name}.png")
        if os.path.exists(image_path):
            shutil.copy(image_path, os.path.join(TRAIN_DIR, os.path.basename(image_path)))
            shutil.copy(label_path, os.path.join(LABEL_DIR, os.path.basename(label_path)))

with open(DATA_YAML_PATH, "w") as f:
    f.write(f"""
path: {DATASET_DIR}
train: images/train
val: images/train
names: {CLASS_NAMES}
""")

print("✅ Dataset prepared for YOLOv11x training.")

# Optional training command:
# subprocess.run([
#     "yolo", "task=detect", "mode=train",
#     f"model={MODEL_ARCH}",
#     f"data={DATA_YAML_PATH}",
#     "epochs=50", "imgsz=640"
# ])
