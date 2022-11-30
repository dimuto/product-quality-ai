import os
import shutil

from detection.detect import run as detection
from classification.run import Run as classification

import yaml
data_yaml_path = "detection/config/data.yaml"
with open(data_yaml_path, errors='ignore') as f:
    names = yaml.safe_load(f)['names']

image_input_dir = "/home/ubuntu/product-quality-ai/data/raw/sample/images/val/IMG_20190816_155802.jpg"

# 1. Fruit detection
detection_output_dir = detection(source=image_input_dir)

# 2. Defect classification
mode = "predict"
r = classification()
pq_score = r.run_prediction(os.path.join(detection_output_dir, "crops", names[0]))

# 3. Delete crops images and folder
shutil.rmtree(detection_output_dir)
print("Deleted crops")
print(pq_score)