import os
import shutil

from detection.detect import run as detection
from classification.run import Run as classification

import yaml
data_yaml_path = "detection/config/data.yaml"
with open(data_yaml_path, errors='ignore') as f:
    names = yaml.safe_load(f)['names']

def ai_pipeline(image_input_dir):
    # 1. Fruit detection
    detection_output_dir = detection(source=image_input_dir)

    # 2. Defect classification
    mode = "predict"
    r = classification()
    pq_score = r.run_prediction(os.path.join(detection_output_dir, "crops", names[0]))
    if pq_score < 5:
        defect_acceptance_level = "unacceptable"
    else:
        defect_acceptance_level = "acceptable"

    # 3. Delete crops images and folder
    shutil.rmtree(detection_output_dir)
    # print("Deleted crops")

    return defect_acceptance_level, pq_score

if __name__ == "__main__":
    image_input_dir = "/home/ubuntu/product-quality-ai/data/raw/sample/images/val/IMG_20190816_155802.jpg"
    defect_acceptance_level, pq_score = ai_pipeline(image_input_dir)
    print(defect_acceptance_level, pq_score)