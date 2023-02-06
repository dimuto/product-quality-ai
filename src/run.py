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
    
    # Check if OD model detected any fruits
    # If detected, run defect classification model
    if os.path.exists(os.path.join(detection_output_dir, "crops")):

        # 2. Defect classification
        mode = "predict"
        r = classification()
        pq_score = r.run_prediction(os.path.join(detection_output_dir, "crops", names[0]))

        if pq_score < 5:
            defect_acceptance_level = "unacceptable"
        else:
            defect_acceptance_level = "acceptable"

        # # 3. Delete crops images and folder
        # shutil.rmtree(detection_output_dir)

    # If no fruits detected, return error
    else:
        defect_acceptance_level = "unacceptable"
        pq_score = "unable to detect fruit"

    return defect_acceptance_level, pq_score

if __name__ == "__main__":
    # image_input_dir = "../data/evaluate/PPCO0014800130"
    image_input_dir = "../results/exp5/image_1672266902453-1672266906.png"
    defect_acceptance_level, pq_score = ai_pipeline(image_input_dir)
    print(defect_acceptance_level, pq_score)