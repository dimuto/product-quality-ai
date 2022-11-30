# import os
# import sys
# from pathlib import Path
# FILE = Path(__file__).resolve()
# ROOT = FILE.parents[0]  # YOLOv5 root directory
# if str(ROOT) not in sys.path:
#     sys.path.append(str(ROOT))  # add ROOT to PATH
# ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import torch
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as nnf

from classification.dataloader import Dataloader
from classification.model import Model
from classification.config import MODEL_PATH, IDX_TO_CLASS

class Predict:
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        m = Model()
        self.model_ft = m.initialize_model()
        self.model_ft.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))

        self.model_ft = self.model_ft.to(self.device)

    def predict_model(self, test_image_path):
        d = Dataloader()
        transform = d.data_transforms['val']

        test_image = Image.open(test_image_path).convert('RGB')
        plt.imshow(test_image)
        test_image_tensor = transform(test_image)
        if torch.cuda.is_available():
            test_image_tensor = test_image_tensor.view(1, 3, 224, 224).cuda()
        else:
            test_image_tensor = test_image_tensor.view(1, 3, 224, 224)
        with torch.no_grad():
            self.model_ft.eval()
            # Model outputs log probabilities
            out = self.model_ft(test_image_tensor)
            # ps = torch.exp(out)
            # topk, topclass = ps.topk(1, dim=1)
            # print("Output class :  ", IDX_TO_CLASS[topclass.cpu().numpy()[0][0]])

            prob = nnf.softmax(out, dim=1)
            top_p, top_class = prob.topk(1, dim = 1)
            # print(prob)
            # print("Output class :  ", IDX_TO_CLASS[top_class.cpu().numpy()[0][0]])
            # print("PQ score: ", prob.cpu().numpy()[0][1])
            return prob.cpu().numpy()[0][1] # probability of defect class
             