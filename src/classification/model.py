import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights

class Model:
    # def __init__(self):

    def initialize_model(self):
        model_ft = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        num_ftrs = model_ft.fc.in_features
        # Here the size of each output sample is set to 2.
        # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
        model_ft.fc = nn.Linear(num_ftrs, 2)

        return model_ft
        
        