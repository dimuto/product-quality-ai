# https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
from __future__ import print_function, division

import torch
import torch.backends.cudnn as cudnn

from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os

cudnn.benchmark = True
plt.ion()   # interactive mode

from config import DATA_DIR

# Data augmentation and normalization for training
# Just normalization for validation

class Dataloader:
    def __init__(self):
        self.data_transforms = {
                    'train': transforms.Compose([
                        transforms.RandomResizedCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ]),
                    'val': transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ]),
                }

    def data_transform(self):
        image_datasets = {x: datasets.ImageFolder(os.path.join(DATA_DIR, x),
                                                self.data_transforms[x])
                        for x in ['train', 'val']}
        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64,
                                                    shuffle=True, num_workers=4)
                        for x in ['train', 'val']}
        dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
        class_names = image_datasets['train'].classes

        return dataloaders, dataset_sizes, class_names

    
    
