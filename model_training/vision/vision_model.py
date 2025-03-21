# model_training/vision/vision_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleVisionModel(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleVisionModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(32 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # shape -> (N,16,112,112)
        x = self.pool(F.relu(self.conv2(x)))  # shape -> (N,32,56,56)
        x = x.view(x.size(0), -1)  # flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
