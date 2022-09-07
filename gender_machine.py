import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import resnet50, ResNet50_Weights


class AgeMachine(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        number_input = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(number_input, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )

        for param in self.resnet.layer1.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.resnet(x)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)
