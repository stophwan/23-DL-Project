import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet50

class MyModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MyModel, self).__init__()
        self.resnet = resnet50(pretrained=True)
        self.num_classes = num_classes
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        out = self.resnet(x)
        return out