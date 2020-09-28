import torch.nn as nn
from torchvision import models
class ClassifierNet(nn.Module):
    def __init__(self,NUM_CLASSES):
        super(ClassifierNet,self).__init__()
        resnet = models.resnet152(pretrained=True)
        self.backbone = nn.Sequential(*(list(resnet.children())[:-1]))
        self.linear_1 = nn.Linear(2048,512)
        self.linear_2 = nn.Linear(512,NUM_CLASSES)
        self.dropout = nn.Dropout(0.3)
    def forward(self,x):
        x = self.backbone(x)
        x = x.view(-1,2048)
        x = self.linear_1(x)
        x = self.dropout(x)
        x = self.linear_2(x)
        return x