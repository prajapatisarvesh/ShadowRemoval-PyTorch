import torch
import torch.nn as nn
from torch.nn import functional as F
from base.base_model import BaseModel
import torchvision
from torchvision.models.feature_extraction import create_feature_extractor

class RSNet(BaseModel):
    def __init__(self):
        super().__init__()
        self.required_layer = {
            'features.1':'relu1_1'
        }
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.vgg16_pretrained = torchvision.models.vgg16 (pretrained = True)
        '''
        Replace ReLU with PReLU
        '''
        for x, feature in enumerate(self.vgg16_pretrained.features):
            if feature.__str__() == 'ReLU(inplace=True)':
                self.vgg16_pretrained.features[x] = nn.PReLU()
        '''
        Modify Maxpooling to change stride to 1 for block 1,3 and 5
        '''
        self.vgg16_pretrained.features[4] = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, dilation=1, ceil_mode=False)
        self.vgg16_pretrained.features[16] = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, dilation=1, ceil_mode=False)
        self.vgg16_pretrained.features[30] = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, dilation=1, ceil_mode=False)
        '''
        Remove the classifier layer
        '''
        self.vgg16_pretrained.classifier = nn.Sequential(*list(self.vgg16_pretrained.classifier.children())[:-7])
    
    def forward(self, x):
        x = self.vgg16_pretrained(x)
        return x