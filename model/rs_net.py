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
        '''
        ###############
        Encoder Network
        ###############
        Using VGG-16 pretrained
        '''
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
        Remove the classifier layer and avgpool layeroutput_padding=1
        '''
        self.vgg16_pretrained = nn.Sequential(*list(self.vgg16_pretrained.children())[:-2])
        '''
        add 1x1 conv
        '''
        idx = 31
        self.vgg16_pretrained.add_module(f'{idx}', nn.Conv2d(512, 512, kernel_size=(1,1), padding=(0,0)))
        idx+=1

        '''
        ###############
        Decoder Network
        ###############
        Using model proposed by Zeiler
        '''
        self.prelu = nn.PReLU()
        self.decoder_conv1 = nn.Conv2d(512, 512, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.decoder_convt_1 = nn.ConvTranspose2d(512, 512, kernel_size=(2,2), stride=(1,1), padding=(0,0))
        self.decoder_conv2 = nn.Conv2d(512, 256, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.decoder_convt_2 = nn.ConvTranspose2d(256, 256, kernel_size=(2,2), stride=(2,2), padding=(0,0))
        self.decoder_conv3 = nn.Conv2d(256, 256, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.decoder_conv4 = nn.Conv2d(256, 128, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.decoder_convt_3 = nn.ConvTranspose2d(128, 128, kernel_size=(2,2), stride=(1,1), padding=(0,0))
        self.decoder_conv5 = nn.Conv2d(128, 128, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.decoder_conv6 = nn.Conv2d(128, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.decoder_convt_4 = nn.ConvTranspose2d(64, 64, kernel_size=(3,3), stride=(2,2), padding=(0,0))
        self.decoder_conv7 = nn.Conv2d(64, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.decoder_conv8 = nn.Conv2d(64, 3, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.decoder_convt_5 = nn.ConvTranspose2d(3, 3, kernel_size=(2,2), stride=(1,1), padding=(0,0))
        self.decoder_conv9 = nn.Conv2d(3, 3, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.decoder_conv10 = nn.Conv2d(3, 3, kernel_size=(3,3), stride=(1,1), padding=(1,1))

    
    def forward(self, x):
        x = self.vgg16_pretrained(x)
        x = self.prelu(self.decoder_conv1(x))
        x = self.decoder_convt_1(x)
        x = self.prelu(self.decoder_conv2(x))
        x = self.decoder_convt_2(x)
        x = self.prelu(self.decoder_conv3(x))
        x = self.prelu(self.decoder_conv4(x))
        x = self.decoder_convt_3(x)
        x = self.prelu(self.decoder_conv5(x))
        x = self.prelu(self.decoder_conv6(x))
        x = self.decoder_convt_4(x)
        x = self.prelu(self.decoder_conv7(x))
        x = self.prelu(self.decoder_conv8(x))
        x = self.decoder_convt_5(x)
        x = self.prelu(self.decoder_conv9(x))
        x = self.decoder_conv10(x)
        return x