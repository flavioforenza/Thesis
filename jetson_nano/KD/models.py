# borrowed from "https://github.com/marvis/pytorch-mobilenet"

import torch.nn as nn
import torch.nn.functional as F


class MobileNetV1_Teach(nn.Module):
    def __init__(self, num_classes=1024):
        super(MobileNetV1_Teach, self).__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
                
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
                #nn.Dropout(0.1),

                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
                #nn.Dropout(0.1),

            )

        self.model = nn.Sequential(
            conv_bn(  3,  32, 2), 
            conv_dw( 32,  64, 1),
            conv_dw( 64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
        )
        self.fc = nn.Linear(1024, num_classes)
        #self.dropout_20 = nn.Dropout(0.2)
        #self.dropout_50 = nn.Dropout(0.5)

    def forward(self, x):
        #x = self.dropout_20(x)
        x = self.model(x)
        #x = self.dropout_50(x)
        #added from jetson repo
        x = F.avg_pool2d(x, 7)
        x = x.view(-1, 1024)
        #x = self.dropout_50(x)
        x = self.fc(x)
        return x

alpha=1

class MobileNetV1_Stud(nn.Module):
    def __init__(self, num_classes=512, wm=1):
        alpha = wm
        print("Alpha: ", alpha)

        super(MobileNetV1_Stud, self).__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),

                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        self.model = nn.Sequential(
            conv_bn(  3,  int(32*alpha), 2), 
            conv_dw( int(32*alpha),  int(64*alpha), 1),
            conv_dw( int(64*alpha), int(128*alpha), 2),
            conv_dw(int(128*alpha), int(128*alpha), 1),
            conv_dw(int(128*alpha), int(256*alpha), 2),
            conv_dw(int(256*alpha), int(256*alpha), 1),
            conv_dw(int(256*alpha), int(512*alpha), 2),
            conv_dw(int(512*alpha), int(512*alpha), 1),
            conv_dw(int(512*alpha), int(512*alpha), 1),
            conv_dw(int(512*alpha), int(512*alpha), 1),
            conv_dw(int(512*alpha), int(512*alpha), 1),
            conv_dw(int(512*alpha), int(512*alpha), 1),
            conv_dw(int(512*alpha), int(1024*alpha), 2),
            conv_dw(int(1024*alpha), int(1024*alpha), 1),
        )
        self.fc = nn.Linear(int(1024*alpha), num_classes)

    def forward(self, x):
        x = self.model(x)
        x = F.avg_pool2d(x, 7)
        x = x.view(-1, 256)
        x = self.fc(x)
        return x
