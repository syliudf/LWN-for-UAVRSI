# -*- coding:utf-8 -*-

import torch
import torch.nn as nn 
import torch.nn.functional as F 

class init_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(init_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class BasicConv2d(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=False) # verify bias false
        self.bn = nn.BatchNorm2d(out_planes,
                                 eps=0.001, # value found in tensorflow
                                 momentum=0.1, # default pytorch value
                                 affine=True)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        
        return x

class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x

class InceptionLayer(nn.Module):

    def __init__(self, in_ch, out_ch, scale=1.0):
        super(InceptionLayer, self).__init__()
        
        if in_ch % 2 != 0:
            raise "wrong in_ch, it should be 2 times"

        self.scale = scale

        self.branch0 = BasicConv2d(in_ch, int(in_ch/2), kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(in_ch, int(in_ch/2), kernel_size=1, stride=1),
            BasicConv2d(int(in_ch/2), int(in_ch/2), kernel_size=3, stride=1, padding=1)
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(in_ch, int(in_ch/2), kernel_size=1, stride=1),
            BasicConv2d(int(in_ch/2), int(in_ch/2 + in_ch), kernel_size=3, stride=1, padding=1),
            BasicConv2d(int(in_ch/2 + in_ch), int(in_ch + in_ch), kernel_size=3, stride=1, padding=1)
        )

        self.conv2d = nn.Conv2d(in_ch*3, out_ch, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

        self.pool = nn.MaxPool2d(kernel_size=2,)

    def _forward(self, x):
    
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        out = self.conv2d(out)
#         out = out * self.scale + x #
        out = out * self.scale
        out = self.relu(out)
        return out

    def forward(self, x):

        x = self.pool(x)
        x = self._forward(x)

        return x


class UpLayer(nn.Module):

    def __init__(self, in_ch, out_ch, bilinear=True, Inception=True):
        super(UpLayer, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)
        
        if Inception:
            self.conv = InceptionLayer(in_ch, out_ch,) 
        else:
            self.conv = BasicConv2d(in_ch, out_ch, kernel_size=1, stride=1)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))
        
        # for padding issues, see 
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

# full assembly of the sub-parts to form the complete net

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.inc = init_conv(n_channels, 64)
        
        self.down1 = InceptionLayer(64, 128)
        self.down2 = InceptionLayer(128, 256)
        self.down3 = InceptionLayer(256, 512)
        self.down4 = InceptionLayer(512, 512)

        self.up1 = UpLayer(1024, 256, Inception=False)
        self.up2 = UpLayer(512, 128, Inception=False)
        self.up3 = UpLayer(256, 64, Inception=False)
        self.up4 = UpLayer(128, 64, Inception=False)
        
        self.outc = outconv(64, n_classes)

    def forward(self, x):

        x1 = self.inc(x)
        # print(f"x1 size: {x1.size()}")
        x2 = self.down1(x1)  # size: 1/2
        # print(f"x2 shape: {x2.size()}")        
        x3 = self.down2(x2)  # size: 1/4
        # print(f"x3 shape: {x3.size()}")        
        x4 = self.down3(x3)  # size: 1/8
        # print(f"x4 shape: {x4.size()}")        
        x5 = self.down4(x4)  # size: 1/16 
        # print(f"x5 shape: {x5.size()}")        
        
        x = self.up1(x5, x4) # size: 1/8
        # print(f"up1 shape: {x.size()}")
        x = self.up2(x, x3)  # size: 1/4
        # print(f"up2 shape: {x.size()}")
        x = self.up3(x, x2)  
        # print(f"up3 shape: {x.size()}")
        x = self.up4(x, x1)
        # print(f"up4 shape: {x.size()}")
        x = self.outc(x)
        
        return torch.sigmoid(x)


if __name__ == "__main__":

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    input_tensor = torch.randn(2, 3, 224, 224).to(device)
    model = UNet(n_channels=3, n_classes=2).to(device)
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
    print(output.size())
    
