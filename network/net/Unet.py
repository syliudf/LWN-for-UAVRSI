import torch
import torch.nn as nn
import torch.nn.functional as F


# 网络中的基本卷积结构
# (conv => BN => ReLU) * 2
# 3*3 卷积核，步长 1，padding 1
class double_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
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


# 输入卷积，对输入图像的直接卷积
class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


# 下采样结构
# 先 2*2 池化，再卷积
# 图像尺寸 / 2
class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


# 上采样结构
# 如果 bilinear == True, 用双线性差值进行上采样, 尺寸 * 2
# 如果 bilinear == False, 用转置卷积进行上采样, 其输入通道数 = 输出通道数 = in_ch // 2, stride=2，表示尺寸 * 2
# 该层的输入是相邻的两个下采样层的输出
# x1 是由 x2 下采样得到的
# 先对 x1 进行上采样，比较上采样后的 x1 与 x2 的尺寸, 如果不同那么一定是 x1 的尺寸大于 x2 的尺寸
# 在 x2 的四周进行补 0, 使其与 x1 有相同的尺寸
# 对 x1 和 x2 进行级联，级联后的维度就是 in_ch
# 然后对 cat(x1, x2) 进行卷积，卷积后的维度为 out_ch
class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2),
                        diffY // 2, int(diffY / 2)))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


# 输出卷积，输出的就是最终结果
class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()

        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_classes)

    def forward(self, x):
        # x -> x1: 3 -> 64, shape = h * w
        x1 = self.inc(x)
        # x1 -> x2: 64 -> 128, shape = h/2 * w/2
        x2 = self.down1(x1)
        # x2 -> x3: 128 -> 256, shape = h/4 * w/4
        x3 = self.down2(x2)
        # x3 -> x4: 256 -> 512, shape = h/8 * w/8
        x4 = self.down3(x3)
        # x4 -> x5: 512 -> 512, shape = h/16 * w/16
        x5 = self.down4(x4)
        # 先对 x5 上采样，然后级联 x5 和 x4，执行卷积
        # x.shape = (256, h/8, w/8)
        x = self.up1(x5, x4)
        # 先对 x 上采样，然后级联 x 和 x3，执行卷积
        # x.shape = (128, h/4, w/4)
        x = self.up2(x, x3)
        # 先对 x 上采样，然后级联 x 和 x2，执行卷积
        # x.shape = (64, h/2, w/2)
        x = self.up3(x, x2)
        # 先对 x 上采样，然后级联 x 和 x1，执行卷积
        # x.shape = (64, h, w)
        x = self.up4(x, x1)
        # x -> x: 64 -> n_classes, shape = h * w
        x = self.outc(x)
        return x
