import torch
import torch.nn as nn
import torch.nn.functional as F


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
            # nn.MaxPool2d(2),
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

class Resize(nn.Module):
    def __init__(self, size, out_ch, conv_flag = True, mode="pool"):
        super(Resize, self).__init__()
        self.size = size
        self.conv_flag = conv_flag
        self.conv = BasicConv2d(in_planes=3, out_planes=out_ch, kernel_size=5, stride=1, padding=2)
        self.mode = mode
        if mode not in ["pool", "PIL"]:
            raise "mode must in one of ['pool','PIL']."

    def resize(self, input_tensors):
        imgs = list()
        for img in input_tensors:
            img_PIL = torchvision.transforms.ToPILImage()(img)
            img_PIL = torchvision.transforms.Resize(self.size)(img_PIL)
            img_PIL = torchvision.transforms.ToTensor()(img_PIL)
            img_PIL = img_PIL.unsqueeze(0)
            imgs.append(img_PIL)
        output = torch.cat(imgs, 0)
        return output

    def forward(self, input_tensor):
        if self.mode == "pool":
            resized_tensor = F.adaptive_avg_pool2d(input_tensor, self.size).data
            output = self.conv(resized_tensor)
        elif self.mode == "PIL":
            resized_tensor = self.resize(input_tensor).data
            output = self.conv(resized_tensor)
        else:
            raise NotImplementedError

        return output



class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()

        self.inc = inconv(n_channels, 64)
        
        self.downsize1 = nn.MaxPool2d(2)
        self.resize1 = Resize((256, 256), 64,) 
        self.down1 = down(128, 128)

        self.downsize2 = nn.MaxPool2d(2)
        self.resize2 = Resize((128, 128), 128,)
        self.down2 = down(256, 256)

        self.downsize3 = nn.MaxPool2d(2)
        self.resize3 = Resize((64, 64), 256,)
        self.down3 = down(512, 512)

        self.downsize4 = nn.MaxPool2d(2)
        self.resize4 = Resize((32, 32), 512,)
        self.down4 = down(1024, 512)
        
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_classes)

    def forward(self, x):
        # x -> x1: 3 -> 64, shape = h * w
        x1 = self.inc(x)  # [N, 64, 512, 512]

        x1 = self.downsize1(x1)
        x1_ = self.resize1(x)  # [N, 64, 512, 512]
        x2_input = torch.cat((x1, x1_), 1)

        # x1 -> x2: 64 -> 128, shape = h/2 * w/2
        x2 = self.down1(x2_input)  # [N, 128, 256, 256]
        x2 = self.downsize2(x2)
        x2_ = self.resize2(x)  # [N, 128, 256, 256]
        x3_input = torch.cat((x2, x2_), 1)
        
        # x2 -> x3: 128 -> 256, shape = h/4 * w/4
        x3 = self.down2(x3_input)  # [N, 256, 128, 128]
        x3 = self.downsize3(x3)
        x3_ = self.resize3(x)
        x4_input = torch.cat((x3, x3_), 1)

        # x3 -> x4: 256 -> 512, shape = h/8 * w/8
        x4 = self.down3(x4_input)  # [N, 512, 64, 64]
        x4 = self.downsize4(x4)
        x4_ = self.resize4(x)
        x5_input = torch.cat((x4, x4_), 1)

        # x4 -> x5: 512 -> 512, shape = h/16 * w/16
        x5 = self.down4(x5_input)  # [N, 512, 32, 32]

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

if __name__ == "__main__":

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    input_tensor = torch.randn(2, 3, 512, 512).to(device)
    model = UNet(n_channels=3, n_classes=2).to(device)
    print(model)
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
    print(output.size())
