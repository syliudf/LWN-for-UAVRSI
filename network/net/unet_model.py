# full assembly of the sub-parts to form the complete net
import torch
# import torch.nn.functional as F

from unet_parts import *

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
        x1 = self.inc(x)
        print(f"x1 size: {x1.size()}")
        x2 = self.down1(x1)
        print(f"x2 size: {x2.size()}")
        x3 = self.down2(x2)
        print(f"x3 size: {x3.size()}")
        x4 = self.down3(x3)
        print(f"x4 size: {x4.size()}")
        x5 = self.down4(x4)
        print(f"x5 size: {x5.size()}")

        x = self.up1(x5, x4)
        print(f"up1 size: {x.size()}")
        x = self.up2(x, x3)
        print(f"up2 size: {x.size()}")
        x = self.up3(x, x2)
        print(f"up3 size: {x.size()}")
        x = self.up4(x, x1)
        print(f"up4 size: {x.size()}")
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
    