import torch
from torchvision.models import resnet50
from thop import profile, clever_format
from network.efficientnet.Efficientnet_DAN import EfficientNet_1_up as model_now
num_classes = 8
model = model_now.from_name('efficientnet-b1',override_params={'num_classes' : num_classes})
input = torch.randn(1,3,512,512)
macs, params = profile(model, inputs=(input, ))
macs, params = clever_format([macs, params], "%.3f")
print(macs)
print(params)