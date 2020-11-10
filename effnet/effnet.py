import torch
from torch import nn

from build_net import EfficientNet_mod



model = EfficientNet_mod.from_name('efficientnet-b0')

print(model)

x=model(torch.randn([4,3,512,512]))
print(x.size())

# torch.save(model, './eff_mod_pretrained.pth')
# print(model_2)
# x = model.forward(torch.randn([1,3,512,512]))
# print(x.size())
# print(model_2)
# print(model)
# u=UNet(n_channels=3, n_classes=1000)
