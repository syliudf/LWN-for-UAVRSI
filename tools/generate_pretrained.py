import torch
from torch import nn
# from network.efficientnet.Efficientnet_mod import EfficientNet_1_upsample
from network.efficientnet.Efficientnet_uav import EfficientNet_1_up, EfficientNet_1_nofusion
from network.efficientnet.model import EfficientNet
from network.efficientnet.Efficientnet_DAN import EfficientNet_1_CAM as model_now




# print(state_dict)
pretrain_state_dict = torch.load("./pretrained/b1_up.pth")
# print(pretrain_state_dict)
# print(type(pretrain_state_dict))
# state_dict = torch.hub.load('rwightman/gen-efficientnet-pytorch', 'efficientnet_b1', pretrained=True)
# model = EfficientNet_postnonlocal.from_name('efficientnet-b0')
model = model_now.from_name('efficientnet-b1', override_params={'num_classes': 8})
# model = EfficientNet_1_up.from_name('efficientnet-b1')
# print(model)
# torch.save(model, 'tmp.pth')
# model = torch.load('tmp.pth')
model_dict = model.state_dict()
# print(model_dict).state_dict
new_dict = {k: v for k,v in pretrain_state_dict.state_dict().items() if k in model_dict}

# del new_dict['outconv_320_8']
model_dict.update(new_dict)
# model_dict.popitem('outconv_320_8')


model.load_state_dict(model_dict)


torch.save(model, './pretrained/b1_cam_8.pth')
# print(model)
x = model.forward(torch.randn([1,3,512,512]))
print(x.size())
# print(x.size())
# print(model_2)
# print(model)

# transfer_state_dict()


