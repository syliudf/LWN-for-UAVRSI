import torch
from torch import nn
from network.efficientnet.Efficientnet_mod import EfficientNet_1_upsample
from network.efficientnet.Efficientnet_uav import EfficientNet_1_up
from network.efficientnet.model import EfficientNet



# print(state_dict)
pretrain_state_dict = torch.load("./pretrained/b1_pretrained.pth")
print(pretrain_state_dict)
# print(type(pretrain_state_dict))
# state_dict = torch.hub.load('rwightman/gen-efficientnet-pytorch', 'efficientnet_b1', pretrained=True)
# model = EfficientNet_postnonlocal.from_name('efficientnet-b0')
model = EfficientNet_1_up.from_name('efficientnet-b1')
# model = EfficientNet_1_up.from_name('efficientnet-b1')
# print(model)
# torch.save(model, 'tmp.pth')
# model = torch.load('tmp.pth')
model_dict = model.state_dict()
# print(model_dict).state_dict
new_dict = {k: v for k,v in pretrain_state_dict.state_dict().items() if k in model_dict}

model_dict.update(new_dict)

model.load_state_dict(model_dict)

torch.save(model, './pretrained/b1_up.pth')
# print(model)
x = model.forward(torch.randn([1,3,512,512]))
print(x.size())
# print(x.size())
# print(model_2)
# print(model)

# transfer_state_dict()

def transfer_state_dict(pretrained_dict, model_dict):
    '''
    根据model_dict,去除pretrained_dict一些不需要的参数,以便迁移到新的网络
    url: https://blog.csdn.net/qq_34914551/article/details/87871134
    :param pretrained_dict:
    :param model_dict:
    :return:
    '''
    # state_dict2 = {k: v for k, v in save_model.items() if k in model_dict.keys()}
    state_dict = {}
    for k, v in pretrained_dict.items():
        if k in model_dict.keys():
            # state_dict.setdefault(k, v)
            state_dict[k] = v
        else:
            print("Missing key(s) in state_dict :{}".format(k))
    return state_dict