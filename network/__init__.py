import torch
import torchvision

def generate_pretrained(model, num_classes):
    # pass
    pretrain_state_dict = torch.load("./pretrained/b1_up.pth")
    model_dict = model.state_dict()
    new_dict = {k: v for k,v in pretrain_state_dict.state_dict().items() if k in model_dict}
    model_dict.update(new_dict)
    model.load_state_dict(model_dict)
    return model


def build_network(model_type, num_classes=8, pretrained=True):
    if model_type == "b1_dan":
        from network.efficientnet.Efficientnet_DAN import EfficientNet_1_DAN
        model = EfficientNet_1_DAN.from_name('efficientnet-b1',override_params={'num_classes' : num_classes}).cuda()
        if pretrained:
            model = generate_pretrained(model,num_classes)

    elif model_type == "b1_up":
        from network.efficientnet.Efficientnet_uav import EfficientNet_1_up
        model = EfficientNet_1_up.from_name('efficientnet-b1',override_params={'num_classes' : num_classes}).cuda()
        if pretrained:
            model = generate_pretrained(model,num_classes)

    elif model_type == "b1_nof":
        from network.efficientnet.Efficientnet_DAN import EfficientNet_1_Nof
        model = EfficientNet_1_Nof.from_name('efficientnet-b1',override_params={'num_classes' : num_classes}).cuda()
        if pretrained:
            model = generate_pretrained(model,num_classes)

    elif model_type == "b1_pam":
        from network.efficientnet.Efficientnet_DAN import EfficientNet_1_PAM
        model = EfficientNet_1_PAM.from_name('efficientnet-b1',override_params={'num_classes' : num_classes}).cuda()
        if pretrained:
            model = generate_pretrained(model,num_classes)

    elif model_type == "b1_cam":
        from network.efficientnet.Efficientnet_DAN import EfficientNet_1_CAM
        model = EfficientNet_1_CAM.from_name('efficientnet-b1',override_params={'num_classes' : num_classes}).cuda()
        if pretrained:
            model = generate_pretrained(model,num_classes)

    elif model_type == 'deeplabv3+_resnet50':
        from network.net import deeplab_resnet50
        model = deeplab_resnet50.DeepLabv3_plus(
                    nInputChannels=3,
                    n_classes=num_classes,
                    os=8,
                    pretrained=True
                    ).cuda()

    elif model_type == 'deeplabv3+_resnet101':
        from network.net import deeplab_resnet
        model = deeplab_resnet.DeepLabv3_plus(
                    nInputChannels=3,
                    n_classes=num_classes,
                    os=8,
                    pretrained=True
                    ).cuda()
    
    return model

