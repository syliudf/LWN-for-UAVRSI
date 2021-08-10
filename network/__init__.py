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
    elif model_type == 'deeplabv3_resnet101':
        from network.net import deeplabv3_resnet
        model = deeplabv3_resnet.DeepLabv3(nInputChannels=3, n_classes=num_classes, os=16, pretrained=True, _print=True).cuda()
    elif model_type == 'deeplabv3+_resnet101':
        from network.net import deeplab_resnet
        model = deeplab_resnet.DeepLabv3_plus(
                    nInputChannels=3,
                    n_classes=num_classes,
                    os=8,
                    pretrained=True
                    ).cuda()
    elif model_type == 'deeplabv2_resnet101':
        from network.net import deeplabv2
        model = deeplabv2.DeepLabV2(
        n_classes=num_classes, n_blocks=[3, 4, 23, 3], atrous_rates=[6, 12, 18, 24]).cuda()
    elif model_type == 'fcn':
        from network.fcn import VGGNet, FCN32s, FCN16s, FCN8s, FCNs
        vgg_model = VGGNet(requires_grad=True, remove_fc=False).cuda()
        model = FCN8s(pretrained_net=vgg_model, n_class=num_classes).cuda()

    elif model_type == "segnet":
        from network.segnet import SegNet
        model = SegNet(3,num_classes).cuda()

    elif model_type == "unet":
        from .net.Unet import UNet
        model =UNet(n_channels=3, n_classes=num_classes ).cuda()
    elif model_type == "bisenetv2":
        from network.bisenetv2 import BiSeNetV2
        model = BiSeNetV2(n_classes=num_classes)
    elif model_type == "unet_new":
        from .net.unet_model import UNet
        model =UNet(n_channels=3, n_classes=num_classes ).cuda()
        model_dict = model.state_dict()
        pretrain_state_dict = torch.hub.load('milesial/Pytorch-UNet', 'unet_carvana')
        new_dict = {k: v for k,v in pretrain_state_dict.state_dict().items() if k in model_dict}
        model_dict.update(new_dict)
        model.load_state_dict(model_dict)
    elif model_type == "dfa":
        from network.dfanet import DFANet,load_backbone, XceptionA
        ENCODER_CHANNEL_CFG=ch_cfg=[[8,48,96],
                                [240,144,288],
                                [240,144,288]]
        net = DFANet(ENCODER_CHANNEL_CFG,decoder_channel=64,num_classes=num_classes).cuda()
        bk=XceptionA(ch_cfg[0],num_classes=num_classes)
        torch.save(bk.state_dict(),'./pretrained/dfa.pth')
        
        model = load_backbone(net,"pretrained/dfa.pth").cuda()
    elif model_type == "erfnet":
        from network.erfnet import Net
        from network.erfnet_imagenet import ERFNet as ERFNet_imagenet
        pretrainedEnc = torch.nn.DataParallel(ERFNet_imagenet(1000))
        pretrainedEnc.load_state_dict(torch.load("./pretrained/erfnet_encoder_pretrained.pth.tar")['state_dict'])
        pretrainedEnc = next(pretrainedEnc.children()).features.encoder
        model = Net(num_classes, encoder=pretrainedEnc).cuda()

    return model


