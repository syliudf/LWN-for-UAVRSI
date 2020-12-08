from  .fcn import VGGNet, FCN32s, FCN16s, FCN8s, FCNs

def build_network(model_type, num_classes=8, pretrained=True, num_channels=3):
    if model_type == 'b1_up' :
        from .efficientnet.Efficientnet_uav import EfficientNet_1_up

