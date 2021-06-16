# !/usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys
import time
import math
import random
import argparse
import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from loader.load_uavid import uavidloader
# from network.unet import UNet
from network.efficientnet.Efficientnet_DAN import EfficientNet_1_Nof
import sys
sys.path.append('./')

def get_Hrrs_label():
    return np.asarray(
                      [
              [  0,   0,   0],  # background clutter
              [128,   0,   0],  # building
		      [128,  64, 128],  # road
		      [  0, 128,   0],  # tree
		      [128, 128,   0],  # low vegetation
		      [ 64,   0, 128],  # moving car
		      [192,   0, 192],  # static car
		      [ 64,  64,   0]   # human
                    ])

def decode_segmap(label_mask, n_classes = 8):
    """Decode segmentation class labels into a color image
    Args:
        label_mask (np.ndarray): an (M,N) array of integer values denoting
        the class label at each spatial location.
        plot (bool, optional): whether to show the resulting color image
        in a figure.
    Returns:
        (np.ndarray, optional): the resulting decoded color image.
    """
    label_colours = get_Hrrs_label()

    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range(0, n_classes):
        r[label_mask == ll] = label_colours[ll, 0]
        g[label_mask == ll] = label_colours[ll, 1]
        b[label_mask == ll] = label_colours[ll, 2]
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3), dtype=np.uint8)
    rgb[:, :, 0] = r
    rgb[:, :, 1] = g
    rgb[:, :, 2] = b
    return rgb

def tta_inference(inp, model, num_classes=8, scales=[1.0], flip=True):
    b, _, h, w = inp.size()
    preds = inp.new().resize_(b, num_classes, h, w).zero_().to(inp.device)
    for scale in scales:
        size = (int(scale*h), int(scale*w))
        resized_img = F.interpolate(inp, size=size, mode='bilinear', align_corners=True,)
        pred = model_inference(model, resized_img.to(inp.device), flip)
        pred = F.interpolate(pred, size=(h, w), mode='bilinear', align_corners=True,)
        preds += pred

    return preds/(len(scales))

def model_inference(model, image, flip=True):
    output = model(image)
    if flip:
        fimg = image.flip(2)
        output += model(fimg).flip(2)
        fimg = image.flip(3)
        output += model(fimg).flip(3)
        return output/3
    return output

def slide(model, scale_image, num_classes=8, crop_size=512, overlap=1/3, scales=[1.0], flip=True):

    N, C, H_, W_ = scale_image.shape
    # print(f"Height: {H_} Width: {W_}")
    
    full_probs = torch.zeros((N, num_classes, H_, W_), device=scale_image.device) #
    count_predictions = torch.zeros((N, num_classes, H_, W_), device=scale_image.device) #

    h_overlap_length = int((1-overlap)*crop_size) #
    w_overlap_length = int((1-overlap)*crop_size) # 

    h = 0
    slide_finish = False
    while not slide_finish:

        if h + crop_size <= H_:
            # print(f"h: {h}")
            # set row flag
            slide_row = True
            # initial row start
            w = 0
            while slide_row:
                if w + crop_size <= W_:
                    # print(f" h={h} w={w} -> h'={h+crop_size} w'={w+crop_size}")
                    patch_image = scale_image[:, :, h:h+crop_size, w:w+crop_size]
                    #
                    patch_pred_image = tta_inference(patch_image, model, num_classes=num_classes, scales=scales, flip=flip)
                    count_predictions[:,:,h:h+crop_size, w:w+crop_size] += 1
                    full_probs[:,:,h:h+crop_size, w:w+crop_size] += patch_pred_image

                else:
                    # print(f" h={h} w={W_-crop_size} -> h'={h+crop_size} w'={W_}")
                    patch_image = scale_image[:, :, h:h+crop_size, W_-crop_size:W_]
                    #
                    patch_pred_image = tta_inference(patch_image, model, num_classes=num_classes, scales=scales, flip=flip)
                    count_predictions[:,:,h:h+crop_size, W_-crop_size:W_] += 1
                    full_probs[:,:,h:h+crop_size, W_-crop_size:W_] += patch_pred_image
                    slide_row = False

                w += w_overlap_length

        else:
            # print(f"h: {h}")
            # set last row flag
            slide_last_row = True
            # initial row start
            w = 0
            while slide_last_row:
                if w + crop_size <= W_:
                    # print(f"h={H_-crop_size} w={w} -> h'={H_} w'={w+crop_size}")
                    patch_image = scale_image[:,:,H_-crop_size:H_, w:w+crop_size]
                    #
                    patch_pred_image = tta_inference(patch_image, model, num_classes=num_classes, scales=scales, flip=flip)
                    count_predictions[:,:,H_-crop_size:H_, w:w+crop_size] += 1
                    full_probs[:,:,H_-crop_size:H_, w:w+crop_size] += patch_pred_image

                else:
                    # print(f"h={H_-crop_size} w={W_-crop_size} -> h'={H_} w'={W_}")
                    patch_image = scale_image[:,:,H_-crop_size:H_, W_-crop_size:W_]
                    #
                    patch_pred_image = tta_inference(patch_image, model, num_classes=num_classes, scales=scales, flip=flip)
                    count_predictions[:,:,H_-crop_size:H_, W_-crop_size:W_] += 1
                    full_probs[:,:,H_-crop_size:H_, W_-crop_size:W_] += patch_pred_image

                    slide_last_row = False
                    slide_finish = True

                w += w_overlap_length

        h += h_overlap_length

    full_probs /= count_predictions

    return full_probs
    
def predict_sliding(model, image, num_classes=8, crop_size=512, overlap=1/3, scales=[1.0], flip=True):

    N, C, H, W = image.shape
    # scale_image = checksize(image, crop_size=crop_size)
    # N, C, H_, W_ = scale_image.shape
    
    # if H_ == W_ and H_ == 512:
    #     full_probs = tta_inference(
    #         inp=scale_image,
    #         model=model,
    #         num_classes=num_classes,
    #         scales=scales,
    #         flip=flip)
    # else:
    #     full_probs = slide(
    #         model, 
    #         scale_image, 
    #         num_classes=num_classes, 
    #         crop_size=crop_size, 
    #         overlap=overlap, 
    #         scales=scales, 
    #         flip=flip)
    full_probs = slide(
        model, 
        image, 
        num_classes=num_classes, 
        crop_size=crop_size, 
        overlap=overlap, 
        scales=scales, 
        flip=flip)

    full_probs = F.interpolate(full_probs, size=(H, W), mode='bilinear', align_corners=True)

    return full_probs

def test(testloader, model, savedir, device):
    '''
    args:
        test_loaded for test dataset
        model: model
    return:
        mean,Iou,IoU class
    '''
    if not os.path.exists(savedir):
        os.mkdir(savedir)

    model.eval()
    total_batches = len(testloader)
    with torch.no_grad():
        for idx, batch in enumerate(testloader):
            
            # load data
            # print(batch)
            image, _,  name = batch
            print(name)
            image = image.to(device)
            N, C, H, W = image.shape

            # if H == W and H == 512:
            #     # H == W == 512, directly tta inference.
            #     print(f"H={H} and W={W} using tta.")
            #     output = tta_inference(
            #         inp=image,
            #         model=model,
            #         num_classes=8,                                      
            #         scales=[0.75, 1.0, 1.25],
            #         flip=True)
            # else:
                # slide.
            # print(f"H={H} and W={W} using slide.")
            output = predict_sliding(
                model=model,
                image=image,
                num_classes=8,
                crop_size=512,
                overlap=1/4,
                scales=[0.75, 1.0, 1.25],
                flip=True)

            _, output = torch.max(output, 1)
     

            assert len(output.shape) == 3, f"Wrong shape!"
            # convert torch to array
            output = np.asarray(output.permute(1,2,0).data.cpu().numpy(), dtype=np.uint8)

            # input: [H, W, 3]
            imageout = decode_segmap(output.squeeze())

            # std output
            img_save_name = os.path.basename(name[0])

            img_save_name = os.path.splitext(img_save_name)[0]

            # infile = open("/workspace/code/sample.xml", "r",encoding='utf-8')  #打开文件
            # outfile = open(os.path.join(savedir, img_save_name+'.xml'), "w", encoding='utf-8') # 内容输出

            # for line in infile:
            #     outfile.write(line.replace('sample', img_save_name))
            # infile.close()
            # outfile.close()

            img_save_path = os.path.join(savedir, img_save_name[:5], 'Labels', img_save_name[6:]+'.png')
            if not os.path.exists(os.path.join(savedir, img_save_name[:5], 'Labels')):
                os.makedirs(os.path.join(savedir, img_save_name[:5], 'Labels'))
            imageout = Image.fromarray(imageout)
            print(img_save_path)
            imageout.save(img_save_path)

def main(input_path_testA, output_path_testA, model_path):

    cudnn.enabled = True     # Enables bencnmark mode in cudnn, to enable the inbuilt
    cudnn.benchmark = True   # cudnn auto-tuner to find the best algorithm to use for
                             # our hardware
    # Setup device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # torch.cuda.set_device(0)

    T = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.3918, 0.4114, 0.3726], [0.1553, 0.1528, 0.1456]),
    ])

    # testA_set = onlinezkxt(root=input_path_testA, transform=T)
    # testA_set = vaihingenloader(root=args.root, split='test')
    testA_set = uavidloader(root=input_path_testA, split=split_conf)
    testA_loader = DataLoader(testA_set, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    # testB_set = onlinezkxt(root=input_path_testB, transform=T)
    # testB_loader = DataLoader(testB_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    # FCN
    # from network.fcn import VGGNet, FCN32s, FCN16s, FCN8s, FCNs
    # vgg_model = VGGNet(requires_grad=True, remove_fc=False).cuda()
    # model = FCN8s(pretrained_net=vgg_model, n_class=8).cuda()

    # SegNet
    # from network.segnet import SegNet
    # model = SegNet(3,num_classes).cuda()

    # from network.net.Unet import UNet
    # model =UNet(n_channels=3, n_classes=num_classes ).cuda()
    
    # Efficient Net
    # model = model_now.from_name('efficientnet-b1',override_params={'num_classes' : 8}).cuda()

    # DeepLab-V3+
    # from network.net import deeplab_resnet50
    # model = deeplab_resnet50.DeepLabv3_plus(
    #                 nInputChannels=3,
    #                 n_classes=8,
    #                 os=8,
    #                 pretrained=True
    #                 ).cuda()

#     new_dict = {k: v for k,v in pretrain_state_dict.state_dict().items() if k in model_dict}
# # del new_dict['outconv_320_8']
#     model_dict.update(new_dict)
#     model.load_state_dict(model_dict)

    # model = deeplab_resnet50.DeepLabv3_plus(
    #                     nInputChannels=3,
    #                     n_classes=8,
    #                     os=8,
    #                     pretrained=True
    #                     ).cuda()

    # model = torch.nn.DataParallel(model, device_ids=[0])
    # deeplab.gffhead.cls[6] = nn.Conv2d(256, 9, kernel_size=(1, 1), stride=(1, 1))
    # deeplab.auxlayer.conv5[4] = nn.Conv2d(256, 9, kernel_size=(1, 1), stride=(1, 1))
    # print(checkpoint)
    # model = EfficientNet_1_Nof.from_name('efficientnet-b1').cuda()

    model = build_network(model_init, num_classes)
    model_dict = model.state_dict()


    checkpoint = torch.load(model_path ,map_location="cuda:0")

    new_dict = {k: v for k,v in checkpoint.items() if k in model_dict}
    # print(checkpoint.state_dict())
    # del checkpoint['outconv_320_6']
    model_dict.update(new_dict)
    checkpoint = model_dict
    # new_state_dict = OrderedDict()
    # for k, v in checkpoint.items():
    #     # name = k[7:] # remove 'module.'
    #     new_state_dict[name] = v
    # model.load_state_dict(new_state_dict)
    model.load_state_dict(checkpoint) 

    
    
    
    deeplab = model.to(device)

    start = time.time()
    test(testA_loader, deeplab, output_path_testA, device)
    print(f"testA finish")
    runtime1 = time.time() - start
    print(f"Spend Time: {math.floor(runtime1//3600):2d}h:"
    f"{math.floor(runtime1%3600//60):2d}m:{math.floor(runtime1%60):2d}s")
    # test(testB_loader, deeplab, output_path_testB, device)
    # print(f"testB finish")
    # runtime = time.time() - start
    # print(f"Spend Time: {math.floor(runtime//3600):2d}h:"
    # f"{math.floor(runtime%3600//60):2d}m:{math.floor(runtime%60):2d}s")

if __name__ == '__main__':

    import os
    import torch
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from os.path import basename
    from PIL import Image
    import sys
    from network import build_network
    # from network.efficientnet.Efficientnet_DAN import EfficientNet_1_up as model_now
    os.environ["CUDA_VISIBLE_DEVICES"] = "4"

    input_path_testA = './data/uavid'

    output_path_testA = './data/results/deeplabv3+_resnet101_6_test'
    # output_path_testA = "./data_paper/uavid/deeplabv3+_resnet101_3_2"

    model_path = 'runs_uavid/deeplabv3+_resnet101_100_6/deeplabv3+_resnet101bs8gpu4/model.pth'

    num_classes = 8
    model_init = "deeplabv3+_resnet101"

    cudnn.benchmark = True
    cudnn.enabled = True
    num_classes = 8

    split_conf = "test"

    main(input_path_testA,  output_path_testA, model_path)
###TODO: