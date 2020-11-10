#!/usr/bin/env python
# coding: utf-8

# In[55]:


#!/usr/bin/env python
# coding: utf-8

# In[2]


# !/usr/bin/python
# -*- coding: utf-8 -*-
from network.segnet import SegNet
import os
import time
import torch.nn as nn
import random
import argparse
import torch
import numpy as np
# from network.unet.u_net_resnet_50_encoder import UNetWithresnet101Encoder
# from network.unet.u_net_resnet_101_encoder import UNetWithresnet101Encoder
# from network.DenseDeepResUNet import UNetWithResnet50Encoder
# from network.unet.Unet import UNet
#from network.IGARSS.UNet101_ASPP_Decoder_ResnetBlockonly import FCNRes101
# from network.unet import deeplab_resnet
from network.efficientnet.Efficientnet_mod import EfficientNet_1_upsample as M
from network.fcn import VGGNet, FCN32s, FCN16s, FCN8s, FCNs

import torch.optim as optim
from loader.load_vaihingen import vaihingenloader
from torch.utils.data import DataLoader
from metrics.metrics import runningScore, averageMeter
import torch.backends.cudnn as cudnn
from utils.modeltools import netParams
from utils.set_logger import get_logger
from utils.SE import SELoss
import utils.utils
import matplotlib.pyplot as plt
import pylab
import warnings
warnings.filterwarnings('ignore')

def get_Vaihingen_label():
    return np.asarray(
                        [
                        [255, 255, 255],  # 不透水面
                        [  0,   0, 255],  # 建筑物
                        [  0, 255, 255],  # 低植被
                        [  0, 255,   0],  # 树
                        [255, 255,   0],  # 车
                        [255,   0,   0],  # Clutter/background
                        [  0,   0,   0]   # ignore
                        ]
                        )

def decode_segmap(label_mask):
    """Decode segmentation class labels into a color image
    Args:
        label_mask (np.ndarray): an (M,N) array of integer values denoting
        the class label at each spatial location.
        plot (bool, optional): whether to show the resulting color image
        in a figure.
    Returns:
        (np.ndarray, optional): the resulting decoded color image.
    """
    n_classes = 7
    label_colours = get_Vaihingen_label()

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

def test(args, testloader, model, criterion, device, epoch, logger ):
    '''
    args:
        test_loaded for test dataset
        model: model
    return:
        mean,Iou,IoU class
    '''
    model.eval()
    tloss = 0.
    #Setup Metrics
    running_Metrics = runningScore(args.num_classes)
    total_batches = len(testloader)
    print("=====> the number of iterations per epoch: ", total_batches)
    with torch.no_grad():
        for iter, batch in enumerate(testloader):
            # start_time = time.time()
            image, label, name = batch
            image = image[:, 0:3, :, :].to(device)
            label = label.to(device)
            output = model(image)
            loss = criterion[0](output, label)
#             loss1 = criterion[0](output[0], label)
#             loss2 = criterion[1](output[1], label)
#             loss = loss1 + 0*loss2
            tloss += loss.item()
            # inter_time = time.time() - start_time
#             output = output[0].cpu().detach()[0].numpy()
            output = output.cpu().detach()[0].numpy()
            gt = np.asarray(label[0].cpu().detach().numpy(), dtype=np.uint8)
            # print('gt size {}, output shape {}'.format(gt.shape, output.shape))
            output = output.transpose(1, 2, 0)
            output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)
            imageout = decode_segmap(output)
            img_save_name = os.path.basename(name[0])
#             print(img_save_name)
            img_save_path = os.path.join(args.savedir, img_save_name)
            imageout = Image.fromarray(imageout)
#             print(img_save_path)
            imageout.save(img_save_path)
            
#             plt.imshow(imageout)
#             pylab.show()
#         plt.subplot(121)
#         plt.imshow(img)
#         plt.subplot(122)
#         plt.imshow(seg_pred_image)
#         pylab.show()
            
            running_Metrics.update(gt, output)

            # print_format = "Epoch [{:d}/{:d}] Iter [{:d}/{:d}] time: {:.4f}".format(
            #                                                         epoch, \
            #                                                         args.max_epochs, \
            #                                                         iter, \
            #                                                         total_batches, \
            #                                                         inter_time \
            #                                                                         )
            # print(print_format)
            # logger.info(print_format)
    print(f"test phase: Epoch [{epoch:d}/{args.max_epochs:d}] loss: {tloss/total_batches:.5f}")
    score, class_iou, class_F1 = running_Metrics.get_scores()
    # for k, v in score.items():
    #     print(k, v)
    #     logger.info('{}: {}'.format(k, v))
    
    # for k, v in class_iou.items()
    #     logger.info('{}: {}'.format(k, v))

    running_Metrics.reset()

    return score, class_iou, class_F1


def main(args, logger):

    cudnn.enabled = True     # Enables bencnmark mode in cudnn, to enable the inbuilt
    cudnn.benchmark = True   # cudnn auto-tuner to find the best algorithm to use for
                             # our hardware
    
    seed = random.randint(1, 10000)
    print('======>random seed {}'.format(seed))
    logger.info('======>random seed {}'.format(seed))
    
    random.seed(seed)  # python random seed
    np.random.seed(seed)  # set numpy random seed

    torch.manual_seed(seed)  # set random seed for cpu
    if torch.cuda.is_available():
        # torch.cuda.manual_seed(seed) # set random seed for GPU now
        torch.cuda.manual_seed_all(seed)  # set random seed for all GPU

    # Setup device
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")

    # setup DatasetLoader
    test_set = vaihingenloader(root=args.root, split='test')

    kwargs = {'num_workers': args.workers, 'pin_memory': True}

    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, **kwargs)

    # setup optimization criterion
    criterion1 = utils.utils.cross_entropy2d
    criterion2 = SELoss()
    criterion = (criterion1, criterion2)

    # setup model
    print('======> building network')
    logger.info('======> building network')
    #model = M.from_name('efficientnet-b1').cuda()
    #vgg_model = VGGNet(requires_grad=True, remove_fc=False).to(device)
    #model = FCN8s(pretrained_net=vgg_model, n_class=6).to(device)
    model = SegNet(3,6).cuda()
    # model = FCNRes34().cuda()
    # model = UNet(n_channels=3, n_classes=6,).to(device)
    if torch.cuda.device_count() >1:
        device_ids = list(map(int, args.gpu.split(',')))
        model = nn.DataParallel(model, device_ids=device_ids)
    # model = UNetWithResnet50Encoder().to(device)

#     model = deeplab_resnet.DeepLabv3_plus(   
#                         nInputChannels=3, 
#                         n_classes=6, 
#                         os=16, 
#                         pretrained=False
#                         ).to(device)
    checkpoint = torch.load('/media/ssd/lsy_data/igarss/runs_seg/segment_5/effnetbs8gpu4,5/model.pth',
                           map_location=lambda storage, loc: storage.cuda(0))
    # model.load_state_dict(checkpoint['model'])
    model.load_state_dict(checkpoint)

    # setup savedir      
    args.savedir = (args.savedir + '/' + args.model + 'bs'
                    + str(args.batch_size) + 'gpu' + str(args.gpu) + '/')
    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)

    start_epoch = 0
    flag = True

    best_overall = 0.
    best_mIoU = 0.
    best_F1 = 0. 

    while flag == True:
        for epoch in range(start_epoch, args.max_epochs):
            if epoch % 1 == 0 or (epoch + 1) == args.max_epochs:
                print('Now Epoch {}, starting evaluate on Test dataset.'.format(epoch))
                logger.info('Now starting evaluate on Test dataset.')
                print('length of test set:', len(test_loader))
                logger.info('length of test set: {}'.format(len(test_loader)))

                score, class_iou, class_F1 = test(args, test_loader, model, criterion, device, epoch, logger)
        
                for k, v in score.items():
                    print('{}: {:.5f}'.format(k, v))
                    logger.info('======>{0:^18} {1:^10}'.format(k, v))
                
                print('Now print class iou')
                for k, v in class_iou.items():
                    print('{}: {:.5f}'.format(k, v))
                    logger.info('======>{0:^18} {1:^10}'.format(k, v))

                print('Now print class_F1')
                for k, v in class_F1.items():
                    print('{}: {:.5f}'.format(k, v))
                    logger.info('======>{0:^18} {1:^10}'.format(k, v))
                
                if score["Mean IoU : \t"] > best_mIoU:
                    best_mIoU = score["Mean IoU : \t"]
                
                if score["Overall Acc : \t"] > best_overall:
                    best_overall = score["Overall Acc : \t"]
                
                if score["Mean F1 : \t"] > best_F1:
                    best_F1 = score["Mean F1 : \t"]

                print(f"best mean IoU: {best_mIoU}")
                print(f"best overall : {best_overall}")
                print(f"best F1: {best_F1}")
                print()

        if (epoch + 1) == args.max_epochs:
            # print('the best pred mIoU: {}'.format(best_pred))
            flag = False
            break
            
if __name__ == '__main__':
    
    import torch
    import os
    from torch.utils.data import DataLoader
    from torchvision import transforms
    import matplotlib.pyplot as plt
    import pylab
    import timeit
    from os.path import basename
    from PIL import Image
    start = timeit.default_timer()

    parser = argparse.ArgumentParser(description='Semantic Segmentation...')

    parser.add_argument('--model', default='dan', type=str)
    parser.add_argument('--root', default='./data/vaismall/', help='data directory')

    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--max_epochs', type=int, default=1, help='the number of epochs: default 100 ')
    parser.add_argument('--num_classes', default=6, type=int)
    parser.add_argument('--lr', default=0.002, type=float)
    parser.add_argument('--weight_decay', default=4e-5, type=float)
    parser.add_argument('--workers', type=int, default=2, help=" the number of parallel threads")
    parser.add_argument('--show_interval', default=10, type=int)
    parser.add_argument('--show_val_interval', default=1000, type=int)
    parser.add_argument('--savedir', default="./results_seg/", help="directory to save the model snapshot")
    parser.add_argument('--logFile', default= "log.txt", help = "storing the training and validation logs")
    parser.add_argument('--gpu', type=str, default="1,2", help="default GPU devices (1,2)")

    args = parser.parse_args()

    def get_Vaihingen_label():
        return np.asarray(
                            [
                            [255, 255, 255],  # 不透水面
                            [  0,   0, 255],  # 建筑物
                            [  0, 255, 255],  # 低植被
                            [  0, 255,   0],  # 树
                            [255, 255,   0],  # 车
                            [255,   0,   0],  # Clutter/background
                            [  0,   0,   0]   # ignore
                            ]
                            )

    def decode_segmap(label_mask):
        """Decode segmentation class labels into a color image
        Args:
            label_mask (np.ndarray): an (M,N) array of integer values denoting
            the class label at each spatial location.
            plot (bool, optional): whether to show the resulting color image
            in a figure.
        Returns:
            (np.ndarray, optional): the resulting decoded color image.
        """
        n_classes = 7
        label_colours = get_Vaihingen_label()

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


    root = '/media/ssd/lsy_data/igarss/data/vaismall'

    isprs = vaihingenloader(root=root, split='test')
    print(len(isprs))
    dataloader = DataLoader(isprs, batch_size=1, shuffle=True, num_workers=2)
    print(len(dataloader))
    
    run_id = random.randint(1, 100000)
    print('Now run_id {}'.format(run_id))
    args.savedir = os.path.join(args.savedir, str(run_id))
    print(args.savedir)
    
    for image, label, name in dataloader:
        print('images type is {},  labels type is {}'.format(image.type(), label.type()))
        print('images size is {},  labels size is {}'.format(image.size(), label.size()))
        img = image.numpy()[0, 0:3, :, :].transpose((1, 2, 0))
        label = label.numpy().squeeze().astype(np.uint8)
        seg_pred_image = decode_segmap(label)
        break
#         seg_pred_image = Image.fromarray(seg_pred_image, mode='RGB')
#         seg_pred_image.save(img_save_path)
       
        # print(np.unique(label))
#         plt.subplot(121)
#         plt.imshow(img)
#         plt.subplot(122)
#         plt.imshow(seg_pred_image)
#         pylab.show()
#         break    

    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)
    logger = get_logger(args.savedir)
    logger.info('just do it')
    print('Input arguments:')
    logger.info('======>Input arguments:')
    
    for key, val in vars(args).items():
        print('======>{:16} {}'.format(key, val))
        logger.info('======> {:16} {}'.format(key, val))

    torch.cuda.set_device(int(args.gpu.split(',')[0]))

    main(args, logger)
    end = timeit.default_timer()
    print("test time:", 1.0*(end-start)/3600)


# In[1]:


try:   
    get_ipython().system('jupyter nbconvert --to python test.ipynb')
    # python即转化为.py，script即转化为.html
    # file_name.ipynb即当前module的文件名
except:
    pass


# In[ ]:




