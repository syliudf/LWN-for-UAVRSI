#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8
# !/usr/bin/python
# -*- coding: utf-8 -*-

import os
import time
import random
import argparse
import torch
import numpy as np
import timeit
import configparser
import torch.optim as optim
from tools.trainer import Trainer
from tools.tester import Tester
from torch.utils.data import DataLoader
from metrics.metrics_uavid import runningScore, averageMeter
import torch.backends.cudnn as cudnn
from utils.modeltools import netParams
from utils.set_logger import get_logger
import utils.utils
from network import build_network

import warnings
warnings.filterwarnings('ignore')

cfg = configparser.RawConfigParser()
cfg.read("config.ini")

dataset_type = cfg.get("init", "DATASET")

MODEL_INIT = cfg.get(dataset_type, "MODEL_INIT")
ROOT = cfg.get(dataset_type, "ROOT")
BATCH_SIZE = cfg.getint(dataset_type, "BATCH_SIZE")
MAX_EPOCHES = cfg.getint(dataset_type, "MAX_EPOCHES")
LR_INIT = cfg.getfloat(dataset_type, "LR_INIT")
NUM_CLASSES = cfg.getint(dataset_type, "NUM_CLASSES")
SAVE_DIR = cfg.get(dataset_type, "SAVE_DIR")
GPU = cfg.get(dataset_type, "GPU")
REPEAT_TIME = cfg.getint(dataset_type, "REPEAT")

RUN_ID = MODEL_INIT+'_'+str(MAX_EPOCHES)+'_'+str(REPEAT_TIME)

def main(args, logger):

    cudnn.enabled = True     # Enables bencnmark mode in cudnn, to enable the inbuilt
    cudnn.benchmark = True   # cudnn auto-tuner to find the best algorithm to use for
                             # our hardware
    #Setup random seed
    # cudnn.deterministic = True # ensure consistent results
                                 # if benchmark = True, deterministic will be False.
    
    seed = random.randint(1, 10000)
    print('======>random seed {}'.format(seed))
    logger.info('======>random seed {}'.format(seed))
    
    random.seed(seed)  # python random seed
    np.random.seed(seed)  # set numpy random seed
    start = timeit.default_timer()

    torch.manual_seed(seed)  # set random seed for cpu
    if torch.cuda.is_available():
        # torch.cuda.manual_seed(seed) # set random seed for GPU now
        torch.cuda.manual_seed_all(seed)  # set random seed for all GPU

    # Setup device
    # device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    # setup DatasetLoader
    if dataset_type == 'uavid':
        from loader.load_uavid import uavidloader
        train_set = uavidloader(root=args.root, split='train')
        test_set = uavidloader(root=args.root, split='val')
    elif dataset_type == 'udd6':
        from loader.load_udd6 import udd6loader
        train_set = udd6loader(root=args.root, split='train')
        test_set = udd6loader(root=args.root, split='val')
    elif dataset_type == 'vai':
        from loader.load_vaihingen import vaihingenloader
        train_set = vaihingenloader(root=args.root, split='train')
        test_set = vaihingenloader(root=args.root, split='test')
    else:
        from loader.load_udd6 import udd6loader
        train_set = udd6loader(root=args.root, split='train')
        test_set = udd6loader(root=args.root, split='val')

    kwargs = {'num_workers': args.workers, 'pin_memory': True}
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
    # test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, **kwargs)

    # setup optimization criterion
    criterion = utils.utils.cross_entropy2d

    # setup model
    print('======> building network')
    logger.info('======> building network')

    model = build_network(MODEL_INIT, NUM_CLASSES)
    pretrain_state_dict = torch.load(model_test, map_location="cuda:0")

    model.load_state_dict(pretrain_state_dict)
    if torch.cuda.device_count() > 1:

        device_ids = list(map(int, args.gpu.split(',')))
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    print("======> computing network parameters")
    logger.info("======> computing network parameters")

    total_paramters = netParams(model)
    print("the number of parameters: " + str(total_paramters))
    logger.info("the number of parameters: " + str(total_paramters))

    # setup optimizer
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    # optimizer = torch.optim.Adam(model.parameters(), args.lr, (0.9, 0.999), eps=1e-08, weight_decay=5e-4)

    # setup savedir      
    args.savedir = (args.savedir + '/' + args.model + 'bs'
                    + str(args.batch_size) + 'gpu' + str(args.gpu) + '/')
    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)

    start_epoch = 0
    flag = True

    best_epoch = 0.
    best_overall = 0.
    best_mIoU = 0.
    best_F1 = 0. 
    epoch = 0
    train_1_epoch = Trainer(args, train_loader, model, criterion, optimizer, epoch, logger)
    testing = Tester(args, test_loader, model, criterion, optimizer, epoch, logger)
    
    score, class_iou, class_F1 = testing.test_net()

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
        model_file_name = args.savedir + '/model.pth'
        # torch.save(model.module.state_dict(), model_file_name)
        best_epoch = epoch
    
    if score["Overall Acc : \t"] > best_overall:
        best_overall = score["Overall Acc : \t"]
        # save model in best overall Acc
                
             


if __name__ == '__main__':

    import timeit
    start = timeit.default_timer()

    parser = argparse.ArgumentParser(description='Semantic Segmentation...')

    parser.add_argument('--model', default=MODEL_INIT, type=str)
    parser.add_argument('--root', default=ROOT, help='data directory')

    parser.add_argument('--batch_size', default=BATCH_SIZE, type=int)
    parser.add_argument('--max_epochs', type=int, default=MAX_EPOCHES, help='the number of epochs: default 100 ')
    parser.add_argument('--num_classes', default=NUM_CLASSES, type=int)
    parser.add_argument('--lr', default=LR_INIT, type=float)
    parser.add_argument('--weight_decay', default=4e-5, type=float)
    parser.add_argument('--workers', type=int, default=2, help=" the number of parallel threads")
    parser.add_argument('--show_interval', default=10, type=int)
    parser.add_argument('--show_val_interval', default=1000, type=int)
    parser.add_argument('--savedir', default=SAVE_DIR, help="directory to save the model snapshot")
    # parser.add_argument('--logFile', default= "log.txt", help = "storing the training and validation logs")
    parser.add_argument('--gpu', type=str, default=GPU, help="default GPU devices (3)")

    args = parser.parse_args()
    model_test = "./runs_udd6/b1_nof_100_5/b1_nofbs8gpu0/model2.pth"

    RUN_ID = "test_"+"b1_nof_100_5"
    print('Now run_id {}'.format(RUN_ID))
    args.savedir = os.path.join(args.savedir, str(RUN_ID), "test")
    print(args.savedir)

    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)
    logger = get_logger(args.savedir)
    logger.info('just do it')
    
    print('Input arguments:')
    logger.info('======>Input arguments:')
    
    for key, val in vars(args).items():
        print('======>{:16} {}'.format(key, val))
        logger.info('======> {:16} {}'.format(key, val))

    if torch.cuda.device_count() > 1:
        torch.cuda.set_device(int(args.gpu.split(',')[0]))
        # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu.split(',')[0]
    else:
        # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        torch.cuda.set_device(int(args.gpu))
    
    main(args, logger)
    end = timeit.default_timer()
    print("training time:", 1.0*(end-start)/3600)
    print('model save in {}.'.format(RUN_ID))




