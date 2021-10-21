import os
import time
import random
import argparse
import torch
import numpy as np
import timeit
import configparser
import torch.optim as optim
from torch.utils.data import dataloader

from loader.load_uavid import uavidloader
from torch.utils.data import DataLoader
from metrics.metrics_uavid import runningScore
import torch.backends.cudnn as cudnn
from utils.modeltools import netParams
from utils.set_logger import get_logger
import utils.utils

class Tester(object):
    def __init__(self, args, dataloader, model, criterion, optimizer, epoch, logger):
        """
        """
        self.args = args
        self.dataloader = dataloader
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.epoch = epoch
        self.logger = logger

    def test_net(self):
        '''
        args:
            trainloader: loaded for train dataset
            model: model
            criterion: loss function
            optimizer: optimizer algorithm, such as Adam or SGD
            epoch: epoch_number
        return:
            average loss
        '''
        self.model.eval()
        tloss = 0.
        testloader = self.dataloader
        # Setup Metrics
        running_Metrics = runningScore(self.args.num_classes)
        total_batches = len(testloader)
        print("=====> the number of iterations per epoch: ", total_batches)
        with torch.no_grad():
            for iter, batch in enumerate(testloader):
                # print(iter)
                # start_time = time.time()
                image, label, name = batch
                image = image[:, 0:3, :, :].cuda()
                label = label.cuda()
                output = self.model(image)[0]
                # inter_time = time.time() - start_time
                output = output.cpu().detach()[0].numpy()
                gt = np.asarray(label[0].cpu().detach().numpy(), dtype=np.uint8)
                # print('gt size {}, output shape {}'.format(gt.shape, output.shape))
                output = output.transpose(1, 2, 0)
                output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)
                running_Metrics.update(gt, output)
        # print(f"test phase: Epoch [{epoch:d}/{args.max_epochs:d}] loss: {tloss/total_batches:.5f}")
        score, class_iou, class_F1 = running_Metrics.get_scores()


        running_Metrics.reset()

        return score, class_iou, class_F1


