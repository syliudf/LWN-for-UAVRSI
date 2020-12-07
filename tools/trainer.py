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

class Trainer(object):
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
        # if args.resume:
        #     self.resume()

    # def resume(self):
    #     self.logger.info("---------------resume beginning....")
    #     checkpoint=torch.load(self.args.resume)
    #     self.model.load_state_dict(checkpoint['model'])
    #     self.optimizer.load_state_dict(checkpoint['optimizer'])
    #     self.criterion.load_state_dict(checkpoint['criterion'])
    #     self.start_epoch=checkpoint['epoch']
    #     self.logger.info("---------------resume end....")
    
    def adjust_learning_rate(self, cur_epoch, max_epoch, curEpoch_iter, perEpoch_iter, baselr):
        """
        poly learning stategyt
        lr = baselr*(1-iter/max_iter)^power
        """
        cur_iter = cur_epoch*perEpoch_iter + curEpoch_iter
        max_iter = max_epoch*perEpoch_iter
        lr = baselr*pow((1 - 1.0*cur_iter/max_iter), 0.9)
        return lr
    
    def train_net(self, epoch):
        '''
        args:
            trainloader: loaded for traain dataset
            model: model
            criterion: loss function
            optimizer: optimizer algorithm, such as Adam or SGD
            epoch: epoch_number
        return:
            average loss
        '''
        self.model.train()
        trainloader = self.dataloader
        total_batches = len(trainloader)
        for iter, batch in enumerate(trainloader, 0):
            lr = self.adjust_learning_rate(
                                    cur_epoch=epoch,
                                    max_epoch=self.args.max_epochs,
                                    curEpoch_iter=iter,
                                    perEpoch_iter=total_batches,
                                    baselr=self.args.lr
                                    )
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            start_time = time.time()
            images, labels, name = batch
            images = images[:, 0:3, :, :].cuda()
            labels = labels.type(torch.long).cuda()
            output = self.model(images)
            loss = self.criterion(output, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            interval_time = time.time() - start_time
            if iter+1 == total_batches:
                fmt_str = '======> epoch [{:d}/{:d}] cur_lr: {:.6f} loss: {:.5f} time: {:.2f}'
                print_str = fmt_str.format(
                                        epoch,
                                        self.args.max_epochs,
                                        lr,
                                        loss.item(),
                                        interval_time
                                        )
                print(print_str)
                self.logger.info(print_str)


