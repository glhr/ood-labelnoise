import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import openood.utils.comm as comm
from openood.utils import Config

from .lr_scheduler import cosine_annealing

from collections import namedtuple


class BaseTrainerMLPMixer:
    def __init__(self, net: nn.Module, train_loader: DataLoader,
                 config: Config) -> None:

        self.net = net
        self.train_loader = train_loader
        self.config = config

        if config.optimizer.name == 'sgd':
            self.optimizer = torch.optim.SGD(
                net.parameters(),
                config.optimizer.lr,
                momentum=config.optimizer.momentum,
                weight_decay=config.optimizer.weight_decay,
                nesterov=True,
            )
        elif config.optimizer.name == 'adam':
            self.optimizer = torch.optim.Adam(
                net.parameters(),
                config.optimizer.lr
            )

        if config.scheduler.name == 'CosineAnnealingLR':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=config.scheduler.n_epochs,
            )
        else: raise NotImplementedError

    def train_epoch(self, epoch_idx):
        self.net.train()

        loss_avg = 0.0
        train_dataiter = iter(self.train_loader)

        for train_step in tqdm(range(1,
                                     len(train_dataiter) + 1),
                               desc='Epoch {:03d}: '.format(epoch_idx),
                               position=0,
                               leave=True,
                               disable=not comm.is_main_process()):
            batch = next(train_dataiter)
            data = batch['data'].cuda()
            target = batch['label'].cuda()

            # forward
            logits_classifier = self.net(data)
            loss = F.cross_entropy(logits_classifier, target)

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            

            # exponential moving average, show smooth values
            with torch.no_grad():
                loss_avg = loss_avg * 0.8 + float(loss) * 0.2

        # comm.synchronize()

        self.scheduler.step(epoch_idx)

        metrics = {}
        metrics['epoch_idx'] = epoch_idx
        metrics['loss'] = self.save_metrics(loss_avg)

        return self.net, metrics

    def save_metrics(self, loss_avg):
        all_loss = comm.gather(loss_avg)
        total_losses_reduced = np.mean([x for x in all_loss])

        return total_losses_reduced
