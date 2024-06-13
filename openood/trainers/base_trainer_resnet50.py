import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import openood.utils.comm as comm
from openood.utils import Config

from .lr_scheduler import cosine_annealing, CosineAnnealingWarmupRestarts_New


class BaseTrainerResnet50:
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
                weight_decay=config.optimizer.weight_decay
            )
        else:
            raise NotImplementedError

        if config.scheduler.name == 'cosine_warm_restarts':
            try: num_restarts = config.scheduler.num_restarts
            except: print('Warning: Num restarts not specified...using 2'); num_restarts = 2

            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer,
                                                                            T_0=int(config.optimizer.num_epochs / (num_restarts + 1)),
                                                                            eta_min=config.optimizer.lr* 1e-3)
        elif config.scheduler.name == 'cosine_warm_restarts_warmup':

            try: num_restarts = config.scheduler.num_restarts
            except: print('Warning: Num restarts not specified...using 2'); num_restarts = 2

            self.scheduler = CosineAnnealingWarmupRestarts_New(warmup_epochs=config.scheduler.warmup_epochs,
                                                               optimizer=self.optimizer,
                                                               T_0=int(config.optimizer.num_epochs / (num_restarts + 1)),
                                                               eta_min=config.optimizer.lr * 1e-3)
        else:
            raise NotImplementedError

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
            # self.scheduler.step() note: switched to epoch

            # exponential moving average, show smooth values
            with torch.no_grad():
                loss_avg = loss_avg * 0.8 + float(loss) * 0.2

        # comm.synchronize()
        self.scheduler.step(epoch=epoch_idx-1)
        print(f"epoch {epoch_idx} - stepped scheduler, lr:", self.optimizer.param_groups[0]['lr'])

        metrics = {}
        metrics['epoch_idx'] = epoch_idx
        metrics['loss'] = self.save_metrics(loss_avg)

        return self.net, metrics

    def save_metrics(self, loss_avg):
        all_loss = comm.gather(loss_avg)
        total_losses_reduced = np.mean([x for x in all_loss])

        return total_losses_reduced
