import numpy as np
import torch
import math

def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * \
                (1 + np.cos(step / total_steps * np.pi))

# from https://github.com/sgvaze/osr_closed_set_all_you_need/blob/3258f058d4fd75cb88e23d0bd638d24b629e60d3/utils/schedulers.py#L6
class CosineAnnealingWarmupRestarts_New(torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):

    def __init__(self, warmup_epochs, *args, **kwargs):

        super(CosineAnnealingWarmupRestarts_New, self).__init__(*args, **kwargs)

        # Init optimizer with low learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.eta_min

        self.warmup_epochs = warmup_epochs

        # Get target LR after warmup is complete
        target_lr = self.eta_min + (self.base_lrs[0] - self.eta_min) * (1 + math.cos(math.pi * warmup_epochs / self.T_i)) / 2

        # Linearly interpolate between minimum lr and target_lr
        linear_step = (target_lr - self.eta_min) / self.warmup_epochs
        self.warmup_lrs = [self.eta_min + linear_step * (n + 1) for n in range(warmup_epochs)]

    def step(self, epoch=None):

        # Called on super class init
        if epoch is None:
            super(CosineAnnealingWarmupRestarts_New, self).step(epoch=epoch)

        else:
            if epoch < self.warmup_epochs:
                lr = self.warmup_lrs[epoch]
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr

                # Fulfill misc super() funcs
                self.last_epoch = math.floor(epoch)
                self.T_cur = epoch
                self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

            else:

                super(CosineAnnealingWarmupRestarts_New, self).step(epoch=epoch)