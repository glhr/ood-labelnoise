from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_postprocessor import BasePostprocessor


class ASHPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super(ASHPostprocessor, self).__init__(config)
        self.args = self.config.postprocessor.postprocessor_args
        self.percentile = self.args.percentile
        self.variant = self.args.variant
        self.temperature = self.args.temperature

        self.args_dict = self.config.postprocessor.postprocessor_sweep

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        output = net.forward_threshold(data, self.percentile, variant=self.variant)
        _, pred = torch.max(output, dim=1)
        #energyconf = torch.logsumexp(output.data.cpu(), dim=1)
        energyconf = self.temperature * torch.logsumexp(output.data.cpu() / self.temperature,
                                                  dim=1)
        return pred, energyconf

    def set_hyperparam(self, hyperparam: list):
        self.percentile = hyperparam[0]
        self.temperature = hyperparam[1]

    def get_hyperparam(self):
        return [self.percentile, self.temperature]
