from typing import Any

import torch
import torch.nn as nn

from .base_postprocessor import BasePostprocessor


class RankFeatPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super(RankFeatPostprocessor, self).__init__(config)
        self.config = config
        self.args = self.config.postprocessor.postprocessor_args

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        inputs = data.cuda()

        # Logit of Block 4 feature
        feat1 = net.intermediate_forward(inputs, layer_index=-1)
        #print(f"{feat1.shape=}")
        try:
            B, C, H, W = feat1.size()
            feat1 = feat1.view(B, C, H * W)
            needs_reshape = True
        except:
            B, C, H = feat1.size()
            W = 1
            needs_reshape = False
        if self.args.accelerate:
            feat1 = feat1 - power_iteration(feat1, iter=20)
        else:
            u, s, v = torch.linalg.svd(feat1, full_matrices=False)
            feat1 = feat1 - s[:, 0:1].unsqueeze(2) * u[:, :, 0:1].bmm(
                v[:, 0:1, :])
        if needs_reshape: feat1 = feat1.view(B, C, H, W)
        logits1 = net.intermediate_feature_to_logits(feat1,next_layers_idx=[])

        # Logit of Block 3 feature
        feat2 = net.intermediate_forward(inputs, layer_index=-2)

        try:
            B, C, H, W = feat2.size()
            feat2 = feat2.view(B, C, H * W)
            needs_reshape = True
        except:
            B, C, H = feat2.size()
            W = 1
            needs_reshape = False
        if self.args.accelerate:
            feat2 = feat2 - power_iteration(feat2, iter=20)
        else:
            u, s, v = torch.linalg.svd(feat2, full_matrices=False)
            feat2 = feat2 - s[:, 0:1].unsqueeze(2) * u[:, :, 0:1].bmm(
                v[:, 0:1, :])
        if needs_reshape: feat2 = feat2.view(B, C, H, W)
        logits2 = net.intermediate_feature_to_logits(feat2,next_layers_idx=[-1])

        #print(f"{logits1.shape=}")
        #print(f"{logits2.shape=}")

        # Fusion at the logit space
        logits = (logits1 + logits2) / 2
        conf = self.args.temperature * torch.logsumexp(
            logits / self.args.temperature, dim=1)

        _, pred = torch.max(logits, dim=1)

        #print(f"{pred.shape=}")
        #print(f"{conf.shape=}")
        return pred, conf


def _l2normalize(v, eps=1e-10):
    return v / (torch.norm(v, dim=2, keepdim=True) + eps)


# Power Iteration as SVD substitute for acceleration
def power_iteration(A, iter=20):
    u = torch.FloatTensor(1, A.size(1)).normal_(0, 1).view(
        1, 1, A.size(1)).repeat(A.size(0), 1, 1).to(A)
    v = torch.FloatTensor(A.size(2),
                          1).normal_(0, 1).view(1, A.size(2),
                                                1).repeat(A.size(0), 1,
                                                          1).to(A)
    for _ in range(iter):
        v = _l2normalize(u.bmm(A)).transpose(1, 2)
        u = _l2normalize(A.bmm(v).transpose(1, 2))
    sigma = u.bmm(A).bmm(v)
    sub = sigma * u.transpose(1, 2).bmm(v.transpose(1, 2))
    return sub
