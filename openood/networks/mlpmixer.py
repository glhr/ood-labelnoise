# from https://github.com/kentaroy47/vision-transformers-cifar10/blob/main/train_cifar10.py

# https://github.com/lucidrains/mlp-mixer-pytorch/blob/main/mlp_mixer_pytorch/mlp_mixer_pytorch.py
from torch import nn
from functools import partial
from einops.layers.torch import Rearrange, Reduce

pair = lambda x: x if isinstance(x, tuple) else (x, x)

class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x

def FeedForward(dim, expansion_factor = 4, dropout = 0., dense = nn.Linear):
    inner_dim = int(dim * expansion_factor)
    return nn.Sequential(
        dense(dim, inner_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(inner_dim, dim),
        nn.Dropout(dropout)
    )

class MLPMixer(nn.Module):
    def __init__(self, image_size, channels, patch_size, dim, depth, num_classes, expansion_factor = 4, expansion_factor_token = 0.5, dropout = 0., *args, **kwargs):
        super(MLPMixer, self).__init__()
        image_h, image_w = pair(image_size)
        assert (image_h % patch_size) == 0 and (image_w % patch_size) == 0, 'image must be divisible by patch size'
        num_patches = (image_h // patch_size) * (image_w // patch_size)
        chan_first, chan_last = partial(nn.Conv1d, kernel_size = 1), nn.Linear

        self.rearrange = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size)
        self.linear1 = nn.Linear((patch_size ** 2) * channels, dim)
        self.mixer_layers = nn.ModuleList([])
        for _ in range(depth):
            self.mixer_layers.append(nn.Sequential(
                PreNormResidual(dim, FeedForward(num_patches, expansion_factor, dropout, chan_first)),
                PreNormResidual(dim, FeedForward(dim, expansion_factor_token, dropout, chan_last))
            ))
        self.layer_norm = nn.LayerNorm(dim)
        self.reduce = Reduce('b n c -> b c', 'mean')
        self.fc= nn.Linear(dim, num_classes)

    def forward(self, x, return_feature = False, return_feature_list = False):
        x = self.rearrange(x)
        x = self.linear1(x)
        feature_list = [x]
        for mixer_layer in self.mixer_layers:
            x = mixer_layer(x)
            feature_list.append(x)
        x = self.layer_norm(x)
        feat = self.reduce(x)
        logits_cls = self.fc(feat)
        if return_feature:
            return logits_cls, feat
        if return_feature_list:
            return logits_cls, feature_list
        return logits_cls

    def intermediate_forward(self, x, layer_index):
        return self.forward(x, return_feature_list=True)[1][layer_index]

    def rankfeat_logits(self, feat):
        feat = self.layer_norm(feat)
        feat = self.reduce(feat)
        logits_cls = self.fc(feat)
        return logits_cls

    def get_fc(self):
        fc = self.fc
        return fc.weight.cpu().detach().numpy(), fc.bias.cpu().detach().numpy()

    def get_fc_layer(self):
        return self.fc

    def intermediate_feature_to_logits(self, x, next_layers_idx=[]):
        for layer_idx in next_layers_idx:
            x = self.mixer_layers[layer_idx](x)

        x = self.layer_norm(x)
        feat = self.reduce(x)
        logits_cls = self.fc(feat)
        
        return logits_cls

    # return nn.Sequential(
    #     Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
    #     nn.Linear((patch_size ** 2) * channels, dim),
    #     *[nn.Sequential(
    #         PreNormResidual(dim, FeedForward(num_patches, expansion_factor, dropout, chan_first)),
    #         PreNormResidual(dim, FeedForward(dim, expansion_factor_token, dropout, chan_last))
    #     ) for _ in range(depth)],
    #     nn.LayerNorm(dim),
    #     Reduce('b n c -> b c', 'mean'),
    #     nn.Linear(dim, num_classes)
    # )