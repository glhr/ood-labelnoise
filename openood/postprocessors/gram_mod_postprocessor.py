from __future__ import division, print_function

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from .base_postprocessor import BasePostprocessor
from .info import get_num_classes

import itertools
import six


class GRAMPostprocessorMod(BasePostprocessor):
    def __init__(self, config):
        self.config = config
        self.postprocessor_args = config.postprocessor.postprocessor_args
        self.num_classes = get_num_classes(self.config.dataset.name)
        self.powers = self.postprocessor_args.powers

        self.feature_min, self.feature_max = None, None
        self.args_dict = self.config.postprocessor.postprocessor_sweep
        self.setup_flag = False
        self.has_data_based_setup = True
        self.divide_by_conf = self.postprocessor_args.divide_by_conf
        self.normalize_devs = self.postprocessor_args.normalize_devs

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict, id_loader_split="train"):
        print(f"Setup on ID data - {id_loader_split} split")
        if not self.setup_flag:
            self.feature_min, self.feature_max = get_min_max(
                net, id_loader_dict[id_loader_split], self.powers, self.num_classes)
            # self.feature_min, self.feature_max = sample_estimator(
            #     net, id_loader_dict[id_loader_split], self.num_classes, self.powers)

            if self.normalize_devs:
                dev_split = "val" if id_loader_split == "train" else "train" if id_loader_split == "val" else "unknown"
                for i,batch in enumerate(tqdm(id_loader_dict[dev_split], desc='Validation deviations')):
                    _, val_deviations_batch = get_deviations(net, batch['data'].cuda(), self.feature_min, self.feature_max, self.powers, num_classes=self.num_classes, divide_by_conf=self.divide_by_conf)
                    if i == 0:
                        val_deviations = val_deviations_batch
                    else:
                        val_deviations = torch.cat((val_deviations,val_deviations_batch),dim=0)
                
                self.val_avg_dev_per_layer = val_deviations.mean(dim=0)+10**-10
                self.val_avg_dev_per_layer = self.val_avg_dev_per_layer.cpu().detach().numpy()
                print(f"Validation average deviations per layer: {self.val_avg_dev_per_layer}")
            else:
                self.val_avg_dev_per_layer = None
            self.setup_flag = True
        else:
            pass

    def postprocess(self, net: nn.Module, data: Any):
        preds, deviations = get_deviations(net, data, self.feature_min, self.feature_max, self.powers, num_classes=self.num_classes, divide_by_conf=self.divide_by_conf, normalize_devs=self.val_avg_dev_per_layer)
        #print(preds.shape, deviations.shape)
        deviations = -deviations.sum(axis=1)
        return preds, deviations

    def set_hyperparam(self, hyperparam: list):
        pass
        #self.powers = hyperparam

    def get_hyperparam(self):
        return self.powers


def tensor2list(x):
    return x.data.cuda().tolist()


def G_p(ob, p):
    try:
        temp = ob.detach()
        
        temp = temp**p
        temp = temp.reshape(temp.shape[0],temp.shape[1],-1)
        #assert torch.isnan(temp).sum() == 0, f"G_p has NaNs"
        #assert torch.isinf(temp).sum() == 0, f"G_p has Infs"
        temp = ((torch.matmul(temp,temp.transpose(dim0=2,dim1=1))))
        # print the tensor type
        #print(temp.type())
        #assert torch.isnan(temp).sum() == 0, f"G_p has NaNs"
        #assert torch.isinf(temp).sum() == 0, f"G_p has Infs"
        
        temp = temp.sum(dim=2) 
        #assert torch.isnan(temp).sum() == 0, f"G_p has NaNs"
        temp = (temp.sign()*torch.abs(temp)**(1/p)).reshape(temp.shape[0],-1)

        assert torch.isnan(temp).sum() == 0, f"G_p has Nans, trying with float64"
        assert torch.isinf(temp).sum() == 0, f"G_p has Infs, trying with float64"
        #assert torch.isnan(temp).sum() == 0, f"G_p has NaNs"
    except AssertionError as e:
        #print("...trying with float64")
        temp = ob.double().detach()
        
        temp = temp**p
        temp = temp.reshape(temp.shape[0],temp.shape[1],-1)
        temp = ((torch.matmul(temp,temp.transpose(dim0=2,dim1=1))))
        temp = temp.sum(dim=2) 
        temp = (temp.sign()*torch.abs(temp)**(1/p)).reshape(temp.shape[0],-1)
    
    return temp

@torch.no_grad()
def get_min_max_per_cls(g_ps_list, power, num_layer):
    mins = []
    maxs = []
    
    #for g_p_list in g_ps_list:
    assert len(g_ps_list) == num_layer, f"{len(g_ps_list)} != {num_layer}"
    assert len(g_ps_list[0]) == len(power), f"{len(g_ps_list[0])} != {len(power)}"

    for L,_ in enumerate(g_ps_list):
        if L==len(mins):
            mins.append([None]*len(power))
            maxs.append([None]*len(power))
        
        for p,P in enumerate(power):
            g_p = g_ps_list[L][p]
            #assert torch.isnan(g_p).sum() == 0, f"g_p has NaNs"
            #print(g_p)
            
            current_min = g_p.min(dim=0,keepdim=True)[0]
            current_max = g_p.max(dim=0,keepdim=True)[0]
            
            if mins[L][p] is None:
                mins[L][p] = current_min
                maxs[L][p] = current_max
            else:
                mins[L][p] = torch.min(current_min,mins[L][p])
                maxs[L][p] = torch.max(current_max,maxs[L][p])
            
            #assert torch.isnan(mins[L][p]).sum() == 0, f"mins[{L}][{p}] has NaNs"
            #assert torch.isnan(maxs[L][p]).sum() == 0, f"maxs[{L}][{p}] has NaNs"
    
    return mins,maxs

@torch.no_grad()
def get_min_max(model, train_loader, power, num_classes):

    model.eval()

    dataset_len = len(train_loader.dataset)

    preds_cls = []
    labels_cls = []

    print(power)
    
    
    for i,batch in enumerate(tqdm(train_loader, desc='Compute min/max')):
        #if i > 5: break
        data = batch['data'].cuda()
        labels = batch['label'].cuda()
        batch_size = data.shape[0]
        preds, feat_list = model(data, return_feature_list=True)
        preds = preds.cpu().detach().numpy()
        feat_list = [feat.detach() for feat in feat_list]
        if i == 0:
            num_layer = len(feat_list)
            mins = [[[None for x in range(len(power))] for y in range(num_layer)] for z in range(num_classes)]
            maxs = [[[None for x in range(len(power))] for y in range(num_layer)] for z in range(num_classes)]
            g_p_list = [[[None for x in range(len(power))]
                      for y in range(num_layer)] for z in range(dataset_len)]
        pred_cls = np.argmax(preds, axis=1)
        

        g_feature = [None]*num_layer
        for layer_idx, feat_L in enumerate(feat_list):
            g_feature[layer_idx] = [None]*len(power)
            for power_idx, P in enumerate(power):
                g_feature[layer_idx][power_idx] = G_p(feat_L,P)
                #print(len(g_feature[layer_idx][power_idx]))
                #assert torch.isnan(g_feature[layer_idx][power_idx]).sum() == 0, f"g_feature[layer_idx][power_idx] has NaNs"
                for sample_idx in range(batch_size):
                    g_p_list[len(preds_cls)+sample_idx][layer_idx][power_idx] = g_feature[layer_idx][power_idx][sample_idx]

        preds_cls.extend(pred_cls)
        labels_cls.extend(labels.cpu().detach().numpy())
    
    
    preds_cls = np.array(preds_cls)
    labels_cls = np.array(labels_cls)

    assert len(preds_cls) == len(g_p_list), f"preds_cls {len(preds_cls)} != g_p_list {len(g_p_list)}"

    mins_per_cls = dict()
    maxs_per_cls = dict()

    for cls in range(num_classes):
        idx = np.argwhere(preds_cls==cls).flatten()
        if len(idx) == 0:
            print(f"Warning - Class {cls} has no predictions, using label instead")
            idx = np.argwhere(labels_cls==cls).flatten()
        #print(f"Class {cls} has {len(idx)} samples, {idx[:10]}")
        #if not len(idx): continue
        #assert max(idx) < len(g_p_list[0][0]), f"idx {max(idx)} is out of range {len(g_p_list[0][0])}"
        g_p_list_cls = [g_p_list[i] for i in idx]
        #print(g_p_list_cls[-1])

        gp_list_cls_flipped = [[None for x in range(len(power))]
                      for y in range(num_layer)]
        for layer_idx in range(num_layer):
            for power_idx in range(len(power)):
                #print(len(g_p_list_cls[:][layer_idx][power_idx]))
                sublist = [g_p_list_cls[i][layer_idx][power_idx] for i in range(len(g_p_list_cls))]
                
                assert len(sublist) == len(idx), f"sublist {len(sublist)} != idx {len(idx)}"
                gp_list_cls_flipped[layer_idx][power_idx] = torch.stack(sublist).cuda()
                #assert torch.isnan(gp_list_cls_flipped[layer_idx][power_idx]).sum() == 0, f"gp_list_cls_flipped[layer_idx][power_idx] has NaNs"
        min_cls, max_cls = get_min_max_per_cls(gp_list_cls_flipped, power, num_layer)
        mins_per_cls[cls] = min_cls
        maxs_per_cls[cls] = max_cls

        #assert torch.isnan(torch.Tensor(mins_per_cls[cls])).sum() == 0, f"mins_per_cls[cls] has NaNs"
        #assert torch.isnan(torch.Tensor(maxs_per_cls[cls])).sum() == 0, f"maxs_per_cls[cls] has NaNs"
        torch.cuda.empty_cache()

    
        
    return mins_per_cls,maxs_per_cls


def get_deviations_per_cls(feat_list, mins, maxs, powers):
    batch_deviations = []
    #print(f"{len(mins)=}, {len(maxs)=}")
    for L,feat_L in enumerate(feat_list):
        feat_L = torch.stack(feat_L).cuda()
        #print(feat_L.shape)
        dev = 0
        for p,P in enumerate(powers):
            g_p = G_p(feat_L,P)
            
            dev +=  (F.relu(mins[L][p]-g_p)/torch.abs(mins[L][p]+10**-6)).sum(dim=1,keepdim=True)
            dev +=  (F.relu(g_p-maxs[L][p])/torch.abs(maxs[L][p]+10**-6)).sum(dim=1,keepdim=True)
        batch_deviations.append(dev.cpu().detach().numpy())
    batch_deviations = np.concatenate(batch_deviations,axis=1)
    return batch_deviations

def get_deviations(model, data, mins_per_cls, maxs_per_cls, powers, num_classes, divide_by_conf=True, normalize_devs=None):
                
    batch = data
    logits, feat_list = model(data, return_feature_list=True)
    num_layers = len(feat_list)
    confs = np.max(F.softmax(logits,dim=1).cpu().detach().numpy(),axis=1)
    preds_cls = torch.argmax(logits, axis=1).cpu().detach().numpy().flatten()

    deviations = np.empty((len(preds_cls),num_layers))

    feat_list = [list(i) for i in zip(*feat_list)]
    assert len(feat_list) == len(preds_cls), f"feat_list {len(feat_list)} != preds_cls {len(preds_cls)}"
    
    for cls in np.unique(preds_cls):
        idx = np.argwhere(preds_cls==cls).flatten()
        assert len(idx) > 0, f"Class {cls} has no samples"

        feat_list_cls = [feat_list[i] for i in idx]
        feat_list_cls = [list(i) for i in zip(*feat_list_cls)]
        #print(len(feat_list_cls), len(feat_list_cls[0]))
        mins = mins_per_cls[cls]
        maxs = maxs_per_cls[cls]
        batch_deviations_cls = get_deviations_per_cls(feat_list_cls, mins, maxs, powers)
        #assert np.isnan(batch_deviations_cls).sum() == 0, f"batch_deviations_cls has {np.isnan(batch_deviations_cls).sum()} NaNs"
        #print(batch_deviations_cls.shape)
        deviations[idx,:] = batch_deviations_cls

    #assert np.isnan(deviations).sum() == 0, f"deviations has {np.isnan(deviations).sum()} NaNs"
    #assert np.isinf(deviations).sum() == 0, f"deviations has {np.isinf(deviations).sum()} Infs"

    if divide_by_conf:
        # make sure there's no division by 0
        assert (confs != 0).all()
        
        #assert np.isnan(confs).sum() == 0, f"confs has {np.isnan(confs).sum()} NaNs"
        #assert np.isinf(confs).sum() == 0, f"confs has {np.isinf(confs).sum()} Infs"

        deviations = deviations / confs[:,np.newaxis]

    if normalize_devs is not None:
        deviations = deviations / normalize_devs

    #deviations = -deviations.sum(axis=1)

    #assert np.isnan(deviations).sum() == 0, f"deviations has {np.isnan(deviations).sum()} NaNs"
    
        
    return torch.tensor(preds_cls).cuda(),torch.tensor(deviations).cuda()