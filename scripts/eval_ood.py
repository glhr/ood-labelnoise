import os, sys
ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.append(ROOT_DIR)
import numpy as np
import pandas as pd
import argparse
import pickle
import collections
from glob import glob
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F

from openood.evaluation_api import Evaluator

from openood.networks import ResNet18_32x32, ResNet18_224x224, ResNet50, ResNet18_64x64, MLPMixer
from openood.networks.conf_branch_net import ConfBranchNet
from openood.networks.godin_net import GodinNet
from openood.networks.rot_net import RotNet
from openood.networks.csi_net import CSINet
from openood.networks.udg_net import UDGNet
from openood.networks.cider_net import CIDERNet
from openood.networks.npos_net import NPOSNet
from openood.networks.mcd_net import MCDNet
from openood.networks.dropout_net import DropoutNet
from openood.networks.cct import cct_model_dict, pe_check, fc_check


def update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


parser = argparse.ArgumentParser()
parser.add_argument('--root', required=True)
parser.add_argument('--architecture', default='resnet')
parser.add_argument('--postprocessor', default='msp')
parser.add_argument(
    '--id-data',
    type=str,
    default='cifar10')
parser.add_argument('--batch-size', type=int, default=200)
parser.add_argument('--save-csv', action='store_true')
parser.add_argument('--save-score', action='store_true')
parser.add_argument('--fsood', action='store_true')
parser.add_argument('--overwrite', action='store_true', default=False)
parser.add_argument('--redo_setup', action='store_true', default=False)
parser.add_argument('--skip_setup', action='store_true', default=False)
parser.add_argument('--num_workers',  default=8, type=int)
parser.add_argument('--id_loader_split',  default="train", type=str)
parser.add_argument('--checkpoint',  default="best.ckpt", type=str)
args = parser.parse_args()

root = args.root



# specify an implemented postprocessor
# 'openmax', 'msp', 'temp_scaling', 'odin'...
postprocessor_name = args.postprocessor

NUM_CLASSES = {'cifar10': 10, 'cifar100': 100, 'imagenet200': 200,
               'clothing1M': 14,
               'fgvc-aircraft': 50,
               'fgvc-cub': 100}



from openood.postprocessors.info import get_num_classes
try:
    num_classes = get_num_classes(args.id_data)
    
except KeyError:
    raise NotImplementedError(f'ID dataset {args.id_data} is not supported.')

# assume that the root folder contains subfolders each corresponding to
# a training run, e.g., s0, s1, s2
# this structure is automatically created if you use OpenOOD for train
if len(glob(os.path.join(root, 's*'))) == 0:
    raise ValueError(f'No subfolders found in {root}')

# iterate through training runs
all_metrics = []
for subfolder in sorted(glob(os.path.join(root, 's*'))):
    print(f'--> Processing {postprocessor_name} - {subfolder}...')

    # read yaml config file
    with open(os.path.join(subfolder, 'config.yml'), 'r') as stream:
        network_config = yaml.load(stream, Loader=yaml.Loader)
        try:
            input_size = network_config.dataset.image_size

            if args.architecture == 'resnet':
                MODEL = {
                    'cifar10': ResNet18_32x32,
                    'cifar100': ResNet18_32x32,
                    'imagenet200': ResNet18_224x224,
                    'clothing1M': ResNet18_224x224,
                    'fgvc-aircraft': ResNet50,
                    'fgvc-cub': ResNet50,
                }
                model_arch = MODEL[args.id_data.split('_')[0]]
            elif args.architecture == 'resnet50':
                model_arch = ResNet50
            elif args.architecture == 'mlpmixer':
                model_arch = MLPMixer
            elif args.architecture == 'cct':
                model_arch = cct_model_dict[network_config.network.model]
                if "sine" in network_config.network.model:
                    positional_embedding = 'sine'
                else:
                    positional_embedding = 'learnable'
            else:
                raise NotImplementedError(f"Architecture {args.architecture} not supported")
            
            print(f"Using model {model_arch} for dataset {args.id_data}")
            
        except Exception as e:
            print(e)
            pass

    # load pre-setup postprocessor if exists
    # if os.path.isfile(
    #         os.path.join(subfolder, 'postprocessors',
    #                      f'{postprocessor_name}.pkl')):
    #     with open(
    #             os.path.join(subfolder, 'postprocessors',
    #                          f'{postprocessor_name}.pkl'), 'rb') as f:
    #         postprocessor = pickle.load(f)
    # else:
    postprocessor = None

    assert len(glob(os.path.join(subfolder, f"last*.ckpt")))
    if "*" in args.checkpoint:
        checkpoint_paths = glob(os.path.join(subfolder, args.checkpoint))
        if len(checkpoint_paths) == 0:
            raise ValueError(f"No checkpoints found in {subfolder} with pattern {args.checkpoint}")
        elif len(checkpoint_paths) > 1:
            raise ValueError(f"Multiple checkpoints found in {subfolder} with pattern {args.checkpoint}")
        else:
            checkpoint_path = checkpoint_paths[0]
    else:
        checkpoint_path = os.path.join(subfolder, args.checkpoint)


    # load the pretrained model provided by the user
    if postprocessor_name == 'conf_branch':
        net = ConfBranchNet(backbone=model_arch(num_classes=num_classes),
                            num_classes=num_classes)
    elif postprocessor_name == 'godin':
        backbone = model_arch(num_classes=num_classes)
        net = GodinNet(backbone=backbone,
                       feature_size=backbone.feature_size,
                       num_classes=num_classes)
    elif 'rotpred' in postprocessor_name:
        net = RotNet(backbone=model_arch(num_classes=num_classes),
                     num_classes=num_classes)
    elif 'csi' in root:
        backbone = model_arch(num_classes=num_classes)
        net = CSINet(backbone=backbone,
                     feature_size=backbone.feature_size,
                     num_classes=num_classes)
    elif 'udg' in root:
        backbone = model_arch(num_classes=num_classes)
        net = UDGNet(backbone=backbone,
                     num_classes=num_classes,
                     num_clusters=1000)
    elif postprocessor_name == 'cider':
        backbone = model_arch(num_classes=num_classes)
        net = CIDERNet(backbone,
                       head='mlp',
                       feat_dim=128,
                       num_classes=num_classes)
    elif postprocessor_name == 'npos':
        backbone = model_arch(num_classes=num_classes)
        net = NPOSNet(backbone,
                      head='mlp',
                      feat_dim=128,
                      num_classes=num_classes)
    elif postprocessor_name == 'mcd':
        backbone = model_arch(num_classes=num_classes)
        backbone.fc = nn.Identity()
        net = MCDNet(backbone,
                      num_classes=num_classes)
    elif postprocessor_name == 'dropout':
        backbone = model_arch(num_classes=num_classes)
        net = DropoutNet(backbone, dropout_p=network_config.network.dropout_p)
    else:
        if args.architecture == 'cct':
            net = model_arch(num_classes=num_classes, image_size=input_size)

            print(f"Loading CCT model from {checkpoint_path}")

            state_dict = torch.load(checkpoint_path, map_location='cpu')
            if positional_embedding == 'learnable':
                state_dict = pe_check(net, state_dict)
            elif positional_embedding == 'sine':
                state_dict['classifier.positional_emb'] = net.state_dict()['classifier.positional_emb']
            state_dict = fc_check(net, state_dict)
            net.load_state_dict(state_dict, strict=True)
        elif args.architecture == 'mlpmixer':
            cfg = network_config.__getstate__()
            net = model_arch(**cfg["network"])
            net.load_state_dict(
                torch.load(checkpoint_path, map_location='cpu'))
        else:
            net = model_arch(num_classes=num_classes)

            net.load_state_dict(
            torch.load(checkpoint_path, map_location='cpu'))

    
    net.cuda()
    net.eval()

    evaluator = Evaluator(
        net,
        id_name=args.id_data,  # the target ID dataset
        #data_root=os.path.join(ROOT_DIR, 'data'),
        config_root=os.path.join(ROOT_DIR, 'configs'),
        preprocessor=None,  # default preprocessing
        postprocessor_name=postprocessor_name,
        postprocessor=
        postprocessor,  # the user can pass his own postprocessor as well
        batch_size=args.
        batch_size,  # for certain methods the results can be slightly affected by batch size
        shuffle=False,
        num_workers=args.num_workers,
        id_loader_split=args.id_loader_split)

    scores_folder = f"scores-{args.checkpoint}"

    score_save_root = os.path.join(subfolder, scores_folder)
    if not os.path.exists(score_save_root):
        os.makedirs(score_save_root)

    if hasattr(evaluator.postprocessor,"has_data_based_setup") and evaluator.postprocessor.has_data_based_setup:
        scores_path = os.path.join(subfolder, scores_folder, f'{postprocessor_name}-{args.id_loader_split}.pkl')
    else:
        scores_path = os.path.join(subfolder, scores_folder, f'{postprocessor_name}.pkl')
    print(f"Setting scores save path to {scores_path}")

    aps_path = scores_path.replace(".pkl", "-APS.pkl")

    # if all files already exist, skip
    if (not args.redo_setup) and (not args.overwrite) and os.path.isfile(scores_path):
        if args.skip_setup:
            print(f".....Skipping {subfolder} because scores already exist at {scores_path}")
            continue
        elif not os.path.isfile(aps_path):
            pass
        

    if (not args.redo_setup) and (os.path.isfile(aps_path)) and (evaluator.postprocessor.APS_mode):
        #print(f".....Skipping {subfolder} because scores already exist at {scores_path}")
        print(f"Found existing APS results at {aps_path}")
        with open(aps_path,'rb') as f:
            aps_dict = pickle.load(f)

        if not isinstance(aps_dict["APS_results"], list):
            aps_dict["APS_results"] = [aps_dict["APS_results"]]
        try:
                evaluator.postprocessor.set_hyperparam(aps_dict["APS_results"])
                evaluator.postprocessor.hyperparam_search_done = True
        except Exception as e:
                print(e)
        #evaluator.postprocessor.setup_flag = True
        #continue

    # only skip the setup if scores_path exists and no APS required
    if (args.redo_setup) or (not os.path.isfile(scores_path)) or (evaluator.postprocessor.APS_mode):
        evaluator.prepare_everything()
#    else:
#        evaluator.prepare_everything()

    APS_params = {
        'APS_results': evaluator.APS_results,
    }

    # save the APS results for future reuse
    if not os.path.isfile(aps_path):
        with open(aps_path,'wb') as f:
            pickle.dump(APS_params, f, pickle.HIGHEST_PROTOCOL)

    # load pre-computed scores if exist
    if not args.overwrite and os.path.isfile(scores_path):
        with open(scores_path,'rb') as f:
            scores = pickle.load(f)
        update(evaluator.scores, scores)
        print('Loaded pre-computed scores from file.')

    # # save the postprocessor for future reuse
    # if hasattr(evaluator.postprocessor, 'setup_flag'
    #            ) or evaluator.postprocessor.hyperparam_search_done is True:
    #     pp_save_root = os.path.join(subfolder, 'postprocessors')
    #     if not os.path.exists(pp_save_root):
    #         os.makedirs(pp_save_root)

    #     if not os.path.isfile(
    #             os.path.join(pp_save_root, f'{postprocessor_name}.pkl')):
    #         with open(os.path.join(pp_save_root, f'{postprocessor_name}.pkl'),
    #                   'wb') as f:
    #             pickle.dump(evaluator.postprocessor, f,
    #                         pickle.HIGHEST_PROTOCOL)

    metrics = evaluator.eval_ood(fsood=args.fsood)
    all_metrics.append(metrics.to_numpy())

    #metrics['params'] = evaluator.APS_results

    
    # save computed scores
    if args.save_score:
        with open(scores_path,
                  'wb') as f:
            pickle.dump(evaluator.scores, f, pickle.HIGHEST_PROTOCOL)

if len(all_metrics):
    # compute mean metrics over training runs
    all_metrics = np.stack(all_metrics, axis=0)
    metrics_mean = np.mean(all_metrics, axis=0)
    metrics_std = np.std(all_metrics, axis=0)

    final_metrics = []
    for i in range(len(metrics_mean)):
        temp = []
        for j in range(metrics_mean.shape[1]):
            temp.append(u'{:.2f} \u00B1 {:.2f}'.format(metrics_mean[i, j],
                                                    metrics_std[i, j]))
        final_metrics.append(temp)
    df = pd.DataFrame(final_metrics, index=metrics.index, columns=metrics.columns)

    if args.save_csv:
        saving_root = os.path.join(root, 'ood' if not args.fsood else 'fsood')
        if not os.path.exists(saving_root):
            os.makedirs(saving_root)
        df.to_csv(os.path.join(saving_root, f'{postprocessor_name}.csv'))
    else:
        print(df)
