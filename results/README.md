This folder contains the checkpoints (+ logs & config files) of trained classifiers, as well as the .pkl files produced by the OOD detection evaluator, with the following directory structure:

```shell
.
├── cifar100_asymm_coarse_cct_7_3x1_32_base_cct_e300_lr0.0006_default
│   ├── s0
│   │   ├── scores-best.ckpt
│   │   │   ├── ash_s-APS.pkl
│   │   │   ├── ash_s.pkl
│   │   │   ├── ...
│   │   │   ├── ...
│   │   │   ├── vim-APS.pkl
│   │   │   └── vim.pkl
│   │   ├── scores-last*.ckpt
│   │   │   ├── ash_s-APS.pkl
│   │   │   ├── ash_s.pkl
│   │   │   ├── ...
│   │   │   ├── ...
│   │   │   ├── vim-APS.pkl
│   │   │   └── vim.pkl
│   │   ├── best.ckpt
│   │   ├── config.yml
│   │   ├── last_epoch300_acc0.5940.ckpt
│   │   └── log.txt
│   ├── s2
│   │   ├── scores-best.ckpt
│   │   ├── scores-last*.ckpt
│   │   ├── best.ckpt
│   │   ├── config.yml
│   │   ├── last_epoch300_acc0.6070.ckpt
│   │   └── log.txt
│   └── s20
├── cifar100_asymm_coarse_mlpmixer_base_mlpmixer_e500_lr0.001_default
│   ├── s0
│   ├── s2
│   └── s20
├── cifar100_asymm_coarse_resnet18_32x32_base_e100_lr0.1_default
└──...
```

`s0`, `s2` and `s20` correspond to 3 different random seeds used for training. The ``scores`` folders are produced by the OOD evaluation script (e.g. [run/cifar10_eval.sh](../run/cifar10_eval.sh)). "Best" and "last" correspond to the 2 types of checkpoints ("best" is saved based on best validation accuracy, "last" is saved when the max. number of epochs has been reached).

The evaluation script produces a .pkl file for each post-hoc OOD detector, following the OpenOOD evaluator, with the following structure:

```shell
{
    'id': {
        'train': None, # unused
        'val': None, # unused
        'test': [array([]), array([], dtype=float32), array([])] # list of 3 arrays: [class predictions (int), OOD scores, class labels (int)]
        },
    'csid': {}, # unused
    'ood': {
        'val': None, # unused
        'near': { # note: we do not make a distinction between near and far OOD in our evaluation
            'imagenet6_test': []  # list of 3 arrays, same as above (only the OOD scores are used)
            },
        'far': {
            'mnist': [],  # list of 3 arrays, same as above (only the OOD scores are used)
            'svhn': [],  # etc
            'texture': [],  # ...
            'food101': [],  # ...
            'stanfordproducts': [],  # ...
            'eurosat': []  # ...
            }
        }
    'id_preds': tensor([]), # class predictions on the ID test set (int) - identical to the first array in 'id'->'test' above
    'id_labels': tensor([]), # class labels from the ID test set (int) - identical to the third array in 'id'->'test' above
    'csid_preds': {}, # unused
    'csid_labels': {} # unused
} 
```