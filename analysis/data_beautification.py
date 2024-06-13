nice_method_names = {
    "knn-train":          "KNN",
    "klm-train":          "KLM",
    "ash":          "ASH-b",
    "ash_s":          "ASH-s",
    "react_mod-train":        "ReAct",
    "odin_mod":         "ODIN-mod",
    "odin_mod2":         "ODIN-mod2",
    "odin_notemp":         "ODIN-notemp",
    "odin_notemp2":         "ODIN-notemp2",
    "odin_nopert":         "ODIN-nopert",
    "msp":          "MSP",
    "mls":          "MLS",
    "mds-val":          "MDS-val",
    "mds-train":          "MDS",
    "mds_ensemble_mod2-train": "MDSEns-mod2",
    "mds_ensemble_mod2-val": "MDSEns-mod2-val",
    "rankfeat":     "RankFeat",
    "dice_mod-train":         "DICE-mod",
    "gram_mod-train":         "GRAM-mod",
    "gram_mod-val":         "GRAM-mod-val",
    "gram_mod2-val":         "GRAM-mod2-val",
    "gram_mod2-train":         "GRAM-mod2-train",
    "vim":          "VIM",
    "rmds-val":         "RMDS-val",
    "rmds-train":         "RMDS",
    "she-val":          "SHE-val",
    "she-train":          "SHE",
    "temp_scaling-val": "TempScale-val",
    "temp_scaling-train": "TempScale",
    "gradnorm":     "GradNorm",
    "openmax-val":      "OpenMax-val",
    "openmax-train":      "OpenMax",
    "gen":          "GEN",
    "ebo":          "EBO",
}

nice_dataset_groups = {
    "cifar10": "CIFAR-10",
    "cifar100-coarse": "CIFAR-100-Coarse",
    "cifar100-fine": "CIFAR-100-Fine",
    "clothing1M": "Clothing1M",
    "fgvc-aircraft": "FGVC-Aircraft",
    "fgvc-cub": "CUB"
}

nice_checkpoints = {
    "best.ckpt": "early",
    "last*.ckpt": "last"
}

nice_archs = {
    "mlpmixer": "MLPMixer",
    "cct": "CCT",
    "resnet": "ResNet18",
    "resnet50": "ResNet50"
}

CHECKPOINTS = {
    "fgvc-aircraft":
        [
            "fgvc-aircraft_clean",
            "fgvc-aircraft_symm0.1",
            "fgvc-aircraft_symm0.2",
        ],
    "fgvc-cub":
        [
            "fgvc-cub_clean",
            "fgvc-cub_symm0.1",
            "fgvc-cub_symm0.2",
        ],
    "cifar10":
        [
            "cifar10",
            "cifar10_noisy_agg",
            "cifar10_noisy_random1",
            "cifar10_noisy_worse"
        ],
    "cifar10_symm": [
        "cifar10",
        "cifar10_symm_agg",
        "cifar10_symm_random1",
        "cifar10_symm_worse",
    ],
    "cifar10_asymm": [
        "cifar10",
        "cifar10_asymm_agg",
        "cifar10_asymm_random1",
        "cifar10_asymm_worse",
    ],
    "cifar100_fine": [
        "cifar100",
        "cifar100_noisy_fine",
    ],
    "cifar100_coarse": [
        "cifar100_clean_coarse",
        "cifar100_noisy_coarse",
    ],
    "cifar100_symm-fine": [
        "cifar100",
        "cifar100_symm_fine",
    ],
    "cifar100_asymm-fine": [
        "cifar100",
        "cifar100_asymm_fine",
    ],
    "cifar100_symm-coarse": [
        "cifar100_clean_coarse",
        "cifar100_symm_coarse"
    ],
    "cifar100_asymm-coarse": [
        "cifar100_clean_coarse",
        "cifar100_asymm_coarse"
    ],
    "clothing1M_symm": [
        "clothing1M_clean",
        "clothing1M_cleanval_symm",
    ],
    "clothing1M_asymm": [
        "clothing1M_clean",
        "clothing1M_cleanval_asymm",
    ],
    "clothing1M": [
        "clothing1M_clean",
        "clothing1M_cleanval",
    ],
}

NETWORKS = {
    "resnet50": {
        "fgvc-aircraft": "resnet50",
        "fgvc-cub": "resnet50"
    },
    "resnet": {
        "cifar10": "resnet18_32x32",
        "cifar100": "resnet18_32x32",
        "clothing1M": "resnet18_224x224",
    },
    "cct": {
        "cifar10": "cct_7_3x1_32",
        "cifar100": "cct_7_3x1_32",
        "clothing1M": "cct_7_7x2_224",
    },
    "mlpmixer": {
        "cifar10": "mlpmixer",
        "cifar100": "mlpmixer",
        "clothing1M": "mlpmixer",
    },
}

EPOCHS = {
    "resnet50": {
        "fgvc-aircraft": 600,
        "fgvc-cub": 600
    },
    "resnet": {
        "cifar10": 100,
        "cifar100": 100,
        "clothing1M": 100,
    },
    "cct": {
        "cifar10": 300,
        "cifar100": 300,
        "clothing1M": 300,
    },
    "mlpmixer": {
        "cifar10": 500,
        "cifar100": 500,
        "clothing1M": 500,
    },
}

ARCHS = {
    "resnet": {
        "cifar10": "resnet18_32x32",
        "cifar100": "resnet18_32x32",
        "clothing1M": "resnet18_224x224",
    },
    "cct": {
        "cifar10": "cct_7_3x1_32",
        "cifar100": "cct_7_3x1_32",
        "clothing1M": "cct_7_7x2_224",
    },
    "mlpmixer": {
        "cifar10": "mlpmixer",
        "cifar100": "mlpmixer",
        "clothing1M": "mlpmixer",
    },
}

TRAINERS = {
    "resnet50": {
        "fgvc-aircraft": "base",
        "fgvc-cub": "base"
    },
    "resnet": {
        "cifar10": "base",
        "cifar100": "base",
        "clothing1M": "base",
    },
    "cct": {
        "cifar10": "base_cct",
        "cifar100": "base_cct",
        "clothing1M": "base_cct",
    },
    "mlpmixer": {
        "cifar10": "base_mlpmixer",
        "cifar100": "base_mlpmixer",
        "clothing1M": "base_mlpmixer",
    },
}

LRS = {
    "resnet50": {
        "fgvc-aircraft": 0.01,
        "fgvc-cub": 0.01
    },
    "resnet": {
        "cifar10": 0.1,
        "cifar100": 0.1,
        "clothing1M": 0.1,
    },
    "cct": {
        "cifar10": 0.00055,
        "cifar100": 0.0006,
        "clothing1M": 0.0005,
    },
    "mlpmixer": {
        "cifar10": 0.001,
        "cifar100": 0.001,
        "clothing1M": 0.001,
    },
}

MARKS = {
    "resnet50": {
        "fgvc-aircraft": "default",
        "fgvc-cub": "default"
    },
    "resnet": {
        "cifar10": "default",
        "cifar100": "default",
        "clothing1M": "default",
    },
    "cct": {
        "cifar10": "default",
        "cifar100": "default",
        "clothing1M": "default",
    },
    "mlpmixer": {
        "cifar10": "default",
        "cifar100": "default",
        "clothing1M": "default",
    },
}



def noisy_or_not(noise_type):
    if noise_type == "clean": return "clean"
    else: return "noisy"

def group_noise(ds):
    if ("_clean" in ds or ds in ["cifar10","cifar100"]) and not "cleanval" in ds:
        return "clean"
    if "asymm" in ds:
        return "asymm"
    elif "symm" in ds:
        return "symm"
    return "real"

def rename_iid_ds(ds):
    for grp in ["cifar10","clothing1M"]:
        if grp in ds and "cifar100" not in ds: return grp

    if ds == "cifar100": return "cifar100-fine"
    if "coarse" in ds:
        return "cifar100-coarse"
    elif "fine" in ds:
        return "cifar100-fine"
    
    return ds

def get_clean_ds(ds):
    if ds == "cifar100" or "fine" in ds: return "cifar100"
    if "coarse" in ds: return "cifar100_clean_coarse"
    if "cifar10" in ds: return "cifar10"
    if "clothing" in ds: return "clothing1M_clean"
    if "fgvc-aircraft" in ds: return "fgvc-aircraft_clean"
    elif "fgvc-cub" in ds: return "fgvc-cub_clean"
    else: raise ValueError

def get_dataset_group(ds):
    for grp in ["cifar10","cifar100"]:
        if grp == ds: return grp
    for cifar10grp in ["agg","random1","worse"]:
       if cifar10grp in ds:
           return f"cifar10-{cifar10grp}"
    for cifar100grp in ["fine","coarse"]:
        if cifar100grp in ds:
           return f"cifar100-{cifar100grp}"
    if "clothing" in ds:
        return "clothing1M"
    if "fgvc-aircraft" in ds:
        return "fgvc-aircraft"
    elif "fgvc-cub" in ds:
        return "fgvc-cub"

    print(f"WTF is {ds}")
    raise ValueError