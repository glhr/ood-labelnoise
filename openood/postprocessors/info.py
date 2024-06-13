num_classes_dict = {
    'cifar10': 10,
    'cifar100': 100,
    'imagenet200': 200,
    'imagenet': 1000,
    "mnist": 10,
    "clothing1M": 14,
    "fgvc-aircraft": 50,
    "fgvc-cub": 100,
}

def get_num_classes(dataset):
    if "coarse" in dataset and "cifar100" in dataset:
        return 20
    return num_classes_dict[dataset.split("_")[0]]