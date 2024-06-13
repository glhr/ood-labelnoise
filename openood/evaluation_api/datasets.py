import os
#import gdown
import zipfile

from torch.utils.data import DataLoader
import torchvision as tvs
if tvs.__version__ >= '0.13':
    tvs_new = True
else:
    tvs_new = False

from openood.datasets.imglist_dataset import ImglistDataset
from openood.preprocessors import BasePreprocessor
from openood.datasets.utils import get_feat_dataset

from .preprocessor import get_default_preprocessor, ImageNetCPreProcessor

ood_dicts = {
    'food101': {
                    'data_dir': 'images_classic/',
                    'imglist_path':
                    'benchmark_imglist/food101/food101_test.txt'
                },
    'stanfordproducts': {
        'data_dir': 'images_classic/',
        'imglist_path':
        'benchmark_imglist/stanfordproducts/stanfordproducts_test_ood.txt'
    },
    'mnist': {
                    'data_dir': 'images_classic/',
                    'imglist_path': 'benchmark_imglist/cifar100/test_mnist.txt'
                },
    'svhn': {
        'data_dir': 'images_classic/',
        'imglist_path': 'benchmark_imglist/cifar100/test_svhn.txt'
    },
    'texture': {
        'data_dir': 'images_classic/',
        'imglist_path':
        'benchmark_imglist/cifar100/test_texture.txt'
    },
    'places365': {
        'data_dir': 'images_classic/',
        'imglist_path':
        'benchmark_imglist/cifar100/test_places365.txt'
    },
     'eurosat': {
        'data_dir': 'images_classic/',
        'imglist_path':
        'benchmark_imglist/eurosat/eurosat_test_ood.txt'
    },
    'tin6_test': {
                    'data_dir': 'images_classic/',
                    'imglist_path': 'benchmark_imglist/tin/tin6_ood_test.txt'
                },
    'tin6_val': {
                    'data_dir': 'images_classic/',
                    'imglist_path': 'benchmark_imglist/tin/tin6_ood_val.txt'
                },
    'imagenet6_val': {
        'data_dir': 'images_classic/',
                    'imglist_path': 'benchmark_imglist/imagenet/imagenet6_ood_val.txt'
    },
    'imagenet6_test': {
        'data_dir': 'images_classic/',
                    'imglist_path': 'benchmark_imglist/imagenet/imagenet6_ood_test.txt'
    },
    'fgvc-aircraft_OSR_easy': {
        'data_dir': 'images_classic/',
                    'imglist_path': 'benchmark_imglist/fgvc-aircraft/ood_fgvc-variant_easy.txt'
    },
    'fgvc-aircraft_OSR_medium': {
        'data_dir': 'images_classic/',
                    'imglist_path': 'benchmark_imglist/fgvc-aircraft/ood_fgvc-variant_medium.txt'
    },
    'fgvc-aircraft_OSR_hard': {
        'data_dir': 'images_classic/',
                    'imglist_path': 'benchmark_imglist/fgvc-aircraft/ood_fgvc-variant_hard.txt'
    },
    'fgvc-cub_OSR_easy': {
        'data_dir': 'images_classic/',
                    'imglist_path': 'benchmark_imglist/fgvc-cub/ood_fgvc-cub_easy.txt'
    },
    'fgvc-cub_OSR_medium': {
        'data_dir': 'images_classic/',
                    'imglist_path': 'benchmark_imglist/fgvc-cub/ood_fgvc-cub_medium.txt'
    },
    'fgvc-cub_OSR_hard': {
        'data_dir': 'images_classic/',
                    'imglist_path': 'benchmark_imglist/fgvc-cub/ood_fgvc-cub_hard.txt'
    }
}

def get_ood_dict():
    return {
            'val': ood_dicts['imagenet6_val'],
            'near': {
                'imagenet6_test': ood_dicts['imagenet6_test']
            },
            'far': {
                #'datasets': ['mnist', 'svhn', 'texture', 'food101', 'stanfordproducts', 'eurosat'],
                'mnist': ood_dicts['mnist'],
                'svhn': ood_dicts['svhn'],
                'texture': ood_dicts['texture'],
                'food101': ood_dicts['food101'],
                'stanfordproducts': ood_dicts['stanfordproducts'],
                'eurosat': ood_dicts['eurosat'],
            },
        }

def get_dict_cifar100(train_txt="train_cifar100.txt", val_txt="val_cifar100.txt", test_txt="test_cifar100.txt", num_classes=100):
    return {
        'num_classes': num_classes,
        'id': {
            'train': {
                'data_dir': 'images_classic/',
                'imglist_path': f'benchmark_imglist/cifar100/{train_txt}'
            },
            'val': {
                'data_dir': 'images_classic/',
                'imglist_path': f'benchmark_imglist/cifar100/{val_txt}'
            },
            'test': {
                'data_dir': 'images_classic/',
                'imglist_path': f'benchmark_imglist/cifar100/{test_txt}'
            }
        },
        'csid': {
            #'datasets': [],
        },
        'ood': get_ood_dict()
    }

def get_dict_cifar(train_txt, val_txt="val_cifar10.txt", test_txt="test_cifar10.txt"):
    return {
        'num_classes': 10,
        'id': {
            'train': {
                'data_dir': 'images_classic/',
                'imglist_path': f'benchmark_imglist/cifar10/{train_txt}'
            },
            'val': {
                'data_dir': 'images_classic/',
                'imglist_path': f'benchmark_imglist/cifar10/{val_txt}'
            },
            'test': {
                'data_dir': 'images_classic/',
                'imglist_path': f'benchmark_imglist/cifar10/{test_txt}'
            }
        },
        'csid': {
            #'datasets': [],
        },
        'ood': get_ood_dict()
    }

def get_dict_clothing1M(train_txt, val_txt="val_clothing1M_clean.txt", test_txt="test_clothing1M_clean.txt"):
    return {
        'num_classes': 14,
        'id': {
            'train': {
                'data_dir': 'images_classic/',
                'imglist_path': f'benchmark_imglist/clothing1M/{train_txt}'
            },
            'val': {
                'data_dir': 'images_classic/',
                'imglist_path': f'benchmark_imglist/clothing1M/{val_txt}'
            },
            'test': {
                'data_dir': 'images_classic/',
                'imglist_path': f'benchmark_imglist/clothing1M/{test_txt}'
            }
        },
        'csid': {
            #'datasets': [],
        },
        'ood': get_ood_dict()
        }

def get_dict_fgvc_aircraft(train_txt="train_fgvc-variant_clean.txt", val_txt="val_fgvc-variant_clean.txt", test_txt="test_fgvc-variant_clean.txt"):
    return {
        'num_classes': 50,
        'id': {
            'train': {
                'data_dir': 'images_classic/',
                'imglist_path': f'benchmark_imglist/fgvc-aircraft/{train_txt}'
            },
            'val': {
                'data_dir': 'images_classic/',
                'imglist_path': f'benchmark_imglist/fgvc-aircraft/{val_txt}'
            },
            'test': {
                'data_dir': 'images_classic/',
                'imglist_path': f'benchmark_imglist/fgvc-aircraft/{test_txt}'
            }
        },
        'csid': {
            #'datasets': [],
        },
        'ood': {
            'val': ood_dicts['fgvc-aircraft_OSR_medium'],
            'near': {
                'fgvc-aircraft_OSR_hard': ood_dicts['fgvc-aircraft_OSR_hard'],
                'fgvc-aircraft_OSR_medium': ood_dicts['fgvc-aircraft_OSR_medium'],
            },
            'far': {
                'fgvc-aircraft_OSR_easy': ood_dicts['fgvc-aircraft_OSR_easy']
            },
        }
    }

def get_dict_fgvc_cub(train_txt="train_fgvc-cub_clean.txt", val_txt="val_fgvc-cub_clean.txt", test_txt="test_fgvc-cub_clean.txt"):
    return {
        'num_classes': 100,
        'id': {
            'train': {
                'data_dir': 'images_classic/',
                'imglist_path': f'benchmark_imglist/fgvc-cub/{train_txt}'
            },
            'val': {
                'data_dir': 'images_classic/',
                'imglist_path': f'benchmark_imglist/fgvc-cub/{val_txt}'
            },
            'test': {
                'data_dir': 'images_classic/',
                'imglist_path': f'benchmark_imglist/fgvc-cub/{test_txt}'
            }
        },
        'csid': {
            #'datasets': [],
        },
        'ood': {
            'val': ood_dicts['fgvc-cub_OSR_medium'],
            'near': {
                'fgvc-cub_OSR_hard': ood_dicts['fgvc-cub_OSR_hard'],
                'fgvc-cub_OSR_medium': ood_dicts['fgvc-cub_OSR_medium'],
            },
            'far': {
                'fgvc-cub_OSR_easy': ood_dicts['fgvc-cub_OSR_easy']
            },
        }
    }

DATA_INFO = {
    'fgvc-cub_clean': get_dict_fgvc_cub("train_fgvc-cub_clean.txt"),
    'fgvc-cub_symm0.1': get_dict_fgvc_cub("train_fgvc-cub_symm0.1.txt"),
    'fgvc-cub_symm0.2': get_dict_fgvc_cub("train_fgvc-cub_symm0.2.txt"),
    'fgvc-aircraft_clean': get_dict_fgvc_aircraft("train_fgvc-variant_clean.txt"),
    'fgvc-aircraft_symm0.1': get_dict_fgvc_aircraft("train_fgvc-variant_symm0.1.txt"),
    'fgvc-aircraft_symm0.2': get_dict_fgvc_aircraft("train_fgvc-variant_symm0.2.txt"),
    'clothing1M_clean': get_dict_clothing1M("train_clothing1M_clean.txt"),
    'clothing1M_cleanval': get_dict_clothing1M("train_clothing1M_noisy.txt"),
    'clothing1M_cleanval_symm': get_dict_clothing1M("train_clothing1M_symm.txt"),
    'clothing1M_cleanval_asymm': get_dict_clothing1M("train_clothing1M_asymm.txt"),
    'cifar10': get_dict_cifar("train_cifar10.txt"),
    'cifar10_noisy_worse': get_dict_cifar("train_cifar10n_worse.txt"),
    'cifar10_noisy_agg': get_dict_cifar("train_cifar10n_agg.txt"),
    'cifar10_noisy_random1': get_dict_cifar("train_cifar10n_random1.txt"),
    'cifar10_noisy_random2': get_dict_cifar("train_cifar10n_random2.txt"),
    'cifar10_noisy_random3': get_dict_cifar("train_cifar10n_random3.txt"),
    'cifar10_symm_agg': get_dict_cifar("train_cifar10symm_agg.txt"),
    'cifar10_symm_random1': get_dict_cifar("train_cifar10symm_random1.txt"),
    'cifar10_symm_worse': get_dict_cifar("train_cifar10symm_worse.txt"),
    'cifar10_asymm_agg': get_dict_cifar("train_cifar10asymm_agg.txt"),
    'cifar10_asymm_random1': get_dict_cifar("train_cifar10asymm_random1.txt"),
    'cifar10_asymm_worse': get_dict_cifar("train_cifar10asymm_worse.txt"),
    'cifar100': get_dict_cifar100(),
    'cifar100_noisy_fine': get_dict_cifar100(train_txt="train_cifar100n_noisyfine.txt"),
    'cifar100_clean_coarse': get_dict_cifar100(num_classes=20, train_txt="train_cifar100n_cleancoarse.txt", val_txt=f"val_cifar100n_cleancoarse.txt", test_txt="test_cifar100n_cleancoarse.txt"),
    'cifar100_noisy_coarse': get_dict_cifar100(num_classes=20, train_txt="train_cifar100n_noisycoarse.txt", val_txt=f"val_cifar100n_cleancoarse.txt", test_txt="test_cifar100n_cleancoarse.txt"),
    'cifar100_symm_fine': get_dict_cifar100(train_txt="train_cifar100symm_noisyfine.txt"),
    'cifar100_symm_coarse': get_dict_cifar100(num_classes=20, train_txt="train_cifar100symm_noisycoarse.txt", val_txt=f"val_cifar100n_cleancoarse.txt", test_txt="test_cifar100n_cleancoarse.txt"),
    'cifar100_asymm_fine': get_dict_cifar100(train_txt="train_cifar100asymm_noisyfine.txt"),
    'cifar100_asymm_coarse': get_dict_cifar100(num_classes=20, train_txt="train_cifar100asymm_noisycoarse.txt", val_txt=f"val_cifar100n_cleancoarse.txt", test_txt="test_cifar100n_cleancoarse.txt")
}


def get_feat_dataloader_test(id_name, data_root, **loader_kwargs):
    dataloader_dict = {}
    data_info = DATA_INFO[id_name]
    # id
    sub_dataloader_dict = {}
    for split in data_info['id'].keys():
        feat_path = data_info['id'][split]['feat_path']
        dataset = get_feat_dataset(feat_path)
        dataloader = DataLoader(dataset, **loader_kwargs)
        sub_dataloader_dict[split] = dataloader
    dataloader_dict['id'] = sub_dataloader_dict

    dataloader_dict['csid'] = {}

    # ood
    sub_dataloader_dict = {}
    dataloader_dict['ood'] = {}
    for split in data_info['ood'].keys():
        feat_path = data_info['ood'][split]['feat_path']
        dataset = get_feat_dataset(feat_path)
        dataloader = DataLoader(dataset, **loader_kwargs)
        if split == 'val':
            sub_dataloader_dict[split] = dataloader
        else:
            sub_dataloader_dict[split] = dict()
            sub_dataloader_dict[split]['ood'] = dataloader
    
    dataloader_dict['ood'] = sub_dataloader_dict
    
    print(dataloader_dict)
    return dataloader_dict



def get_id_ood_dataloader(id_name, data_root, preprocessor, **loader_kwargs):
    if 'imagenet' in id_name:
        if tvs_new:
            if isinstance(preprocessor,
                          tvs.transforms._presets.ImageClassification):
                mean, std = preprocessor.mean, preprocessor.std
            elif isinstance(preprocessor, tvs.transforms.Compose):
                temp = preprocessor.transforms[-1]
                mean, std = temp.mean, temp.std
            elif isinstance(preprocessor, BasePreprocessor):
                temp = preprocessor.transform.transforms[-1]
                mean, std = temp.mean, temp.std
            else:
                raise TypeError
        else:
            if isinstance(preprocessor, tvs.transforms.Compose):
                temp = preprocessor.transforms[-1]
                mean, std = temp.mean, temp.std
            elif isinstance(preprocessor, BasePreprocessor):
                temp = preprocessor.transform.transforms[-1]
                mean, std = temp.mean, temp.std
            else:
                raise TypeError
        imagenet_c_preprocessor = ImageNetCPreProcessor(mean, std)

    # weak augmentation for data_aux
    test_standard_preprocessor = get_default_preprocessor(id_name)

    dataloader_dict = {}
    data_info = DATA_INFO[id_name]

    # id
    sub_dataloader_dict = {}
    for split in data_info['id'].keys():
        dataset = ImglistDataset(
            name='_'.join((id_name, split)),
            imglist_pth=os.path.join(data_root,
                                     data_info['id'][split]['imglist_path']),
            data_dir=os.path.join(data_root,
                                  data_info['id'][split]['data_dir']),
            num_classes=data_info['num_classes'],
            preprocessor=preprocessor,
            data_aux_preprocessor=test_standard_preprocessor)
        dataloader = DataLoader(dataset, **loader_kwargs)
        sub_dataloader_dict[split] = dataloader
    dataloader_dict['id'] = sub_dataloader_dict

    # csid
    sub_dataloader_dict = {}
    for dataset_name in data_info['csid'].keys():
        dataset = ImglistDataset(
            name='_'.join((id_name, 'csid', dataset_name)),
            imglist_pth=os.path.join(
                data_root, data_info['csid'][dataset_name]['imglist_path']),
            data_dir=os.path.join(data_root,
                                  data_info['csid'][dataset_name]['data_dir']),
            num_classes=data_info['num_classes'],
            preprocessor=preprocessor
            if dataset_name != 'imagenet_c' else imagenet_c_preprocessor,
            data_aux_preprocessor=test_standard_preprocessor)
        dataloader = DataLoader(dataset, **loader_kwargs)
        sub_dataloader_dict[dataset_name] = dataloader
    dataloader_dict['csid'] = sub_dataloader_dict

    # ood
    dataloader_dict['ood'] = {}
    for split in data_info['ood'].keys():
        split_config = data_info['ood'][split]

        if split == 'val':
            # validation set
            dataset = ImglistDataset(
                name='_'.join((id_name, 'ood', split)),
                imglist_pth=os.path.join(data_root,
                                         split_config['imglist_path']),
                data_dir=os.path.join(data_root, split_config['data_dir']),
                num_classes=data_info['num_classes'],
                preprocessor=preprocessor,
                data_aux_preprocessor=test_standard_preprocessor)
            dataloader = DataLoader(dataset, **loader_kwargs)
            dataloader_dict['ood'][split] = dataloader
        else:
            # dataloaders for nearood, farood
            sub_dataloader_dict = {}
            for dataset_name in split_config.keys():
                dataset_config = split_config[dataset_name]
                dataset = ImglistDataset(
                    name='_'.join((id_name, 'ood', dataset_name)),
                    imglist_pth=os.path.join(data_root,
                                             dataset_config['imglist_path']),
                    data_dir=os.path.join(data_root,
                                          dataset_config['data_dir']),
                    num_classes=data_info['num_classes'],
                    preprocessor=preprocessor,
                    data_aux_preprocessor=test_standard_preprocessor)
                dataloader = DataLoader(dataset, **loader_kwargs)
                sub_dataloader_dict[dataset_name] = dataloader
            dataloader_dict['ood'][split] = sub_dataloader_dict

    return dataloader_dict
