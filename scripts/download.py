import argparse
import os
import zipfile

import gdown

benchmarks_dict = {
    'cifar-10': [
        'cifar10', 'cifar100', 'tin', 'mnist', 'svhn', 'texture',
    ],
}

dir_dict = {
    'images_classic/': [
        'cifar10', 'cifar100', 'tin', 'svhn', 'mnist', 'cifar10', 'texture', 
    ]
}

download_id_dict = {
    'cifar100': '1PGKheHUsf29leJPPGuXqzLBMwl8qMF8_',
    'cifar10': '1Co32RiiWe16lTaiOU6JMMnyUYS41IlO1',
    'svhn': '1DQfc11HOtB1nEwqS4pWUFp8vtQ3DczvI',
    'mnist': '1CCHAGWqA1KJTFFswuF9cbhmB-j98Y1Sb',
    'tin': '1PZ-ixyx52U989IKsMA2OT-24fToTrelC',
    'texture': '1OSz1m3hHfVWbRdmMwKbUzoU8Hg9UKcam',
}


def require_download(filename, path):
    for item in os.listdir(path):
        if item.startswith(filename) or filename.startswith(
                item) or path.endswith(filename):
            return False

    else:
        print(filename + ' needs download:')
        return True


def download_dataset(dataset, args):
    for key in dir_dict.keys():
        if dataset in dir_dict[key]:
            store_path = os.path.join(args.save_dir[0], key, dataset)
            if not os.path.exists(store_path):
                os.makedirs(store_path)
            break
    else:
        print('Invalid dataset detected {}'.format(dataset))
        return

    if require_download(dataset, store_path):
        print(store_path)
        if not store_path.endswith('/'):
            store_path = store_path + '/'
        gdown.download(id=download_id_dict[dataset], output=store_path)

        file_path = os.path.join(store_path, dataset + '.zip')
        with zipfile.ZipFile(file_path, 'r') as zip_file:
            zip_file.extractall(store_path)
        os.remove(file_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Download datasets and checkpoints')
    parser.add_argument('--contents',
                        nargs='+',
                        default=['datasets', 'checkpoints'])
    parser.add_argument('--datasets', nargs='+', default=['default'])
    parser.add_argument('--checkpoints', nargs='+', default=['all'])
    parser.add_argument('--save_dir',
                        nargs='+',
                        default=['./data', './results'])
    parser.add_argument('--dataset_mode', default='default')
    args = parser.parse_args()

    if args.datasets[0] == 'default':
        args.datasets = ['cifar-10']
    elif args.datasets[0] == 'all':
        args.datasets = list(benchmarks_dict.keys())

    for content in args.contents:
        if content == 'datasets':

            store_path = args.save_dir[0]
            if not store_path.endswith('/'):
                store_path = store_path + '/'

            if args.dataset_mode == 'default' or \
                    args.dataset_mode == 'benchmark':
                for benchmark in args.datasets:
                    for dataset in benchmarks_dict[benchmark]:
                        download_dataset(dataset, args)

            if args.dataset_mode == 'dataset':
                for dataset in args.datasets:
                    download_dataset(dataset, args)
