import torch
import torchvision
from tqdm import tqdm

from torchvision.datasets.folder import default_loader

import os

# create an imagefolder dataset
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((256, 256)),
    torchvision.transforms.ToTensor(),
])

import pandas as pd
# get list of train images]
df = pd.read_csv('../../benchmark_imglist/clothing1M/train_clothing1M_clean.txt', sep=' ', header=None)
# remove leading 'clothing1M/' from image paths
df[0] = df[0].str.replace('clothing1M/', '')
# get real image paths (following symlinks)
#df[0] = df[0].apply(lambda x: os.path.realpath(x))
img_list = df[0].tolist()

# create custom dataset
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, img_list, transform):
        self.samples = img_list
        self.transform = transform
        self.loader = default_loader

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        return sample
    
dataset = CustomDataset(img_list, transform)

# create a dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

# calculate mean and std
mean = 0.
std = 0.
nb_samples = 0.
for data in tqdm(dataloader):
    batch_samples = data.size(0)
    data = data.view(batch_samples, data.size(1), -1)
    mean += data.mean(2).sum(0)
    std += data.std(2).sum(0)
    nb_samples += batch_samples

mean /= nb_samples
std /= nb_samples
print(f"mean: {mean}", f"std: {std}", "nb_samples: ", nb_samples)

# result:
# mean: tensor([0.7249, 0.6878, 0.6702]) std: tensor([0.2514, 0.2656, 0.2657]) nb_samples:  47570.0