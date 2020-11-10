import torchvision as tv
import numpy as np
from PIL import Image
import time
import torch
from torch.utils.data import Dataset
import os
import csv
from tqdm import tqdm
from mypath import Path

class MiniImagenet84(Dataset):
    # including hard labels & soft labels
    def __init__(self, data, labels, transform=None, target_transform=None):
        self.data, self.targets =  data, labels
        self.transform = transform
        self.target_transform = target_transform
        self.clean_noisy = None
        self.num_classes = 100
        
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
            
        img = Image.open(img).convert('RGB')
        
        if self.transform is not None:
            img = self.transform(img)
            
        if self.target_transform is not None:
            target = self.target_transform(target)
            
        sample = {'image':img, 'target':target, 'index':index}
        if self.clean_noisy is not None:
            sample['clean_noisy'] = self.clean_noisy[index]
        return sample


    def __len__(self):
        return len(self.data)


def make_dataset(root=Path.db_root_dir('miniimagenet_preset'), noise_ratio="0.2", noise_type="red"):
    np.random.seed(42)
    nclass = 100
    img_paths = []
    labels = []
    for split in ["training", "validation"]:
        if split == "training":
            class_split_path = os.path.join(root, split, '{}_noise_nl_{}'.format(noise_type, noise_ratio))
        else:
            class_split_path = os.path.join(root, split)
        for c in range(nclass):
            class_img_paths = os.listdir(os.path.join(class_split_path, str(c)))
            for paths in class_img_paths:
                img_paths.append(os.path.join(class_split_path, str(c), paths))
                labels.append(c)

    train_num = 50000
    val_num = 10000

    train_paths = img_paths[:train_num]
    train_labels = labels[:train_num]
    val_paths = img_paths[train_num:]
    val_labels = labels[train_num:]
    clean_noisy = np.ones(len(train_paths))
        
    return train_paths, train_labels, val_paths, val_labels, None, None, clean_noisy
