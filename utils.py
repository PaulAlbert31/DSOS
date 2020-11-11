import torch
import torchvision
import os
import numpy as np
import datasets
import random
import faiss
import torch.nn.functional as F

import itertools
from RandAugment import RandAugment
import copy

def multi_class_loss(pred, target):
    pred = F.log_softmax(pred, dim=1)
    loss = - torch.sum(target*pred, dim=1)
    return loss

def mixup_data(x, y, alpha=1.0, device='cuda'):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
        lam = max(lam, 1-lam)
    else:
        lam = 1

    device = x.get_device()
    batch_size = x.size()[0]
    
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam, index


def make_data_loader(args, no_aug=False, transform=None, **kwargs):
    
    if args.dataset == "miniimagenet_preset":
        mean = [0.4728, 0.4487, 0.4031]
        std = [0.2744, 0.2663 , 0.2806]
        size1 = 320
        size = 299
    elif args.dataset == 'webvision':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        size1 = 256
        size = 227
    elif args.dataset == 'clothing':
        mean = [0.6959, 0.6537, 0.6371]
        std = [0.3113, 0.3192, 0.3214]
        size1 = 256
        size = 224


    transform_train = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size1),
        torchvision.transforms.RandomCrop(size),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean, std),
    ])
        
    transform_test = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size1),
        torchvision.transforms.CenterCrop(size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean, std)
    ])

    if args.dataset == "miniimagenet_preset":
        from datasets.miniimagenet_preset import make_dataset, MiniImagenet84
        train_data, train_labels, val_data, val_labels, test_data, test_labels, clean_noisy = make_dataset(noise_ratio=args.noise_ratio)
        trainset = MiniImagenet84(train_data, train_labels, transform=transform_train)
        trainset.clean_noisy = clean_noisy
        testset = MiniImagenet84(val_data, val_labels, transform=transform_test)
    elif args.dataset == "webvision":
        from datasets.webvision import webvision_dataset, imagenet_dataset
        trainset = webvision_dataset(transform=transform_train, mode="train", num_class=50)
        testset = webvision_dataset(transform=transform_test, mode="test", num_class=50)
    elif args.dataset == 'clothing':
        from datasets.clothing import clothing_dataset
        trainset = clothing_dataset(transform=transform_train, split="train")
        testset = clothing_dataset(transform=transform_test, split="test")
    else:
        raise NotImplementedError("Dataset {} is not implemented".format(args.dataset))
    

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs) #Normal training    
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, **kwargs)
        
    return train_loader, test_loader

def create_save_folder(args):
    try:
        os.mkdir(args.save_dir)
    except:
        pass
    try:
        os.mkdir(os.path.join(args.save_dir, args.net + '_'  + args.dataset))
    except:
        pass
    try:
        os.mkdir(os.path.join(args.save_dir, args.net + '_'  + args.dataset, args.exp_name))
    except:
        pass
    try:
        os.mkdir(os.path.join(args.save_dir, args.net + '_'  + args.dataset, args.exp_name, str(args.seed)))
    except:
        pass
       
class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor
    

class DSOS(torch.nn.Module):
    def __init__(self, args, a=.1, alpha=1):
        super(DSOS, self).__init__()
        self.a = a
        self.alpha = alpha
        self.args = args

    def forward(self, x, y, weights1, weights2, preds, eps):
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
            lam = max(lam, 1-lam)
        else:
            lam = 1
            
        device = x.get_device()
        batch_size = x.size()[0]
        
        index = torch.randperm(batch_size).to(device)
        if eps:
            if self.args.boot:
                y[weights1] = preds[weights1]
            if self.args.soft:
                y_s = F.softmax(y/self.a*(1-weights2.view(-1,1)), dim=1)
            else:
                y_s = F.softmax(y/.2, dim=1)
        else:
            y_s = F.softmax(y/.2, dim=1)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y_s, y_s[index]
            
        return mixed_x, y_a, y_b, lam, index
