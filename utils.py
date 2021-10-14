import torch
import torchvision
import os
import numpy as np
import datasets
import torch.nn.functional as F

import copy
import PIL
from mypath import Path

def min_max(x):
    return (x - x.min())/(x.max() - x.min())

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
    if args.dataset == 'cifar100':
        mean = [0.5071, 0.4867, 0.4408]
        std = [0.2675, 0.2565, 0.2761]
        size1 = 32
        size = 32
    elif args.dataset == "miniimagenet_preset":
        mean = [0.4728, 0.4487, 0.4031]
        std = [0.2744, 0.2663 , 0.2806]
        size1 = 320
        size = 299
    elif args.dataset == 'webvision':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        size1 = 256
        size = 227

    if args.aug == 'rc':
        transform_train = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size1, interpolation=PIL.Image.BICUBIC),
            torchvision.transforms.RandomCrop(size, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean, std),
        ])
    elif args.aug == 'rrc':
        transform_train = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size1, interpolation=PIL.Image.BICUBIC),
            torchvision.transforms.RandomResizedCrop(size, interpolation=PIL.Image.BICUBIC),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean, std),
        ])

    if args.dataset == "miniimagenet_preset": #We find that testing at a different solution works better for miniimagenet
        sizer = size
    else:
        sizer = size1
    
    transform_test = torchvision.transforms.Compose([
        torchvision.transforms.Resize(sizer, interpolation=PIL.Image.BICUBIC),
        torchvision.transforms.CenterCrop(size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean, std)
    ])

    if args.dataset == "cifar100":
        from datasets.cifar import CIFAR100
        trainset = CIFAR100(Path.db_root_dir("cifar100"), ood_noise=args.ood_ratio, ind_noise=args.ind_ratio, train=True, transform=transform_train)
        trackset = CIFAR100(Path.db_root_dir("cifar100"), ood_noise=args.ood_ratio, ind_noise=args.ind_ratio, train=True, transform=transform_test)
        trackset.data, trackset.targets = trainset.data, trainset.targets
        testset = CIFAR100(Path.db_root_dir("cifar100"), ood_noise=args.ood_ratio, ind_noise=args.ind_ratio, train=False, transform=transform_test)
    elif args.dataset == "miniimagenet_preset":
        from datasets.miniimagenet_preset import make_dataset, MiniImagenet84
        train_data, train_labels, val_data, val_labels, test_data, test_labels = make_dataset(noise_ratio=args.noise_ratio)
        trainset = MiniImagenet84(train_data, train_labels, transform=transform_train)
        trackset = MiniImagenet84(train_data, train_labels, transform=transform_test)
        testset = MiniImagenet84(val_data, val_labels, transform=transform_test)
    elif args.dataset == "webvision":
        from datasets.webvision import webvision_dataset
        trainset = webvision_dataset(transform=transform_train, mode="train", num_class=50)
        trackset = webvision_dataset(transform=transform_test, mode="train", num_class=50)
        testset = webvision_dataset(transform=transform_test, mode="test", num_class=50)
    else:
        raise NotImplementedError("Dataset {} is not implemented".format(args.dataset))
    

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs) #Normal training
    track_loader = torch.utils.data.DataLoader(trackset, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, **kwargs)
    
    return train_loader, track_loader, test_loader

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
       
class DSOS(torch.nn.Module):
    def __init__(self, args, a=0.05, alpha=1):
        super(DSOS, self).__init__()
        self.a = a
        self.alpha = alpha
        self.args = args

    def forward(self, y, id_metric, ood_metric, preds):
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
            lam = max(lam, 1-lam)
        else:
            lam = 1
            
        device = y.get_device()
        batch_size = y.size()[0]
        
        index = torch.randperm(batch_size).to(device)

        if self.args.boot:
            y[id_metric] = preds[id_metric]
        if self.args.soft:
            y_s = F.softmax(y/self.a*ood_metric.view(-1,1), dim=1)
        else:
            y_s = y
            
        return y_s
