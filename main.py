import argparse
import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from utils import make_data_loader, create_save_folder, UnNormalize, multi_class_loss, mixup_data, DSOS

import os
from torch.multiprocessing import set_sharing_strategy
set_sharing_strategy('file_system')
import matplotlib
matplotlib.use('agg')

import matplotlib.pyplot as plt
import concurrent.futures

from PIL import Image
import copy
import random

class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.epoch = 0
        
        #Lots of networks, didn't try all of them
        if args.net == 'inception':
            from nets.inceptionresnetv2 import InceptionResNetV2
            model = InceptionResNetV2(num_classes=self.args.num_class)
        elif args.net == 'preresnet18':
            from nets.preresnet import PreActResNet18
            model = PreActResNet18(num_classes=self.args.num_class)
        elif args.net == 'resnet50':
            from torchvision.models import resnet50
            model = resnet50(pretrained=True)
            model.fc = torch.nn.linear(2048, self.args.num_class)
        else:
            raise NotImplementedError("Network {} is not implemented".format(args.net))
        
        print('Number of parameters', sum(p.numel() for p in model.parameters() if p.requires_grad))
        
        self.model = nn.DataParallel(model).cuda()
        self.DSOSMix = DSOS(args, a=.1, alpha=1)

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=0.9, weight_decay=5e-4)
        self.criterion_nored = nn.CrossEntropyLoss(reduction='none')
        
        self.kwargs = {'num_workers': 4, 'pin_memory': True}        
        self.train_loader, self.val_loader = make_data_loader(args, **self.kwargs)
   
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.args.steps, gamma=self.args.gamma)
        
        self.best = 0
        self.best_epoch = 0
        self.acc = []
        self.train_acc = []

        self.toPIL = torchvision.transforms.ToPILImage()

        self.unorm = UnNormalize(mean=(0.4728, 0.4487, 0.4031), std=(0.2744, 0.2663, 0.2806))

        self.forget_rank = torch.zeros(len(self.train_loader.dataset)).long()
        self.previous_acc = torch.zeros(len(self.train_loader.dataset)).long()
        
        self.previous_preds = torch.zeros((len(self.train_loader.dataset), self.args.num_class))
        self.entropies = torch.zeros((len(self.train_loader.dataset)))
        self.accuracy = torch.zeros((len(self.train_loader.dataset)))
        
        if self.args.cuda:
            self.DSOSMix = self.DSOSMix.cuda()
        
    def train(self, epoch, feature_center=None):
        running_loss = 0.0
        self.model.train()
        
        acc = 0
        tbar = tqdm(self.train_loader)
        m_dists = torch.tensor([])
        l = torch.tensor([])
        self.epoch = epoch
        total_sum = 0
        
        for i, sample in enumerate(tbar):
            image, target, ids = sample['image'], sample['target'], sample['index']
            weights = sample['clean_noisy']
            if epoch < self.args.steps[0]:
                weights = torch.ones(len(image)).cuda()
            
            if self.args.cuda:
                target, image = target.cuda(), image.cuda()
                weights = weights.cuda()
                                            
            if self.args.mixup or (epoch < self.args.steps[0] and self.args.softmix):
                image, la, lb, lam, o = mixup_data(image, target)
            elif self.args.softmixup:
                image, la, lb, lam, o = self.DSOSMix(image, target, weights, self.entropies[ids].cuda(), self.previous_preds[ids].cuda(), epoch >= self.args.steps[0])
                
            if False:
                for i, im in enumerate(image):
                    if target[i][20] == 1:
                        image_s = self.toPIL(self.unorm(im.cpu()))
                        if weights[i] == 1:
                            image_s.save('preds/im_clean{}.png'.format(sample['index'][i]))
                        else:
                            image_s.save('preds/im_noisy{}.png'.format(sample['index'][i]))

            self.optimizer.zero_grad()
            
            outputs = self.model(image)
            if self.args.mixup or self.args.softmixup:
                loss_b = lam * multi_class_loss(outputs, la) + (1-lam) * multi_class_loss(outputs, lb)
            else:
                loss_b = multi_class_loss(outputs, target)

            if self.args.entro and epoch >= self.args.steps[0]:
                if self.args.soft:
                    a = 1-self.entropies[ids].mean()
                else:
                    a = .4
                loss_b += a * torch.sum(-F.softmax(outputs, dim=1) * F.log_softmax(outputs, dim=1), dim=1) #* (weights * lam + weights[o] * (1-lam)) / (lam * weights + (1-lam) * weights[o]).sum()

            loss = torch.mean(loss_b)
            if not (self.args.mixup or self.args.softmixup):
                preds = torch.argmax(F.log_softmax(outputs, dim=1), dim=1)
                acc += torch.sum(preds == torch.argmax(target, dim=1))

                total_sum += preds.size(0)

            loss.backward()
            if i % 10 == 0:
                tbar.set_description('Training loss {0:.2f}, LR {1:.6f}, L0 {2:.2f}, L1 {3:.2f}, L2 {4:.2f}'.format(loss, self.optimizer.param_groups[0]['lr'], loss, 0.0, 0.0))
            self.optimizer.step()
        self.scheduler.step()
        if epoch == self.args.steps[0] - 1 or epoch == 49:
            self.save_model(epoch, t=True)
        print('[Epoch: {}, numImages: {}, numClasses: {}]'.format(epoch, total_sum, self.args.num_class))
        if not (self.args.mixup or self.args.softmixup):
            print('Training Accuracy: {0:.4f}'.format(float(acc)/total_sum))
            self.train_acc.append(float(acc)/total_sum)
            torch.save(torch.tensor(self.train_acc), os.path.join(self.args.save_dir, '{0}_trainacc.pth.tar'.format(self.args.checkname)))
            return float(acc)/total_sum
        
    def val(self, epoch, dataset='val', save=True):
        self.model.eval()
        acc = 0

        vbar = tqdm(self.val_loader)
        total = 0
        losses, accs = torch.tensor([]), torch.tensor([])
        with torch.no_grad():
            for i, sample in enumerate(vbar):
                image, target = sample['image'], sample['target']
                if self.args.cuda:
                    image, target = image.cuda(), target.cuda()

                outputs = self.model(image)
                    
                loss = self.criterion_nored(outputs, target)
                losses = torch.cat((losses, loss.cpu()))
                
                preds = torch.argmax(F.log_softmax(outputs, dim=1), dim=1)
                accs = torch.cat((accs, (preds==target.data).float().cpu()))
                
                acc += torch.sum(preds == target.data)
                total += preds.size(0)
            
                if i % 10 == 0:
                    if dataset == 'val':
                        vbar.set_description('Validation loss: {0:.2f}'.format(loss.mean()))
                    else:
                        vbar.set_description('Test loss: {0:.2f}'.format(loss.mean()))
        final_acc = float(acc)/total
        if i % 10 == 0:
            print('[Epoch: {}, numImages: {}]'.format(epoch, (len(self.val_loader)-1)*self.args.batch_size + image.shape[0]))
        self.acc.append(final_acc)
        torch.save(torch.tensor(self.acc), os.path.join(self.args.save_dir, '{0}_acc.pth.tar'.format(self.args.checkname)))
        if final_acc > self.best and save:
            self.best = final_acc
            self.best_epoch = epoch
        self.save_model(epoch)
            
        print('Validation Accuracy: {0:.4f}, best accuracy {1:.4f} at epoch {2}'.format(final_acc, self.best, self.best_epoch))
        return final_acc, losses.mean(), accs.mean()

    def save_model(self, epoch, t=False):
        if t:
            torch.save({
                'epoch': epoch+1,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best': self.best,
                'best_epoch':self.best_epoch
            }, os.path.join(self.args.save_dir, '{}_{}.pth.tar'.format(self.args.checkname, epoch)))
        else:
            torch.save({
                'epoch': epoch+1,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best': self.best,
                'best_epoch':self.best_epoch
            }, os.path.join(self.args.save_dir, '{}.pth.tar'.format(self.args.checkname)))
        with open(os.path.join(self.args.save_dir, 'bestpred_{}.txt'.format(self.args.checkname)), 'w') as f:
            f.write(str(self.best))

    def track_loss(self, relabel, plot=False):
        self.model.train()
        acc = 0
        total_sum = 0
        with torch.no_grad():
            tr = copy.deepcopy(self.train_loader.dataset.transform)
            if self.args.dataset == 'miniimagenet_preset':
                size1 = 84
                size = 84
                mean = [0.4728, 0.4487, 0.4031]
                std = [0.2744, 0.2663 , 0.2806]
            elif self.args.dataset == 'webvision':
                size1 = 256
                size = 227
                mean = [0.485, 0.456, 0.406]
                std = [0.229, 0.224, 0.225]
            elif self.args.dataset == 'clothing':
                size1 = 256
                size = 224
                mean = [0.6959, 0.6537, 0.6371]
                std = [0.3113, 0.3192, 0.3214]
                
            track_tr = torchvision.transforms.Compose([
                torchvision.transforms.Resize(size1),
                torchvision.transforms.CenterCrop(size),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean, std),
            ])
            
            self.train_loader.dataset.transform = track_tr
                               
            tbar = tqdm(self.train_loader)
            tbar.set_description('Tracking loss')
            
            losses = torch.zeros(len(self.train_loader.dataset))
            accuracies = torch.zeros(len(self.train_loader.dataset))
            entropy = torch.zeros(len(self.train_loader.dataset))
            display_acc = 0
            
            for i, sample in enumerate(tbar):
                image, target, ids = sample['image'], sample['target'], sample['index']
                if self.args.cuda:
                    target, image = target.cuda(), image.cuda()
                   
                outputs = self.model(image)
                
                #Track loss
                losses[ids] = multi_class_loss(outputs, target).detach().cpu()
                #Track accuracy
                accuracy_soft = F.softmax(outputs, dim=1).detach().cpu()
                target = torch.argmax(target, dim=1).cpu()
                for i, t in enumerate(target):
                    accuracies[ids[i]] = accuracy_soft[i][t]
                #Track forgetting events
                preds = torch.argmax(accuracy_soft, dim=1)
                correct_pred = (preds == target).long()
                forgetting_event = (self.previous_acc[ids] > correct_pred).long()
                never_learnt  = ((self.previous_acc[ids] == correct_pred) * (correct_pred == torch.zeros(len(correct_pred), dtype=torch.long))).long()
                forgetting_event += never_learnt
                self.previous_acc[ids] = correct_pred
                self.forget_rank[ids] += forgetting_event
                #Track entropy
                entropy[ids] = (- accuracy_soft * torch.log(accuracy_soft)).sum(dim=1)

                #Track train accuracy
                if self.args.mixup or self.args.softmixup:
                    display_acc += (preds == target).sum()
                    total_sum += preds.size(0)
                    
            if self.args.mixup or self.args.softmixup:
                print('Training Accuracy: {0:.4f}'.format(float(display_acc)/total_sum))
                
            self.train_loader.dataset.transform = tr
            
            return losses, accuracies, self.forget_rank, entropy
   
def main():


    parser = argparse.ArgumentParser(description="PyTorch noisy labels intermediary features")
    parser.add_argument('--net', type=str, default='preresnet18',
                        choices=['resnet50', 'preresnet18', 'inception'],
                        help='net name (default: preresnet18)')
    parser.add_argument('--dataset', type=str, default='miniimagenet_preset', choices=['miniimagenet_preset', 'miniimagenet', 'webvision', 'clothing'])
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.1, help='Multiplicative factor for lr decrease, default .1')
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--steps', type=int, default=None, nargs='+', help='Epochs when to reduce lr')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Probably not working')
    parser.add_argument('--save-dir', type=str, default='checkpoints')
    parser.add_argument('--checkname', type=str, default=None)
    parser.add_argument('--exp-name', type=str, default='')
    parser.add_argument('--seed', default=1, type=float)
    parser.add_argument('--mixup', default=False, action='store_true')
    parser.add_argument('--softmixup', default=False, action='store_true')
    parser.add_argument('--entro', default=False, action='store_true')

    parser.add_argument('--noisy-labels', default=None, type=str)
    parser.add_argument('--lam', default=0, type=float)
    parser.add_argument('--noise-ratio', default="0.2", type=str)
    parser.add_argument('--track', default=False, action='store_true')
    parser.add_argument('--boot', default=False, action='store_true')
    parser.add_argument('--soft', default=False, action='store_true')
    parser.add_argument('--resume', default=None, type=str)

    args = parser.parse_args()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    seeds = {'1': round(torch.exp(torch.ones(1)).item()*1e6), '2': round(torch.acos(torch.zeros(1)).item() * 2), '3':round(torch.sqrt(torch.tensor(2.)).item()*1e6)}
    try:
        torch.manual_seed(seeds[str(args.seed)])
        torch.cuda.manual_seed_all(seeds[str(args.seed)])  # GPU seed
        random.seed(seeds[str(args.seed)])  # python seed for image transformation                                                                                                                                                                                               
    except:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)  # GPU seed
        random.seed(args.seed)

    dict_class = {'miniimagenet':100, 'webvision':50, 'miniimagenet_preset':100, 'clothing':14}
    
    args.num_class = dict_class[args.dataset]
        
    if args.steps is None:
        args.steps = [args.epochs]
    if args.checkname is None:
        args.checkname = "{}_{}".format(args.net, args.dataset)
        
    create_save_folder(args)
    args.save_dir = os.path.join(args.save_dir, args.net + '_'  + args.dataset, args.exp_name, str(args.seed))
    args.cuda = not args.no_cuda
    
    for u in range(1):
        _trainer = Trainer(args)
        
        relabel = torch.tensor(_trainer.train_loader.dataset.targets)

        # One hot if not
        try:
            relabel[0][0]
        except:
            relabel_oh = torch.zeros((len(relabel), args.num_class))
            for i, r in enumerate(relabel):
                relabel_oh[i][r] = 1
            relabel = relabel_oh
                
        _trainer.train_loader.dataset.targets = relabel
        
        losses_t = torch.zeros(args.epochs, len(relabel)) #Average over the epochs
        accuracies_t = torch.zeros(args.epochs, len(relabel)) 
        forget_t = torch.zeros(args.epochs, len(relabel)) 
        entropy_t = torch.zeros(args.epochs, len(relabel))

        save_dict = {}

        #Refinement of the class labels
        start_ep = 0
        if args.resume is not None:
            load_dict = torch.load(args.resume)
            _trainer.model.module.load_state_dict(load_dict['state_dict'])
            _trainer.optimizer.load_state_dict(load_dict['optimizer'])
            start_ep = load_dict['epoch']
            steps = [s-start_ep for s in args.steps if s-start_ep > 0]

            _trainer.scheduler = torch.optim.lr_scheduler.MultiStepLR(_trainer.optimizer, milestones=steps, gamma=_trainer.args.gamma)
            for _ in range(0):
                _trainer.optimizer.step()
                _trainer.scheduler.step()
                start_ep += 1
            if args.track:
                losses, accuracies, forget, entropy = _trainer.track_loss(relabel, plot=False)
                
                l = accuracies
                l = (l - l.min()) / (l.max() - l.min())
                e = entropy
                e = (e-e.min()) / (e.max() - e.min())
                
                _trainer.entropies = e
                perc = np.percentile(l*e, 5)
                _trainer.train_loader.dataset.clean_noisy = (l*e<perc)#mm_soft
                
                accuracies_t[start_ep-1] = accuracies
                entropy_t[start_ep-1] = entropy
                losses_t[start_ep-1] = losses
                forget_t[start_ep-1] = forget
                
                save_dict = {'losses': losses_t, 'accuracies': accuracies_t, 'forget': forget_t, 'entropy': entropy_t}
                
                torch.save(save_dict, os.path.join(args.save_dir, '{0}_metrics_ep{1}'.format(args.checkname, start_ep-1+1)))
                
            v, loss, acc = _trainer.val(start_ep)

        for eps in range(start_ep, args.epochs):
            _trainer.train(eps)

            if eps >= args.steps[0] - 1 and args.track:
                losses, accuracies, forget, entropy = _trainer.track_loss(relabel, plot=False)
                #Metrics tracking
                losses_t[eps] = losses
                accuracies_t[eps] = accuracies
                forget_t[eps] = forget
                entropy_t[eps] = entropy
            
                l = accuracies_t.mean(dim=0)
                l = (l - l.min()) / (l.max() - l.min())

                e = entropy_t.mean(dim=0)
                e = (e-e.min()) / (e.max() - e.min())
                _trainer.entropies = e
                
                perc = np.percentile(l*e, 5)
                _trainer.train_loader.dataset.clean_noisy = (l*e<perc)
                
                save_dict = {'losses': losses_t, 'accuracies': accuracies_t, 'forget': forget_t, 'entropy': entropy_t}
                
                torch.save(save_dict, os.path.join(args.save_dir, '{0}_metrics_ep{1}'.format(args.checkname, eps+1)))

                if eps > args.steps[0]-1:
                    os.remove(os.path.join(args.save_dir, '{0}_metrics_ep{1}'.format(args.checkname, eps)))

            v, loss, acc = _trainer.val(eps)    

if __name__ == "__main__":
   main()
