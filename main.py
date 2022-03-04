import argparse
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from utils import make_data_loader, create_save_folder, multi_class_loss, mixup_data, DSOS, min_max

import os
from torch.multiprocessing import set_sharing_strategy
set_sharing_strategy('file_system')
import random
from beta_model import BetaMixture1D
import shutil

class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.epoch = 0
        
        if args.net == 'inception':
            from nets.inceptionresnetv2 import InceptionResNetV2
            model = InceptionResNetV2(num_classes=self.args.num_class, drop_ratio=self.args.drop_ratio)
        elif args.net == 'preresnet18':
            from nets.preresnet import PreActResNet18
            model = PreActResNet18(num_classes=self.args.num_class)
        elif args.net == 'resnet50':
            from torchvision.models import resnet50
            model = resnet50(pretrained=True)
            model.fc = torch.nn.Linear(2048, self.args.num_class)
        else:
            raise NotImplementedError("Network {} is not implemented".format(args.net))
        
        print('Number of parameters', sum(p.numel() for p in model.parameters() if p.requires_grad))
        
        self.model = nn.DataParallel(model).cuda()
        if self.args.dsos:
            self.DSOS = DSOS(args, a=self.args.alpha, alpha=self.args.mixup_alpha)

        wd = 5e-4
        if self.args.dataset == 'clothing':
            wd = 1e-3
            
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=0.9, weight_decay=wd)
            
        self.criterion_nored = nn.CrossEntropyLoss(reduction='none')
        
        self.kwargs = {'num_workers': 12, 'pin_memory': True}        
        self.train_loader, self.track_loader, self.val_loader = make_data_loader(args, **self.kwargs)
        print(len(self.train_loader.dataset), len(self.val_loader.dataset))

        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.args.steps, gamma=self.args.gamma)
        
        self.best = 0
        self.best_epoch = 0
        self.acc = []
        self.train_acc = []

        self.previous_preds = torch.zeros((len(self.train_loader.dataset), self.args.num_class), dtype=torch.long)
        
        self.entropies = torch.ones(len(self.train_loader.dataset))

        
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

            if self.args.cuda:
                target, image = target.cuda(), image.cuda()
                
            if self.args.dsos and (epoch > self.args.correct_ep):
                target = self.DSOS(target, self.id_metric[ids].cuda(), self.ood_metric[ids].cuda(), self.previous_preds[ids].cuda())
                                            
            if self.args.mixup:
                image, la, lb, lam, o = mixup_data(image, target, alpha=self.args.mixup_alpha)
                target = lam * la + (1-lam) * lb
                
            self.optimizer.zero_grad()
            
            outputs = self.model(image)

            if self.args.mixup:
                loss_b = lam * multi_class_loss(outputs, la) + (1-lam) * multi_class_loss(outputs, lb)
            else:
                loss_b = multi_class_loss(outputs, target)
                
            if self.args.entro:
                a = .4
                if epoch > self.args.correct_ep:
                    #Dynamic entropy
                    a = self.entropies[ids].cuda().mean() * a
                loss_entro = (a * torch.sum(-F.softmax(outputs, dim=1) * F.log_softmax(outputs, dim=1), dim=1)).mean()
                loss_b += loss_entro
            else:
                loss_entro = 0.
                
            loss = torch.mean(loss_b)
            if not self.args.mixup:
                preds = torch.argmax(F.log_softmax(outputs, dim=1), dim=1)
                acc += torch.sum(preds == torch.argmax(target, dim=1))

                total_sum += preds.size(0)

            loss.backward()
            if i % 10 == 0:
                tbar.set_description('Training loss {0:.2f}, LR {1:.6f}, L0 {2:.2f}, L1 {3:.2f}, L2 {4:.2f}'.format(loss, self.optimizer.param_groups[0]['lr'], loss, loss_entro, 0.0))
            self.optimizer.step()
        self.scheduler.step()
        if epoch == self.args.correct_ep - 1 or epoch == 49 or epoch == 79 or epoch == 24:
            self.save_model(epoch, t=True)
        print('[Epoch: {}, numImages: {}, numClasses: {}]'.format(epoch, total_sum, self.args.num_class))
        if self.args.dataset == "clothing":
            #Load another 1000 batches every epoch
            self.train_loader, self.track_loader, self.val_loader = make_data_loader(self.args, **self.kwargs)
            relabel = torch.tensor(self.train_loader.dataset.targets)
            relabel = F.one_hot(relabel, self.args.num_class)
            self.train_loader.dataset.targets = relabel
            self.track_loader.dataset.targets = relabel

        if not self.args.mixup:
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
            self.save_model(epoch, best=True)
        else:
            self.save_model(epoch)
        with open(os.path.join(self.args.save_dir, 'lastpred_{}.txt'.format(self.args.checkname)), 'w') as f:
            f.write(str(final_acc))
            
        print('Validation Accuracy: {0:.4f}, best accuracy {1:.4f} at epoch {2}'.format(final_acc, self.best, self.best_epoch))
        return final_acc, losses.mean(), accs.mean()

    def save_model(self, epoch, t=False, best=False):
        if t:
            checkname = os.path.join(self.args.save_dir, '{}_{}.pth.tar'.format(self.args.checkname, epoch))
        elif best:
            checkname = os.path.join(self.args.save_dir, '{}_best.pth.tar'.format(self.args.checkname, epoch))
            with open(os.path.join(self.args.save_dir, 'bestpred_{}.txt'.format(self.args.checkname)), 'w') as f:
                f.write(str(self.best))
        else:
            checkname = os.path.join(self.args.save_dir, '{}.pth.tar'.format(self.args.checkname, epoch))
            
        torch.save({
            'epoch': epoch+1,
            'state_dict': self.model.module.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'best': self.best,
            'best_epoch':self.best_epoch
        }, checkname)
        
        with open(os.path.join(self.args.save_dir, 'bestpred_{}.txt'.format(self.args.checkname)), 'w') as f:
            f.write(str(self.best))

    def track_loss(self, plot=False):
        self.model.train() #Batchnorm tuning on unmixed samples
        total_sum = 0
        display_acc = 0
        with torch.no_grad():
            tbar = tqdm(self.track_loader)
            tbar.set_description('Tracking loss')
            
            nentropy = torch.zeros(len(self.train_loader.dataset))
            
            for i, sample in enumerate(tbar):
                image, target, ids = sample['image'], sample['target'], sample['index']
                if self.args.cuda:
                    target, image = target.cuda(), image.cuda()

                outputs = self.model(image)
                preds = F.softmax(outputs, dim=1)
                
                #Track entropy
                mid_lab = (preds + target) / 2
                nentropy[ids] = - torch.log(((mid_lab)**2).sum(dim=1)).cpu() #collision entropy
                
                if self.args.cons:
                    image2 = sample['image2'].cuda()
                    outputs2 = self.model(image2)
                    outputs = (outputs + F.softmax(outputs2, dim=-1)) / 2
                    outputs = outputs **2 #temp sharp
                    preds = outputs / outputs.sum(dim=1, keepdim=True) #normalization

                self.previous_preds[ids] = F.one_hot(torch.argmax(preds, dim=1), num_classes=self.args.num_class).cpu()
                
                #Track train accuracy
                if self.args.mixup:
                    display_acc += (torch.argmax(preds, dim=-1) == torch.argmax(target, dim=-1)).sum()
                    total_sum += preds.size(0)
                    
            if self.args.mixup:
                print('Training Accuracy: {0:.4f}'.format(float(display_acc)/total_sum))
                            
        return nentropy
   
def main():


    parser = argparse.ArgumentParser(description="PyTorch noisy labels intermediary features")
    parser.add_argument('--net', type=str, default='preresnet18',
                        choices=['resnet50', 'preresnet18', 'inception'],
                        help='net name (default: preresnet18)')
    parser.add_argument('--dataset', type=str, default='miniimagenet_preset', choices=['miniimagenet_preset', 'webvision', 'clothing', 'stanford_preset', 'cifar100'])
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
    parser.add_argument('--dsos', default=False, action='store_true')
    parser.add_argument('--entro', default=False, action='store_true')

    parser.add_argument('--noisy-labels', default=None, type=str)
    parser.add_argument('--noise-ratio', default="0.3", type=str)
    parser.add_argument('--track', default=False, action='store_true')
    parser.add_argument('--boot', default=False, action='store_true')
    parser.add_argument('--soft', default=False, action='store_true')
    parser.add_argument('--resume', default=None, type=str)
    parser.add_argument('--alpha', default=.15, type=float)
    parser.add_argument('--mixup-alpha', default=1, type=float)
    parser.add_argument('--ood-ratio', default=0.0, type=float)
    parser.add_argument('--ind-ratio', default=0.0, type=float)
    parser.add_argument('--correct-ep', default=None, type=int)
    
    parser.add_argument('--drop-ratio', default=0.0, type=float)
    parser.add_argument('--aug', default='rc', type=str)
    parser.add_argument('--cons', default=False, action='store_true')
    
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

    dict_class = {'miniimagenet':100, 'webvision':50, 'miniimagenet_preset':100, 'clothing':14, 'stanford_preset':196, 'cifar100':100, 'cifar10':10}
    
    args.num_class = dict_class[args.dataset]
        
    if args.steps is None:
        args.steps = [args.epochs]
    if args.checkname is None:
        args.checkname = "{}_{}".format(args.net, args.dataset)
    if args.correct_ep is None:
        args.correct_ep = args.steps[0]
        
    create_save_folder(args)
    args.save_dir = os.path.join(args.save_dir, args.net + '_'  + args.dataset, args.exp_name, str(args.seed))
    args.cuda = not args.no_cuda
    if True:
        if not os.path.isdir(os.path.join(args.save_dir + "/code")):
            os.mkdir(os.path.join(args.save_dir, "code"))
        if os.path.isdir(os.path.join(args.save_dir, "code", "nets")):
            shutil.rmtree(os.path.join(args.save_dir, "code", "nets"))
        if os.path.isdir(os.path.join(args.save_dir, "code", "datasets")):
            shutil.rmtree(os.path.join(args.save_dir, "code", "datasets"))
            
        os.mkdir(os.path.join(args.save_dir, "code", "datasets"))
        os.mkdir(os.path.join(args.save_dir, "code", "nets"))
        
        shutil.copyfile("main.py", "{}/main.py".format(os.path.join(args.save_dir,"code")))
        shutil.copyfile("utils.py", "{}/utils.py".format(os.path.join(args.save_dir,"code")))
        for f in os.listdir("nets"):
            if os.path.isfile("nets/{}".format(f)) and f[-1] != "~":
                shutil.copyfile("nets/{}".format(f), os.path.join(args.save_dir, "code", "nets/{}".format(f)))
        for f in os.listdir("datasets"):
            if os.path.isfile("datasets/{}".format(f)) and f[-1] != "~":
                shutil.copyfile("datasets/{}".format(f), os.path.join(args.save_dir, "code", "datasets/{}".format(f)))

        torch.save(args, os.path.join(args.save_dir, "code","args"))
        
    for u in range(1):
        _trainer = Trainer(args)
                
        relabel = torch.tensor(_trainer.train_loader.dataset.targets)
        # One hot if not
        relabel = F.one_hot(relabel, num_classes=args.num_class)
                
        _trainer.train_loader.dataset.targets = relabel
        _trainer.track_loader.dataset.targets = relabel
        
        nentropy_t = torch.zeros(args.epochs, len(relabel))
                
        save_dict = {}

        #Refinement of the class labels
        start_ep = 0
        if args.resume is not None:
            load_dict = torch.load(args.resume, map_location='cpu')
            _trainer.model.module.load_state_dict(load_dict['state_dict'])
            _trainer.optimizer.load_state_dict(load_dict['optimizer'])
            _trainer.scheduler.load_state_dict(load_dict['scheduler'])
            start_ep = load_dict['epoch'] + 1
            if args.track and (start_ep >= args.correct_ep):
                nentropy = _trainer.track_loss()
                nentropy_t[start_ep] = nentropy
                
                lim = -torch.log(torch.tensor(.5))

                #Fitting a 2 components BMM to the noisy samples
                noisy = (nentropy >= lim)
                interest = min_max(nentropy[noisy])
                bmm = BetaMixture1D(n_components=2).fit(interest)
                
                #Computing the BMM modes to order the detection
                modes = (bmm.alphas - 1) / (bmm.alphas + bmm.betas - 2)
                order = torch.argsort(modes)

                proba = torch.zeros((len(nentropy), 3))
                proba[~noisy, 0] = 1

                if modes[1] > -0.1 and modes[1] < 1.1:
                    #If the OOD noise was captured (mode in [-0.1, 1.1])
                    proba[noisy, 2] = bmm.posterior(interest, 1).float()
                else:
                    #No OOD noise was captured (mode outside [-0.1, 1.1]), using the min_max value itself
                    proba[noisy, 2] = interest
                    
                if modes[0] > -0.1 and modes[0] < 1.1:
                    #If the ID noise was captured (mode in [-0.1, 1.1])
                    proba[noisy, 1] = bmm.posterior(interest, 0).float()
                else:
                    #BMM did not manage to capture the ID mode, using the min_max value itself
                    proba[noisy, 1] = ((1-interest) > .5) * 1.

                #Filtering errors
                proba[proba != proba] = 0

                _trainer.ood_metric = 1-proba[:, 2]
                _trainer.id_metric = (proba[:, 1] > .9)
            
            v, loss, acc = _trainer.val(start_ep)

        for eps in range(start_ep, args.epochs):
            _trainer.train(eps)
            if args.track and (eps >= args.correct_ep):
                nentropy = _trainer.track_loss()
                nentropy_t[eps] = nentropy
                
                lim = -torch.log(torch.tensor(.5))

                #Fitting a 2 components BMM to the noisy samples
                noisy = (nentropy >= lim)
                interest = min_max(nentropy[noisy])
                
                bmm = BetaMixture1D(n_components=2).fit(interest)
                
                #Computing the BMM modes to order the detection
                modes = (bmm.alphas - 1) / (bmm.alphas + bmm.betas - 2)
                order = torch.argsort(modes)

                proba = torch.zeros((len(nentropy), 3))
                proba[~noisy, 0] = 1

                if modes[1] > -0.1 and modes[1] < 1.1:
                    #If the OOD noise was captured (mode in [-0.1, 1.1])
                    proba[noisy, 2] = bmm.posterior(interest, 1).float()
                else:
                    #No OOD noise was captured (mode outside [-0.1, 1.1]), using the min_max value itself
                    proba[noisy, 2] = interest
                    
                if modes[0] > -0.1 and modes[0] < 1.1:
                    #If the ID noise was captured (mode in [-0.1, 1.1])
                    proba[noisy, 1] = bmm.posterior(interest, 0).float()
                else:
                    #BMM did not manage to capture the ID mode, using the min_max value itself
                    proba[noisy, 1] = ((1-interest) > .5) * 1.

                #Filtering errors
                proba[proba != proba] = 0

                _trainer.ood_metric = 1-proba[:, 2]
                _trainer.id_metric = (proba[:, 1] > .9)
                
            v, loss, acc = _trainer.val(eps)    


if __name__ == "__main__":
   main()
