import torchvision
import numpy as np
from PIL import Image
import math
from mypath import Path
    
class CIFAR100(torchvision.datasets.CIFAR100):
    def __init__(self, root, ood_noise, ind_noise, train=True, transform=None, target_transform=None, download=False):
        super(CIFAR100, self).__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
        #CIFAR100red
        self.train = train
        self.ood_noise = ood_noise
        self.ind_noise = ind_noise

        self.data = np.array(self.data)
        self.targets = np.array(self.targets)
        n_class = 100
        
        #Same dataset accross experiments
        np.random.seed(round(math.exp(1) * 1000))
        #OOD noise
        if train and self.ood_noise > 0:
            from datasets.imagenet32 import ImageNet
            imagenet32 = ImageNet(root=Path.db_root_dir('imagenet32'), size=32, train=True)
            self.ids_ood = [i for i, t in enumerate(self.targets) if np.random.random() < self.ood_noise]
            ood_images = imagenet32.data[np.random.permutation(np.arange(len(imagenet32)))[:len(self.ids_ood)]]
            self.data[self.ids_ood] = ood_images
            del imagenet32
            
        #sym ID noise
        if train and self.ind_noise > 0:
            self.ids_not_ood = [i for i in range(len(self.targets)) if i not in self.ids_ood]
            self.ids_id = [i for i in self.ids_not_ood if np.random.random() < (self.ind_noise/(1-self.ood_noise))]
            self.targets = np.array([t if i not in self.ids_id else int(np.random.random() * n_class) for i, t in enumerate(self.targets)])

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return {'image':img, 'target':target, 'index':index}

    
