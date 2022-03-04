from torch.utils.data import Dataset
from PIL import Image
import os
from mypath import Path
import random
import torch

class clothing_dataset(Dataset): 
    def __init__(self, transform, split, root=Path.db_root_dir('clothing'), num_samples=32000):
        super(clothing_dataset, self).__init__()
        self.root = root
        self.transform = transform
        self.split = split
        self.nclass = 14
        self.data = []
        self.targets = {}
        num_samples = num_samples
        num_class = 14
        
        with open('{}/noisy_label_kv.txt'.format(self.root),'r') as f:
            lines = f.read().splitlines()
            for l in lines:
                entry = l.split()           
                img_path = '{}/images/{}'.format(self.root,entry[0][7:])
                self.targets[img_path] = int(entry[1])                         
        with open('{}/clean_label_kv.txt'.format(self.root),'r') as f:
            lines = f.read().splitlines()
            for l in lines:
                entry = l.split()           
                img_path = '{}/images/{}'.format(self.root,entry[0][7:])
                self.targets[img_path] = int(entry[1])
               
        if self.split == 'train':       
            train_imgs=[]
            with open('{}/noisy_train_key_list.txt'.format(self.root),'r') as f:
                lines = f.read().splitlines()
                for l in lines:
                    img_path = '{}/images/{}'.format(self.root, l[7:])
                    train_imgs.append(img_path)                                
            random.shuffle(train_imgs)
            class_num = torch.zeros(num_class)
            self.data = []
            for impath in train_imgs:
                label = self.targets[impath] 
                if class_num[label]<(num_samples/14) and len(self.data)<num_samples:
                    self.data.append(impath)
                    class_num[label]+=1
            random.shuffle(self.data)
        elif self.split == 'test':
            self.test_imgs = []
            with open('%s/clean_test_key_list.txt'%self.root,'r') as f:
                lines = f.read().splitlines()
                for l in lines:
                    img_path = '{}/images/{}'.format(self.root,l[7:])
                    self.data.append(img_path)            
        elif self.split == 'val':
            self.val_imgs = []
            with open('%s/clean_val_key_list.txt'%self.root,'r') as f:
                lines = f.read().splitlines()
                for l in lines:
                    img_path = '{}/images/{}'.format(self.root,l[7:])
                    self.data.append(img_path)
                    
        self.clean_noisy = [1 for _ in range(len(self.data))]
        self.targets = [self.targets[k] for k in self.data]
        print(len(self.data))
                    
    def __getitem__(self, index):
        img_path = self.data[index]
        target = self.targets[index]
        image = Image.open(img_path).convert('RGB') 
        img = self.transform(image)
        return {'image':img, 'target':target, 'index':index, 'clean_noisy':self.clean_noisy[index]}
        
    def __len__(self):
        return len(self.data)
