from torch.utils.data import Dataset
from PIL import Image
from mypath import Path

class webvision_dataset(Dataset): 
    def __init__(self, transform, mode, num_class, pred=[], probability=[], log=''): 
        self.root = Path.db_root_dir("webvision")
        self.transform = transform
        self.mode = mode
        self.num_classes = 50
        
        if self.mode=='test':
            with open(self.root+'info/val_filelist.txt') as f:
                lines=f.readlines()
            self.val_imgs = []
            self.val_labels = []
            for line in lines:
                img, target = line.split()
                target = int(target)
                if target<num_class:
                    self.val_imgs.append(img)
                    self.val_labels.append(target)
        else:    
            with open(self.root+'info/train_filelist_google.txt') as f:
                lines=f.readlines()    
            train_imgs = []
            self.targets = []
            for line in lines:
                img, target = line.split()
                target = int(target)
                if target<num_class:
                    train_imgs.append(img)
                    self.targets.append(target)
            self.data = train_imgs
                            
    def __getitem__(self, index):
        if self.mode=='train':
            img_path = self.data[index]
            target = self.targets[index]
            image = Image.open(self.root+img_path).convert('RGB') 
            img = self.transform(image)
            return {'image':img, 'target':target, 'index':index}
        elif self.mode=='test':
            img_path = self.val_imgs[index]
            target = self.val_labels[index]
            image = Image.open(self.root+'val_images_256/'+img_path).convert('RGB')
            img = self.transform(image) 
            return {'image':img, 'target':target, 'index':index}
           
    def __len__(self):
        if self.mode!='test':
            return len(self.data)
        else:
            return len(self.val_imgs)    
