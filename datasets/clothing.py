from torch.utils.data import Dataset
from PIL import Image
import os
from mypath import Path

class clothing_dataset(Dataset): 
    def __init__(self, transform, split, root=Path.db_root_dir('clothing')):
        super(clothing_dataset, self).__init__()
        self.root = root
        self.transform = transform
        self.split = split
        self.nclass = 14
        self.data = []
        self.targets = []

        if self.split == 'train':
            im_dir = os.path.join(self.root, 'noisy_train')
        elif self.split == 'val':
            im_dir = os.path.join(self.root, 'clean_val')
        elif self.split == 'test':
            im_dir = os.path.join(self.root, 'clean_test')
            
        for c in range(self.nclass):
            im_names = os.listdir(os.path.join(im_dir, str(c)))
            for im in im_names:
                self.data.append(os.path.join(im_dir, str(c), im))
                self.targets.append(c)
        self.clean_noisy = [1 for _ in range(len(self.data))]
        print(len(self.data))
                    
    def __getitem__(self, index):
        img_path = self.data[index]
        target = self.targets[index]
        image = Image.open(img_path).convert('RGB') 
        img = self.transform(image)
        return {'image':img, 'target':target, 'index':index, 'clean_noisy':self.clean_noisy[index]}
        
    def __len__(self):
        return len(self.data)
