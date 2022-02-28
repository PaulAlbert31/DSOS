import sys
from torch.utils.data import Dataset, DataLoader
import torch
import torchvision
import json
import os
from tqdm import tqdm
from PIL import Image
import torch.nn.functional as F
import torchnet
from mypath import Path

PATH_TO_IN = Path.db_root_dir("imagenet_val")
PATH_TO_LABELS = "val_labels_webvision"

class imagenet_dataset(Dataset):
    def __init__(self, transform):
        self.root = PATH_TO_IN
        self.transform = transform
        self.val_labels = PATH_TO_LABELS
        self.val_data = []
        d = torch.load(PATH_TO_LABELS)
        for c, k in enumerate(d.keys()):
            imgs = os.listdir(self.root+str(d[k]))
            for img in imgs:
                self.val_data.append([c,os.path.join(self.root,str(d[k]),img)])                
                
    def __getitem__(self, index):
        data = self.val_data[index]
        target = data[0]
        image = Image.open(data[1]).convert('RGB')   
        img = self.transform(image) 
        return img, target
    
    def __len__(self):
        return len(self.val_data)
    
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(227),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean, std),
])

dataset = imagenet_dataset(transforms)
dataloader = DataLoader(dataset, batch_size=50, shuffle=False, num_workers=4, pin_memory=True)

from nets.inceptionresnetv2 import InceptionResNetV2
model = InceptionResNetV2(num_classes=50)
load_dict = torch.load(sys.argv[1])
model.load_state_dict(load_dict['state_dict'])
model.cuda()
model.eval()

ensemble = False
try:
    sys.argv[2]
    ensemble = True
except:pass

if ensemble:
    model2 = InceptionResNetV2(num_classes=50)
    load_dict = torch.load(sys.argv[2])
    model2.load_state_dict(load_dict['state_dict'])
    model2.eval()

acc = 0
vbar = tqdm(dataloader)
total = 0
losses, accs = torch.tensor([]), torch.tensor([])
accmeter = torchnet.meter.ClassErrorMeter(topk=[1,5], accuracy=True)
with torch.no_grad():
    for i, sample in enumerate(vbar):
        image, target = sample[0], sample[1]

        image, target = image.cuda(), target.cuda()

        outputs = model(image)
        if ensemble:
            model.cpu()
            model2.cuda()
            outputs += model2(image)
            model2.cpu()
            model.cuda()

        topk = torch.topk(F.log_softmax(outputs, dim=1), 5)[1]
        
        preds = torch.argmax(F.log_softmax(outputs, dim=1), dim=1)
        accs = torch.cat((accs, (preds==target.data).float().cpu()))

        accmeter.add(outputs,target)
        
        acc += torch.sum(preds == target.data)
        total += preds.size(0)

    final_acc = float(acc)/total
    
print(accmeter.value())
print('Validation Accuracy on ImageNet val set: {0:.4f}'.format(final_acc))


from datasets.webvision import webvision_dataset
testset = webvision_dataset(transform=transforms, mode="test", num_class=50)
dataloader = DataLoader(testset, batch_size=50, shuffle=False, num_workers=4, pin_memory=True)

acc = 0
vbar = tqdm(dataloader)
total = 0
losses, accs = torch.tensor([]), torch.tensor([])
accmeter = torchnet.meter.ClassErrorMeter(topk=[1,5], accuracy=True)
with torch.no_grad():
    for i, sample in enumerate(vbar):
        image, target = sample['image'], sample['target']

        image, target = image.cuda(), target.cuda()

        outputs = model(image)
        if ensemble:
            model.cpu()
            model2.cuda()
            outputs += model2(image)
            model2.cpu()
            model.cuda()

        topk = torch.topk(F.log_softmax(outputs, dim=1), 5)[1]
        
        preds = torch.argmax(F.log_softmax(outputs, dim=1), dim=1)
        accs = torch.cat((accs, (preds==target.data).float().cpu()))

        accmeter.add(outputs,target)
        
        acc += torch.sum(preds == target.data)
        total += preds.size(0)

    final_acc = float(acc)/total
    
print(accmeter.value())
print('Validation Accuracy on Webvision val set: {0:.4f}'.format(final_acc))


