import os
from PIL import Image
import torch 
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import random

class DIV2KDataset(Dataset):
    def __init__(self, lr_dir, hr_dir, crop_size=128, transform=None):
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.crop_size = crop_size
        self.transform = transform 

        # generate pairs of LR and HR (LR: 002x2.png - HR: 002.png)
        self.lr_files = sorted([f for f in os.listdir(lr_dir) if f.endswith(".png")])
        self.hr_files = sorted([f for f in os.listdir(hr_dir) if f.endswith(".png")])

        self.pairs = []
        for lr_file in self.lr_files:
            hr_file = lr_file.replace("2x", "")
            if hr_file in self.hr_files:
                self.pairs.append((lr_file, hr_file))

    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, index):
        lr_path = os.path.join(self.lr_dir, self.pairs[index][0])
        hr_path = os.path.join(self.hr_dir, self.pairs[index][1])

        lr_img = Image.open(lr_path).convert("RGB")
        hr_img = Image.open(hr_path).convert("RGB")

        crop_size_hr = self.crop_size
        crop_size_lr = self.crop_size // 2
        
        seed = index
        random.seed(seed)
        i = random.randint(0, hr_img.height - crop_size_hr)
        j = random.randint(0, hr_img.width - crop_size_hr)
        
        hr_img = transforms.functional.crop(hr_img, i, j, crop_size_hr, crop_size_hr)
        lr_img = transforms.functional.crop(lr_img, i // 2, j // 2, crop_size_lr, crop_size_lr)

        if self.transform:
            lr_img = self.transform(lr_img)
            hr_img = self.transform(hr_img)

        return lr_img, hr_img
    
    