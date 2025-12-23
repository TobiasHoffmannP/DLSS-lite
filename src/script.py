from div2k_dataset import DIV2KDataset
import os
from PIL import Image
import torch 
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from pathlib import Path

transform = transforms.Compose([
    transforms.ToTensor()
])

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / 'data'

lr_dir = DATA_DIR / 'DIV2K_train_LR_bicubic_X2/DIV2K_train_LR_bicubic/X2/'
hr_dir = DATA_DIR / 'DIV2K_train_HR/DIV2K_train_HR/'

print(f"Using LR: {lr_dir}")
print(f"Using HR: {hr_dir}")

train_ds = DIV2KDataset(
    lr_dir=str(lr_dir),     
    hr_dir=str(hr_dir),
    crop_size=128,
    transform=transform
)

print(f"LR files: {len(train_ds.lr_files)}")
print(f"HR files: {len(train_ds.hr_files)}")
print(f"Pairs found: {len(train_ds.pairs)}")
print("Sample LR:", train_ds.lr_files[:2])
print("Sample HR:", train_ds.hr_files[:2])

train_loader = torch.utils.data.DataLoader(
    train_ds, 
    batch_size=16, 
    shuffle=True, 
    num_workers=0
)

lr, hr = train_ds[0]
print(f"Single sample shapes: LR {lr.shape}, HR {hr.shape}")
print(f"LR range: [{lr.min():.3f}, {lr.max():.3f}]")
print(f"HR range: [{hr.min():.3f}, {hr.max():.3f}]")

train_loader = torch.utils.data.DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0)
for lr_batch, hr_batch in train_loader:
    print(f"✅ Batch shapes: LR {lr_batch.shape}, HR {hr_batch.shape}")
    break

