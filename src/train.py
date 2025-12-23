import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from div2k_dataset import DIV2KDataset
from SimpleSR import SimpleSR 
import torchvision.transforms as transforms
from pathlib import Path
import numpy as np
from PIL import Image
import os

def psnr(img1, img2): # Peak Signal to Noise Ratio
    # https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def main():
    # paths
    SCRIPT_DIR = Path(__file__).parent
    DATA_DIR = SCRIPT_DIR.parent / 'data'
    lr_dir = DATA_DIR / 'DIV2K_train_LR_bicubic_X2/DIV2K_train_LR_bicubic/X2/'
    hr_dir = DATA_DIR / 'DIV2K_train_HR/DIV2K_train_HR/'

    # data prep
    transform = transforms.Compose([transforms.ToTensor()])
    train_ds = DIV2KDataset(
        lr_dir=str(lr_dir),     
        hr_dir=str(hr_dir),
        crop_size=128,
        transform=transform
    )
    train_loader = torch.utils.data.DataLoader(
        train_ds, 
        batch_size=16, 
        shuffle=True, 
        num_workers=0
    )

    # model, loss, optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleSR(
        scale=2,
        num_channels=64 
    ).to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(
        model.parameters(), 
        lr=1e-4
    )
    scaler = torch.amp.GradScaler("cuda") # mixed precision

    print(f"Training on {device}")
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Dataset size: {len(train_ds)}")

    num_epochs = 400
    best_psnr = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        num_batches = 0

        for lr_batch, hr_batch in train_loader:
            lr_batch = lr_batch.to(device, non_blocking=True)
            hr_batch = hr_batch.to(device, non_blocking=True)

            optimizer.zero_grad()

            with torch.amp.autocast(device_type="cuda"):
                sr_batch = model(lr_batch)
                loss = criterion(sr_batch, hr_batch)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            num_batches += 1
        
        avg_loss = train_loss / num_batches

        model.eval()
        with torch.no_grad():
            lr_val, hr_val = train_ds[-1] # last sample
            lr_val = lr_val.unsqueeze(0).to(device)
            sr_val = model(lr_val)
            val_psnr = psnr(sr_val, hr_val.unsqueeze(0).to(device))

        print(f"Epoch {epoch+1:2d}/{num_epochs} | Loss: {avg_loss:.4f} | PSNR: {val_psnr:.2f}dB")

        MODEL_DIR = SCRIPT_DIR.parent / 'models'
        os.makedirs(str(MODEL_DIR), exist_ok=True)
        
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save(model.state_dict(), str(MODEL_DIR / 'best_simple_sr.pth'))
            print(f"  -> New best PSNR: {best_psnr:.2f}dB, saved to { MODEL_DIR / 'best_simple_sr.pth' }")
    
    print(f"\nTraining complete! Best PSNR: {best_psnr:.2f}dB")
    print("Model saved as 'models/best_simple_sr.pth'")

if __name__ == "__main__":
    main()






