import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from div2k_dataset import DIV2KDataset
from pathlib import Path
import os
import torchvision.transforms as transforms
import torchvision.models as models

class DenseBlock(nn.Module):
    def __init__(self, channels=64, growth_channels=32):
        super().__init__()
        # Dense Block: inputs grow in channel size at each step
        self.conv1 = nn.Conv2d(channels, growth_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(channels + growth_channels, growth_channels, 3, 1, 1)
        self.conv3 = nn.Conv2d(channels + 2 * growth_channels, growth_channels, 3, 1, 1)
        self.conv4 = nn.Conv2d(channels + 3 * growth_channels, growth_channels, 3, 1, 1)
        self.conv5 = nn.Conv2d(channels + 4 * growth_channels, channels, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        # Valid Dense Block scales the output by 0.2
        return x5 * 0.2 + x

class RRDBBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # RRDB contains 3 internal Dense Blocks
        self.dense1 = DenseBlock(channels)
        self.dense2 = DenseBlock(channels)
        self.dense3 = DenseBlock(channels)

    def forward(self, x):
        out = self.dense1(x)
        out = self.dense2(out)
        out = self.dense3(out)
        # Residual-in-Residual scaling (beta=0.2)
        return out * 0.2 + x

# ESRGAN Generator (23 RRDB blocks!)
class ESRGAN(nn.Module):
    def __init__(self, scale=2, channels=64):
        super().__init__()
        self.scale = scale
        
        # Shallow feature extraction
        self.head = nn.Sequential(
            nn.Conv2d(3, channels, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # 23 RRDB blocks 
        rrdb_blocks = []
        for _ in range(23):
            rrdb_blocks.append(RRDBBlock(channels))
        self.body = nn.Sequential(*rrdb_blocks)
        
        # Upsampling
        # self.upscale = nn.Sequential(
        #     nn.Conv2d(channels, channels * scale * scale, 3, 1, 1),
        #     nn.PixelShuffle(scale)
        # )
        self.upscale = nn.Sequential(
            nn.Upsample(scale_factor=scale, mode='nearest'),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True) # Optional, depends on specific variant
        )
        
        # Reconstruction
        self.tail = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels // 2, 3, 3, 1, 1)
        )
    
    def forward(self, x):
        feat = self.head(x)
        body_out = self.body(feat)
        out = body_out + feat  # Long skip!
        out = self.upscale(out)
        out = self.tail(out)
        return out
    
class VGUloss(nn.Module):
    def __init__(self, layer_ids=None, use_input_norm=True):
        super().__init__()
        if layer_ids is None:
            layer_ids = [34] 

        self.layer_ids = layer_ids
        self.use_input_norm = use_input_norm 

        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
        self.vgg_layers = vgg.features

        for param in self.parameters():
            param.requires_grad = False

        self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, sr, hr):
        if self.use_input_norm:
            sr = (sr - self.mean) / self.std
            hr = (hr - self.mean) / self.std

        sr_features = []
        hr_features = []
        
        x_sr = sr
        x_hr = hr

        for i, layer in enumerate(self.vgg_layers):
            x_sr = layer(x_sr)
            x_hr = layer(x_hr)
            
            if i in self.layer_ids:
                sr_features.append(x_sr)
                hr_features.append(x_hr)
                
            # Stop early if we have all needed features
            if i >= max(self.layer_ids):
                break

        loss = 0
        for f_sr, f_hr in zip(sr_features, hr_features):
            loss += nn.functional.l1_loss(f_sr, f_hr)
            
        return loss
    
class Discriminator(nn.Module):
    def __init__(self, input_shape=(3, 128, 128)):
        super().__init__()
        
        self.input_shape = input_shape
        in_channels, in_height, in_width = input_shape
        patch_h, patch_w = in_height // 16, in_width // 16

        def discriminator_block(in_filters, out_filters, first_block=False):
            layers = []
            layers.append(nn.Conv2d(in_filters, out_filters, 3, 1, 1, bias=False))
            layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Conv2d(out_filters, out_filters, 3, 2, 1, bias=False)) # Downsample
            layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
    
        layers = []

        layers.append(nn.Conv2d(in_channels, 64, 3, 1, 1))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        layers.append(nn.Conv2d(64, 64, 3, 2, 1)) # Downsample -> 64x64
        layers.append(nn.BatchNorm2d(64))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        # Contracting blocks
        layers.extend(discriminator_block(64, 128))   # -> 32x32
        layers.extend(discriminator_block(128, 256))  # -> 16x16
        layers.extend(discriminator_block(256, 512))  # -> 8x8

        self.feature_extraction = nn.Sequential(*layers)

        self.classifier = nn.Sequential(
            nn.Linear(512 * 8 * 8, 100),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(100, 1) # Output raw score (logits)
        )

    def forward(self, x):
        features = self.feature_extraction(x)
        features = features.view(features.size(0), -1) # Flatten
        validity = self.classifier(features)
        return validity

def perceptual_loss(sr, hr):
    return torch.mean(((sr - hr) ** 2) * 0.1 + torch.mean(torch.abs(sr - hr))) # L1 + mse

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

    MODEL_DIR = SCRIPT_DIR.parent / 'models/esrganSR'
    os.makedirs(str(MODEL_DIR), exist_ok=True)

    # print(f"LR directory: {lr_dir}\nHR directory: {hr_dir}")
    # print(f"Model save location: {MODEL_DIR}")

    transform = transforms.Compose([transforms.ToTensor()])
    train_ds = DIV2KDataset(
        lr_dir=str(lr_dir),     
        hr_dir=str(hr_dir),
        crop_size=128,
        transform=transform
    )
    train_loader = torch.utils.data.DataLoader(
        train_ds, 
        batch_size=4, 
        shuffle=True, 
        num_workers=0
    )


    # ESRGAN model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ESRGAN(scale=2, channels=64).to(device)

    vgg_loss = VGUloss().to(device)
    discriminator = Discriminator(input_shape=(3, 128, 128)).to(device)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.9, 0.999))
    criterion_GAN = nn.BCEWithLogitsLoss().to(device)

    print(f"ESRGAN training on {device}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    # optimizers 
    optimizer_g = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999))

    # training (GAN-ish but simple for speed)
    best_psnr = 0
    epochs = 100

    # Warmup generator with L1 loss (generator needs to be trained a bit first otherwise it is too hard for discriminator)
    if not os.path.exists(MODEL_DIR / 'warmup_generator.pth'):
        print("Starting Warmup (L1 Loss only)...")
        for epoch in range(5):
            model.train()
            for lr, hr in train_loader:
                lr, hr = lr.to(device), hr.to(device)
                optimizer_g.zero_grad()
                sr = model(lr)
                loss = nn.functional.l1_loss(sr, hr)
                loss.backward()
                optimizer_g.step()
            print(f"Warmup {epoch+1} done.")
        torch.save(model.state_dict(), str(MODEL_DIR / 'warmup_generator.pth'))
    else:
        print("Loading warmed up weights...")
        model.load_state_dict(torch.load(str(MODEL_DIR / 'warmup_generator.pth')))

    print("Warmup complete. Starting GAN training...")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        for lr_batch, hr_batch in train_loader:
            lr_batch, hr_batch = lr_batch.to(device), hr_batch.to(device)
            sr_batch = model(lr_batch)
            
            optimizer_D.zero_grad()
            

            pred_real = discriminator(hr_batch)
            pred_fake = discriminator(sr_batch.detach())

            # Relativistic Average GAN (RaGAN) Logic:
            # Real should be "more real" than Fake average
            loss_real = criterion_GAN(pred_real - torch.mean(pred_fake), torch.ones_like(pred_real))
            # Fake should be "less real" than Real average
            loss_fake = criterion_GAN(pred_fake - torch.mean(pred_real), torch.zeros_like(pred_fake))

            loss_D = (loss_real + loss_fake) / 2
            loss_D.backward()
            optimizer_D.step()


            optimizer_g.zero_grad()

            # Recalculate validity for generator update (grad flows this time)
            pred_real = discriminator(hr_batch.detach()) # Detach real, we don't update D here
            pred_fake = discriminator(sr_batch)          # SR needs grad

            # Generator wants to fool D:
            # Fake should look "more real" than Real average
            loss_G_Ra = criterion_GAN(pred_fake - torch.mean(pred_real), torch.ones_like(pred_fake))
            loss_G_Ra_Inverse = criterion_GAN(pred_real - torch.mean(pred_fake), torch.zeros_like(pred_real))
            loss_GAN = (loss_G_Ra + loss_G_Ra_Inverse) / 2

            # Calculate Feature Loss
            loss_vgg = vgg_loss(sr_batch, hr_batch)
            loss_l1 = nn.functional.l1_loss(sr_batch, hr_batch)

            total_loss_G = 0.005 * loss_GAN + 1.0 * loss_vgg + 0.01 * loss_l1

            total_loss_G.backward()
            optimizer_g.step()

            epoch_loss += total_loss_G.item()

        avg_loss = epoch_loss / len(train_loader)
    
        # PSNR validation
        model.eval()
        with torch.no_grad():
            lr_val, hr_val = next(iter(train_loader)) 
            lr_val, hr_val = lr_val[0:1].to(device), hr_val[0:1].to(device)  
            sr_val = model(lr_val)
            val_psnr = psnr(sr_val, hr_val)
        
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | PSNR: {val_psnr:.2f}dB")
        
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save(model.state_dict(), str(MODEL_DIR / 'best_esrgan_sr.pth'))
            print(f"  -> New best PSNR: {best_psnr:.2f}dB, saved!")

    print(f"ESRGAN training complete! Saved {MODEL_DIR / 'best_esrgan_sr.pth'}")

if __name__ == "__main__":
    os.makedirs('models', exist_ok=True)
    main()