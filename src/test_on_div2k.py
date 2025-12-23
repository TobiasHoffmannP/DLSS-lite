import torch
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt
from SimpleSR import SimpleSR 
from esrgan import ESRGAN  
import os
import glob
from pathlib import Path

# Load your models (exact same as realtime_demo)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

simplesr = SimpleSR().to(device)
simplesr.load_state_dict(torch.load('models/simpleSR/v1.1/best_simple_sr.pth'))

# esrgan = ESRGAN(scale=2, num_feat=64, num_block=23, num_grow_ch=32).to(device)
esrgan = ESRGAN(scale=2, channels=64).to(device)
esrgan.load_state_dict(torch.load('models/esrganSR/best_esrgan_sr.pth', map_location=device))

esrgan.eval()
simplesr.eval()

def test_div2k_game_size(div2k_path):
    """Test DIV2K at GAME resolution (640x360 → 1280x720)"""
    # Pick random HR
    hr_files = glob.glob(os.path.join(div2k_path, '**', '*.png'), recursive=True)
    hr_path = np.random.choice(hr_files)
    print(f"Testing GAME SIZE on: {os.path.basename(hr_path)}")
    
    # Load HR
    hr_img = cv2.imread(hr_path)
    hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2RGB)
    
    # SIMULATE YOUR GAME PIPELINE:
    # 1. Downscale to game LR size (640x360)
    lr_game = cv2.resize(hr_img, (640, 360), interpolation=cv2.INTER_LINEAR)
    
    # 2. Model input tensor
    lr_tensor = torch.from_numpy(lr_game).permute(2, 0, 1).float() / 255.0
    lr_tensor = lr_tensor.unsqueeze(0).to(device)
    
    # 3. Upscale x2 (to 1280x720)
    with torch.no_grad():
        sr_simple = simplesr(lr_tensor)
        sr_esrgan = esrgan(lr_tensor)
    
    # 4. Display (all 1280x720)
    lr_np = (lr_tensor.squeeze().cpu().numpy().transpose(1,2,0) * 255).astype(np.uint8)
    lr_display = cv2.resize(lr_np, (1280, 720))  # Upscale LR for comparison
    sr_simple_np = (sr_simple.squeeze().cpu().numpy().transpose(1,2,0) * 255).astype(np.uint8)
    sr_esrgan_np = (sr_esrgan.squeeze().cpu().numpy().transpose(1,2,0) * 255).astype(np.uint8)
    hr_display = cv2.resize(hr_img, (1280, 720), interpolation=cv2.INTER_AREA)
    
    # Perfect 2x2 layout
    top_row = np.hstack([lr_display, hr_display])
    bottom_row = np.hstack([sr_simple_np, sr_esrgan_np])
    combined = np.vstack([top_row, bottom_row])
    
    # Shrink 3x
    combined_small = cv2.resize(combined, (combined.shape[1]//1, combined.shape[0]//1))
    
    cv2.imshow("GAME SIZE: LR | HR | SimpleSR | ESRGAN", combined_small)
    print("ESC to close. Now exact game resolution test!")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# RUN TEST
if __name__ == "__main__":
    # CHANGE THIS TO YOUR DIV2K PATH
    SCRIPT_DIR = Path(__file__).parent
    DATA_DIR = SCRIPT_DIR.parent / 'data'
    DIV2K_PATH = DATA_DIR / 'DIV2K_train_HR/DIV2K_train_HR/'  # or wherever your HR PNGs are
    
    test_div2k_game_size(DIV2K_PATH)

