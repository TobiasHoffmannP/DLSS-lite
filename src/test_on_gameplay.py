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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

simplesr = SimpleSR().to(device)
simplesr.load_state_dict(torch.load('models/simpleSR/v1.1/best_simple_sr.pth'))

esrgan = ESRGAN(scale=2, channels=64).to(device)
esrgan.load_state_dict(torch.load('models/esrganSR/best_esrgan_sr.pth', map_location=device))

esrgan.eval()
simplesr.eval()

def test_gameplay_frames_game_size(div2k_path):
    """Test Gameplay frames at GAME resolution (1280x720 → 2560x1440)"""
    # Pick random HR
    hr_files = glob.glob(os.path.join(div2k_path, '**', '*.png'), recursive=True)
    hr_path = np.random.choice(hr_files)
    print(f"Testing GAME SIZE on: {os.path.basename(hr_path)}")
    
    # Load HR
    hr_img = cv2.imread(hr_path)
    hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2RGB)

    # Downscale to game LR size (1280x720)
    lr_game = cv2.resize(hr_img, (1280, 720), interpolation=cv2.INTER_LINEAR)
    
    # Model input tensor
    lr_tensor = torch.from_numpy(lr_game).permute(2, 0, 1).float() / 255.0
    lr_tensor = lr_tensor.unsqueeze(0).to(device)
    
    # Upscale x2 (to 2040, 1356 (native image res))
    with torch.no_grad():
        sr_simple = simplesr(lr_tensor)
        sr_esrgan = esrgan(lr_tensor)

    # Clamp to [0,1]
    sr_simple = torch.clamp(sr_simple, 0.0, 1.0)
    sr_esrgan = torch.clamp(sr_esrgan, 0.0, 1.0)
    
    # Display
    lr_np = (lr_tensor.squeeze().cpu().numpy().transpose(1,2,0) * 255).astype(np.uint8)
    lr_display = cv2.resize(lr_np, (2560, 1440))  # Upscale LR for comparison
    sr_simple_np = (sr_simple.squeeze().cpu().numpy().transpose(1,2,0) * 255).astype(np.uint8)
    sr_esrgan_np = (sr_esrgan.squeeze().cpu().numpy().transpose(1,2,0) * 255).astype(np.uint8)
    hr_display = cv2.resize(hr_img, (2560, 1440), interpolation=cv2.INTER_AREA)

    # convert to RGB
    lr_display = cv2.cvtColor(cv2.resize(lr_np, (2560, 1440)), cv2.COLOR_RGB2BGR)
    sr_simple_display = cv2.cvtColor(sr_simple_np, cv2.COLOR_RGB2BGR)
    sr_esrgan_display = cv2.cvtColor(sr_esrgan_np, cv2.COLOR_RGB2BGR)
    hr_display = cv2.cvtColor(cv2.resize(hr_img, (2560, 1440), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2BGR)

    # 2x2 layout
    top_row = np.hstack([lr_display, hr_display])
    bottom_row = np.hstack([sr_simple_display, sr_esrgan_display])
    combined = np.vstack([top_row, bottom_row])
    
    combined_small = cv2.resize(combined, (combined.shape[1]//2, combined.shape[0]//2))
    
    cv2.imshow("GAME SIZE: LR | HR | SimpleSR | ESRGAN", combined_small)
    print("ESC to close. Now exact game resolution test!")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# RUN TEST
if __name__ == "__main__":
    SCRIPT_DIR = Path(__file__).parent
    DATA_DIR = SCRIPT_DIR.parent / 'data'
    DIV2K_PATH = DATA_DIR / 'gameplay_frames/war_thunder/hr' 
    
    test_gameplay_frames_game_size(DIV2K_PATH)

