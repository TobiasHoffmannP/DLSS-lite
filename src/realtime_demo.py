import torch
import torch.nn as nn
from SimpleSR import SimpleSR 
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path
import mss
import time
from torch.amp import autocast
from esrgan import ESRGAN  

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

############## SimpleSR model loading ##############
# model = SimpleSR(
#     scale=2, 
#     num_channels=64
# ).to(device)
# model.load_state_dict(torch.load('models/v1.1/best_simple_sr.pth', map_location=device))
# model.eval()

model = ESRGAN(scale=2, channels=64).to(device)
model.load_state_dict(torch.load('models/esrganSR/best_esrgan_sr.pth', map_location=device))
model.eval()

transform = transforms.Compose([transforms.ToTensor()])

sct = mss.mss()
monitor = {
    "top": 50,      # X pos from screen top
    "left": 50,     # Y pos from screen top
    "width": 1280,  # Game window width 
    "height": 720   # Game window height
}

prev_time = time.time()
frame_count = 0

# print("DLSS-lite ready! PSNR 22.25dB model loaded.")
print("DLSS-lite ready! PSNR 35.64dB model loaded.")
print("Press ctrl + c to quit")

while True:
    screenshot = sct.grab(monitor)
    frame = np.array(screenshot)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)

    lr_frame = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_LINEAR)  

    lr_tensor = torch.from_numpy(lr_frame).permute(2, 0, 1).float() / 255.0
    lr_tensor = lr_tensor.unsqueeze(0).to(device, non_blocking=True)

    with torch.no_grad():
        sr_tensor = model(lr_tensor)
        sr_frame = sr_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()
        sr_frame = np.clip(sr_frame * 255, 0, 255).astype(np.uint8)

   

    

    curr_time = time.time()
    frame_count += 1
    if curr_time - prev_time >= 1.0:  # Print FPS every second
        print(f"60 FPS target | Actual: {frame_count:.1f} FPS")
        frame_count = 0
        prev_time = curr_time

    #cv2.imshow("DLSS-lite: LR (640x360) | SR (1280x720)  (Press Q to quit)", combined)
    #lr_display = cv2.resize(lr_frame, (1280, 720))      # Small LR: 640x360
    bicubic_sr = cv2.resize(lr_frame, (1280, 720), interpolation=cv2.INTER_CUBIC)
    model_sr  = cv2.resize(sr_frame, (1280, 720))    # Full SR: 1280x720
    combined = np.hstack([
        cv2.cvtColor(bicubic_sr, cv2.COLOR_RGB2BGR),
        cv2.cvtColor(model_sr, cv2.COLOR_RGB2BGR)
    ])
    cv2.imshow("Bicubic | Your Model", combined)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:  # q or ESC
        print("Exiting DLSS-lite...")
        break

cv2.destroyAllWindows()
cv2.waitKey(1)  
print("DLSS-lite closed successfully!")