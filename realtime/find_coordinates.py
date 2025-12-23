import mss
import cv2
import numpy as np
monitor = {
    "top": 100,      # Y position from screen top
    "left": 100,     # X position from screen left  
    "width": 960,    # Game window width
    "height": 540    # Game window height
}

sct = mss.mss()
while True:
    screenshot = sct.grab(sct.monitors[1])  # Primary monitor
    img = np.array(screenshot)
    cv2.imshow('Hover mouse over game window', cv2.cvtColor(img, cv2.COLOR_BGRA2BGR))
    
    # Press 's' to print current mouse position
    k = cv2.waitKey(1) & 0xFF
    if k == ord('s'):
        print(f"Mouse at: {cv2.getMouseCallback()}")  # Use print(sct.monitors[1])
    if k == ord('q'):
        break