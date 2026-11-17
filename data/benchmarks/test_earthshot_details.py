import numpy as np
import cv2
from PIL import Image

p_earth = r"C:\Users\singh\.gemini\antigravity\brain\665a9fec-00da-48fe-b182-dbe26c3b7721\.user_uploaded\media_1794938867451_a73fa26d.jpg"
p_nadia = r"C:\Users\singh\.gemini\antigravity\brain\665a9fec-00da-48fe-b182-dbe26c3b7721\.user_uploaded\media_1794937346098_9eed34f1.jpg"

img_e = Image.open(p_earth).convert('RGB')
img_n = Image.open(p_nadia).convert('RGB')

arr_e = np.asarray(img_e, dtype=np.float32)
arr_n = np.asarray(img_n, dtype=np.float32)

print("EARTHSHOT:", arr_e.shape, "mean:", np.mean(arr_e, axis=(0,1)), "std:", np.std(arr_e, axis=(0,1)))
print("NADIA:    ", arr_n.shape, "mean:", np.mean(arr_n, axis=(0,1)), "std:", np.std(arr_n, axis=(0,1)))

# Let's inspect the dark background in Earthshot:
# The backdrop behind Prince William and the children is a dark grey wall (RGB ~ 50, 52, 60).
# In a real camera, a dark wall at ISO 800/1600 has significant sensor shot noise.
dark_mask_e = (arr_e[:,:,0] < 80) & (arr_e[:,:,1] < 80) & (arr_e[:,:,2] < 90)
print("Earthshot dark wall pixels:", np.sum(dark_mask_e))
if np.sum(dark_mask_e) > 1000:
    # High-pass residual on the dark wall
    gray_e = cv2.cvtColor(np.asarray(img_e), cv2.COLOR_RGB2GRAY).astype(np.float32)
    k = np.array([[-1,2,-2,2,-1],[2,-6,8,-6,2],[-2,8,-12,8,-2],[2,-6,8,-6,2],[-1,2,-2,2,-1]], dtype=np.float32)/12.0
    res_e = cv2.filter2D(gray_e, -1, k)
    print("Earthshot dark wall noise std:", np.std(res_e[dark_mask_e]))

# In Nadia, dark suit jacket (RGB < 60):
dark_mask_n = (arr_n[:,:,0] < 60) & (arr_n[:,:,1] < 60) & (arr_n[:,:,2] < 60)
gray_n = cv2.cvtColor(np.asarray(img_n), cv2.COLOR_RGB2GRAY).astype(np.float32)
res_n = cv2.filter2D(gray_n, -1, k)
print("Nadia dark clothing noise std:", np.std(res_n[dark_mask_n]))

