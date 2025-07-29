import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from albumentations.pytorch import ToTensorV2
import albumentations as A
from models.segmentation.unet import UNet

# -------- CONFIG --------
MODEL_PATH = r"D:\ForgeScanX\models\segmentation\best_segmentation_model.pth"
TEST_IMAGE_PATH = r"D:\ForgeScanX\data\forgescan_dataset\classification\forged\001_F.png"  # 🔁 change
IMG_SIZE = 256

# -------- DEVICE --------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------- Load Model --------
model = UNet(in_channels=3, out_channels=1)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# -------- Transforms --------
transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

# -------- Load & Preprocess Image --------
original = cv2.imread(TEST_IMAGE_PATH)
original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
image_resized = cv2.resize(original, (IMG_SIZE, IMG_SIZE))

augmented = transform(image=image_resized)
input_tensor = augmented["image"].unsqueeze(0).to(device)  # shape: [1, 3, H, W]

# -------- Predict --------
with torch.no_grad():
    output = model(input_tensor)
    output = torch.sigmoid(output)  # convert logits to probabilities
    mask_pred = (output > 0.5).float()  # threshold to binary
    mask_pred = mask_pred.squeeze().cpu().numpy()

# -------- Visualize --------
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(image_resized)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(mask_pred, cmap="gray")
plt.title("Predicted Binary Mask")
plt.axis("off")

# -------- Overlay Forged Area in RED --------
overlay = image_resized.copy()
overlay[mask_pred > 0.5] = [255, 0, 0]  # Red for forged areas

plt.subplot(1, 3, 3)
plt.imshow(overlay)
plt.title("Red Overlay (Forged)")
plt.axis("off")

plt.tight_layout()
plt.show()
