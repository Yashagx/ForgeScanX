import os
import numpy as np
from pathlib import Path
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import cv2

from models.segmentation.unet import UNet

# === Paths ===
classification_model_path = "models/classification/classification_model.pth"
segmentation_model_path = "models/segmentation/best_segmentation_model.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Classification Transform ===
classification_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# === Segmentation Transform ===
segmentation_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]) #normalizing about 0.229,0.224,0.225
])

# === Load Classification Model ===
classification_model = models.resnet18(pretrained=False)
classification_model.fc = torch.nn.Linear(classification_model.fc.in_features, 2)
classification_model.load_state_dict(torch.load(classification_model_path, map_location=device))
classification_model = classification_model.to(device)
classification_model.eval()

# === Load Segmentation Model ===
segmentation_model = UNet(in_channels=3, out_channels=1)
segmentation_model.load_state_dict(torch.load(segmentation_model_path, map_location=device))
segmentation_model = segmentation_model.to(device)
segmentation_model.eval()

# === Classification Function ===
def classify_image(image_path):
    image_rgb = Image.open(image_path).convert("RGB")
    input_tensor = classification_transform(image_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = classification_model(input_tensor)
        probs = F.softmax(logits, dim=1).cpu().numpy().flatten()
        predicted_class = int(probs.argmax())
        confidence = float(probs[predicted_class])

    label_map = {0: "forged", 1: "unforged"}
    return {
        "class": predicted_class,
        "label": label_map[predicted_class],
        "confidence": confidence
    }

# === Segmentation Function ===
def segment_image(image_path, output_mask_path):
    try:
        image_rgb = Image.open(image_path).convert("RGB")
        original_size = image_rgb.size

        input_tensor = segmentation_transform(image_rgb).unsqueeze(0).to(device)

        with torch.no_grad():
            output = segmentation_model(input_tensor)
            prob_mask = torch.sigmoid(output)
            binary_mask = (prob_mask > 0.5).float()

        mask_tensor = binary_mask.squeeze(0).squeeze(0)
        mask_np = (mask_tensor.cpu().numpy() * 255).astype(np.uint8)

        # Red overlay
        mask_rgb = np.zeros((mask_np.shape[0], mask_np.shape[1], 3), dtype=np.uint8)
        mask_rgb[..., 0] = mask_np  # Red

        mask_img = Image.fromarray(mask_rgb).resize(original_size)
        os.makedirs(os.path.dirname(output_mask_path), exist_ok=True)
        mask_img.save(output_mask_path)

    except Exception as e:
        print(f"[❌] Error in segment_image(): {e}")

# === Copy-Move Forgery Detection using ORB ===
def detect_copy_move_forgery(image_path, output_path):
    try:
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        orb = cv2.ORB_create(nfeatures=1000)
        kp, des = orb.detectAndCompute(gray, None)

        if des is None or len(kp) < 2:
            print("[⚠️] Not enough keypoints detected.")
            return None

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des, des)
        matches = sorted(matches, key=lambda x: x.distance)[:50]

        result_img = img.copy()
        for m in matches:
            pt1 = tuple(map(int, kp[m.queryIdx].pt))
            pt2 = tuple(map(int, kp[m.trainIdx].pt))
            if pt1 != pt2:
                cv2.line(result_img, pt1, pt2, (0, 255, 0), 1)
                cv2.circle(result_img, pt1, 3, (0, 0, 255), -1)
                cv2.circle(result_img, pt2, 3, (255, 0, 0), -1)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, result_img)
        return output_path

    except Exception as e:
        print(f"[❌] Error in detect_copy_move_forgery(): {e}")
        return None

# === Full Prediction Pipeline ===
def run_prediction_pipeline(image_path, output_dir="static/masks"):
    os.makedirs(output_dir, exist_ok=True)
    result = classify_image(image_path)

    filename = Path(image_path).stem
    result["mask_path"] = None
    result["copy_move_path"] = None

    if result["label"] == "forged":
        # 1. Segmentation mask
        mask_filename = f"{filename}_mask.png"
        full_mask_path = os.path.join(output_dir, mask_filename)
        segment_image(image_path, full_mask_path)
        result["mask_path"] = f"masks/{mask_filename}".replace("\\", "/")

        # 2. Keypoint forgery visualization
        forgery_vis_filename = f"{filename}_keypoints.png"
        full_vis_path = os.path.join(output_dir, forgery_vis_filename)
        keypoint_img_path = detect_copy_move_forgery(image_path, full_vis_path)
        if keypoint_img_path:
            result["copy_move_path"] = f"masks/{forgery_vis_filename}".replace("\\", "/")

    return result
