import os
import cv2
import numpy as np
from tqdm import tqdm

def preprocess_image(image_path, size=(512, 512)):
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    img_resized = cv2.resize(img, size)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    denoised = cv2.GaussianBlur(gray, (5, 5), 0)

    return denoised

def preprocess_directory(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    images = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

    for img_name in tqdm(images, desc=f"Processing {os.path.basename(input_dir)}"):
        input_path = os.path.join(input_dir, img_name)
        output_path = os.path.join(output_dir, img_name)

        processed = preprocess_image(input_path)
        if processed is not None:
            cv2.imwrite(output_path, processed)

if __name__ == "__main__":
    datasets = {
        "CoMoFoD": "data/raw/CoMoFoD",
        "MICC-F220": "data/raw/MICC-F220",
        "MICC-F2000": "data/raw/MICC-F2000"
    }

    for name, path in datasets.items():
        output_path = f"data/processed/{name}"
        preprocess_directory(path, output_path)

    print("\n✅ All datasets preprocessed successfully!")
