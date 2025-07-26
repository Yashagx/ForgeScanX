import os
import cv2
from tqdm import tqdm

# Set constants
TARGET_SIZE = (512, 512)
DATASETS = ['CoMoFoD', 'MICC-F220', 'MICC-F2000']
RAW_BASE = 'data/raw'
PREPROC_BASE = 'data/preprocessed'

def preprocess_image(image_path):
    """Load, convert, resize, denoise, and return the image."""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, TARGET_SIZE)
    denoised = cv2.GaussianBlur(resized, (3, 3), 0)
    return denoised

def preprocess_dataset(dataset_name):
    input_dir = os.path.join(RAW_BASE, dataset_name)
    output_dir = os.path.join(PREPROC_BASE, dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n⚙️ Preprocessing: {dataset_name}")
    count = 0

    for filename in tqdm(os.listdir(input_dir)):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            in_path = os.path.join(input_dir, filename)
            out_path = os.path.join(output_dir, os.path.splitext(filename)[0] + '.png')
            try:
                processed = preprocess_image(in_path)
                cv2.imwrite(out_path, processed)
                count += 1
            except Exception as e:
                print(f"❌ Failed: {filename} — {e}")

    print(f"✅ Saved {count} images to {output_dir}")

if __name__ == "__main__":
    for dataset in DATASETS:
        preprocess_dataset(dataset)
