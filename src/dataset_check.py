import os
from PIL import Image

DATASETS = ['CoMoFoD', 'MICC-F220', 'MICC-F2000']  # Dataset names
base_path = 'data/raw'  # Update if you use a different folder

for dataset in DATASETS:
    folder = os.path.join(base_path, dataset)
    image_count = 0
    resolutions = set()
    formats = set()

    if not os.path.exists(folder):
        print(f"[!] Folder not found: {folder}")
        continue

    for file in os.listdir(folder):
        if file.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp')):
            image_count += 1
            try:
                with Image.open(os.path.join(folder, file)) as img:
                    resolutions.add(img.size)
                    formats.add(img.format)
            except Exception as e:
                print(f"Failed to read {file}: {e}")

    print(f"\n📂 Dataset: {dataset}")
    print(f"🖼️ Total Images: {image_count}")
    print(f"🧩 Resolutions found: {resolutions}")
    print(f"📸 Image formats: {formats}")
