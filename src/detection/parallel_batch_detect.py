import os
import cv2
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

PREPROC_BASE = "data/preprocessed"
RESULTS_BASE = "data/results"

def process_image(args):
    """Detect forgery in a single image and save the result."""
    img_path, output_path, debug = args

    try:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return f"❌ Skipping (cannot read): {os.path.basename(img_path)}"

        sift = cv2.SIFT_create()
        kp, desc = sift.detectAndCompute(img, None)

        if desc is None:
            return f"⚠️ No descriptors in: {os.path.basename(img_path)}"

        matcher = cv2.BFMatcher()
        matches = matcher.knnMatch(desc, desc, k=2)

        good = []
        for m, n in matches:
            if debug:
                if m.queryIdx != m.trainIdx:
                    good.append(m)
            else:
                if m.distance < 0.75 * n.distance and m.queryIdx != m.trainIdx:
                    good.append(m)

        if len(good) < 4:
            return f"⚠️ Skipping {os.path.basename(img_path)}: Not enough good matches"

        src_pts = np.float32([kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if mask is None:
            return f"⚠️ Skipping {os.path.basename(img_path)}: RANSAC failed"

        inliers = mask.ravel().tolist()
        result_img = cv2.drawMatches(img, kp, img, kp, good, None,
                                     matchColor=(0, 255, 0),
                                     matchesMask=inliers,
                                     singlePointColor=None, flags=2)

        cv2.imwrite(output_path, result_img)
        return f"✅ Saved: {os.path.basename(output_path)}"

    except Exception as e:
        return f"❌ Error in {os.path.basename(img_path)}: {str(e)}"


def parallel_batch_detect(dataset_name="CoMoFoD", limit=None, debug=True):
    """Parallel batch detection for a given dataset."""
    input_dir = os.path.join(PREPROC_BASE, dataset_name)
    output_dir = os.path.join(RESULTS_BASE, dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg')) and "_f_" in f.lower()]
    if limit:
        files = files[:limit]

    print(f"\n🚀 Running parallel batch detection on {len(files)} forged images from {dataset_name}...\n")

    args_list = []
    for fname in files:
        img_path = os.path.join(input_dir, fname)
        out_path = os.path.join(output_dir, fname)
        args_list.append((img_path, out_path, debug))

    with Pool(processes=cpu_count()) as pool:
        for msg in tqdm(pool.imap_unordered(process_image, args_list), total=len(args_list)):
            print(msg)


if __name__ == "__main__":
    # Example usage
    parallel_batch_detect("CoMoFoD", limit=100, debug=True)
