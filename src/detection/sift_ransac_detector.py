import cv2
import os
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

# Constants
PREPROC_BASE = "data/preprocessed"
RESULTS_BASE = "data/results"

def detect_forgery(image_path, show=True, debug=False):
    """Run SIFT + RANSAC forgery detection on a single image."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"❌ Could not load image: {image_path}")
        return

    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(img, None)

    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(descriptors, descriptors, k=2)

    # Step 2: Apply Lowe’s ratio test or debug mode
    good_matches = []
    for m, n in matches:
        if debug:
            if m.queryIdx != m.trainIdx:
                good_matches.append(m)
        else:
            if m.distance < 0.75 * n.distance and m.queryIdx != m.trainIdx:
                good_matches.append(m)

    if len(good_matches) < 4:
        print("⚠️ Not enough good matches for RANSAC")
        return

    src_pts = np.float32([keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if mask is None:
        print("⚠️ RANSAC failed to find a valid homography")
        return

    inliers = mask.ravel().tolist()
    matched_img = cv2.drawMatches(img, keypoints, img, keypoints, good_matches, None,
                                  matchColor=(0, 255, 0), matchesMask=inliers,
                                  singlePointColor=None, flags=2)

    if show:
        plt.figure(figsize=(12, 6))
        plt.title("Forgery Detection (SIFT + RANSAC)")
        plt.imshow(matched_img, cmap='gray')
        plt.axis('off')
        plt.show()


def batch_detect(dataset_name, limit=None, debug=False):
    """Batch detect forgery in only forged images and save results."""
    input_dir = os.path.join(PREPROC_BASE, dataset_name)
    output_dir = os.path.join(RESULTS_BASE, dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    # Only forged images (filename contains '_F_')
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg')) and '_f_' in f.lower()]
    if limit:
        files = files[:limit]

    print(f"\n📁 Batch Processing {len(files)} forged images from {dataset_name}...\n")

    for fname in tqdm(files):
        img_path = os.path.join(input_dir, fname)
        output_path = os.path.join(output_dir, fname)

        try:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"❌ Skipping (cannot read): {fname}")
                continue

            sift = cv2.SIFT_create()
            kp, desc = sift.detectAndCompute(img, None)

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
                print(f"⚠️ Skipping {fname}: Not enough good matches")
                continue

            src_pts = np.float32([kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if mask is None:
                print(f"⚠️ Skipping {fname}: RANSAC failed")
                continue

            inliers = mask.ravel().tolist()
            result_img = cv2.drawMatches(img, kp, img, kp, good, None,
                                         matchColor=(0, 255, 0),
                                         matchesMask=inliers,
                                         singlePointColor=None, flags=2)
            cv2.imwrite(output_path, result_img)
            print(f"✅ Saved: {output_path}")

        except Exception as e:
            print(f"❌ Error in {fname}: {e}")


if __name__ == "__main__":
    # 🔍 Test single image if needed
    # detect_forgery("data/preprocessed/CoMoFoD/001_F_BC1.png", show=True, debug=True)

    # 🔁 Batch process forged images from CoMoFoD
    batch_detect("CoMoFoD", limit=100, debug=True)

    # 🔁 Try MICC dataset
    # batch_detect("MICC-F220", limit=100, debug=True)
