import cv2
import os
import numpy as np
from matplotlib import pyplot as plt

def detect_forgery(image_path, show=True):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Step 1: Detect SIFT keypoints & descriptors
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(img, None)
    
    # Step 2: Brute-Force matcher (could also use FLANN)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors, descriptors, k=2)

    # Step 3: Filter matches using Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance and m.queryIdx != m.trainIdx:
            good_matches.append(m)

    # Step 4: Get matched keypoint coordinates
    src_pts = np.float32([keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Step 5: Apply RANSAC to filter geometric inliers
    if len(good_matches) > 4:
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        inliers = mask.ravel().tolist()
    else:
        print("⚠️ Not enough matches for RANSAC")
        return

    # Step 6: Draw matches
    img_matches = cv2.drawMatches(img, keypoints, img, keypoints, good_matches, None,
                                  matchColor=(0, 255, 0), singlePointColor=None,
                                  matchesMask=inliers, flags=2)

    if show:
        plt.figure(figsize=(12, 6))
        plt.title("Detected Forgery (SIFT + RANSAC)")
        plt.imshow(img_matches, cmap='gray')
        plt.axis('off')
        plt.show()

if __name__ == "__main__":
    # Example image (from preprocessed CoMoFoD)
    sample_path = "data/preprocessed/CoMoFoD/Sp_T_1001.png"
    detect_forgery(sample_path)
