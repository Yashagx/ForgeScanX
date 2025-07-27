import os
import cv2
import numpy as np
from tqdm import tqdm

# Parameters
BLOCK_SIZE = 8
QUANTIZATION = 16
TSIMILARITY = 5     # Euclidean distance between DCT descriptors
TDISTANCE = 20      # Minimum pixel distance between blocks
VECTOR_LIMIT = 20   # Minimum shift vector frequency

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def sift_detect(img, gray, min_inliers=5):
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(gray, None)

    if des is None or len(kp) < 2:
        return None, None

    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    try:
        matches = flann.knnMatch(des, des, k=2)
    except:
        return None, None

    good = []
    for m, n in matches:
        if m.queryIdx != m.trainIdx and m.distance < 0.85 * n.distance:
            pt1 = kp[m.queryIdx].pt
            pt2 = kp[m.trainIdx].pt
            if np.linalg.norm(np.array(pt1) - np.array(pt2)) > 10:
                good.append(m)

    if len(good) < min_inliers:
        return None, None

    src_pts = np.float32([kp[m.queryIdx].pt for m in good])
    dst_pts = np.float32([kp[m.trainIdx].pt for m in good])
    M, mask = cv2.estimateAffine2D(src_pts, dst_pts, ransacReprojThreshold=5.0)

    if M is None or mask is None:
        return None, None

    inliers = mask.ravel().tolist()
    if sum(inliers) < min_inliers:
        return None, None

    binary = np.zeros(gray.shape, dtype=np.uint8)
    vis = img.copy()
    for i, m in enumerate(good):
        if inliers[i]:
            pt1 = tuple(np.round(kp[m.queryIdx].pt).astype(int))
            pt2 = tuple(np.round(kp[m.trainIdx].pt).astype(int))
            cv2.circle(binary, pt1, 5, 255, -1)
            cv2.circle(binary, pt2, 5, 255, -1)
            cv2.line(vis, pt1, pt2, (0, 255, 0), 1)
    return binary, vis

def dct_detect(gray):
    h, w = gray.shape
    arr = np.array(gray)
    prediction_mask = np.zeros((h, w), dtype=np.uint8)
    dcts = []

    # Scan blocks
    for i in range(0, h - BLOCK_SIZE):
        for j in range(0, w - BLOCK_SIZE):
            patch = arr[i:i + BLOCK_SIZE, j:j + BLOCK_SIZE]
            imf = np.float32(patch) / 255.0
            dct = cv2.dct(imf)
            flat = []
            for k in range(BLOCK_SIZE + BLOCK_SIZE - 1):
                for l in range(BLOCK_SIZE):
                    m = k - l
                    if 0 <= m < BLOCK_SIZE:
                        flat.append(dct[l][m])
            flat = np.array(flat[:16]) // QUANTIZATION
            flat = np.append(flat, [i, j])
            dcts.append(flat)

    dcts = np.array(dcts)
    dcts = dcts[np.lexsort(np.rot90(dcts))]

    # Similarity matching
    sim_array = []
    for i in range(len(dcts) - 10):
        for j in range(i + 1, i + 10):
            if j >= len(dcts): break
            pixelsim = np.linalg.norm(dcts[i][:16] - dcts[j][:16])
            pointdis = np.linalg.norm(dcts[i][16:] - dcts[j][16:])
            if pixelsim <= TSIMILARITY and pointdis >= TDISTANCE:
                sim_array.append([
                    *dcts[i][16:], *dcts[j][16:],  # locations
                    dcts[i][16] - dcts[j][16],     # dx
                    dcts[i][17] - dcts[j][17]      # dy
                ])

    # Eliminate rare vectors
    sim_array = np.array(sim_array)
    vectors = sim_array[:, 4:6]
    vector_freq = {}
    for v in map(tuple, vectors):
        vector_freq[v] = vector_freq.get(v, 0) + 1
    filtered = np.array([s for s in sim_array if vector_freq[tuple(s[4:6])] >= VECTOR_LIMIT])

    # Paint result
    for s in filtered:
        for i in range(BLOCK_SIZE):
            for j in range(BLOCK_SIZE):
                try:
                    prediction_mask[int(s[0]) + i, int(s[1]) + j] = 255
                    prediction_mask[int(s[2]) + i, int(s[3]) + j] = 255
                except:
                    continue
    return prediction_mask

def hybrid_detect(image_path, binary_save_path, visual_save_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"[❌] Could not load: {image_path}")
        return False

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Try SIFT first
    binary, visual = sift_detect(img, gray)
    if binary is not None:
        fname = os.path.splitext(os.path.basename(image_path))[0]
        cv2.imwrite(os.path.join(binary_save_path, fname + ".png"), binary)
        cv2.imwrite(os.path.join(visual_save_path, fname + ".png"), visual)
        print(f"[✅] SIFT success: {fname}")
        return True

    # Fallback to DCT
    binary = dct_detect(gray)
    if np.sum(binary) == 0:
        print(f"[⚠️] DCT also failed: {os.path.basename(image_path)}")
        return False

    fname = os.path.splitext(os.path.basename(image_path))[0]
    visual = img.copy()
    visual[binary > 0] = [0, 0, 255]
    cv2.imwrite(os.path.join(binary_save_path, fname + ".png"), binary)
    cv2.imwrite(os.path.join(visual_save_path, fname + ".png"), visual)
    print(f"[🟡] Fallback DCT success: {fname}")
    return True

def run_hybrid_batch(input_folder, binary_out, visual_out):
    ensure_dir(binary_out)
    ensure_dir(visual_out)

    files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"🚀 Starting hybrid detection on {len(files)} images...\n")

    success = 0
    for f in tqdm(files):
        image_path = os.path.join(input_folder, f)
        if hybrid_detect(image_path, binary_out, visual_out):
            success += 1

    print(f"\n✅ Completed. {success}/{len(files)} images detected and saved.")

if __name__ == "__main__":
    run_hybrid_batch(
        input_folder='data/test_images',
        binary_out='data/results/test_output/Binary',
        visual_out='data/results/test_output/visual'
    )
