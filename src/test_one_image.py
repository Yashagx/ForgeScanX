import cv2
import os

def highlight_forgery(original_path, binary_mask_path, output_path):
    original = cv2.imread(original_path)
    mask = cv2.imread(binary_mask_path, 0)  # grayscale

    if original is None:
        print(f"❌ Could not read original: {original_path}")
        return

    if mask is None:
        print(f"❌ Could not read mask: {binary_mask_path}")
        return

    if original.shape[:2] != mask.shape:
        print(f"❌ Dimension mismatch between original and mask.")
        return

    # Create red mask overlay
    red_overlay = original.copy()
    red_overlay[mask > 0] = [0, 0, 255]  # Red color on forged pixels

    # Blend original and red overlay
    blended = cv2.addWeighted(original, 0.7, red_overlay, 0.3, 0)

    # Save result
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, blended)
    print(f"✅ Highlighted forgery saved to: {output_path}")

if __name__ == "__main__":
    # Change filenames here
    original_img = "data/test_images/001_F_BC1.png"
    detected_mask = "data/results/single_test/Binary/001_F_BC1.png"
    save_as = "data/results/single_test/highlighted/001_F_BC1_highlighted.png"

    highlight_forgery(original_img, detected_mask, save_as)
