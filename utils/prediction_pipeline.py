import os
import numpy as np
from pathlib import Path
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image, ImageChops, ImageFilter, ImageEnhance
import cv2
from scipy import ndimage
from skimage import feature, segmentation, measure
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
import warnings
warnings.filterwarnings('ignore')

# Ensure your UNet model definition is correctly imported.
from models.segmentation.unet import UNet

# === Paths ===
classification_model_path = "models/classification/classification_model.pth"
segmentation_model_path = "models/segmentation/best_segmentation_model.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Enhanced Transforms ===
classification_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Enhanced segmentation transforms with better preprocessing
segmentation_transform_rgb = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

segmentation_transform_gray = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1] range
])

# === Advanced Forensic Feature Extraction ===
def compute_ela(image_path, quality=90):
    """Enhanced ELA computation with better preprocessing"""
    try:
        original_image = Image.open(image_path).convert("RGB")
        temp_path = "temp_ela_image.jpg"
        original_image.save(temp_path, quality=quality)
        recompressed_image = Image.open(temp_path).convert("RGB")
        
        # Calculate difference
        ela_image = ImageChops.difference(original_image, recompressed_image)
        
        # Convert to numpy for enhanced processing
        ela_np = np.array(ela_image, dtype=np.float32)
        
        # Better grayscale conversion
        ela_gray = 0.299 * ela_np[:,:,0] + 0.587 * ela_np[:,:,1] + 0.114 * ela_np[:,:,2]
        
        # Enhanced contrast with multiple techniques
        ela_gray_uint8 = np.clip(ela_gray, 0, 255).astype(np.uint8)
        
        # Histogram equalization
        ela_eq = cv2.equalizeHist(ela_gray_uint8)
        
        # CLAHE for better local contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        ela_enhanced = clahe.apply(ela_eq)
        
        # Gamma correction for better visibility
        gamma = 1.5
        ela_gamma = np.power(ela_enhanced / 255.0, gamma) * 255.0
        ela_gamma = np.clip(ela_gamma, 0, 255).astype(np.uint8)
        
        # Light denoising
        ela_smoothed = cv2.bilateralFilter(ela_gamma, 5, 75, 75)
        
        os.remove(temp_path)
        return Image.fromarray(ela_smoothed)
    except Exception as e:
        print(f"[❌] Error computing ELA: {e}")
        return None

def compute_noise_residual(image_path, blur_radius=2):
    """Enhanced noise residual computation"""
    try:
        original_image = Image.open(image_path).convert("L")
        original_np = np.array(original_image, dtype=np.float32)
        
        # Multi-scale noise detection with edge preservation
        residuals = []
        for sigma in [0.8, 1.2, 1.8]:
            blurred = cv2.GaussianBlur(original_np, (0, 0), sigma)
            residual = np.abs(original_np - blurred)
            residuals.append(residual)
        
        # Combine residuals using maximum
        combined_residual = np.max(residuals, axis=0)
        
        # Enhance using adaptive histogram equalization
        residual_uint8 = np.clip(combined_residual, 0, 255).astype(np.uint8)
        
        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced_residual = clahe.apply(residual_uint8)
        
        # Gamma correction
        gamma = 1.2
        enhanced_residual = np.power(enhanced_residual / 255.0, gamma) * 255.0
        enhanced_residual = np.clip(enhanced_residual, 0, 255).astype(np.uint8)
        
        return Image.fromarray(enhanced_residual)
    except Exception as e:
        print(f"[❌] Error computing Noise Residual: {e}")
        return None

def compute_advanced_gradient_analysis(image_path):
    """Advanced gradient-based forgery detection"""
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
            
        # Compute gradients using multiple methods
        grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        
        # Scharr operator for better accuracy
        scharr_x = cv2.Scharr(img, cv2.CV_64F, 1, 0)
        scharr_y = cv2.Scharr(img, cv2.CV_64F, 0, 1)
        
        # Combine gradients
        sobel_mag = np.sqrt(grad_x**2 + grad_y**2)
        scharr_mag = np.sqrt(scharr_x**2 + scharr_y**2)
        
        # Gradient inconsistency detection
        h, w = img.shape
        block_size = 16
        gradient_anomaly = np.zeros((h, w), dtype=np.float32)
        
        for i in range(0, h-block_size, block_size//4):
            for j in range(0, w-block_size, block_size//4):
                # Local gradient statistics
                sobel_block = sobel_mag[i:i+block_size, j:j+block_size]
                scharr_block = scharr_mag[i:i+block_size, j:j+block_size]
                
                # Statistical measures
                sobel_mean = np.mean(sobel_block)
                scharr_mean = np.mean(scharr_block)
                sobel_std = np.std(sobel_block)
                scharr_std = np.std(scharr_block)
                
                # Anomaly score based on gradient inconsistency
                mean_diff = abs(sobel_mean - scharr_mean) / (sobel_mean + scharr_mean + 1e-6)
                std_diff = abs(sobel_std - scharr_std) / (sobel_std + scharr_std + 1e-6)
                
                anomaly_score = mean_diff + std_diff
                gradient_anomaly[i:i+block_size, j:j+block_size] = anomaly_score
        
        # Normalize and enhance
        gradient_map = cv2.normalize(gradient_anomaly, None, 0, 255, cv2.NORM_MINMAX)
        gradient_map = gradient_map.astype(np.uint8)
        
        # Apply enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gradient_map = clahe.apply(gradient_map)
        
        return Image.fromarray(gradient_map)
    except Exception as e:
        print(f"[❌] Error in gradient analysis: {e}")
        return None

def compute_texture_inconsistency(image_path):
    """Texture-based inconsistency detection using LBP and GLCM"""
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
            
        # Local Binary Pattern
        radius = 2
        n_points = 8 * radius
        lbp = feature.local_binary_pattern(img, n_points, radius, method='uniform')
        
        # Texture analysis in overlapping blocks
        h, w = img.shape
        block_size = 32
        texture_anomaly = np.zeros((h, w), dtype=np.float32)
        
        for i in range(0, h-block_size, block_size//4):
            for j in range(0, w-block_size, block_size//4):
                # Get local texture patch
                lbp_block = lbp[i:i+block_size, j:j+block_size]
                img_block = img[i:i+block_size, j:j+block_size]
                
                # LBP histogram
                lbp_hist, _ = np.histogram(lbp_block.ravel(), bins=n_points+2, 
                                         range=(0, n_points+2), density=True)
                
                # Texture uniformity measure
                uniformity = np.sum(lbp_hist**2)
                
                # Contrast measure
                contrast = np.var(img_block)
                
                # Combine measures
                texture_score = (1 - uniformity) * np.log(contrast + 1)
                texture_anomaly[i:i+block_size, j:j+block_size] = texture_score
        
        # Normalize
        texture_map = cv2.normalize(texture_anomaly, None, 0, 255, cv2.NORM_MINMAX)
        texture_map = texture_map.astype(np.uint8)
        
        # Enhance with CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        texture_map = clahe.apply(texture_map)
        
        return Image.fromarray(texture_map)
    except Exception as e:
        print(f"[❌] Error in texture analysis: {e}")
        return None

def prepare_enhanced_forensic_inputs(image_path):
    """Enhanced forensic input preparation with multiple techniques"""
    try:
        rgb_image = Image.open(image_path).convert("RGB")
        
        # Generate forensic features
        ela_image = compute_ela(image_path, quality=85)
        noise_res_image = compute_noise_residual(image_path)
        gradient_image = compute_advanced_gradient_analysis(image_path)
        texture_image = compute_texture_inconsistency(image_path)
        
        # Check which techniques succeeded
        techniques = [ela_image, noise_res_image, gradient_image, texture_image]
        valid_techniques = [img for img in techniques if img is not None]
        
        if len(valid_techniques) < 2:
            print("Warning: Some forensic techniques failed, using available ones")
        
        # Prepare RGB input
        rgb_tensor = segmentation_transform_rgb(rgb_image)
        
        # Prepare grayscale inputs
        gray_tensors = []
        for technique_img in valid_techniques:
            if technique_img is not None:
                gray_tensor = segmentation_transform_gray(technique_img)
                gray_tensors.append(gray_tensor)
        
        # Ensure we have exactly the right number of channels
        while len(gray_tensors) < 2:  # Minimum 2 forensic channels
            if len(gray_tensors) > 0:
                gray_tensors.append(gray_tensors[-1])  # Duplicate last valid one
            else:
                # Create zero tensor if all failed
                zero_tensor = torch.zeros_like(segmentation_transform_gray(Image.new('L', (256, 256))))
                gray_tensors.append(zero_tensor)
        
        # Combine inputs - RGB + forensic channels
        combined_input_tensor = torch.cat([rgb_tensor] + gray_tensors[:2], dim=0)  # Limit to 5 total channels
        
        print(f"Combined input channels: {combined_input_tensor.shape[0]}")
        return combined_input_tensor.unsqueeze(0), len(gray_tensors)
        
    except Exception as e:
        print(f"[❌] Error in prepare_enhanced_forensic_inputs: {e}")
        # Fallback to basic method
        rgb_image = Image.open(image_path).convert("RGB")
        rgb_tensor = segmentation_transform_rgb(rgb_image)
        
        # Create dummy forensic channels
        dummy_tensor = torch.zeros(1, 256, 256)
        combined_input_tensor = torch.cat([rgb_tensor, dummy_tensor, dummy_tensor], dim=0)
        return combined_input_tensor.unsqueeze(0), 0

def adaptive_threshold(prob_mask, method='otsu_multi'):
    """Adaptive thresholding for better segmentation"""
    # Convert to uint8 for thresholding
    prob_uint8 = (prob_mask * 255).astype(np.uint8)
    
    if method == 'otsu_multi':
        # Multi-level Otsu thresholding
        try:
            thresh1 = cv2.threshold(prob_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
            thresh2 = thresh1 * 0.7  # Lower threshold
            
            high_conf_mask = (prob_uint8 > thresh1).astype(np.uint8)
            med_conf_mask = (prob_uint8 > thresh2).astype(np.uint8)
            
            return high_conf_mask, med_conf_mask
        except:
            # Fallback to fixed thresholds
            high_conf_mask = (prob_mask > 0.6).astype(np.uint8)
            med_conf_mask = (prob_mask > 0.4).astype(np.uint8)
            return high_conf_mask, med_conf_mask
    
    elif method == 'adaptive':
        # Adaptive thresholding
        adaptive_mask = cv2.adaptiveThreshold(prob_uint8, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                            cv2.THRESH_BINARY, 11, 2)
        return (adaptive_mask > 0).astype(np.uint8), (adaptive_mask > 0).astype(np.uint8)

def enhanced_post_process_mask(prob_mask, original_size, min_area=500):
    """Enhanced post-processing with better region detection"""
    print(f"Post-processing mask - Shape: {prob_mask.shape}, Min: {prob_mask.min():.3f}, Max: {prob_mask.max():.3f}")
    
    # Apply adaptive thresholding
    high_conf_mask, med_conf_mask = adaptive_threshold(prob_mask, method='otsu_multi')
    
    print(f"High confidence pixels: {np.sum(high_conf_mask)}")
    print(f"Medium confidence pixels: {np.sum(med_conf_mask)}")
    
    # Morphological operations with smaller kernels
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_med = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    # Clean high confidence mask
    high_cleaned = cv2.morphologyEx(high_conf_mask * 255, cv2.MORPH_OPEN, kernel_small, iterations=1)
    high_cleaned = cv2.morphologyEx(high_cleaned, cv2.MORPH_CLOSE, kernel_small, iterations=1)
    
    # Clean medium confidence mask
    med_cleaned = cv2.morphologyEx(med_conf_mask * 255, cv2.MORPH_OPEN, kernel_small, iterations=1)
    
    # Combine masks intelligently
    final_mask = high_cleaned.copy()
    
    # Add medium confidence regions that are close to high confidence
    high_dilated = cv2.dilate(high_cleaned, kernel_med, iterations=2)
    med_near_high = cv2.bitwise_and(med_cleaned, high_dilated)
    final_mask = np.maximum(final_mask, med_near_high)
    
    # Apply final cleaning
    final_mask = cv2.medianBlur(final_mask, 3)
    
    # Resize to original size
    final_mask_resized = cv2.resize(final_mask, original_size, interpolation=cv2.INTER_NEAREST)
    
    # Connected component analysis with relaxed criteria
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(final_mask_resized, 8, cv2.CV_32S)
    
    refined_mask = np.zeros_like(final_mask_resized)
    detected_bboxes = []
    
    print(f"Found {num_labels-1} connected components")
    
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        
        aspect_ratio = w / h if h > 0 else float('inf')
        
        # More lenient criteria for region acceptance
        if (area >= min_area and 
            0.1 <= aspect_ratio <= 10.0 and  # More lenient aspect ratio
            w > 10 and h > 10):  # Minimum dimensions
            
            detected_bboxes.append((x, y, w, h))
            refined_mask[labels == i] = 255
            print(f"✓ Kept component {i}: Area={area}, AR={aspect_ratio:.2f}")
        else:
            print(f"✗ Filtered component {i}: Area={area}, AR={aspect_ratio:.2f}")
    
    print(f"Final detected regions: {len(detected_bboxes)}")
    
    return refined_mask, detected_bboxes

def create_enhanced_highlight_visualization(image_path, mask, alpha=0.5, highlight_color=(0, 0, 255)):
    """Create enhanced visualization with better highlighting"""
    # Load original image
    original = cv2.imread(image_path)
    if original is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Ensure mask is same size as original
    if mask.shape[:2] != original.shape[:2]:
        mask = cv2.resize(mask, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # Create binary mask
    binary_mask = (mask > 127).astype(np.uint8) * 255
    
    # Count forged pixels
    forged_pixels = np.sum(binary_mask > 0)
    total_pixels = binary_mask.shape[0] * binary_mask.shape[1]
    forged_percentage = (forged_pixels / total_pixels) * 100
    
    print(f"Highlighting {forged_pixels} pixels ({forged_percentage:.2f}% of image)")
    
    # Create visualization
    highlighted = original.copy()
    
    if forged_pixels > 0:
        # Create colored overlay
        overlay = original.copy()
        overlay[binary_mask > 0] = highlight_color
        
        # Blend with original
        mask_3ch = np.stack([binary_mask, binary_mask, binary_mask], axis=2) / 255.0
        highlighted = highlighted.astype(np.float32)
        overlay = overlay.astype(np.float32)
        
        highlighted = highlighted * (1 - mask_3ch * alpha) + overlay * (mask_3ch * alpha)
        highlighted = highlighted.astype(np.uint8)
        
        # Add contours for better visibility
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(highlighted, contours, -1, highlight_color, thickness=2)
        
        # Add bounding boxes
        for contour in contours:
            if cv2.contourArea(contour) > 500:  # Only for significant regions
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(highlighted, (x, y), (x + w, y + h), (255, 255, 0), 2)
    
    return highlighted

# === Load Models ===
try:
    classification_model = models.resnet18(weights=None)
    classification_model.fc = torch.nn.Linear(classification_model.fc.in_features, 2)
    classification_model.load_state_dict(torch.load(classification_model_path, map_location=device))
    classification_model.to(device).eval()
    print(f"✓ Classification model loaded from {classification_model_path}")
except Exception as e:
    print(f"❌ Error loading classification model: {e}")
    exit()

try:
    segmentation_model = UNet(in_channels=5, out_channels=1)
    segmentation_model.load_state_dict(torch.load(segmentation_model_path, map_location=device))
    segmentation_model.to(device).eval()
    print(f"✓ Segmentation model loaded from {segmentation_model_path}")
except Exception as e:
    print(f"❌ Error loading segmentation model: {e}")
    exit()

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

# === Enhanced Segmentation Function ===
def segment_image(image_path, output_mask_path):
    """Enhanced segmentation with improved highlighting"""
    try:
        image_rgb = Image.open(image_path).convert("RGB")
        original_size = image_rgb.size
        
        print("🔍 Running enhanced forensic analysis...")
        
        # Prepare enhanced inputs
        combined_tensor, num_techniques = prepare_enhanced_forensic_inputs(image_path)
        combined_tensor = combined_tensor.to(device)
        
        print(f"✓ Using {num_techniques} forensic techniques")
        
        # Multi-scale inference for robustness
        scales = [224, 256, 288]
        all_predictions = []
        
        for scale in scales:
            try:
                # Resize to scale
                resized_input = F.interpolate(combined_tensor, size=(scale, scale), 
                                            mode='bilinear', align_corners=False)
                
                with torch.no_grad():
                    output = segmentation_model(resized_input)
                    prob_mask = torch.sigmoid(output).squeeze().cpu().numpy()
                
                # Resize back to standard size
                prob_mask_resized = cv2.resize(prob_mask, (256, 256), interpolation=cv2.INTER_LINEAR)
                all_predictions.append(prob_mask_resized)
                
                print(f"✓ Scale {scale}: Prediction range [{prob_mask.min():.3f}, {prob_mask.max():.3f}]")
                
            except Exception as e:
                print(f"❌ Scale {scale} failed: {e}")
                continue
        
        if not all_predictions:
            raise ValueError("All scales failed in segmentation")
        
        # Ensemble predictions
        if len(all_predictions) > 1:
            ensemble_mask = np.mean(all_predictions, axis=0)
        else:
            ensemble_mask = all_predictions[0]
        
        print(f"✓ Ensemble prediction range: [{ensemble_mask.min():.3f}, {ensemble_mask.max():.3f}]")
        
        # Apply Gaussian smoothing for better results
        ensemble_mask = cv2.GaussianBlur(ensemble_mask, (5, 5), 1.0)
        
        # Enhanced post-processing with multiple thresholds
        detected_bboxes = []
        refined_mask_np = None
        
        # Try different minimum area thresholds
        for min_area in [300, 500, 800]:
            print(f"\n📊 Trying minimum area: {min_area}")
            refined_mask_np, detected_bboxes = enhanced_post_process_mask(
                ensemble_mask, original_size, min_area=min_area
            )
            
            if detected_bboxes:
                print(f"✅ Found {len(detected_bboxes)} regions with min_area={min_area}")
                break
            else:
                print(f"❌ No regions found with min_area={min_area}")
        
        # If still no regions, try with very permissive settings
        if not detected_bboxes:
            print("\n🔄 Applying permissive detection...")
            
            # Very low threshold approach
            prob_resized = cv2.resize(ensemble_mask, original_size, interpolation=cv2.INTER_LINEAR)
            
            # Lower threshold
            low_thresh_mask = (prob_resized > 0.2).astype(np.uint8) * 255
            
            # Light morphological cleaning
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            low_thresh_mask = cv2.morphologyEx(low_thresh_mask, cv2.MORPH_OPEN, kernel, iterations=1)
            low_thresh_mask = cv2.morphologyEx(low_thresh_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            
            # Find contours
            contours, _ = cv2.findContours(low_thresh_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            refined_mask_np = np.zeros_like(low_thresh_mask)
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 200:  # Very permissive
                    cv2.drawContours(refined_mask_np, [contour], -1, 255, -1)
                    x, y, w, h = cv2.boundingRect(contour)
                    detected_bboxes.append((x, y, w, h))
            
            print(f"✅ Permissive detection found {len(detected_bboxes)} regions")
        
        # Create enhanced visualization
        highlighted_image = create_enhanced_highlight_visualization(
            image_path, 
            refined_mask_np, 
            alpha=0.4,
            highlight_color=(0, 0, 255)  # Red highlighting
        )

        # Save results
        os.makedirs(os.path.dirname(output_mask_path), exist_ok=True)
        cv2.imwrite(output_mask_path, highlighted_image)
        
        print(f"✅ Enhanced segmentation saved to: {output_mask_path}")
        print(f"🎯 Total detected forged regions: {len(detected_bboxes)}")

        return detected_bboxes

    except Exception as e:
        print(f"❌ Error in segment_image(): {e}")
        return []

# === Enhanced Copy-Move Detection ===
def detect_copy_move_forgery(image_path, output_path):
    """Enhanced copy-move detection"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"❌ Could not load image: {image_path}")
            return None

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Enhanced feature detection
        orb = cv2.ORB_create(nfeatures=2000, scaleFactor=1.2, nlevels=8)
        kp_orb, des_orb = orb.detectAndCompute(gray, None)

        # Try SIFT if available
        try:
            sift = cv2.SIFT_create(nfeatures=1500)
            kp_sift, des_sift = sift.detectAndCompute(gray, None)
        except:
            kp_sift, des_sift = [], None

        result_img = img.copy()
        all_good_matches = []

        # ORB matching
        if des_orb is not None and len(kp_orb) >= 2:
            bf_orb = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
            raw_matches_orb = bf_orb.knnMatch(des_orb, des_orb, k=2)

            for match_pair in raw_matches_orb:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.75 * n.distance:
                        pt1 = np.array(kp_orb[m.queryIdx].pt)
                        pt2 = np.array(kp_orb[m.trainIdx].pt)
                        if np.linalg.norm(pt1 - pt2) > 30:  # Minimum distance
                            all_good_matches.append((m, kp_orb, 'ORB'))

        # SIFT matching
        if des_sift is not None and len(kp_sift) >= 2:
            bf_sift = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
            raw_matches_sift = bf_sift.knnMatch(des_sift, des_sift, k=2)

            for match_pair in raw_matches_sift:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.7 * n.distance:
                        pt1 = np.array(kp_sift[m.queryIdx].pt)
                        pt2 = np.array(kp_sift[m.trainIdx].pt)
                        if np.linalg.norm(pt1 - pt2) > 30:
                            all_good_matches.append((m, kp_sift, 'SIFT'))

        # Visualize matches
        for match_info in all_good_matches[:100]:  # Limit to top 100 matches
            m, keypoints, descriptor_type = match_info
            pt1 = tuple(map(int, keypoints[m.queryIdx].pt))
            pt2 = tuple(map(int, keypoints[m.trainIdx].pt))
            
            # Different colors for different descriptors
            if descriptor_type == 'ORB':
                line_color = (0, 255, 0)  # Green
                point_color = (0, 0, 255)  # Red
            else:  # SIFT
                line_color = (255, 0, 0)  # Blue
                point_color = (0, 255, 255)  # Yellow
            
            cv2.line(result_img, pt1, pt2, line_color, 2)
            cv2.circle(result_img, pt1, 4, point_color, -1)
            cv2.circle(result_img, pt2, 4, point_color, -1)

        # Clustering for dense regions
        if len(all_good_matches) > 6:
            points = []
            for match_info in all_good_matches:
                m, keypoints, _ = match_info
                pt1 = keypoints[m.queryIdx].pt
                pt2 = keypoints[m.trainIdx].pt
                points.extend([pt1, pt2])
            
            points = np.array(points)
            
            # DBSCAN clustering
            try:
                clustering = DBSCAN(eps=40, min_samples=6).fit(points)
                labels = clustering.labels_
                
                # Draw cluster bounding boxes
                unique_labels = set(labels)
                for label in unique_labels:
                    if label != -1:  # Ignore noise
                        cluster_points = points[labels == label]
                        if len(cluster_points) > 8:
                            x_min, y_min = np.min(cluster_points, axis=0).astype(int)
                            x_max, y_max = np.max(cluster_points, axis=0).astype(int)
                            
                            cv2.rectangle(result_img, (x_min-10, y_min-10), 
                                        (x_max+10, y_max+10), (255, 255, 0), 3)
                            cv2.putText(result_img, f'Copy-Move {label}', 
                                      (x_min, y_min-15), cv2.FONT_HERSHEY_SIMPLEX, 
                                      0.7, (255, 255, 0), 2)
                        
            except Exception as e:
                print(f"Clustering failed: {e}")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, result_img)
        print(f"✅ Copy-move detection saved: {output_path}")
        print(f"🔍 Found {len(all_good_matches)} potential matches")
        return output_path

    except Exception as e:
        print(f"❌ Error in copy-move detection: {e}")
        return None

# === Main Prediction Pipeline ===
def run_prediction_pipeline(image_path, output_dir="static/masks"):
    """Enhanced prediction pipeline with accurate highlighting"""
    os.makedirs(output_dir, exist_ok=True)
    print(f"🚀 Processing image: {image_path}")

    # Step 1: Classification
    result = classify_image(image_path)
    print(f"📊 Classification: {result['label']} (Confidence: {result['confidence']:.4f})")

    filename = Path(image_path).stem
    result["mask_path"] = None
    result["copy_move_path"] = None
    result["forged_objects_bboxes"] = []
    result["forged_objects_overlay_path"] = None

    # Step 2: If forged, perform localization
    if result["label"] == "forged":
        print("🎯 Image classified as FORGED. Running localization...")

        # Segmentation with highlighting
        mask_filename = f"{filename}_highlighted.png"
        full_mask_path = os.path.join(output_dir, mask_filename)
        forged_bboxes = segment_image(image_path, full_mask_path)
        result["mask_path"] = f"masks/{mask_filename}".replace("\\", "/")
        result["forged_objects_bboxes"] = forged_bboxes

        # Create bounding box overlay
        if forged_bboxes:
            original_img = cv2.imread(image_path)
            if original_img is not None:
                overlay_img = original_img.copy()
                for i, (x, y, w, h) in enumerate(forged_bboxes):
                    # Draw bounding box
                    cv2.rectangle(overlay_img, (x, y), (x + w, y + h), (0, 255, 255), 3)
                    # Add label
                    cv2.putText(overlay_img, f'Forged {i+1}', (x, y-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                bbox_filename = f"{filename}_bboxes.png"
                full_bbox_path = os.path.join(output_dir, bbox_filename)
                cv2.imwrite(full_bbox_path, overlay_img)
                result["forged_objects_overlay_path"] = f"masks/{bbox_filename}".replace("\\", "/")
                print(f"✅ Bounding boxes saved: {full_bbox_path}")

        # Copy-move detection
        copy_move_filename = f"{filename}_copy_move.png"
        full_copy_move_path = os.path.join(output_dir, copy_move_filename)
        copy_move_result = detect_copy_move_forgery(image_path, full_copy_move_path)
        if copy_move_result:
            result["copy_move_path"] = f"masks/{copy_move_filename}".replace("\\", "/")

    else:
        print("✅ Image classified as UNFORGED. Skipping localization.")

    return result

# === Visualization Helpers ===
def create_comparison_visualization(image_path, results, output_path):
    """Create a comparison visualization showing all results"""
    try:
        # Load original image
        original = cv2.imread(image_path)
        h, w = original.shape[:2]
        
        # Create a larger canvas for comparison
        if results["mask_path"] and results["copy_move_path"]:
            canvas = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)
            
            # Original image (top-left)
            canvas[0:h, 0:w] = original
            cv2.putText(canvas, 'Original', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Highlighted forgeries (top-right)
            if os.path.exists(results["mask_path"].replace("masks/", "static/masks/")):
                highlighted = cv2.imread(results["mask_path"].replace("masks/", "static/masks/"))
                if highlighted is not None:
                    canvas[0:h, w:2*w] = cv2.resize(highlighted, (w, h))
                    cv2.putText(canvas, 'Forged Regions', (w+10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Bounding boxes (bottom-left)
            if results["forged_objects_overlay_path"] and os.path.exists(results["forged_objects_overlay_path"].replace("masks/", "static/masks/")):
                bbox_img = cv2.imread(results["forged_objects_overlay_path"].replace("masks/", "static/masks/"))
                if bbox_img is not None:
                    canvas[h:2*h, 0:w] = cv2.resize(bbox_img, (w, h))
                    cv2.putText(canvas, 'Bounding Boxes', (10, h+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            # Copy-move detection (bottom-right)
            if os.path.exists(results["copy_move_path"].replace("masks/", "static/masks/")):
                copy_move = cv2.imread(results["copy_move_path"].replace("masks/", "static/masks/"))
                if copy_move is not None:
                    canvas[h:2*h, w:2*w] = cv2.resize(copy_move, (w, h))
                    cv2.putText(canvas, 'Copy-Move', (w+10, h+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            
            cv2.imwrite(output_path, canvas)
            print(f"✅ Comparison visualization saved: {output_path}")
            return output_path
        
    except Exception as e:
        print(f"❌ Error creating comparison: {e}")
        return None

# === Testing and Example Usage ===
if __name__ == "__main__":
    from PIL import ImageDraw, ImageFont
    import random

    # Create a more realistic test image with simulated forgery
    temp_dir = "temp_test_images"
    os.makedirs(temp_dir, exist_ok=True)
    test_image_path = os.path.join(temp_dir, "test_forgery_image.jpg")

    if not os.path.exists(test_image_path):
        print("🎨 Creating test image with simulated forgery...")
        
        # Create base image
        img = Image.new('RGB', (800, 600), color=(135, 206, 235))  # Sky blue
        draw = ImageDraw.Draw(img)
        
        # Add some natural elements
        # Ground
        draw.rectangle([0, 400, 800, 600], fill=(34, 139, 34))  # Forest green
        
        # Trees
        for i in range(5):
            x = random.randint(50, 750)
            y = random.randint(350, 400)
            # Tree trunk
            draw.rectangle([x-5, y, x+5, y+50], fill=(139, 69, 19))
            # Tree crown
            draw.ellipse([x-25, y-30, x+25, y+10], fill=(0, 100, 0))
        
        # Add a "forged" object - a building that doesn't fit
        # This simulates a copy-paste forgery
        draw.rectangle([300, 250, 400, 350], fill=(128, 128, 128))  # Building
        draw.rectangle([320, 270, 340, 300], fill=(0, 0, 255))      # Blue window
        draw.rectangle([360, 270, 380, 300], fill=(0, 0, 255))      # Blue window
        draw.rectangle([340, 320, 360, 350], fill=(139, 69, 19))    # Door
        
        # Add some clouds
        for i in range(3):
            x = random.randint(100, 700)
            y = random.randint(50, 150)
            draw.ellipse([x, y, x+80, y+40], fill=(255, 255, 255))
        
        # Save with JPEG compression to create artifacts
        img.save(test_image_path, "JPEG", quality=85)
        print(f"✅ Test image created: {test_image_path}")

    # Run the enhanced pipeline
    print("\n" + "="*60)
    print("🚀 RUNNING ENHANCED FORGERY DETECTION PIPELINE")
    print("="*60)
    
    results = run_prediction_pipeline(test_image_path)
    
    print("\n" + "="*60)
    print("📊 PIPELINE RESULTS")
    print("="*60)
    print(f"Classification: {results['label']}")
    print(f"Confidence: {results['confidence']:.4f}")
    
    if results["mask_path"]:
        print(f"✅ Highlighted Image: {results['mask_path']}")
    
    if results["forged_objects_bboxes"]:
        print(f"✅ Detected {len(results['forged_objects_bboxes'])} forged regions:")
        for i, (x, y, w, h) in enumerate(results['forged_objects_bboxes']):
            print(f"   Region {i+1}: ({x}, {y}) - {w}x{h} pixels")
    
    if results.get("forged_objects_overlay_path"):
        print(f"✅ Bounding Boxes: {results['forged_objects_overlay_path']}")
    
    if results["copy_move_path"]:
        print(f"✅ Copy-Move Analysis: {results['copy_move_path']}")
    
    # Create comparison visualization
    comparison_path = os.path.join("static/masks", f"{Path(test_image_path).stem}_comparison.png")
    create_comparison_visualization(test_image_path, results, comparison_path)
    
    print("\n" + "="*60)
    print("🎯 KEY IMPROVEMENTS IMPLEMENTED")
    print("="*60)
    print("✅ Enhanced forensic feature extraction (ELA, noise, gradient, texture)")
    print("✅ Multi-scale inference for better accuracy")
    print("✅ Adaptive thresholding (Otsu + multi-level)")
    print("✅ Improved post-processing with relaxed criteria")
    print("✅ Better morphological operations")
    print("✅ Enhanced visualization with proper highlighting")
    print("✅ Fallback mechanisms for robustness")
    print("✅ Permissive detection mode for edge cases")
    print("✅ Comprehensive comparison visualizations")
    print("✅ Better error handling and logging")
    
    print(f"\n🎉 Processing complete! Check the output directory: static/masks/")

def get_model_predictions_debug(image_path):
    """Debug function to inspect model predictions"""
    try:
        combined_tensor, _ = prepare_enhanced_forensic_inputs(image_path)
        combined_tensor = combined_tensor.to(device)
        
        with torch.no_grad():
            output = segmentation_model(combined_tensor)
            prob_mask = torch.sigmoid(output).squeeze().cpu().numpy()
        
        print(f"Raw model output statistics:")
        print(f"  Shape: {prob_mask.shape}")
        print(f"  Min: {prob_mask.min():.6f}")
        print(f"  Max: {prob_mask.max():.6f}")
        print(f"  Mean: {prob_mask.mean():.6f}")
        print(f"  Std: {prob_mask.std():.6f}")
        print(f"  Pixels > 0.1: {np.sum(prob_mask > 0.1)}")
        print(f"  Pixels > 0.3: {np.sum(prob_mask > 0.3)}")
        print(f"  Pixels > 0.5: {np.sum(prob_mask > 0.5)}")
        print(f"  Pixels > 0.7: {np.sum(prob_mask > 0.7)}")
        
        return prob_mask
    except Exception as e:
        print(f"Debug failed: {e}")
        return None