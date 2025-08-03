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
import matplotlib.patches as patches
from scipy.signal import convolve2d
import warnings
import json
import time
from datetime import datetime
import seaborn as sns
plt.style.use('seaborn-v0_8')
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

# === Analytics and Statistics Tracking ===
class ForgeryAnalytics:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.start_time = time.time()
        self.classification_time = 0
        self.segmentation_time = 0
        self.post_processing_time = 0
        self.copy_move_time = 0
        
        # Classification metrics
        self.classification_confidence = 0
        self.classification_probabilities = []
        
        # Segmentation metrics
        self.segmentation_confidence = 0
        self.mask_statistics = {}
        self.forensic_features_stats = {}
        
        # Detection metrics
        self.detected_regions = 0
        self.total_forged_pixels = 0
        self.forged_percentage = 0
        self.region_areas = []
        self.region_confidences = []
        
        # Model performance
        self.model_activations = {}
        self.feature_importance = {}
        
        # Forensic analysis
        self.ela_stats = {}
        self.noise_stats = {}
        self.gradient_stats = {}
        self.texture_stats = {}
        
    def log_timing(self, phase, duration):
        setattr(self, f"{phase}_time", duration)
    
    def log_classification(self, probabilities, confidence):
        self.classification_probabilities = probabilities.tolist()
        self.classification_confidence = confidence
    
    def log_segmentation(self, mask, confidence_scores=None):
        self.segmentation_confidence = np.mean(confidence_scores) if confidence_scores is not None else 0
        self.total_forged_pixels = np.sum(mask > 127)
        total_pixels = mask.shape[0] * mask.shape[1]
        self.forged_percentage = (self.total_forged_pixels / total_pixels) * 100
        
        # Mask statistics
        self.mask_statistics = {
            'mean_intensity': float(np.mean(mask)),
            'std_intensity': float(np.std(mask)),
            'max_intensity': float(np.max(mask)),
            'min_intensity': float(np.min(mask)),
            'unique_values': len(np.unique(mask))
        }
    
    def log_regions(self, bboxes, region_scores=None):
        self.detected_regions = len(bboxes)
        self.region_areas = [w * h for x, y, w, h in bboxes]
        if region_scores:
            self.region_confidences = region_scores
    
    def log_forensic_features(self, ela_img, noise_img, gradient_img, texture_img):
        if ela_img is not None:
            ela_array = np.array(ela_img)
            self.ela_stats = {
                'mean': float(np.mean(ela_array)),
                'std': float(np.std(ela_array)),
                'energy': float(np.sum(ela_array ** 2))
            }
        
        if noise_img is not None:
            noise_array = np.array(noise_img)
            self.noise_stats = {
                'mean': float(np.mean(noise_array)),
                'std': float(np.std(noise_array)),
                'energy': float(np.sum(noise_array ** 2))
            }
        
        if gradient_img is not None:
            grad_array = np.array(gradient_img)
            self.gradient_stats = {
                'mean': float(np.mean(grad_array)),
                'std': float(np.std(grad_array)),
                'energy': float(np.sum(grad_array ** 2))
            }
        
        if texture_img is not None:
            texture_array = np.array(texture_img)
            self.texture_stats = {
                'mean': float(np.mean(texture_array)),
                'std': float(np.std(texture_array)),
                'energy': float(np.sum(texture_array ** 2))
            }
    
    def get_summary(self):
        total_time = time.time() - self.start_time
        return {
            'timing': {
                'total_processing_time': round(total_time, 3),
                'classification_time': round(self.classification_time, 3),
                'segmentation_time': round(self.segmentation_time, 3),
                'post_processing_time': round(self.post_processing_time, 3),
                'copy_move_time': round(self.copy_move_time, 3)
            },
            'classification': {
                'confidence': round(self.classification_confidence, 4),
                'probabilities': [round(p, 4) for p in self.classification_probabilities]
            },
            'segmentation': {
                'confidence': round(self.segmentation_confidence, 4),
                'mask_statistics': self.mask_statistics
            },
            'detection': {
                'detected_regions': self.detected_regions,
                'total_forged_pixels': int(self.total_forged_pixels),
                'forged_percentage': round(self.forged_percentage, 2),
                'region_areas': self.region_areas,
                'average_region_area': round(np.mean(self.region_areas), 2) if self.region_areas else 0
            },
            'forensic_features': {
                'ela_stats': self.ela_stats,
                'noise_stats': self.noise_stats,
                'gradient_stats': self.gradient_stats,
                'texture_stats': self.texture_stats
            }
        }

# Global analytics instance
analytics = ForgeryAnalytics()

# === Advanced Forensic Feature Extraction (Fixed) ===
def compute_ela(image_path, quality=90):
    """Enhanced ELA computation with analytics"""
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
    """Enhanced forensic input preparation with analytics"""
    try:
        rgb_image = Image.open(image_path).convert("RGB")
        
        # Generate forensic features
        ela_image = compute_ela(image_path, quality=85)
        noise_res_image = compute_noise_residual(image_path)
        gradient_image = compute_advanced_gradient_analysis(image_path)
        texture_image = compute_texture_inconsistency(image_path)
        
        # Log forensic features for analytics
        analytics.log_forensic_features(ela_image, noise_res_image, gradient_image, texture_image)
        
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
    """Enhanced post-processing with analytics - FIXED"""
    start_time = time.time()
    
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
    
    # FIXED: Resize to original size correctly
    final_mask_resized = cv2.resize(final_mask, original_size, interpolation=cv2.INTER_NEAREST)
    
    # Connected component analysis with relaxed criteria
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(final_mask_resized, 8, cv2.CV_32S)
    
    refined_mask = np.zeros_like(final_mask_resized)
    detected_bboxes = []
    region_scores = []
    
    print(f"Found {num_labels-1} connected components")
    
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        
        aspect_ratio = w / h if h > 0 else float('inf')
        
        # Calculate confidence score for this region - FIXED
        region_mask = (labels == i)
        # Resize prob_mask to match original size for proper indexing
        prob_mask_resized = cv2.resize(prob_mask, original_size, interpolation=cv2.INTER_LINEAR)
        region_prob_values = prob_mask_resized[region_mask] if region_mask.sum() > 0 else [0]
        region_confidence = np.mean(region_prob_values)
        
        # More lenient criteria for region acceptance
        if (area >= min_area and 
            0.1 <= aspect_ratio <= 10.0 and  # More lenient aspect ratio
            w > 10 and h > 10):  # Minimum dimensions
            
            detected_bboxes.append((x, y, w, h))
            region_scores.append(region_confidence)
            refined_mask[labels == i] = 255
            print(f"✓ Kept component {i}: Area={area}, AR={aspect_ratio:.2f}, Conf={region_confidence:.3f}")
        else:
            print(f"✗ Filtered component {i}: Area={area}, AR={aspect_ratio:.2f}")
    
    print(f"Final detected regions: {len(detected_bboxes)}")
    
    # Log analytics
    analytics.log_timing("post_processing", time.time() - start_time)
    analytics.log_regions(detected_bboxes, region_scores)
    
    return refined_mask, detected_bboxes, region_scores

def create_enhanced_highlight_visualization(image_path, mask, alpha=0.5, highlight_color=(0, 0, 255)):
    """Create enhanced visualization with analytics - FIXED"""
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
    
    # Log segmentation analytics
    analytics.log_segmentation(mask)
    
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

# === Analytics Visualization Functions ===
def create_performance_charts(output_dir, filename):
    """Create comprehensive performance and analysis charts"""
    chart_paths = {}
    
    try:
        # Get analytics summary
        summary = analytics.get_summary()
        
        # 1. Processing Time Breakdown
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Timing chart
        timing_data = summary['timing']
        phases = ['Classification', 'Segmentation', 'Post-processing', 'Copy-move']
        times = [timing_data['classification_time'], timing_data['segmentation_time'], 
                timing_data['post_processing_time'], timing_data['copy_move_time']]
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        ax1.pie(times, labels=phases, autopct='%1.1f%%', colors=colors, startangle=90)
        ax1.set_title('Processing Time Distribution', fontweight='bold')
        
        # 2. Classification Confidence
        if summary['classification']['probabilities']:
            probs = summary['classification']['probabilities']
            classes = ['Forged', 'Authentic']
            bars = ax2.bar(classes, probs, color=['#FF4757', '#2ED573'])
            ax2.set_title('Classification Probabilities', fontweight='bold')
            ax2.set_ylabel('Probability')
            ax2.set_ylim(0, 1)
            
            # Add value labels on bars
            for bar, prob in zip(bars, probs):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                        f'{prob:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. Forensic Features Analysis
        forensic_data = summary['forensic_features']
        features = []
        energies = []
        means = []
        
        for feature_name, stats in forensic_data.items():
            if stats:  # Check if stats is not empty
                features.append(feature_name.replace('_stats', '').upper())
                energies.append(stats.get('energy', 0))
                means.append(stats.get('mean', 0))
        
        if features:
            x_pos = np.arange(len(features))
            
            # Normalize energies for better visualization
            if max(energies) > 0:
                energies_norm = [e / max(energies) * 100 for e in energies]
            else:
                energies_norm = energies
            
            ax3.bar(x_pos, energies_norm, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
            ax3.set_title('Forensic Feature Energy Levels', fontweight='bold')
            ax3.set_ylabel('Normalized Energy')
            ax3.set_xticks(x_pos)
            ax3.set_xticklabels(features, rotation=45)
        
        # 4. Detection Statistics
        detection_data = summary['detection']
        if detection_data['detected_regions'] > 0:
            # Region area distribution
            areas = detection_data['region_areas']
            ax4.hist(areas, bins=min(10, len(areas)), color='#FFA726', alpha=0.7, edgecolor='black')
            ax4.set_title('Detected Region Area Distribution', fontweight='bold')
            ax4.set_xlabel('Area (pixels)')
            ax4.set_ylabel('Frequency')
            ax4.axvline(detection_data['average_region_area'], color='red', linestyle='--', 
                       label=f'Average: {detection_data["average_region_area"]:.0f}')
            ax4.legend()
        else:
            ax4.text(0.5, 0.5, 'No Regions Detected', transform=ax4.transAxes, 
                    ha='center', va='center', fontsize=16, fontweight='bold')
            ax4.set_title('Detection Results', fontweight='bold')
        
        plt.tight_layout()
        
        # Save performance chart
        perf_chart_path = os.path.join(output_dir, f"{filename}_performance.png")
        plt.savefig(perf_chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        chart_paths['performance'] = f"masks/{filename}_performance.png"
        
        print("✅ Performance charts created successfully")
        return chart_paths
        
    except Exception as e:
        print(f"❌ Error creating performance charts: {e}")
        return {}

def generate_analysis_report(image_path, results):
    """Generate detailed textual analysis report"""
    summary = analytics.get_summary()
    
    report = {
        'image_info': {
            'filename': os.path.basename(image_path),
            'processing_time': summary['timing']['total_processing_time'],
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        },
        'classification_analysis': {
            'prediction': results['label'],
            'confidence': results['confidence'],
            'confidence_interpretation': get_confidence_interpretation(results['confidence']),
            'probabilities': summary['classification']['probabilities']
        },
        'segmentation_analysis': {
            'regions_detected': summary['detection']['detected_regions'],
            'forged_percentage': summary['detection']['forged_percentage'],
            'average_region_size': summary['detection']['average_region_area'],
            'segmentation_quality': get_segmentation_quality(summary['segmentation'])
        },
        'forensic_analysis': {
            'ela_findings': analyze_ela_results(summary['forensic_features']['ela_stats']),
            'noise_findings': analyze_noise_results(summary['forensic_features']['noise_stats']),
            'gradient_findings': analyze_gradient_results(summary['forensic_features']['gradient_stats']),
            'texture_findings': analyze_texture_results(summary['forensic_features']['texture_stats'])
        },
        'technical_details': {
            'processing_breakdown': summary['timing'],
            'model_performance': calculate_model_performance(summary),
            'detection_metrics': summary['detection']
        },
        'recommendations': generate_recommendations(results, summary)
    }
    
    return report

def get_confidence_interpretation(confidence):
    """Interpret confidence levels"""
    if confidence >= 0.9:
        return "Very High - Strong evidence for the classification"
    elif confidence >= 0.8:
        return "High - Confident in the classification"
    elif confidence >= 0.7:
        return "Moderate - Reasonably confident"
    elif confidence >= 0.6:
        return "Low - Some uncertainty in classification"
    else:
        return "Very Low - High uncertainty, manual review recommended"

def get_segmentation_quality(seg_stats):
    """Assess segmentation quality"""
    if not seg_stats['mask_statistics']:
        return "No segmentation performed"
    
    confidence = seg_stats['confidence']
    std = seg_stats['mask_statistics'].get('std_intensity', 0)
    
    if confidence >= 0.8 and std > 50:
        return "Excellent - High confidence with good contrast"
    elif confidence >= 0.6:
        return "Good - Reliable segmentation results"
    elif confidence >= 0.4:
        return "Fair - Moderate reliability"
    else:
        return "Poor - Low confidence results"

def analyze_ela_results(ela_stats):
    """Analyze ELA results"""
    if not ela_stats:
        return "ELA analysis not available"
    
    energy = ela_stats.get('energy', 0)
    mean = ela_stats.get('mean', 0)
    
    if energy > 10000 and mean > 50:
        return "High ELA energy detected - Strong indication of compression artifacts or tampering"
    elif energy > 5000:
        return "Moderate ELA energy - Some compression inconsistencies detected"
    else:
        return "Low ELA energy - Minimal compression artifacts"

def analyze_noise_results(noise_stats):
    """Analyze noise residual results"""
    if not noise_stats:
        return "Noise analysis not available"
    
    std = noise_stats.get('std', 0)
    energy = noise_stats.get('energy', 0)
    
    if std > 20 and energy > 5000:
        return "High noise variance - Possible splicing or copy-move operations"
    elif std > 10:
        return "Moderate noise patterns - Some inconsistencies detected"
    else:
        return "Low noise variance - Consistent noise patterns"

def analyze_gradient_results(grad_stats):
    """Analyze gradient inconsistency results"""
    if not grad_stats:
        return "Gradient analysis not available"
    
    mean = grad_stats.get('mean', 0)
    energy = grad_stats.get('energy', 0)
    
    if mean > 30 and energy > 8000:
        return "High gradient inconsistencies - Strong evidence of boundary tampering"
    elif mean > 15:
        return "Moderate gradient anomalies - Some edge inconsistencies detected"
    else:
        return "Low gradient variance - Consistent edge characteristics"

def analyze_texture_results(texture_stats):
    """Analyze texture inconsistency results"""
    if not texture_stats:
        return "Texture analysis not available"
    
    std = texture_stats.get('std', 0)
    energy = texture_stats.get('energy', 0)
    
    if std > 25 and energy > 6000:
        return "High texture inconsistencies - Possible copy-move or splicing"
    elif std > 12:
        return "Moderate texture anomalies - Some pattern irregularities"
    else:
        return "Consistent texture patterns - No major anomalies detected"

def calculate_model_performance(summary):
    """Calculate overall model performance metrics"""
    timing = summary['timing']
    detection = summary['detection']
    
    # Processing efficiency (inverse of time)
    efficiency = 1.0 / max(timing['total_processing_time'], 0.1)
    
    # Detection accuracy (based on region detection and confidence)
    if detection['detected_regions'] > 0:
        accuracy = min(detection['detected_regions'] / 3.0, 1.0)  # Assume 3 regions is good
    else:
        accuracy = 0.5  # Neutral if no regions
    
    # Overall performance score
    performance_score = (efficiency * 0.3 + accuracy * 0.7) * summary['classification']['confidence']
    
    return {
        'efficiency_score': round(efficiency, 3),
        'detection_accuracy': round(accuracy, 3),
        'overall_score': round(performance_score, 3)
    }

def generate_recommendations(results, summary):
    """Generate actionable recommendations"""
    recommendations = []
    
    confidence = results['confidence']
    detected_regions = summary['detection']['detected_regions']
    
    if confidence < 0.7:
        recommendations.append("Consider manual review due to low classification confidence")
    
    if results['label'] == 'forged' and detected_regions == 0:
        recommendations.append("Forgery detected but no specific regions identified - may require different analysis techniques")
    
    if detected_regions > 5:
        recommendations.append("Multiple regions detected - consider investigating potential systematic tampering")
    
    if summary['timing']['total_processing_time'] > 10:
        recommendations.append("Processing time was high - consider optimizing image size or computational resources")
    
    # Forensic-specific recommendations
    forensic = summary['forensic_features']
    if forensic['ela_stats'] and forensic['ela_stats']['energy'] > 10000:
        recommendations.append("High ELA energy suggests significant compression artifacts - investigate source and history")
    
    if not recommendations:
        recommendations.append("Analysis completed successfully with good confidence levels")
    
    return recommendations

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
    start_time = time.time()
    
    image_rgb = Image.open(image_path).convert("RGB")
    input_tensor = classification_transform(image_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = classification_model(input_tensor)
        probs = F.softmax(logits, dim=1).cpu().numpy().flatten()
        predicted_class = int(probs.argmax())
        confidence = float(probs[predicted_class])

    # Log analytics
    analytics.log_timing("classification", time.time() - start_time)
    analytics.log_classification(probs, confidence)

    label_map = {0: "forged", 1: "unforged"}
    return {
        "class": predicted_class,
        "label": label_map[predicted_class],
        "confidence": confidence
    }

# === Enhanced Segmentation Function - FIXED ===
def segment_image(image_path, output_mask_path):
    """Enhanced segmentation with comprehensive analytics - FIXED"""
    start_time = time.time()
    
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
        
        # Log segmentation timing
        analytics.log_timing("segmentation", time.time() - start_time)
        
        # Enhanced post-processing with multiple thresholds
        detected_bboxes = []
        refined_mask_np = None
        region_scores = []
        
        # Try different minimum area thresholds
        for min_area in [300, 500, 800]:
            print(f"\n📊 Trying minimum area: {min_area}")
            refined_mask_np, detected_bboxes, region_scores = enhanced_post_process_mask(
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
                    region_scores.append(0.5)  # Default score for permissive detection
            
            print(f"✅ Permissive detection found {len(detected_bboxes)} regions")
        
        # Create enhanced visualization - FIXED
        if refined_mask_np is not None:
            highlighted_image = create_enhanced_highlight_visualization(
                image_path, 
                refined_mask_np, 
                alpha=0.4,
                highlight_color=(0, 0, 255)  # Red highlighting
            )
        else:
            # If no mask generated, create a copy of original
            highlighted_image = cv2.imread(image_path)

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_mask_path), exist_ok=True)
        
        # Save results
        if cv2.imwrite(output_mask_path, highlighted_image):
            print(f"✅ Enhanced segmentation saved to: {output_mask_path}")
        else:
            print(f"❌ Failed to save image to: {output_mask_path}")
        
        print(f"🎯 Total detected forged regions: {len(detected_bboxes)}")

        return detected_bboxes, region_scores

    except Exception as e:
        print(f"❌ Error in segment_image(): {e}")
        import traceback
        traceback.print_exc()
        return [], []

# === Enhanced Copy-Move Detection ===
def detect_copy_move_forgery(image_path, output_path):
    """Enhanced copy-move detection with analytics"""
    start_time = time.time()
    
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

        # Log copy-move timing
        analytics.log_timing("copy_move", time.time() - start_time)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, result_img)
        print(f"✅ Copy-move detection saved: {output_path}")
        print(f"🔍 Found {len(all_good_matches)} potential matches")
        return output_path

    except Exception as e:
        print(f"❌ Error in copy-move detection: {e}")
        analytics.log_timing("copy_move", time.time() - start_time)
        return None

# === Main Prediction Pipeline - FIXED ===
def run_prediction_pipeline(image_path, output_dir="static/masks"):
    """Enhanced prediction pipeline with comprehensive analytics - FIXED"""
    analytics.reset()  # Reset analytics for new analysis
    
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
    result["analytics_charts"] = {}
    result["analysis_report"] = {}

    # Step 2: If forged, perform localization
    if result["label"] == "forged":
        print("🎯 Image classified as FORGED. Running localization...")

        # Segmentation with highlighting
        mask_filename = f"{filename}_highlighted.png"
        full_mask_path = os.path.join(output_dir, mask_filename)
        forged_bboxes, region_scores = segment_image(image_path, full_mask_path)
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
                    # Add label with confidence if available
                    conf_text = f" ({region_scores[i]:.2f})" if i < len(region_scores) else ""
                    cv2.putText(overlay_img, f'Forged {i+1}{conf_text}', (x, y-10), 
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

    # Step 3: Generate comprehensive analytics
    print("📈 Generating performance analytics...")
    
    # Create performance charts
    chart_paths = create_performance_charts(output_dir, filename)
    result["analytics_charts"] = chart_paths
    
    # Generate detailed analysis report
    analysis_report = generate_analysis_report(image_path, result)
    result["analysis_report"] = analysis_report
    
    # Save analytics summary as JSON
    summary_filename = f"{filename}_analytics_summary.json"
    summary_path = os.path.join(output_dir, summary_filename)
    try:
        with open(summary_path, 'w') as f:
            json.dump(analysis_report, f, indent=2)
        result["analytics_summary_path"] = f"masks/{summary_filename}".replace("\\", "/")
        print(f"✅ Analytics summary saved: {summary_path}")
    except Exception as e:
        print(f"❌ Error saving analytics summary: {e}")

    return result