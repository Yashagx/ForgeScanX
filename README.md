# ForgeScanX 🔍🖼️
**Advanced Image Forgery Detection and Localization leveraging Deep Learning and Hybrid Feature Extraction**

## 📌 Overview
ForgeScanX is an innovative, web-based platform engineered for robust image forgery detection and precise localization of manipulated regions. This system integrates state-of-the-art deep learning models for classification with sophisticated computer vision techniques for fine-grained segmentation and localization, offering a comprehensive solution for digital image forensics.

---

## 🧠 Core Methodologies

### 1. **Image Classification for Authenticity Assessment**
Our primary classification module is built upon a fine-tuned **ResNet18 CNN architecture**. This model is meticulously trained to discern between authentic and forged images, serving as the initial gatekeeper in our detection pipeline.

* **Model:** ResNet18 (pre-trained on ImageNet and fine-tuned on diverse image forgery datasets)
* **Input:** Normalized RGB image (preprocessed to $224 \times 224$ pixels)
* **Output:** Binary classification label – "Authentic" or "Forged" – accompanied by a confidence score.
* **Key Features:** Leverages hierarchical feature extraction to identify subtle artifacts indicative of various forgery types (e.g., copy-move, image splicing, retouching).

### 2. **Forged Region Localization and Segmentation**
Upon classifying an image as "Forged," ForgeScanX activates its specialized localization and segmentation modules to pinpoint the exact manipulated areas. This multi-pronged approach ensures high accuracy and visual clarity of detected forgeries.

#### 2.1. **Deep Learning-based Semantic Segmentation (U-Net Architecture)**
For precise pixel-level identification of forged regions, we employ a **U-Net convolutional network**. This architecture is particularly adept at semantic segmentation tasks, enabling the generation of high-resolution forgery masks.

* **Model:** U-Net (custom-trained for forgery mask generation)
* **Input:** Full-resolution or downsampled RGB image (depending on U-Net input requirements)
* **Output:** Binary segmentation mask, where pixels corresponding to forged regions are highlighted.
* **Purpose:** Provides a high-level, semantic understanding of manipulated areas, crucial for various forgery types beyond simple copy-move.

#### 2.2. **Hybrid Feature Matching for Copy-Move Forgery Localization**
To specifically address copy-move forgeries, ForgeScanX integrates a powerful hybrid feature matching strategy that combines global keypoint descriptors with local block-based analysis.

* **a. ORB (Oriented FAST and Rotated BRIEF) Keypoint Matching:**
    * **Mechanism:** Detects robust, rotation-invariant keypoints across the image and computes their descriptors. These descriptors are then efficiently matched to identify duplicated patterns.
    * **Visualization:** Matched keypoints are visually represented as **red dots** overlaid on the output image.
    * **Benefit:** Effective for detecting copy-move operations even with slight rotations or scaling.

* **b. PCA-based Block Matching:**
    * **Mechanism:** Divides the image into overlapping blocks and applies Principal Component Analysis (PCA) to reduce the dimensionality of block features. Similar blocks (indicating potential copy-move operations) are then identified by comparing their low-dimensional PCA representations.
    * **Visualization:** Duplicated blocks are highlighted with **green bounding boxes** overlaid on the output image.
    * **Benefit:** Complements keypoint matching by identifying larger, coherent duplicated regions, often more resilient to minor post-processing.

---

- ## 📁 Project Structure

~~~
ForgeScanX/
│
├── models/
│ └── classification/
│ └── classification_model.pth # Trained ResNet18 model
│
├── static/
│ ├── uploads/ # Uploaded test images
│ └── masks/ # Output overlays (forged region maps)
│
├── templates/
│ ├── index.html # Upload page
│ └── result.html # Results page
│
├── utils/
│ └── prediction_pipeline.py # Full classification + forgery detection logic
│
├── main.py # FastAPI backend
└── README.md
~~~

---

## ⚙️ Operational Workflow

1.  **Image Submission:** A user uploads an image via the intuitive web interface.
2.  **Initial Preprocessing:** The uploaded image undergoes standardized preprocessing (e.g., resizing to $224 \times 224$ for classification, normalization using ImageNet statistics).
3.  **Authenticity Classification:** The preprocessed image is fed into the **ResNet18 classifier**.
    * If classified as "Authentic," the system displays the label and confidence score.
    * If classified as "Forged," the process proceeds to advanced localization.
4.  **Forged Region Detection (for "Forged" images only):**
    * **U-Net Segmentation:** The image is processed by the **U-Net model** to generate a precise pixel-level forgery mask.
    * **Hybrid Feature Matching:** Simultaneously, **ORB keypoint detection and matching** and **PCA-based block matching** are executed to identify copy-move instances.
5.  **Result Visualization:** A composite output image is generated, featuring:
    * The original image.
    * The U-Net generated **segmentation mask** (e.g., highlighted in a semi-transparent color).
    * **Red dots** indicating matched ORB keypoints for copy-move localization.
    * **Green rectangles** denoting duplicated blocks identified by PCA for copy-move localization.
    * A summary of the classification result and confidence.
6.  **Output Delivery:** The processed image and analytical findings are presented to the user on the `result.html` page.

---

## 🚀 Getting Started

### ✅ Prerequisites
Ensure you have Python 3.8+ installed.

### ✅ Install Dependencies

```bash
pip install fastapi uvicorn torch torchvision opencv-python scikit-learn pillow jinja2
✅ Start the Server
Bash

uvicorn main:app --reload
Navigate your web browser to http://127.0.0.1:8000 to interact with the ForgeScanX web application.

🖼️ Illustrative Output
Example of a Forged Image Analysis:

(Imagine an image here with the following overlays)

Semi-transparent Blue Overlay: Represents the pixel-level forgery mask generated by the U-Net segmentation model, indicating the precise boundaries of the manipulated region.

🔴 Red Dots: Show matched ORB keypoints, highlighting areas involved in copy-move operations.

🟩 Green Rectangles: Delineate duplicated image blocks found via PCA-based matching, further pinpointing copy-move instances.

📌 Technical Notes
Deep Learning Framework: PyTorch is utilized for all neural network models.

Image Preprocessing: All input images undergo rigorous preprocessing, including resizing and normalization based on ImageNet statistics, to ensure optimal model performance.

Scalability: The modular design of ForgeScanX allows for easy integration of more advanced models or feature extraction techniques in the future.

Robustness: The hybrid approach combining deep learning segmentation with classical feature matching enhances the robustness of forgery detection across diverse manipulation scenarios.






