# ForgeScanX 🔍🖼️
**Image Forgery Detection using Deep Learning and Keypoint/Block Matching (No Segmentation)**

## 📌 Overview
ForgeScanX is a web-based tool that detects image forgeries using:
- ✅ **ResNet18-based Classification** – to detect whether an image is forged or not.
- 🔁 **Combined ORB Keypoint Matching + PCA Block Matching** – to localize forged regions (copy-move forgeries) when the image is classified as forged.

The project **does not use segmentation** masks. Instead, it overlays red dots (keypoints) and green rectangles (matched regions) on potentially forged areas.

---

## 🧠 Models Used
### 1. **ResNet18 Classification**
- Input: RGB image (224x224)
- Output: Binary label – `"forged"` or `"unforged"`

### 2. **Forgery Localization (Only for Forged Images)**
- ✅ **ORB (Oriented FAST and Rotated BRIEF)** – detects and matches keypoints (shown as red dots)
- ✅ **PCA-based block matching** – detects duplicated blocks in image (shown as green rectangles)

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

## ⚙️ How It Works

1. User uploads an image.
2. The image is classified:
   - **Unforged:** Only label + confidence is shown.
   - **Forged:** ORB + PCA are applied to detect forged areas.
3. A new image is saved with overlays showing suspected forged regions.

---

## 🚀 How to Run

### ✅ Install Dependencies

```bash
pip install fastapi uvicorn torch torchvision opencv-python pillow jinja2
✅ Start Server
bash
Copy
Edit
uvicorn main:app --reload
Visit http://127.0.0.1:8000 to use the web app.

🖼️ Output Example
🔴 Red Dots: ORB keypoints (matched).

🟩 Green Rectangles: Block matches found via PCA.

📌 Note
No UNet or segmentation used.

Only forged images trigger localization.

Images are processed as RGB and resized using ImageNet normalization.
