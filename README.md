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






GIVE DOWNLOADABLE VERSION


Creating a "downloadable version" of a FastAPI web application for users to run locally without installing Python or its dependencies can be achieved using tools called "bundlers" or "freezers." These tools package your Python application, its dependencies, and a Python interpreter into a single executable file or a standalone directory.

The most popular and recommended tool for this purpose is PyInstaller.

Here's how you can prepare ForgeScanX for a downloadable distribution using PyInstaller, along with the necessary updates to your README.md.

Guide for a Downloadable Version (using PyInstaller)
Since ForgeScanX is a web-based application (using FastAPI and Uvicorn), packaging it for local desktop use requires embedding a web server. When the user runs the executable, it will launch the web server in the background, and they can then access the application through their web browser by navigating to http://127.0.0.1:8000 (or whatever port you configure).

1. Update your main.py for PyInstaller compatibility
For PyInstaller to correctly bundle Uvicorn, you might need to explicitly tell Uvicorn how to run your app within the main.py file, rather than relying solely on uvicorn main:app --reload.

Modify your main.py like this:

Python

# main.py
import uvicorn
from fastapi import FastAPI
import os
import sys

# Add these lines to ensure PyInstaller finds all necessary paths
# This helps with issues related to static files, templates, and models
if getattr(sys, 'frozen', False):
    # Running in a PyInstaller bundle
    base_path = sys._MEIPASS
else:
    # Running in normal Python environment
    base_path = os.path.dirname(os.path.abspath(__file__))

# Adjust paths in your app if you use relative paths for models, static, or templates
# Example:
# models_dir = os.path.join(base_path, 'models', 'classification')
# templates_dir = os.path.join(base_path, 'templates')
# static_uploads_dir = os.path.join(base_path, 'static', 'uploads')
# static_masks_dir = os.path.join(base_path, 'static', 'masks')

# Ensure these directories exist within the bundled app
# os.makedirs(static_uploads_dir, exist_ok=True)
# os.makedirs(static_masks_dir, exist_ok=True)

app = FastAPI()

# Import your prediction pipeline and other modules
from utils.prediction_pipeline import run_forgery_detection # Assuming this is your main function

# --- Your FastAPI routes will go here ---
@app.get("/")
async def root():
    # You might want to serve your index.html here directly or via a Starlette template response
    from fastapi.responses import HTMLResponse
    with open(os.path.join(base_path, 'templates', 'index.html'), 'r') as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

# Example for serving static files (adjust based on your actual structure)
from starlette.staticfiles import StaticFiles
app.mount("/static", StaticFiles(directory=os.path.join(base_path, "static")), name="static")


@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    # Your existing upload logic
    # Make sure to save to and read from the adjusted paths like static_uploads_dir
    # Call your detection pipeline
    # return results via templates.TemplateResponse or JSONResponse
    pass # Placeholder for your actual upload logic

# Ensure Uvicorn runs when the executable is launched
if __name__ == "__main__":
    # You can choose a different port if 8000 might conflict
    uvicorn.run(app, host="127.0.0.1", port=8000)
Key Adjustments in main.py:

sys._MEIPASS: This is a special attribute set by PyInstaller that points to the temporary directory where it extracts your bundled application. It's crucial for accessing static files, templates, and model weights correctly.

Path Management: All references to files (models, templates, static files) should be made relative to base_path.

Uvicorn Run: The if __name__ == "__main__": uvicorn.run(app, host="127.0.0.1", port=8000) block is essential. When PyInstaller creates the executable, it will run this block, starting the Uvicorn server.

Serving HTML/Static Files: Ensure your FastAPI app can serve the index.html and other static assets (CSS, JS, images) from within the bundled executable.

2. Create a requirements.txt file
This file should list all Python packages your project depends on.

fastapi
uvicorn
torch
torchvision
opencv-python
pillow
jinja2
# Add any other specific libraries used by your models or utilities
3. Install PyInstaller
Bash

pip install pyinstaller
4. Bundle your application with PyInstaller
Navigate to your ForgeScanX/ root directory in your terminal and run the PyInstaller command.

Bash

pyinstaller main.py --name ForgeScanX --onefile --windowed --add-data "models;models" --add-data "static;static" --add-data "templates;templates"
Let's break down this command:

pyinstaller main.py: Tells PyInstaller to build an executable from main.py.

--name ForgeScanX: Sets the name of the executable to "ForgeScanX".

--onefile: Creates a single executable file. This is generally more convenient for distribution, though it might take longer to start as it unpacks itself. (Alternatively, --onedir creates a directory with the executable and its dependencies, which can be faster for repeated launches).

--windowed or -w: This is crucial for a web application. It prevents a console window from popping up when the executable is run. The web server will run in the background.

--add-data "source;destination": This option is vital for including your models/, static/, and templates/ directories.

"models;models": Copies the models directory from your project root into a models directory within the executable's temporary runtime location.

"static;static": Copies the static directory.

"templates;templates": Copies the templates directory.

Important Considerations for --add-data:

The source path is relative to where you run the pyinstaller command.

The destination path is the name of the folder inside the PyInstaller's temporary sys._MEIPASS directory. You use this destination path in your main.py when constructing os.path.join(base_path, 'destination', 'your_file.ext').

You must ensure all files and directories accessed by your application (models, static assets, templates) are included using --add-data.

5. Locate the Executable
After successful execution, PyInstaller will create two new directories: build/ and dist/. Your standalone executable will be in the dist/ForgeScanX/ (if using --onedir) or directly in dist/ (if using --onefile). The exact location and file extension will depend on your operating system (e.g., .exe on Windows, no extension on Linux/macOS).

6. Distribute
You can now zip the dist/ folder (or just the single executable if --onefile was used) and distribute it. Users can simply run this executable.

Updated README.md Section for Downloadable Version
Add a new section to your README.md like this:

Markdown

---

## 📦 Downloadable Version (Standalone Application)

For users who prefer to run ForgeScanX locally without a Python environment setup, we provide a pre-packaged standalone application. This version bundles the entire application, including a minimal web server, into a single executable or a self-contained directory.

### ✅ Download

* **[Link to Latest Windows Executable]** (e.g., `ForgeScanX-Windows.zip`)
* **[Link to Latest macOS Executable]** (e.g., `ForgeScanX-macOS.zip`)
* **[Link to Latest Linux Executable]** (e.g., `ForgeScanX-Linux.tar.gz`)

*(You will need to build these executables on their respective operating systems and then upload them to a release page on GitHub, a file hosting service, or your own website.)*

### ⚙️ How to Use the Downloadable Version

1.  **Download:** Download the appropriate package for your operating system from the links above.
2.  **Extract:** Unzip or extract the downloaded archive.
3.  **Run:**
    * **Windows:** Double-click the `ForgeScanX.exe` file.
    * **macOS / Linux:** Open a terminal, navigate to the extracted directory, and run `./ForgeScanX`.
4.  **Access:** Once the application launches (it might run silently in the background), open your web browser and navigate to `http://127.0.0.1:8000`.
5.  **Exit:** To close the application, simply close the web browser tab/window and then typically terminate the background process (e.g., by finding the process in Task Manager on Windows or using `Ctrl+C` in the terminal if you ran it from there on Linux/macOS, or closing the console window that might appear if `--windowed` wasn't perfectly effective).

### ⚠️ Important Notes for Downloadable Version:

* **Platform Specific:** Executables are built for specific operating systems (Windows, macOS, Linux). A Windows executable will not run on macOS or Linux, and vice-versa.
* **Resource Usage:** The bundled application may have a larger file size due to containing a Python interpreter and all dependencies.
* **Performance:** Initial launch might take a few seconds as the bundled application unpacks its contents.
* **Antivirus Software:** Due to the nature of executable bundling, some antivirus software might flag the application. This is generally a false positive.
* **No Console:** The web server runs in the background. If you need to see server logs or detailed output, you might need to run the executable from a terminal (by removing the `--windowed` flag during bundling, for development/debugging purposes).

---
