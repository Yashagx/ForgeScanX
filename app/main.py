import os
from uuid import uuid4
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from utils.prediction_pipeline import classify_image, segment_image  # Import both
from PIL import Image
from pathlib import Path

app = FastAPI()

# Static and template setup
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Ensure upload & mask folders exist
UPLOAD_DIR = "static/uploads"
MASK_DIR = "static/masks"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(MASK_DIR, exist_ok=True)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    # Save uploaded image
    file_ext = os.path.splitext(file.filename)[-1]
    unique_filename = f"{uuid4().hex}{file_ext}"
    image_path = os.path.join(UPLOAD_DIR, unique_filename)

    with open(image_path, "wb") as f:
        f.write(await file.read())

    # Run classification
    result = classify_image(image_path)
    print("🔍 Classification result:", result)

    # If forged, run segmentation
    mask_path = None
    if result["label"].lower() == "forged":
        mask_filename = f"{Path(unique_filename).stem}_mask.png"
        mask_path = os.path.join(MASK_DIR, mask_filename)
        print("🧠 Segmenting image...")

        try:
            segment_image(image_path, mask_path)
            print(f"✅ Mask saved at: {mask_path}")
        except Exception as e:
            print(f"❌ Segmentation error: {e}")
            mask_path = None

    # Strip "static/" prefix to get relative path for HTML
    relative_mask_path = mask_path.replace("static/", "") if mask_path else None

    return templates.TemplateResponse("result.html", {
        "request": request,
        "filename": unique_filename,
        "label": result["label"],
        "confidence": result["confidence"],
        "mask_path": relative_mask_path
    })
