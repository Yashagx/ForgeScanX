import os
from uuid import uuid4
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from utils.prediction_pipeline import run_prediction_pipeline
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
    
    # Run the complete prediction pipeline
    print(f"🚀 Running complete analysis pipeline for: {unique_filename}")
    
    try:
        # Use the enhanced pipeline that returns all template variables
        result = run_prediction_pipeline(image_path, output_dir=MASK_DIR)
        print("✅ Pipeline completed successfully")
        
        # Add request to the result dictionary
        result["request"] = request
        
        # Debug: Print result keys to verify all template variables are present
        print(f"Result keys: {list(result.keys())}")
        
        # Return the template with all required variables
        return templates.TemplateResponse("result.html", result)
        
    except Exception as e:
        print(f"❌ Pipeline error: {e}")
        import traceback
        traceback.print_exc()
        
        # Fallback response with minimal data to prevent template errors
        fallback_result = {
            "request": request,
            "filename": unique_filename,
            "label": "error",
            "confidence": 0.0,
            "mask_path": None,
            "copy_move_path": None,
            "forged_objects_bboxes": [],
            "forged_objects_overlay_path": None,
            "processing_time": 0.0,
            "regions_detected": 0,
            "forged_percentage": 0.0,
            "current_time": "00:00:00",
            "current_date": "2024-01-01",
            "forged_probability": 0.5,
            "unforged_probability": 0.5,
            "classification_time": 0.0,
            "segmentation_time": 0.0,
            "post_processing_time": 0.0,
            "copy_move_time": 0.0,
            "max_time": 1.0,
            "ela_energy": 0.0,
            "noise_energy": 0.0,
            "gradient_energy": 0.0,
            "texture_energy": 0.0,
            "efficiency_score": 0.0,
            "detection_accuracy": 0.0,
            "overall_score": 0.0,
            "confidence_interpretation": "Error occurred during analysis",
            "region_areas": [],
            "average_region_size": 0.0,
            "recommendations": ["An error occurred during analysis. Please try again."]
        }
        
        return templates.TemplateResponse("result.html", fallback_result)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)