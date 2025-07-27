# 🔍 ForgeScanX



> "Accelerating image authenticity checks using parallel SIFT-based analysis."



## 📁 Project Structure
- `data/` – Datasets (raw, preprocessed, and results)
- `src/` – Core implementation (SIFT, RANSAC, parallelization)
- `notebooks/` – Experimentation and visual debugging
- `visualizations/` – Metrics plots and output overlays
- `gui/` – (Optional) Simple interface for users
- `reports/` – Research paper, summary, and presentation
- `logs/` – Processing stats, performance, and errors

## 🔧 Tech Stack
- Python 3.x
- OpenCV (SIFT, image processing)
- NumPy, Matplotlib, Seaborn
- Joblib / Multiprocessing (for parallelism)
- Optional: Flask/Tkinter for GUI

## 📊 Datasets Used
- [CoMoFoD](http://domingomery.ing.puc.cl/CoMoFoD)
- [MICC-F220](https://www.micc.unifi.it/vim/datasets/micc-f220/)
- [CASIA TIDEv2](https://github.com/OrcunCetintas/CASIA-TIDEv2-Dataset)

## ✅ Objectives
- Detect copy-move forgery in digital images
- Accelerate detection using parallel SIFT matching
- Provide visual and statistical verification

---

Stay tuned for updates as development progresses.
 