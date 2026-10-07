# 🚁 VisDrone Object Detection with YOLO11s

A fine-tuned **YOLO11s** object detector for drone-view imagery, trained on the **VisDrone 2019 DET** benchmark.  
The project ships a **Flask web application** that accepts any image, runs real-time inference, and returns an annotated result.

---

## 📋 Table of Contents
- [Problem Statement](#problem-statement)
- [Features](#features)
- [Model Overview](#model-overview)
- [Dataset](#dataset)
- [Training](#training)
- [Evaluation Results](#evaluation-results)
- [Project Structure](#project-structure)
- [Quick Start — Local](#quick-start--local)
- [API Reference](#api-reference)
- [Environment Variables](#environment-variables)
- [Deployment](#deployment)
- [Model Weights](#model-weights)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)

---

## Problem Statement

Detecting small, densely packed objects from drone-view imagery is significantly harder than standard ground-level detection.  
VisDrone images present extreme scale variation, heavy occlusion, and severe class imbalance (e.g., bicycles and awning-tricycles are 20–50× rarer than cars and pedestrians).

This project explores:
1. A **YOLO11n baseline** (nano model, 640 px) to establish a performance floor.
2. An **oversampled YOLO11n** to address class imbalance for rare categories.
3. A **final YOLO11s** (small model, 960 px) that delivers the best mAP among all experiments.

---

## Features

- 🎯 **10-class detection** — pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor
- 🌐 **Web interface** — drag & drop upload, adjustable confidence / IoU / image-size controls
- 🔌 **REST API** — `/predict` endpoint returns JSON with bounding boxes, class labels, and confidence scores
- 📊 **Full evaluation artifacts** — confusion matrices, PR/F1/P/R curves, per-class AP tables
- 🏋️ **Training reproducibility** — all hyperparameter YAML files included

---

## Model Overview

| Property | Value |
|---|---|
| Architecture | YOLO11s (small) |
| Input size | 960 × 960 px |
| Pretrained base | COCO (via ultralytics) |
| Fine-tuning epochs | 15 |
| Optimizer | Auto (AdamW) |
| Batch size | 8 |
| Classes | 10 |
| Weight file | `models/best.pt` (≈ 18.3 MB) |

---

## Dataset

**VisDrone 2019 DET** — drone-captured object detection benchmark  
[https://github.com/VisDrone/VisDrone-Dataset](https://github.com/VisDrone/VisDrone-Dataset)

| Split | Images |
|---|---|
| Train | 6 471 |
| Validation | 548 |
| Test-dev | 1 610 |

**Classes (10):**
`pedestrian` · `people` · `bicycle` · `car` · `van` · `truck` · `tricycle` · `awning-tricycle` · `bus` · `motor`

Dataset is **not included** in this repository due to size. See [Dataset Setup](#dataset-setup) below.

---

## Training

Three experiments were conducted in sequence:

### Experiment 1 — Baseline YOLO11n (640 px, 10 epochs)
- Small nano model, standard 640 px input
- mAP@50 = **24.1%** · mAP@50-95 = **13.4%**

### Experiment 2 — Oversampled YOLO11n (640 px, 5 epochs)
- Same architecture; rare classes (bicycle ×2, tricycle ×2, awning-tricycle ×3, bus ×3) oversampled
- mAP@50 = **24.2%** · mAP@50-95 = **13.4%** — marginal gain, oversampling helped rare-class recall slightly

### Experiment 3 — **Final YOLO11s (960 px, 15 epochs)** ✅
- Larger model (small vs nano) + higher resolution (960 vs 640)
- mAP@50 = **46.6%** · mAP@50-95 = **27.9%** — nearly **2× improvement** over baseline

All training hyperparameters are documented in [`training/`](training/).

---

## Evaluation Results

Results are from the **VisDrone validation set** (548 images).

### Overall — Final YOLO11s Model

| Metric | Value |
|---|---|
| Precision | 56.5 % |
| Recall | 45.0 % |
| **mAP@50** | **46.6 %** |
| **mAP@50-95** | **27.9 %** |

### Per-Class AP — Final YOLO11s Model

| Class | AP@50 | AP@50-95 |
|---|---|---|
| pedestrian | 54.1 % | 25.9 % |
| people | 40.0 % | 15.7 % |
| bicycle | 23.7 % | 10.8 % |
| **car** | **83.7 %** | **59.4 %** |
| van | 51.5 % | 36.7 % |
| truck | 43.1 % | 29.5 % |
| tricycle | 33.7 % | 19.6 % |
| awning-tricycle | 19.4 % | 11.8 % |
| **bus** | **59.9 %** | **44.6 %** |
| motor | 53.7 % | 24.6 % |

### Evaluation Artifacts

| File | Description |
|---|---|
| [`evaluation/val/confusion_matrix.png`](evaluation/val/confusion_matrix.png) | Confusion matrix |
| [`evaluation/val/confusion_matrix_normalized.png`](evaluation/val/confusion_matrix_normalized.png) | Normalised confusion matrix |
| [`evaluation/val/BoxF1_curve.png`](evaluation/val/BoxF1_curve.png) | F1-score vs confidence |
| [`evaluation/val/BoxP_curve.png`](evaluation/val/BoxP_curve.png) | Precision vs confidence |
| [`evaluation/val/BoxR_curve.png`](evaluation/val/BoxR_curve.png) | Recall vs confidence |
| [`evaluation/val/BoxPR_curve.png`](evaluation/val/BoxPR_curve.png) | Precision–Recall curve |
| [`evaluation/yolo11s_960_training_curves.png`](evaluation/yolo11s_960_training_curves.png) | Training loss & metric curves |
| [`evaluation/metrics/yolo11s_960_results.csv`](evaluation/metrics/yolo11s_960_results.csv) | Per-epoch training metrics |
| [`evaluation/metrics/yolo11s_960_per_class_ap.csv`](evaluation/metrics/yolo11s_960_per_class_ap.csv) | Final per-class AP |
| [`evaluation/comparisons/`](evaluation/comparisons/) | Baseline vs oversampled confusion comparison |

---

## Project Structure

```
visdrone-yolo11s/
│
├── app.py                     # Flask application entry point
├── requirements.txt           # Pinned Python dependencies
├── vercel.json                # Vercel deployment config
├── .env.example               # Environment variable template
├── .gitignore
│
├── api/
│   └── index.py               # Vercel serverless handler
│
├── src/
│   └── inference.py           # YOLO model loading + inference logic
│
├── models/
│   └── best.pt                # ✅ Final YOLO11s weights (18.3 MB)
│
├── data/
│   └── dataset.yaml           # VisDrone dataset config (training/eval)
│
├── training/
│   ├── yolo11s_960_args.yaml  # Final model hyperparameters
│   ├── baseline_yolo11n_args.yaml
│   └── oversampled_yolo11n_args.yaml
│
├── evaluation/
│   ├── yolo11s_960_training_curves.png
│   ├── metrics/               # CSV metrics for all experiments
│   ├── val/                   # Confusion matrices + P/R/F1 curves
│   ├── baselines/             # Baseline experiment charts
│   └── comparisons/           # Baseline vs oversampled comparison
│
├── templates/
│   └── index.html             # Web UI
│
├── static/
│   ├── css/style.css
│   └── js/app.js
│
└── scripts/
    └── export_onnx.py         # One-time ONNX export for lightweight deploy
```

---

## Quick Start — Local

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/visdrone-yolo11s.git
cd visdrone-yolo11s
```

### 2. Create a virtual environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **CPU-only torch** is used by default (faster to install, sufficient for inference).  
> For GPU: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`

### 4. Verify the model weights

The `models/best.pt` file (**18.3 MB**) is included in the repository.  
If it is missing (e.g. after cloning without Git LFS), see [Model Weights](#model-weights).

### 5. Run the application

```bash
python app.py
```

Open [http://localhost:5000](http://localhost:5000) in your browser.

---

## API Reference

### `POST /predict`

Run object detection on an uploaded image.

**Form data:**

| Field | Type | Default | Description |
|---|---|---|---|
| `image` | file | — | Image file (JPG, PNG, BMP, WEBP, TIF) |
| `conf` | float | `0.25` | Minimum confidence threshold (0.01 – 0.99) |
| `iou` | float | `0.45` | NMS IoU threshold (0.01 – 0.99) |
| `imgsz` | int | `960` | Inference image size (320 / 640 / 960 / 1280) |

**Response (JSON):**

```json
{
  "success": true,
  "annotated_b64": "<base64-jpeg>",
  "total_detections": 12,
  "summary": { "car": 7, "pedestrian": 3, "motor": 2 },
  "detections": [
    {
      "class_id": 3,
      "class_name": "car",
      "confidence": 0.8821,
      "bbox": { "x1": 142.0, "y1": 88.5, "x2": 310.2, "y2": 195.1 }
    }
  ],
  "image_size": [1920, 1080]
}
```

### `GET /health`

Returns `{"status": "ok", "model": "yolo11s", "classes": 10}`.

---

## Environment Variables

Copy `.env.example` to `.env` and set as needed:

| Variable | Default | Description |
|---|---|---|
| `SECRET_KEY` | — | Flask session secret (required in production) |
| `FLASK_DEBUG` | `false` | Enable Flask debug mode |
| `PORT` | `5000` | Local server port |
| `MODEL_PATH` | `models/best.pt` | Override model weight path |
| `VISDRONE_PATH` | — | Root path to VisDrone dataset (training only) |

---

## Dataset Setup

*(Required only for retraining / re-evaluation — not needed for inference.)*

1. Download VisDrone 2019 DET from the [official repository](https://github.com/VisDrone/VisDrone-Dataset).
2. Extract into a local directory, e.g. `/data/VisDrone/`.
3. Set the environment variable:
   ```bash
   export VISDRONE_PATH=/data/VisDrone
   ```
4. Update `data/dataset.yaml` if you use a different structure.

---

## Deployment

### Option A — Local / Render / Railway (Recommended)

These platforms support Docker and have no meaningful dependency-size limits.

```bash
# Using gunicorn (included in requirements.txt)
gunicorn --bind 0.0.0.0:$PORT app:app
```

**Render** example (free tier):  
Create a new Web Service → connect repo → set Build Command to `pip install -r requirements.txt` → Start Command to `gunicorn app:app`.

### Option B — HuggingFace Spaces (Free GPU)

1. Create a new Space → select **Gradio** or **Docker** runtime.
2. Push this repo; HuggingFace natively supports PyTorch.

### Option C — Vercel ⚠️

> **Known Limitation:**  
> Vercel serverless functions have a **250 MB compressed** dependency limit.  
> `torch` + `ultralytics` + `opencv` exceed this limit and **cannot be deployed** on Vercel's standard Python runtime.

**Workaround — ONNX + onnxruntime-cpu:**

1. Export the model to ONNX once (requires `torch` locally):
   ```bash
   python scripts/export_onnx.py
   ```
2. This produces `models/best.onnx` (~37 MB).
3. Replace `requirements.txt` dependencies:
   - Remove: `torch`, `torchvision`, `ultralytics`
   - Add: `onnxruntime-cpu==1.20.1`
4. Update `src/inference.py` to use onnxruntime (see inline comments).
5. Deploy to Vercel — the `vercel.json` is pre-configured for the file structure.

---

## Model Weights

The trained weights file `models/best.pt` (18.3 MB) is committed to this repository.

If you are cloning and the file is missing (e.g., repository was cloned without LFS), you can re-download from the original Kaggle training run or contact the project author.

> **Note:** `models/best.pt` is the final best checkpoint from the YOLO11s 960-px, 15-epoch training run.  
> It corresponds to **epoch 15** (training was not interrupted early — patience=100 was not triggered).

---

## Limitations

1. **Small object detection** — VisDrone images contain very small objects; performance on images taken much closer/further than training examples may vary.
2. **Rare classes** — `bicycle`, `awning-tricycle`, and `tricycle` have AP@50 below 25 % due to limited training samples even after oversampling.
3. **Drone-view domain** — The model was trained specifically on drone-captured imagery and may generalise poorly to ground-level or satellite imagery.
4. **Inference speed** — At 960 px on CPU, each image takes 2–8 seconds. A GPU significantly reduces latency.
5. **Vercel constraint** — PyTorch dependencies exceed Vercel's 250 MB serverless limit; see Deployment § Option C above.

---

## Future Improvements

- [ ] Export to ONNX / TensorRT for faster CPU/GPU inference
- [ ] SAHI (Slicing Aided Hyper Inference) for very small object detection
- [ ] Train for more epochs (50+) with cosine LR schedule
- [ ] Data augmentation specifically targeting scale variation (mosaic at multiple resolutions)
- [ ] Integrate class-balanced focal loss to further address class imbalance
- [ ] Deploy to HuggingFace Spaces with GPU acceleration

---

## References

- VisDrone Dataset: [https://github.com/VisDrone/VisDrone-Dataset](https://github.com/VisDrone/VisDrone-Dataset)
- Ultralytics YOLO11: [https://docs.ultralytics.com](https://docs.ultralytics.com)
- Zhu et al. — *"VisDrone-DET2019: The Vision Meets Drone Object Detection in Image Challenge Results"*, ICCVW 2019
