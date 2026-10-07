"""
VisDrone Object Detection — Inference Module
============================================
Loads the trained YOLO11s model and performs inference on images.
"""

import io
import os
import base64
from pathlib import Path

import numpy as np
from PIL import Image

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "best.pt"

# Class names — must match the training dataset order
CLASS_NAMES = [
    "pedestrian",      # 0
    "people",          # 1
    "bicycle",         # 2
    "car",             # 3
    "van",             # 4
    "truck",           # 5
    "tricycle",        # 6
    "awning-tricycle", # 7
    "bus",             # 8
    "motor",           # 9
]

# Colour palette per class (BGR → used for OpenCV, but PIL draws in RGB)
CLASS_COLORS = [
    (255, 87, 51),   # pedestrian   — orange-red
    (255, 195, 0),   # people       — golden
    (0, 204, 102),   # bicycle      — green
    (0, 122, 255),   # car          — blue
    (153, 0, 255),   # van          — purple
    (255, 51, 153),  # truck        — pink
    (0, 204, 204),   # tricycle     — teal
    (255, 128, 0),   # awning-tri   — amber
    (204, 0, 0),     # bus          — dark red
    (51, 153, 255),  # motor        — light blue
]

# ──────────────────────────────────────────────────────────────────────────────
# Model loader (singleton)
# ──────────────────────────────────────────────────────────────────────────────
_model = None


def get_model(model_path: str | Path | None = None):
    """Load (and cache) the YOLO model.  Thread-safe for single-worker use."""
    global _model
    if _model is None:
        from ultralytics import YOLO

        path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        if not path.exists():
            raise FileNotFoundError(
                f"Model weights not found at '{path}'.\n"
                "Make sure 'models/best.pt' exists in the project root.\n"
                "See README.md § 'Model Weights' for download instructions."
            )
        _model = YOLO(str(path))
    return _model


# ──────────────────────────────────────────────────────────────────────────────
# Inference helpers
# ──────────────────────────────────────────────────────────────────────────────

def validate_image(image_bytes: bytes) -> Image.Image:
    """Validate and convert raw bytes to a PIL Image."""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.verify()                    # raises if corrupt
        img = Image.open(io.BytesIO(image_bytes))   # re-open after verify
        img = img.convert("RGB")
        return img
    except Exception as exc:
        raise ValueError(f"Invalid image: {exc}") from exc


def run_inference(
    image_bytes: bytes,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    imgsz: int = 960,
) -> dict:
    """
    Run YOLO11s inference on raw image bytes.

    Parameters
    ----------
    image_bytes    : raw bytes of the uploaded image
    conf_threshold : minimum confidence to keep a detection
    iou_threshold  : NMS IoU threshold
    imgsz          : inference image size (model was trained at 960)

    Returns
    -------
    dict with keys:
        annotated_b64   str   base64-encoded annotated JPEG
        detections      list  raw detection list (boxes, scores, class ids)
        summary         dict  count per class
        image_size      tuple (width, height) of original image
    """
    pil_img = validate_image(image_bytes)
    orig_w, orig_h = pil_img.size

    model = get_model()
    results = model.predict(
        source=pil_img,
        conf=conf_threshold,
        iou=iou_threshold,
        imgsz=imgsz,
        verbose=False,
    )

    result = results[0]
    boxes  = result.boxes

    detections = []
    summary    = {}

    if boxes is not None and len(boxes):
        xyxy   = boxes.xyxy.cpu().numpy()     # [N, 4]
        confs  = boxes.conf.cpu().numpy()     # [N]
        cls_ids = boxes.cls.cpu().int().numpy()  # [N]

        for i in range(len(xyxy)):
            cid   = int(cls_ids[i])
            cname = CLASS_NAMES[cid] if cid < len(CLASS_NAMES) else f"class_{cid}"
            det   = {
                "class_id":   cid,
                "class_name": cname,
                "confidence": float(round(float(confs[i]), 4)),
                "bbox": {
                    "x1": float(round(float(xyxy[i][0]), 2)),
                    "y1": float(round(float(xyxy[i][1]), 2)),
                    "x2": float(round(float(xyxy[i][2]), 2)),
                    "y2": float(round(float(xyxy[i][3]), 2)),
                },
            }
            detections.append(det)
            summary[cname] = summary.get(cname, 0) + 1

    # Build annotated image
    annotated_np = result.plot(line_width=2)          # numpy RGB
    annotated_pil = Image.fromarray(annotated_np)
    buf = io.BytesIO()
    annotated_pil.save(buf, format="JPEG", quality=90)
    annotated_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return {
        "annotated_b64": annotated_b64,
        "detections":    detections,
        "summary":       summary,
        "image_size":    (orig_w, orig_h),
        "total_detections": len(detections),
    }
