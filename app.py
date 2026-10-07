"""
VisDrone Object Detection — Flask Application Entry Point
=========================================================
Run locally:
    python app.py

For production / Vercel see: api/index.py
"""

import os
import sys

# Ensure project root is on the path so `src` is importable
sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask, render_template, request, jsonify

from src.inference import run_inference, CLASS_NAMES

app = Flask(__name__)

# 16 MB upload limit
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "webp", "tif", "tiff"}


def _allowed(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ──────────────────────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html", class_names=CLASS_NAMES)


@app.route("/predict", methods=["POST"])
def predict():
    """Accept an image upload, run YOLO11s inference, return JSON."""
    if "image" not in request.files:
        return jsonify({"error": "No image field in request"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not _allowed(file.filename):
        return jsonify({
            "error": f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        }), 400

    conf  = float(request.form.get("conf",  0.25))
    iou   = float(request.form.get("iou",   0.45))
    imgsz = int(request.form.get("imgsz",   960))

    # Clamp values
    conf  = max(0.01, min(conf,  0.99))
    iou   = max(0.01, min(iou,   0.99))
    imgsz = max(320,  min(imgsz, 1280))

    try:
        image_bytes = file.read()
        result = run_inference(image_bytes, conf_threshold=conf, iou_threshold=iou, imgsz=imgsz)
        return jsonify({"success": True, **result})
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 500
    except Exception as exc:
        return jsonify({"error": f"Inference failed: {exc}"}), 500


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model": "yolo11s", "classes": len(CLASS_NAMES)})


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
