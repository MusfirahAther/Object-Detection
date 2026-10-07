"""
Vercel Serverless API Entry Point
===================================
This module adapts the Flask app for Vercel's serverless Python runtime.

⚠️  DEPLOYMENT LIMITATION — READ BEFORE DEPLOYING TO VERCEL:
    PyTorch + ultralytics combined size exceeds Vercel's 250 MB Lambda limit.

    Recommended alternatives for this model:
      • HuggingFace Spaces (free, supports PyTorch natively)
      • Render.com (free tier, Docker support)
      • Railway.app (Docker support)
      • Google Cloud Run / AWS Lambda container images

    If you still wish to deploy on Vercel, use the ONNX path:
      1. Run:  python scripts/export_onnx.py
      2. Replace `requirements.txt` with `requirements-onnx.txt`
      3. Ensure `models/best.onnx` exists and update MODEL_PATH below.

See README.md § "Deployment" for complete instructions.
"""

import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app import app

# Vercel expects the WSGI app to be named `app` or `application`
# The import above exposes `app` automatically.


# ──────────────────────────────────────────────────────────────────────────────
# Vercel handler
# ──────────────────────────────────────────────────────────────────────────────
# Vercel calls this module as a serverless function.
# The `app` object (Flask WSGI) is auto-detected by Vercel's Python runtime.
