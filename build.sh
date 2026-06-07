#!/usr/bin/env bash
# Build script for Render deployment

set -o errexit

echo "==> Installing dependencies..."
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

echo "==> Generating dataset..."
python generate_dataset.py

echo "==> Training ML models..."
python train_college_model.py

echo "==> Build complete!"
