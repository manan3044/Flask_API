#!/bin/bash

# Install Tesseract
apt-get update
apt-get install -y tesseract-ocr libtesseract-dev libleptonica-dev pkg-config

# Install FFmpeg
apt-get install -y ffmpeg

# Start the Flask app
gunicorn --bind 0.0.0.0:8000 app:app
