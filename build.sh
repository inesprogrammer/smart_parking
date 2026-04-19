#!/bin/bash
pip install -r requirements.txt
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
python -c "import easyocr; easyocr.Reader(['en'], gpu=False)"