"""
plate_detector.py - Fast OCR, optimized for speed
"""

import cv2
import numpy as np
import re
import logging

logger = logging.getLogger(__name__)


def clean_plate(text):
    cleaned = re.sub(r"[^A-Z0-9 ]", "", text.upper().strip())
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned if len(cleaned.replace(" ", "")) >= 3 else ""


def detect_and_read_plate(vehicle_crop, ocr_reader):
    if vehicle_crop is None or vehicle_crop.size == 0:
        return "N/A", None

    h, w = vehicle_crop.shape[:2]
    best_text = ""
    best_conf = 0.0

    # FAST: only 1 size instead of 3
    target_widths = [600]
    regions = []

    for tw in target_widths:
        scale = tw / w
        resized = cv2.resize(vehicle_crop,
                             (tw, int(h * scale)),
                             interpolation=cv2.INTER_CUBIC)
        rh = resized.shape[0]

        # Only 2 regions instead of 4
        regions.append(resized)               # full crop
        regions.append(resized[rh//2:, :])    # bottom half only

    for region in regions:
        if region is None or region.size == 0:
            continue

        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if len(region.shape) == 3 else region

        # Only 2 versions instead of 5
        imgs_to_try = [
            region,   # original color
            gray,     # grayscale
        ]

        for img in imgs_to_try:
            try:
                results = ocr_reader.readtext(img, detail=1, paragraph=False)
                for (bbox_pts, text, conf) in results:
                    cleaned = clean_plate(text)
                    if cleaned and conf > best_conf:
                        best_conf = conf
                        best_text = cleaned
            except Exception as e:
                logger.debug(f"OCR err: {e}")
                continue

        # Early exit if good result found
        if best_conf > 0.5 and len(best_text.replace(" ", "")) >= 4:
            break

    result = best_text.upper().strip() if best_text else "N/A"
    return result, None