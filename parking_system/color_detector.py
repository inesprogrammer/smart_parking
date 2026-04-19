"""
color_detector.py
-----------------
Detects the dominant color of a cropped car image.

Strategy:
  1. Convert the image to HSV color space (better for color segmentation).
  2. Mask out the top 30% of the crop (often sky/background) and bottom 15% (road/shadow).
  3. Use HSV range matching to classify the dominant hue bucket.
  4. Return a human-readable color name and its BGR representation for drawing.
"""

import cv2
import numpy as np


# --- Color definitions in HSV space ---
# Each entry: (color_name, lower_hsv, upper_hsv, bgr_for_display)
COLOR_RANGES = [
    # Red (wraps around 0/180 in HSV, so we define two ranges)
    ("red",   np.array([0,   70, 50]),  np.array([10,  255, 255]), (0,   0,   200)),
    ("red",   np.array([170, 70, 50]),  np.array([180, 255, 255]), (0,   0,   200)),
    # Orange
    ("orange", np.array([11,  70, 50]),  np.array([25, 255, 255]), (0,   128, 255)),
    # Yellow
    ("yellow", np.array([26,  70, 50]),  np.array([35, 255, 255]), (0,   210, 255)),
    # Green
    ("green",  np.array([36,  50, 50]),  np.array([85, 255, 255]), (0,   180, 0)),
    # Blue
    ("blue",   np.array([86,  50, 50]),  np.array([130, 255, 255]), (200, 50,  0)),
    # Purple / Violet
    ("purple", np.array([131, 50, 50]),  np.array([160, 255, 255]), (180, 0,   180)),
    # Pink
    ("pink",   np.array([161, 50, 50]),  np.array([169, 255, 255]), (180, 105, 255)),
    # White — low saturation, high value
    ("white",  np.array([0,   0,  200]), np.array([180, 40, 255]),  (245, 245, 245)),
    # Gray — low saturation, mid value
    ("gray",   np.array([0,   0,  80]),  np.array([180, 40, 200]),  (150, 150, 150)),
    # Black — very low value
    ("black",  np.array([0,   0,  0]),   np.array([180, 255, 79]),  (30,  30,  30)),
]


def detect_car_color(car_crop: np.ndarray) -> tuple[str, tuple]:
    """
    Detect the dominant color of a car from its cropped image.

    Args:
        car_crop: BGR image crop of the detected vehicle.

    Returns:
        A tuple of (color_name: str, bgr_color: tuple)
        e.g. ("blue", (200, 50, 0))
    """
    if car_crop is None or car_crop.size == 0:
        return "unknown", (128, 128, 128)

    h, w = car_crop.shape[:2]

    # Mask the top 30% (sky/background) and bottom 15% (road/tyres)
    top_cut = int(h * 0.30)
    bot_cut = int(h * 0.85)
    roi = car_crop[top_cut:bot_cut, :]

    if roi.size == 0:
        roi = car_crop  # fallback: use full crop

    # Convert to HSV
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Resize to small patch for speed (color analysis doesn't need full res)
    hsv_small = cv2.resize(hsv, (64, 32), interpolation=cv2.INTER_AREA)

    # Count pixels per color bucket
    color_scores = {}
    for entry in COLOR_RANGES:
        name, lower, upper, bgr = entry
        mask = cv2.inRange(hsv_small, lower, upper)
        count = cv2.countNonZero(mask)
        # Accumulate (red has two ranges)
        color_scores[name] = color_scores.get(name, 0) + count

    # Pick the color with the most matching pixels
    if not color_scores or max(color_scores.values()) == 0:
        return "unknown", (128, 128, 128)

    best_color = max(color_scores, key=color_scores.get)

    # Find the BGR representation for this color
    bgr = (128, 128, 128)  # default gray
    for entry in COLOR_RANGES:
        if entry[0] == best_color:
            bgr = entry[3]
            break

    return best_color, bgr
