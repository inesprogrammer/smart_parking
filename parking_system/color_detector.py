"""
color_detector.py
-----------------
Detects the dominant color of a cropped car image using HSV color space.
Fixed: white vs gray vs black distinction improved.
"""

import cv2
import numpy as np


# --- Color definitions in HSV space ---
COLOR_RANGES = [
    # Red (wraps around 0/180 in HSV)
    ("red",    np.array([0,   100, 80]),  np.array([10,  255, 255]), (0,   0,   200)),
    ("red",    np.array([170, 100, 80]),  np.array([180, 255, 255]), (0,   0,   200)),
    # Orange
    ("orange", np.array([11,  100, 80]),  np.array([25,  255, 255]), (0,   128, 255)),
    # Yellow
    ("yellow", np.array([26,  100, 80]),  np.array([35,  255, 255]), (0,   210, 255)),
    # Green
    ("green",  np.array([36,  60,  60]),  np.array([85,  255, 255]), (0,   180, 0)),
    # Blue
    ("blue",   np.array([86,  60,  60]),  np.array([130, 255, 255]), (200, 50,  0)),
    # Purple
    ("purple", np.array([131, 60,  60]),  np.array([160, 255, 255]), (180, 0,   180)),
    # Pink
    ("pink",   np.array([161, 60,  60]),  np.array([169, 255, 255]), (180, 105, 255)),
    # White — very low saturation, high value (lowered to 180 to catch more whites)
    ("white", np.array([0, 0, 150]), np.array([180, 60, 255]), (245, 245, 245)),
    # Gray — low saturation, MEDIUM value (80-180)
    ("gray",   np.array([0,   0,   80]),  np.array([180, 45,  180]), (150, 150, 150)),
    # Black — very low value (below 80)
    ("black",  np.array([0,   0,   0]),   np.array([180, 80,  79]),  (30,  30,  30)),
]


def detect_car_color(car_crop: np.ndarray) -> tuple:
    """
    Detect the dominant color of a car from its cropped image.

    Returns:
        A tuple of (color_name: str, bgr_color: tuple)
    """
    if car_crop is None or car_crop.size == 0:
        return "unknown", (128, 128, 128)

    h, w = car_crop.shape[:2]

    # Mask the top 30% (sky/background) and bottom 15% (road/tyres)
    top_cut = int(h * 0.30)
    bot_cut = int(h * 0.85)
    roi = car_crop[top_cut:bot_cut, :]

    if roi.size == 0:
        roi = car_crop

    # Convert to HSV
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Resize for speed
    hsv_small = cv2.resize(hsv, (64, 32), interpolation=cv2.INTER_AREA)

    # Count pixels per color bucket
    color_scores = {}
    for entry in COLOR_RANGES:
        name, lower, upper, bgr = entry
        mask = cv2.inRange(hsv_small, lower, upper)
        count = cv2.countNonZero(mask)
        color_scores[name] = color_scores.get(name, 0) + count

    # Calculate average brightness and saturation
    avg_v = np.mean(hsv_small[:, :, 2])
    avg_s = np.mean(hsv_small[:, :, 1])

    best_color = max(color_scores, key=color_scores.get)

    # --- White vs Gray vs Black disambiguation ---

    # If detected gray but brightness is high → it's white
    if best_color == "gray" and avg_v > 100:
        best_color = "white"

    # If detected white but brightness is not high enough → it's gray
    elif best_color == "white" and avg_v < 120:
        best_color = "gray"

    # If detected black but brightness > 80 → it's gray
    elif best_color == "black" and avg_v > 80:
        best_color = "gray"

    # If detected gray but brightness < 60 → it's black
    elif best_color == "gray" and avg_v < 60:
        best_color = "black"

    if not color_scores or max(color_scores.values()) == 0:
        return "unknown", (128, 128, 128)

    # Find BGR for the best color
    bgr = (128, 128, 128)
    for entry in COLOR_RANGES:
        if entry[0] == best_color:
            bgr = entry[3]
            break

    return best_color, bgr