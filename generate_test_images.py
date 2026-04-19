"""
generate_test_images.py
-----------------------
Generates synthetic parking lot test images for demonstration purposes.
Run this BEFORE running main.py if you don't have real images.

Usage:
  python generate_test_images.py
"""

import cv2
import numpy as np
import os
import random
from pathlib import Path

OUTPUT_DIR = "dataset/sample_images"
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

COLORS = {
    "red":    (30,  30,  200),
    "blue":   (200, 80,  40),
    "white":  (240, 240, 240),
    "black":  (30,  30,  30),
    "gray":   (140, 140, 140),
    "green":  (40,  160, 60),
    "yellow": (30,  210, 220),
    "orange": (30,  130, 230),
}


def draw_car(img, x, y, w, h, color_bgr, plate_text=""):
    """Draw a simple car shape."""
    # Car body
    cv2.rectangle(img, (x, y), (x + w, y + h), color_bgr, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), 1)

    # Car roof (smaller rectangle on top)
    rx = x + w // 5
    rw = w * 3 // 5
    rh = h * 2 // 5
    roof_color = tuple(max(0, c - 40) for c in color_bgr)
    cv2.rectangle(img, (rx, y - rh), (rx + rw, y + 4), roof_color, -1)
    cv2.rectangle(img, (rx, y - rh), (rx + rw, y + 4), (0, 0, 0), 1)

    # Windshield
    cv2.rectangle(img, (rx + 4, y - rh + 4), (rx + rw - 4, y), (150, 200, 220), -1)

    # Wheels
    wh = h // 3
    for wx in [x + w // 5, x + w * 3 // 4]:
        cv2.ellipse(img, (wx, y + h), (wh, wh), 0, 0, 360, (30, 30, 30), -1)
        cv2.ellipse(img, (wx, y + h), (wh // 2, wh // 2), 0, 0, 360, (80, 80, 80), -1)

    # License plate
    if plate_text:
        px = x + w // 4
        py = y + h - h // 4
        pw = w // 2
        ph = h // 6
        cv2.rectangle(img, (px, py), (px + pw, py + ph), (230, 230, 230), -1)
        cv2.rectangle(img, (px, py), (px + pw, py + ph), (0, 0, 0), 1)
        font_scale = max(0.3, pw / 120)
        cv2.putText(img, plate_text, (px + 3, py + ph - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1)


def draw_parking_lot(n_spaces=8, n_cars=None):
    """
    Create a top-down-style parking lot image.
    Spaces are arranged in a row; some are filled with cars.
    """
    if n_cars is None:
        n_cars = random.randint(2, n_spaces - 1)

    W = 1280
    H = 480
    img = np.ones((H, W, 3), dtype=np.uint8) * 55  # asphalt gray

    # Road lines
    for i in range(0, W, 40):
        cv2.line(img, (i, H // 2), (i + 20, H // 2), (200, 200, 200), 1)

    space_w = W // n_spaces
    space_h = H * 2 // 3
    top_y = H // 6

    # Parking space outlines
    for i in range(n_spaces):
        x1 = i * space_w + 4
        y1 = top_y
        x2 = (i + 1) * space_w - 4
        y2 = top_y + space_h
        cv2.rectangle(img, (x1, y1), (x2, y2), (180, 180, 180), 2)

        # Space number
        cv2.putText(img, str(i + 1), (x1 + 8, y1 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)

    # Randomly pick which spaces have cars
    occupied_indices = random.sample(range(n_spaces), min(n_cars, n_spaces))
    color_names = list(COLORS.keys())

    for i in occupied_indices:
        color_name = random.choice(color_names)
        color_bgr = COLORS[color_name]

        # Car dimensions (slightly smaller than the space)
        margin = 10
        x = i * space_w + margin
        y = top_y + margin + 20
        w = space_w - margin * 2
        h = space_h - margin * 2 - 25

        # Generate random plate
        plate = (
            "".join(random.choices("ABCDEFGHJKLMNPRSTUVWXYZ", k=2))
            + "".join(random.choices("0123456789", k=3))
            + "".join(random.choices("ABCDEFGHJKLMNPRSTUVWXYZ", k=2))
        )

        draw_car(img, x, y, w, h, color_bgr, plate_text=plate)

    # Add some noise/texture
    noise = np.random.randint(0, 12, img.shape, dtype=np.uint8)
    img = cv2.add(img, noise)

    return img, len(occupied_indices), n_spaces - len(occupied_indices)


def main():
    print("Generating synthetic parking lot test images...")
    for i in range(1, 6):
        img, occupied, free = draw_parking_lot(
            n_spaces=random.randint(6, 10),
            n_cars=random.randint(2, 6)
        )
        fname = f"parking_lot_{i:02d}.jpg"
        fpath = os.path.join(OUTPUT_DIR, fname)
        cv2.imwrite(fpath, img)
        print(f"  Saved: {fname}  [occupied={occupied}, free={free}]")

    print(f"\nDone! Images saved to: {OUTPUT_DIR}/")
    print("Run the system with:")
    print("  python main.py --input dataset/sample_images --spaces 10 --save-csv")


if __name__ == "__main__":
    main()
