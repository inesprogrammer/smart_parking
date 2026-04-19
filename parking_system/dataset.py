"""
dataset.py
----------
Handles loading and preprocessing of parking lot images from a folder.
Supports: .jpg, .jpeg, .png, .bmp, .tiff, .webp
"""

import cv2
import os
import logging
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def load_images_from_folder(folder_path: str) -> list[tuple[str, np.ndarray]]:
    """
    Load all supported images from a folder.

    Args:
        folder_path: Path to the directory containing images.

    Returns:
        List of (filename, image_array) tuples.
        Images are BGR NumPy arrays as returned by cv2.imread.
    """
    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"Image folder not found: {folder_path}")

    image_files = sorted([
        f for f in folder.iterdir()
        if f.suffix.lower() in SUPPORTED_EXTENSIONS
    ])

    if not image_files:
        logger.warning(f"No supported images found in: {folder_path}")
        return []

    loaded = []
    for img_path in image_files:
        img = cv2.imread(str(img_path))
        if img is None:
            logger.warning(f"Could not read image: {img_path.name}")
            continue
        loaded.append((img_path.name, img))
        logger.info(f"Loaded: {img_path.name}  [{img.shape[1]}x{img.shape[0]}]")

    logger.info(f"Total images loaded: {len(loaded)}")
    return loaded


def preprocess_image(
    image: np.ndarray,
    target_width: int = 1280,
    normalize: bool = False
) -> np.ndarray:
    """
    Preprocess an image for inference.

    Steps:
      - Resize to target_width while preserving aspect ratio
      - Optionally normalize pixel values to [0, 1]

    Args:
        image: Input BGR image.
        target_width: Max width to resize to (larger images are downscaled).
        normalize: If True, convert to float32 in range [0, 1].

    Returns:
        Preprocessed image array.
    """
    h, w = image.shape[:2]

    # Only resize if wider than target_width
    if w > target_width:
        scale = target_width / w
        new_w = target_width
        new_h = int(h * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        logger.debug(f"Resized from {w}x{h} to {new_w}x{new_h}")

    if normalize:
        image = image.astype(np.float32) / 255.0

    return image


def load_single_image(image_path: str) -> np.ndarray:
    """
    Load a single image from disk.

    Args:
        image_path: Full path to the image file.

    Returns:
        BGR NumPy array, or raises FileNotFoundError.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"OpenCV could not decode image: {image_path}")

    return img
