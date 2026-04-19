"""
exporter.py
-----------
Saves detection results to a CSV file and exports annotated images.
"""

import csv
import os
import cv2
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


def save_results_to_csv(all_results: list[dict], output_path: str = "results/parking_results.csv"):
    """
    Save detection results for multiple images to a CSV file.

    Args:
        all_results: List of result dicts, each containing:
            - 'filename': source image name
            - 'cars': list of car dicts with id, color, plate, confidence
            - 'car_count': int
            - 'free_spaces': int
            - 'occupied_spaces': int
        output_path: Where to save the CSV.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    fieldnames = [
        "timestamp",
        "image_filename",
        "car_id",
        "color",
        "plate",
        "confidence",
        "total_cars",
        "occupied_spaces",
        "free_spaces",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for result in all_results:
            filename = result.get("filename", "unknown")
            car_count = result.get("car_count", 0)
            occupied = result.get("occupied_spaces", 0)
            free = result.get("free_spaces", 0)

            if not result.get("cars"):
                # Write a row even if no cars detected
                writer.writerow({
                    "timestamp": timestamp,
                    "image_filename": filename,
                    "car_id": "-",
                    "color": "-",
                    "plate": "-",
                    "confidence": "-",
                    "total_cars": car_count,
                    "occupied_spaces": occupied,
                    "free_spaces": free,
                })
            else:
                for car in result["cars"]:
                    writer.writerow({
                        "timestamp": timestamp,
                        "image_filename": filename,
                        "car_id": car.get("id", "-"),
                        "color": car.get("color", "unknown"),
                        "plate": car.get("plate", "N/A"),
                        "confidence": car.get("confidence", 0.0),
                        "total_cars": car_count,
                        "occupied_spaces": occupied,
                        "free_spaces": free,
                    })

    logger.info(f"Results saved to: {output_path}")
    return str(output_path)


def save_annotated_image(image, filename: str, output_dir: str = "results/annotated"):
    """
    Save an annotated image to disk.

    Args:
        image: BGR NumPy array.
        filename: Original image filename (used to derive output name).
        output_dir: Directory to save annotated images.

    Returns:
        Path to saved image.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(filename).stem
    out_path = out_dir / f"{stem}_annotated.jpg"

    cv2.imwrite(str(out_path), image)
    logger.info(f"Annotated image saved: {out_path}")
    return str(out_path)


def print_console_report(result: dict, filename: str = ""):
    """
    Print a formatted console report for a single image result.

    Args:
        result: Detection result dict.
        filename: Image filename for the header.
    """
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  SMART PARKING SYSTEM — DETECTION REPORT")
    if filename:
        print(f"  Image: {filename}")
    print(sep)
    print(f"  🚗  Cars detected      : {result['car_count']}")
    print(f"  🟢  Free spaces        : {result['free_spaces']}")
    print(f"  🔴  Occupied spaces    : {result['occupied_spaces']}")
    print(f"  📊  Total spaces       : {result['car_count'] + result['free_spaces']}")
    print("-" * 60)

    if result["cars"]:
        print(f"  {'ID':<5} {'Color':<12} {'Plate':<15} {'Type':<12} {'Conf'}")
        print(f"  {'-'*4} {'-'*11} {'-'*14} {'-'*11} {'-'*5}")
        for car in result["cars"]:
            print(
                f"  {car['id']:<5} "
                f"{car['color']:<12} "
                f"{car['plate']:<15} "
                f"{car['class']:<12} "
                f"{car['confidence']:.2f}"
            )
    else:
        print("  No vehicles detected.")

    print(sep)
