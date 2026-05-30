"""
exporter.py
-----------
Saves detection results to a CSV file (APPEND mode) and exports annotated images.
"""

import csv
import os
import cv2
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

PARKING_CSV = "static/results/parking_history.csv"


def append_result_to_csv(car: dict, entry_time: str, place: str, total_spaces: int, occupied: int):
    """
    Append a single car detection to the global parking CSV.
    Creates the file if it doesn't exist.
    """
    output_path = Path(PARKING_CSV)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "entry_time",
        "place",
        "plate",
        "color",
        "confidence",
        "total_spaces",
        "occupied_spaces",
        "free_spaces",
    ]

    file_exists = output_path.exists()

    with open(output_path, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "entry_time":      entry_time,
            "place":           place,
            "plate":           car.get("plate", "N/A"),
            "color":           car.get("color", "unknown"),
            "confidence":      round(car.get("confidence", 0.0), 2),
            "total_spaces":    total_spaces,
            "occupied_spaces": occupied,
            "free_spaces":     total_spaces - occupied,
        })

    logger.info(f"Car appended to CSV: {car.get('plate')} → {place}")
    return str(output_path)


def read_all_parked_cars():
    """
    Read all currently parked cars from the CSV.
    Returns list of dicts.
    """
    output_path = Path(PARKING_CSV)
    if not output_path.exists():
        return []

    cars = []
    with open(output_path, "r", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            cars.append(row)
    return cars


def get_occupied_count():
    """Return number of cars currently in the parking."""
    return len(read_all_parked_cars())


def reset_parking_csv():
    """Clear the parking CSV (reset all spaces)."""
    output_path = Path(PARKING_CSV)
    if output_path.exists():
        output_path.unlink()
    logger.info("Parking CSV reset.")


def save_results_to_csv(all_results: list, output_path: str = "results/parking_results.csv"):
    """Legacy function — kept for compatibility."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fieldnames = ["timestamp", "image_filename", "car_id", "color", "plate",
                  "confidence", "total_cars", "occupied_spaces", "free_spaces"]
    with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in all_results:
            filename = result.get("filename", "unknown")
            car_count = result.get("car_count", 0)
            occupied = result.get("occupied_spaces", 0)
            free = result.get("free_spaces", 0)
            if not result.get("cars"):
                writer.writerow({"timestamp": timestamp, "image_filename": filename,
                    "car_id": "-", "color": "-", "plate": "-", "confidence": "-",
                    "total_cars": car_count, "occupied_spaces": occupied, "free_spaces": free})
            else:
                for car in result["cars"]:
                    writer.writerow({"timestamp": timestamp, "image_filename": filename,
                        "car_id": car.get("id", "-"), "color": car.get("color", "unknown"),
                        "plate": car.get("plate", "N/A"), "confidence": car.get("confidence", 0.0),
                        "total_cars": car_count, "occupied_spaces": occupied, "free_spaces": free})
    return str(output_path)


def save_annotated_image(image, filename: str, output_dir: str = "results/annotated"):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(filename).stem
    out_path = out_dir / f"{stem}_annotated.jpg"
    cv2.imwrite(str(out_path), image)
    logger.info(f"Annotated image saved: {out_path}")
    return str(out_path)


def print_console_report(result: dict, filename: str = ""):
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  SMART PARKING SYSTEM — DETECTION REPORT")
    if filename:
        print(f"  Image: {filename}")
    print(sep)
    print(f"  Cars detected      : {result['car_count']}")
    print(f"  Free spaces        : {result['free_spaces']}")
    print(f"  Occupied spaces    : {result['occupied_spaces']}")
    print(sep)