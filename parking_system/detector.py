"""
detector.py
-----------
Core detection module for the Smart Parking System.
Fixed: _draw_summary now accepts real occupied/free counts from DB.
"""

import cv2
import numpy as np
import easyocr
import logging
from ultralytics import YOLO
from parking_system.color_detector import detect_car_color
from parking_system.plate_detector import detect_and_read_plate

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class ParkingDetector:
    """
    Main class that orchestrates all detection tasks for a parking image.
    """

    def __init__(self, model_path: str = "yolov8n.pt", total_spaces: int = 30):
        logger.info("Loading YOLOv8 model...")
        self.model = YOLO(model_path)

        logger.info("Initializing EasyOCR reader...")
        self.ocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)

        self.total_spaces = total_spaces

        self.vehicle_classes = {
            2: "car",
            3: "motorcycle",
            5: "bus",
            7: "truck",
        }

        logger.info("ParkingDetector initialized successfully.")

    def detect(self, image: np.ndarray, db_occupied: int = None) -> dict:
        """
        Run full detection pipeline on a single image.

        Args:
            image: BGR image as NumPy array.
            db_occupied: Real occupied count from DB (for banner display).
                         If None, uses only current image count.

        Returns:
            Detection results dict.
        """
        results = {
            "annotated_image": image.copy(),
            "cars": [],
            "car_count": 0,
            "free_spaces": 0,
            "occupied_spaces": 0,
        }

        annotated = image.copy()

        # --- Step 1: YOLO inference ---
        logger.info("Running YOLO inference...")
        yolo_results = self.model(image, verbose=False)[0]
        boxes = yolo_results.boxes

        detected_vehicles = []
        for box in boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            if class_id in self.vehicle_classes and confidence >= 0.35:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detected_vehicles.append({
                    "bbox": (x1, y1, x2, y2),
                    "class": self.vehicle_classes[class_id],
                    "confidence": confidence,
                })

        logger.info(f"Detected {len(detected_vehicles)} vehicle(s).")

        # --- Step 2: Per-vehicle analysis ---
        for i, vehicle in enumerate(detected_vehicles):
            x1, y1, x2, y2 = vehicle["bbox"]
            vehicle_crop = image[y1:y2, x1:x2]
            if vehicle_crop.size == 0:
                continue

            color_name, color_bgr = detect_car_color(vehicle_crop)
            plate_text, plate_bbox = detect_and_read_plate(vehicle_crop, self.ocr_reader)

            car_info = {
                "id": i + 1,
                "bbox": vehicle["bbox"],
                "class": vehicle["class"],
                "confidence": round(vehicle["confidence"], 2),
                "color": color_name,
                "plate": plate_text if plate_text else "N/A",
            }
            results["cars"].append(car_info)

            annotated = self._draw_vehicle_box(
                annotated, car_info, color_bgr, plate_bbox, (x1, y1)
            )

        # --- Step 3: Compute space counts ---
        new_cars = len(results["cars"])

        # Use DB count if provided (for accurate cumulative display)
        if db_occupied is not None:
            # After adding these new cars, total occupied = db_occupied + new_cars
            total_occupied = db_occupied + new_cars
        else:
            total_occupied = new_cars

        free = max(0, self.total_spaces - total_occupied)

        results["car_count"] = new_cars
        results["occupied_spaces"] = total_occupied
        results["free_spaces"] = free

        # Draw banner with REAL cumulative counts
        annotated = self._draw_summary(annotated, total_occupied, free)
        results["annotated_image"] = annotated

        return results

    def _draw_vehicle_box(self, image, car_info, color_bgr, plate_bbox, vehicle_offset):
        x1, y1, x2, y2 = car_info["bbox"]
        car_id = car_info["id"]
        plate = car_info["plate"]
        color_name = car_info["color"]

        # Bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 200, 0), 2)

        # Color swatch
        swatch_x = x2 + 4
        swatch_y = y1
        cv2.rectangle(image, (swatch_x, swatch_y), (swatch_x + 16, swatch_y + 16), color_bgr, -1)
        cv2.rectangle(image, (swatch_x, swatch_y), (swatch_x + 16, swatch_y + 16), (255, 255, 255), 1)

        # Label background
        label = f"#{car_id} | {color_name} | {plate}"
        (lw, lh), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_bg_y1 = max(y1 - lh - baseline - 6, 0)
        cv2.rectangle(image, (x1, label_bg_y1), (x1 + lw + 6, y1), (0, 0, 0), -1)

        # Label text
        cv2.putText(image, label, (x1 + 3, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Plate box
        if plate_bbox is not None:
            px1, py1, px2, py2 = plate_bbox
            ox, oy = vehicle_offset
            cv2.rectangle(image, (ox + px1, oy + py1), (ox + px2, oy + py2), (0, 165, 255), 2)
            cv2.putText(image, plate, (ox + px1, oy + py1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1, cv2.LINE_AA)

        return image

    def _draw_summary(self, image: np.ndarray, occupied: int, free: int) -> np.ndarray:
        """Draw summary banner with REAL cumulative parking counts."""
        overlay = image.copy()
        h, w = image.shape[:2]

        cv2.rectangle(overlay, (0, 0), (w, 50), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, image, 0.25, 0, image)

        summary = (
            f"SMART PARKING SYSTEM  |  "
            f"Occupied: {occupied}  |  "
            f"Free: {free}  |  "
            f"Total: {self.total_spaces}"
        )
        cv2.putText(image, summary, (10, 33),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 160), 2, cv2.LINE_AA)
        return image