"""
detector.py
-----------
Core detection module for the Smart Parking System.
Handles:
  - Car detection using YOLOv8
  - License plate detection
  - Car color detection
  - OCR on license plates
"""

import cv2
import numpy as np
import easyocr
import logging
from ultralytics import YOLO
from parking_system.color_detector import detect_car_color
from parking_system.plate_detector import detect_and_read_plate

# Configure logging
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class ParkingDetector:
    """
    Main class that orchestrates all detection tasks for a parking image.
    """

    def __init__(self, model_path: str = "yolov8n.pt", total_spaces: int = 10):
        """
        Initialize the detector.

        Args:
            model_path: Path to the YOLOv8 model weights.
            total_spaces: Total number of parking spaces in the lot.
        """
        logger.info("Loading YOLOv8 model...")
        self.model = YOLO(model_path)  # Auto-downloads if not found

        logger.info("Initializing EasyOCR reader...")
        self.ocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)

        self.total_spaces = total_spaces

        # COCO class IDs for vehicles
        self.vehicle_classes = {
            2: "car",
            3: "motorcycle",
            5: "bus",
            7: "truck",
        }

        logger.info("ParkingDetector initialized successfully.")

    def detect(self, image: np.ndarray) -> dict:
        """
        Run full detection pipeline on a single image.

        Args:
            image: BGR image as a NumPy array (loaded via OpenCV).

        Returns:
            A dictionary with detection results:
              - 'annotated_image': image with all bounding boxes drawn
              - 'cars': list of detected car dicts
              - 'car_count': int
              - 'free_spaces': int
              - 'occupied_spaces': int
        """
        results = {
            "annotated_image": image.copy(),
            "cars": [],
            "car_count": 0,
            "free_spaces": 0,
            "occupied_spaces": 0,
        }

        annotated = image.copy()

        # --- Step 1: Run YOLO inference ---
        logger.info("Running YOLO inference...")
        yolo_results = self.model(image, verbose=False)[0]
        boxes = yolo_results.boxes

        detected_vehicles = []
        for box in boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            # Only keep vehicles above confidence threshold
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

            # Crop vehicle from image
            vehicle_crop = image[y1:y2, x1:x2]
            if vehicle_crop.size == 0:
                continue

            # 2a: Detect dominant color
            color_name, color_bgr = detect_car_color(vehicle_crop)

            # 2b: Detect license plate + OCR
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

            # --- Step 3: Draw annotations on the image ---
            annotated = self._draw_vehicle_box(
                annotated, car_info, color_bgr, plate_bbox, (x1, y1)
            )

        # --- Step 4: Compute space counts ---
        occupied = len(results["cars"])
        free = max(0, self.total_spaces - occupied)

        results["car_count"] = occupied
        results["occupied_spaces"] = occupied
        results["free_spaces"] = free

        # Draw parking space summary on image
        annotated = self._draw_summary(annotated, occupied, free)
        results["annotated_image"] = annotated

        return results

    def _draw_vehicle_box(
        self,
        image: np.ndarray,
        car_info: dict,
        color_bgr: tuple,
        plate_bbox,
        vehicle_offset: tuple,
    ) -> np.ndarray:
        """Draw bounding box, label, and plate info for a vehicle."""
        x1, y1, x2, y2 = car_info["bbox"]
        car_id = car_info["id"]
        plate = car_info["plate"]
        color_name = car_info["color"]

        # Main vehicle bounding box (green)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 200, 0), 2)

        # Color swatch indicator (filled small rectangle beside box)
        swatch_x = x2 + 4
        swatch_y = y1
        cv2.rectangle(image, (swatch_x, swatch_y), (swatch_x + 16, swatch_y + 16), color_bgr, -1)
        cv2.rectangle(image, (swatch_x, swatch_y), (swatch_x + 16, swatch_y + 16), (255, 255, 255), 1)

        # Background for label
        label = f"#{car_id} | {color_name} | {plate}"
        (lw, lh), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_bg_y1 = max(y1 - lh - baseline - 6, 0)
        cv2.rectangle(image, (x1, label_bg_y1), (x1 + lw + 6, y1), (0, 0, 0), -1)

        # Label text
        cv2.putText(
            image, label,
            (x1 + 3, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
        )

        # Draw plate bounding box inside vehicle crop (if found)
        if plate_bbox is not None:
            px1, py1, px2, py2 = plate_bbox
            # Offset back to full image coords
            ox, oy = vehicle_offset
            cv2.rectangle(
                image,
                (ox + px1, oy + py1),
                (ox + px2, oy + py2),
                (0, 165, 255), 2  # Orange for plate
            )
            cv2.putText(
                image, plate,
                (ox + px1, oy + py1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1, cv2.LINE_AA
            )

        return image

    def _draw_summary(self, image: np.ndarray, occupied: int, free: int) -> np.ndarray:
        """Draw a summary overlay at the top of the image."""
        overlay = image.copy()
        h, w = image.shape[:2]

        # Semi-transparent dark banner
        cv2.rectangle(overlay, (0, 0), (w, 50), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, image, 0.25, 0, image)

        summary = (
            f"SMART PARKING SYSTEM  |  "
            f"Occupied: {occupied}  |  "
            f"Free: {free}  |  "
            f"Total: {occupied + free}"
        )
        cv2.putText(
            image, summary,
            (10, 33),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 160), 2, cv2.LINE_AA
        )
        return image
