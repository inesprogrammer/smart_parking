"""
app.py
------
Flask web interface for the Smart Parking System.

Features:
  - Upload a parking lot image through the browser
  - Displays the annotated image with detected cars, plates, colors
  - Shows a structured results table
  - Download annotated image and CSV

Run:
  python app.py
Then open: http://localhost:5000
"""

import os
import sys
import cv2
import base64
import logging
import numpy as np
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

sys.path.insert(0, str(Path(__file__).parent))

from parking_system import (
    ParkingDetector,
    load_single_image,
    preprocess_image,
    save_results_to_csv,
    save_annotated_image,
)

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# --- App config ---
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB max upload
app.config["UPLOAD_FOLDER"] = "static/uploads"
app.config["RESULTS_FOLDER"] = "static/results"

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "webp"}

# Create required dirs
for folder in [app.config["UPLOAD_FOLDER"], app.config["RESULTS_FOLDER"]]:
    Path(folder).mkdir(parents=True, exist_ok=True)

# Global detector (loaded once)
detector = None


def get_detector():
    global detector
    if detector is None:
        logger.info("Initializing detector for web app...")
        detector = ParkingDetector(model_path="yolov8n.pt", total_spaces=10)
    return detector


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def image_to_base64(image: np.ndarray) -> str:
    """Convert a BGR OpenCV image to a base64-encoded JPEG string."""
    _, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buffer).decode("utf-8")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    """Handle image upload and run detection."""
    if "image" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["image"]
    spaces = int(request.form.get("spaces", 30))

    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type"}), 400

    filename = secure_filename(file.filename)
    upload_path = Path(app.config["UPLOAD_FOLDER"]) / filename
    file.save(str(upload_path))

    try:
        # Load and process image
        image = load_single_image(str(upload_path))
        image = preprocess_image(image, target_width=1280)

        # Run detection
        det = get_detector()
        det.total_spaces = spaces
        result = det.detect(image)
        result["filename"] = filename

        # Save annotated image
        annotated = result["annotated_image"]
        stem = Path(filename).stem
        annotated_filename = f"{stem}_annotated.jpg"
        annotated_path = Path(app.config["RESULTS_FOLDER"]) / annotated_filename
        cv2.imwrite(str(annotated_path), annotated)

        # Save CSV
        csv_path = Path(app.config["RESULTS_FOLDER"]) / f"{stem}_results.csv"
        save_results_to_csv([result], output_path=str(csv_path))

        # Build response
        response_data = {
            "success": True,
            "annotated_image_b64": image_to_base64(annotated),
            "annotated_filename": annotated_filename,
            "csv_filename": csv_path.name,
            "car_count": result["car_count"],
            "free_spaces": result["free_spaces"],
            "occupied_spaces": result["occupied_spaces"],
            "total_spaces": spaces,
            "cars": result["cars"],
        }
        return jsonify(response_data)

    except Exception as e:
        logger.error(f"Detection error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/download/<filename>")
def download_file(filename):
    """Serve a result file for download."""
    safe_name = secure_filename(filename)
    return send_from_directory(app.config["RESULTS_FOLDER"], safe_name, as_attachment=True)


if __name__ == "__main__":
    logger.info("Starting Smart Parking Web App...")
    get_detector()
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)