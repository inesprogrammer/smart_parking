"""
app.py — Smart Parking System v3
Uses SQLite database for persistent storage.
"""

import os
import sys
import cv2
import base64
import logging
import sqlite3
import numpy as np
from pathlib import Path
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

sys.path.insert(0, str(Path(__file__).parent))

from parking_system import (
    ParkingDetector,
    load_single_image,
    preprocess_image,
)

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TOTAL_SPACES = 30
DB_PATH = "/tmp/parking.db"  # /tmp persists during session on Render

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
app.config["UPLOAD_FOLDER"] = "/tmp/uploads"
app.config["RESULTS_FOLDER"] = "/tmp/results"

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "webp"}

for folder in [app.config["UPLOAD_FOLDER"], app.config["RESULTS_FOLDER"]]:
    Path(folder).mkdir(parents=True, exist_ok=True)


# ── Database ──────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS parking (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            entry_time TEXT,
            place      TEXT,
            plate      TEXT,
            color      TEXT,
            confidence REAL
        )
    """)
    conn.commit()
    conn.close()


def add_car(entry_time, place, plate, color, confidence):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO parking (entry_time, place, plate, color, confidence) VALUES (?,?,?,?,?)",
        (entry_time, place, plate, color, confidence)
    )
    conn.commit()
    conn.close()


def get_all_cars():
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT entry_time, place, plate, color, confidence FROM parking ORDER BY id"
    ).fetchall()
    conn.close()
    return [
        {"entry_time": r[0], "place": r[1], "plate": r[2],
         "color": r[3], "confidence": r[4]}
        for r in rows
    ]


def get_occupied_count():
    conn = sqlite3.connect(DB_PATH)
    count = conn.execute("SELECT COUNT(*) FROM parking").fetchone()[0]
    conn.close()
    return count


def reset_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM parking")
    conn.commit()
    conn.close()


def export_csv():
    """Export all parked cars to CSV string."""
    import csv, io
    cars = get_all_cars()
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=[
        "entry_time", "place", "plate", "color", "confidence"
    ])
    writer.writeheader()
    writer.writerows(cars)
    return output.getvalue()


# Initialize DB
init_db()

# ── Detector ──────────────────────────────────────────────
detector = None

def initialize():
    global detector
    if detector is None:
        logger.info("Pre-loading models...")
        detector = ParkingDetector(model_path="yolov8n.pt", total_spaces=TOTAL_SPACES)

initialize()

def get_detector():
    global detector
    if detector is None:
        detector = ParkingDetector(model_path="yolov8n.pt", total_spaces=TOTAL_SPACES)
    return detector

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def image_to_base64(image):
    _, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buffer).decode("utf-8")


# ── Routes ────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["image"]
    if file.filename == "" or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file"}), 400

    occupied = get_occupied_count()
    if occupied >= TOTAL_SPACES:
        return jsonify({"error": "Parking is FULL!"}), 400

    filename = secure_filename(file.filename)
    upload_path = Path(app.config["UPLOAD_FOLDER"]) / filename
    file.save(str(upload_path))

    try:
        image = load_single_image(str(upload_path))
        image = preprocess_image(image, target_width=1280)

        det = get_detector()
        det.total_spaces = TOTAL_SPACES
        result = det.detect(image)

        # Save annotated image
        annotated = result["annotated_image"]
        stem = Path(filename).stem
        annotated_filename = f"{stem}_annotated.jpg"
        annotated_path = Path(app.config["RESULTS_FOLDER"]) / annotated_filename
        cv2.imwrite(str(annotated_path), annotated)

        # Add each car to DB
        entry_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        new_occupied = occupied

        for car in result.get("cars", []):
            new_occupied += 1
            place = f"P-{new_occupied:02d}"
            add_car(
                entry_time=entry_time,
                place=place,
                plate=car.get("plate", "N/A"),
                color=car.get("color", "unknown"),
                confidence=round(car.get("confidence", 0.0), 2)
            )

        total_occupied = get_occupied_count()
        free = max(0, TOTAL_SPACES - total_occupied)
        all_parked = get_all_cars()

        return jsonify({
            "success": True,
            "annotated_image_b64": image_to_base64(annotated),
            "car_count": len(result.get("cars", [])),
            "free_spaces": free,
            "occupied_spaces": total_occupied,
            "total_spaces": TOTAL_SPACES,
            "all_parked_cars": all_parked,
        })

    except Exception as e:
        logger.error(f"Detection error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/parking-status")
def parking_status():
    all_parked = get_all_cars()
    occupied = len(all_parked)
    return jsonify({
        "total_spaces": TOTAL_SPACES,
        "occupied_spaces": occupied,
        "free_spaces": max(0, TOTAL_SPACES - occupied),
        "all_parked_cars": all_parked,
    })


@app.route("/reset", methods=["POST"])
def reset():
    reset_db()
    return jsonify({"success": True, "message": "Parking reset!"})


@app.route("/download/parking_history.csv")
def download_csv():
    from flask import Response
    csv_data = export_csv()
    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=parking_history.csv"}
    )


@app.route("/download/<filename>")
def download_file(filename):
    safe_name = secure_filename(filename)
    return send_from_directory(app.config["RESULTS_FOLDER"], safe_name, as_attachment=True)


if __name__ == "__main__":
    logger.info("Starting Smart Parking Web App...")
    port = int(os.environ.get("PORT", 10000))
    app.run(debug=False, host="0.0.0.0", port=port)