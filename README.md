# 🅿️ Smart Parking System

An intelligent parking lot analysis system using **YOLOv8**, **EasyOCR**, and **OpenCV**.  
Upload a parking lot image → get car count, license plates, car colors, and free space count.

---

## 📁 Project Structure

```
smart_parking/
│
├── main.py                     ← CLI entry point (run images from terminal)
├── app.py                      ← Flask web interface (bonus)
├── generate_test_images.py     ← Generates synthetic test images
├── requirements.txt
│
├── parking_system/             ← Core Python package
│   ├── __init__.py
│   ├── detector.py             ← Main orchestrator (YOLO + color + plate)
│   ├── color_detector.py       ← Car color detection (HSV analysis)
│   ├── plate_detector.py       ← License plate region detection + OCR
│   ├── dataset.py              ← Image loading and preprocessing
│   └── exporter.py             ← CSV export + console reporting
│
├── templates/
│   └── index.html              ← Flask web UI
│
├── static/
│   ├── uploads/                ← Web uploads (auto-created)
│   └── results/                ← Web results (auto-created)
│
├── dataset/
│   └── sample_images/          ← Put your parking lot images here
│
└── results/                    ← Output: annotated images + CSV (auto-created)
    └── annotated/
```

---

## ⚙️ Requirements

- Python 3.9 or higher
- pip

---

## 🚀 Setup — Step by Step

### 1. Clone / download the project

```bash
# If you have git:
git clone <your-repo-url>
cd smart_parking

# Or just unzip and cd into the folder
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# Windows:
venv\Scripts\activate

# macOS/Linux:
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> ⚠️ **Note:** Installing may take a few minutes.  
> EasyOCR will download language model files (~200 MB) on first run.  
> YOLOv8 (`yolov8n.pt`, ~6 MB) auto-downloads on first run.

### 4. Add your images

Place your parking lot `.jpg` or `.png` images in:
```
dataset/sample_images/
```

**No images yet?** Generate synthetic test images:
```bash
python generate_test_images.py
```

---

## ▶️ Running the System

### Option A — Command Line

```bash
# Basic: process all images in a folder
python main.py --input dataset/sample_images

# Specify total parking spaces (default: 10)
python main.py --input dataset/sample_images --spaces 20

# Process a single image
python main.py --input dataset/sample_images/parking_01.jpg

# Save results as CSV + don't open display window
python main.py --input dataset/sample_images --save-csv --no-display

# Full example with all options
python main.py \
  --input dataset/sample_images \
  --spaces 15 \
  --save-csv \
  --output-dir results \
  --model yolov8n.pt
```

**All CLI options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--input` | *(required)* | Image file or folder path |
| `--spaces` | `10` | Total parking spaces in the lot |
| `--model` | `yolov8n.pt` | YOLOv8 model weights file |
| `--output-dir` | `results` | Where to save outputs |
| `--save-csv` | `False` | Save detections to CSV |
| `--save-images` | `True` | Save annotated images |
| `--no-display` | `False` | Skip OpenCV display window |
| `--width` | `1280` | Max image width for processing |

---

### Option B — Web Interface (Flask)

```bash
python app.py
```

Then open your browser at: **http://localhost:5000**

Features:
- 📷 Drag-and-drop image upload
- 🔢 Configurable total spaces
- 🖼️ Annotated output image with bounding boxes
- 📊 Stats: occupied, free, total
- 🎨 Color + plate table per vehicle
- ⬇️ Download annotated image + CSV

---

## 📤 Sample Output

**Console:**
```
============================================================
  SMART PARKING SYSTEM — DETECTION REPORT
  Image: parking_lot_01.jpg
============================================================
  🚗  Cars detected      : 5
  🟢  Free spaces        : 5
  🔴  Occupied spaces    : 5
  📊  Total spaces       : 10
------------------------------------------------------------
  ID    Color        Plate           Type         Conf
  ---- ----------- -------------- ------------ -----
  1     blue         AB123CD        car          0.87
  2     white        XY456ZZ        car          0.91
  3     red          QR789TU        car          0.83
  ...
============================================================
```

**Files produced:**
```
results/
  annotated/
    parking_lot_01_annotated.jpg   ← image with bounding boxes
  parking_results.csv              ← detection data
```

**CSV columns:**
`timestamp, image_filename, car_id, color, plate, confidence, total_cars, occupied_spaces, free_spaces`

---

## 🧠 How It Works

```
Image input
    │
    ▼
YOLOv8 Object Detection
    │  Detects all vehicles (car, truck, bus, motorcycle)
    │  Returns bounding boxes + class + confidence
    │
    ▼
Per-vehicle pipeline
    ├── Color Detector
    │     Convert crop to HSV → match against color ranges
    │     Returns: "blue", "red", "white", etc.
    │
    └── Plate Detector + OCR
          Edge detection → contour filtering → find rectangles
          EasyOCR on best candidate region
          Returns: plate text + bounding box
    │
    ▼
Annotated image + CSV export + console report
```

---

## 🎨 Detected Colors

The system recognizes: `red`, `orange`, `yellow`, `green`, `blue`, `purple`, `pink`, `white`, `gray`, `black`

---

## 🔧 Customization

**Change total spaces:** Pass `--spaces N` to CLI or set in the web form.

**Use a larger YOLOv8 model** (more accurate, slower):
```bash
python main.py --input dataset/sample_images --model yolov8s.pt
```
Available: `yolov8n.pt` (nano), `yolov8s.pt` (small), `yolov8m.pt` (medium)

**GPU acceleration** — If you have an NVIDIA GPU with CUDA:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```
YOLO and EasyOCR will automatically use GPU.

---

## 🐛 Troubleshooting

| Issue | Fix |
|-------|-----|
| `ModuleNotFoundError: ultralytics` | Run `pip install ultralytics` |
| `ModuleNotFoundError: easyocr` | Run `pip install easyocr` |
| First run is slow | EasyOCR + YOLO download models — wait ~1 min |
| No cars detected | Try a clearer/closer image, or lower confidence threshold in `detector.py` |
| OCR shows "N/A" | Plate may not be visible; this is expected for side/rear angle shots |
| Flask port in use | Change port in `app.py`: `app.run(port=5001)` |

---

## 📦 Libraries Used

| Library | Purpose |
|---------|---------|
| `ultralytics` | YOLOv8 object detection |
| `easyocr` | License plate text recognition |
| `opencv-python` | Image loading, drawing, preprocessing |
| `numpy` | Array operations |
| `flask` | Web interface |
| `Pillow` | Image format support |

---

## 📝 License

This project is for educational and research purposes.
