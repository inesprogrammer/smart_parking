# 🧠 Deep Mind Manager
### Intelligent Parking System powered by Deep Learning

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://python.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-blue.svg)](https://ultralytics.com)
[![Flask](https://img.shields.io/badge/Flask-3.0-blue.svg)](https://flask.palletsprojects.com)
[![EasyOCR](https://img.shields.io/badge/EasyOCR-1.7-blue.svg)](https://github.com/JaidedAI/EasyOCR)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Render-blue.svg)](https://smart-parking-u3xs.onrender.com)

---

## 🌐 Live Demo

> **[https://smart-parking-u3xs.onrender.com](https://smart-parking-u3xs.onrender.com)**

---

## 📋 Overview

**Deep Mind Manager** is an intelligent parking management system that uses **Deep Learning** and **Computer Vision** to automatically:

- 🚗 Detect vehicles in parking lot images using **YOLOv8**
- 🎨 Identify the **dominant color** of each car
- 🔤 Read **license plate numbers** using **EasyOCR**
- 🅿️ Track **occupied and free parking spaces** in real time
- 📊 Maintain a **persistent history** of all parked cars
- 🗺️ Display a **visual parking map**

---

## 🧠 How It Works

```
📷 Image Input
      │
      ▼
🧠 YOLOv8 (Deep Learning CNN)
      │  Detects all vehicles with bounding boxes
      │
      ▼
🎨 Color Detector (Computer Vision - HSV)
      │  Identifies dominant car color
      │
      ▼
🔤 EasyOCR (Deep Learning CNN + LSTM)
      │  Reads license plate text
      │
      ▼
🗃️ SQLite Database
      │  Stores: plate, color, time, place
      │
      ▼
🌐 Flask Web Interface
      Displays: annotated image + table + map
```

---

## 🛠️ Technologies Used

| Technology | Version | Purpose |
|-----------|---------|---------|
| **Python** | 3.11 | Core programming language |
| **YOLOv8** (Ultralytics) | 8.x | Vehicle detection (Deep Learning) |
| **EasyOCR** | 1.7 | License plate recognition (Deep Learning) |
| **OpenCV** | 4.8 | Image processing and annotation |
| **Flask** | 3.0 | Web interface |
| **SQLite** | Built-in | Persistent parking database |
| **NumPy** | 1.24 | Array and matrix operations |
| **Pillow** | 10.0 | Image format support |

---

## 📁 Project Structure

```
smart_parking/
│
├── app.py                          ← Flask web app (main entry point)
├── main.py                         ← CLI entry point
├── requirements.txt                ← Python dependencies
├── runtime.txt                     ← Python version for deployment
├── README.md                       ← This file
│
├── parking_system/                 ← Core AI package
│   ├── __init__.py
│   ├── detector.py                 ← YOLOv8 orchestrator
│   ├── color_detector.py           ← HSV color detection
│   ├── plate_detector.py           ← License plate OCR
│   ├── dataset.py                  ← Image loading & preprocessing
│   └── exporter.py                 ← CSV export & reporting
│
├── templates/
│   └── index.html                  ← Web interface (HTML/CSS/JS)
│
├── static/
│   ├── uploads/                    ← Uploaded images
│   └── results/                    ← Annotated output images
│
└── dataset/
    └── sample_images/              ← Test parking images
```

---

## ⚙️ Requirements

- Python 3.11+
- pip

---

## 🚀 Installation & Running

### 1. Clone the repository

```bash
git clone https://github.com/inesprogrammer/smart_parking.git
cd smart_parking
```

### 2. Create virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

> ⏳ First install takes 5-10 minutes (downloads PyTorch, YOLOv8, EasyOCR models)

### 4. Run the web interface

```bash
python app.py
```

Open your browser at: **http://localhost:5000**

### 5. OR run via command line

```bash
# Process a folder of images
python main.py --input dataset/sample_images --spaces 30 --save-csv

# Process a single image
python main.py --input path/to/parking.jpg
```

---

## 📤 Output

### Web Interface
- 🖼️ Annotated image with bounding boxes, colors, plate numbers
- 📊 Real-time statistics: Total / Occupied / Free spaces
- 🗺️ Visual parking map (green = free, red = occupied)
- 📋 Table of all parked cars with entry time and place
- ⬇️ Download annotated image and CSV

### Console Output
```
============================================================
  DEEP MIND MANAGER — DETECTION REPORT
============================================================
  Cars detected      : 7
  Free spaces        : 23
  Occupied spaces    : 7
  Total spaces       : 30
------------------------------------------------------------
  ID    Color        Plate           Type         Conf
  1     gray         N/A             car          0.91
  2     black        N/A             car          0.88
  7     red          11CY SS78       car          0.85
============================================================
```

### CSV File Columns
| Column | Description |
|--------|-------------|
| `entry_time` | Date and time of entry |
| `place` | Assigned parking spot (P-01, P-02...) |
| `plate` | License plate text (OCR) |
| `color` | Detected car color |
| `confidence` | YOLO detection confidence (0-1) |

---

## 🎯 Features

| Feature | Description |
|---------|-------------|
| ✅ **Multi-car tracking** | Each uploaded image adds cars to history |
| ✅ **Persistent storage** | SQLite database keeps all records |
| ✅ **Auto place assignment** | Cars get P-01, P-02... automatically |
| ✅ **Visual parking map** | 30-space grid with live status |
| ✅ **Color detection** | 10 colors: red, blue, white, black, gray... |
| ✅ **License plate OCR** | Reads plates from front/rear view images |
| ✅ **Reset function** | Clear all cars and reset parking |
| ✅ **CSV export** | Download complete parking history |

---

## ⚠️ Limitations

- License plate OCR works best on **front/rear view** images with visible plates
- YOLOv8 pre-trained model may miss cars in **extreme top-down** aerial views
- First request after inactivity takes **~30 seconds** to wake up (free hosting)

---

## 🔧 Configuration

Change total parking spaces in `app.py`:
```python
TOTAL_SPACES = 30  # Change this value
```

---

## 📄 License

This project is for educational and research purposes.