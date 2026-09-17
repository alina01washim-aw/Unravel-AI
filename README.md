# Unravel AI

### Real-Time AI Vision & Object Intelligence

**Unravel AI** is a real-time computer vision system that combines **YOLOv8 object detection**, **OpenCV**, and a locally running **Gemma 2B model through Ollama** to analyze objects captured from a live camera feed.

The system detects objects in real time, tracks their movement, displays intelligent visual overlays, and generates short AI-powered descriptions of detected objects.

---

## What It Does

* Captures live video through a webcam
* Detects objects in real time using **YOLOv8**
* Tracks detected objects as they move through the frame
* Displays bounding boxes, labels, and object information
* Generates short AI insights for detected objects using **Gemma 2B**
* Runs the language model locally through **Ollama**
* Uses background processing so AI inference does not block the live camera display
* Provides an interactive HUD-style vision interface
* Supports saving screenshots of the live detection interface

---

## System Architecture

```text
              Live Camera Feed
                     │
                     ▼
                OpenCV
                     │
                     ▼
              YOLOv8 Detection
                     │
          ┌──────────┴──────────┐
          │                     │
          ▼                     ▼
    Object Detection       Object Tracking
          │                     │
          └──────────┬──────────┘
                     │
                     ▼
              Object Label
                     │
                     ▼
          Gemma 2B via Ollama
                     │
                     ▼
             AI Object Insight
                     │
                     ▼
          Real-Time Vision HUD
```

---

## Tech Stack

| Technology           | Role                                                |
| -------------------- | --------------------------------------------------- |
| **Python**           | Core application and processing                     |
| **YOLOv8**           | Real-time object detection                          |
| **OpenCV**           | Camera capture, image processing, and visualization |
| **Ollama**           | Runs the language model locally                     |
| **Gemma 2B**         | Generates short AI-powered object descriptions      |
| **NumPy**            | Image and numerical processing                      |
| **Python Threading** | Runs AI processing in the background                |

---

## AI Models

### YOLOv8

The project uses **YOLOv8** for real-time object detection.

Two model files are included:

* `yolov8n.pt` — YOLOv8 Nano, optimized for lighter hardware
* `yolov8l.pt` — YOLOv8 Large, providing higher-capacity detection

The current `vision.py` configuration uses:

```python
yolo = YOLO("yolov8l.pt")
```

The Nano model can be used when lower computational requirements are preferred.

### Gemma 2B

**Gemma 2B** is used as the local language model for generating concise descriptions of detected objects.

It is accessed through **Ollama**, allowing the AI insight generation to run locally rather than relying on a cloud API.

The current configuration uses:

```python
MODEL = "gemma:2b"
```

---

## How It Works

1. **OpenCV** captures frames from the connected camera.
2. Frames are processed for real-time vision analysis.
3. **YOLOv8** identifies objects present in the scene.
4. Detected objects are tracked and their positions are calculated.
5. The system displays labels and visual information directly on the live feed.
6. Object information can be passed to **Gemma 2B** through Ollama.
7. Gemma generates a short description of the detected object.
8. The AI insight is displayed as part of the vision HUD.
9. Background threading keeps AI processing separate from the main display loop.

---

## Requirements

### Software

* Python 3.9+
* Ollama
* A working webcam
* Windows, Linux, or macOS

### Python Dependencies

Install the required packages with:

```bash
pip install -r requirements.txt
```

The project uses:

```text
opencv-python
ollama
pillow
ultralytics
```

> **Note:** `ultralytics` is required for YOLOv8. If it is not already present in `requirements.txt`, install it separately with:

```bash
pip install ultralytics
```

---

## Ollama Setup

Install Ollama and make sure it is running.

Then download the model used by Unravel AI:

```bash
ollama pull gemma:2b
```

You can verify that the model is available with:

```bash
ollama list
```

You should see:

```text
gemma:2b
```

---

## Running the Project

Activate your Python virtual environment first.

Then run:

```bash
python vision.py
```

The application will open the live vision interface and begin processing the camera feed.

If Ollama is not already running, start it with:

```bash
ollama serve
```

---

## Controls

| Key | Action               |
| --- | -------------------- |
| `Q` | Quit the application |
| `P` | Pause / Resume       |
| `S` | Save a screenshot    |

Screenshots are saved inside:

```text
screenshots/
```

---

## Project Structure

```text
Unravel-AI/
│
├── vision.py
├── requirements.txt
├── yolov8l.pt
├── yolov8n.pt
├── README.md
│
└── screenshots/
    ├── detection screenshots
    └── saved captures
```

---

## Key Features

### Real-Time Detection

YOLOv8 processes the live camera feed and identifies objects in the scene.

### Object Tracking

Detected objects are tracked across frames to provide smoother and more stable visual labels.

### Local AI Intelligence

Gemma 2B runs locally through Ollama, allowing object insights without depending on a remote AI API.

### HUD-Style Interface

Detection information is presented through a visual interface designed for real-time monitoring and situational awareness.

### Non-Blocking AI Processing

Python threading separates AI processing from the main display loop, helping maintain a responsive camera interface.

---

## Why Unravel AI?

Unravel AI combines **computer vision and local AI intelligence** into a single real-time system.

Instead of relying only on object detection, the system adds a second layer of AI-powered understanding:

**See → Detect → Track → Understand**

This architecture can serve as a foundation for applications in **autonomous systems, robotics, drones, surveillance, situational awareness, and edge AI**.

---

## Future Development

Potential extensions include:

* Deployment on edge devices such as **Raspberry Pi**
* Optimized YOLO models for edge inference
* Drone-mounted camera integration
* Autonomous object-following capabilities
* Advanced gesture recognition
* Multi-object tracking improvements
* Voice-based AI interaction
* Real-time alerts and event detection

---

## Project

**Unravel AI**
Real-Time AI Vision & Object Intelligence

Built with **Python · YOLOv8 · OpenCV · Ollama · Gemma 2B**
