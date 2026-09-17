# Gemma 4 Live Vision

A real-time object detection and gesture recognition app powered by **Gemma 4** running locally via **Ollama**, using **OpenCV** for camera capture and display.

---

## What it does

- Opens your webcam in a native desktop window
- Continuously captures frames and sends them to Gemma 4
- Detects objects, people, children, and hand gestures in real time
- Draws labeled overlays at the actual position of each detected object
- Smoothly tracks objects as they move across the frame

---

## Tech Stack

| Tool | Role |
|------|------|
| [Ollama](https://ollama.com) | Runs Gemma 4 locally |
| Gemma 4 (`gemma4:e2b`) | Vision-language model for detection |
| OpenCV (`cv2`) | Camera capture, frame processing, display |
| Python threading | Pipelines inference so display never blocks |

---

## About Gemma 4

Gemma 4 is Google's latest open-weight multimodal model family, released in April 2025.

- **Multimodal** — understands both text and images
- **Available sizes** — 1B, 4B, 12B, 27B parameters
- **Context window** — 128K tokens
- **Strengths** — strong visual understanding, object recognition, gesture detection, multilingual support
- **Edge-friendly** — optimized to run on consumer hardware
- The `e2b` variant used here is efficient and fast for real-time vision tasks

More info: https://ai.google.dev/gemma 

---

## About Ollama

Ollama lets you run large language models locally on your machine with a simple CLI.

- Website: https://ollama.com
- Model library: https://ollama.com/library
- Pull Gemma 4: `ollama pull gemma4:e2b`

---

## Requirements

- Python 3.9+
- Ollama installed and running
- Gemma 4 pulled in Ollama

```bash
pip3 install -r requirements.txt
```

---

## Run

```bash
ollama serve          # make sure Ollama is running
python3 vision.py     # launch the app
```

| Key | Action |
|-----|--------|
| `Q` | Quit |
| `P` | Pause / Resume |
| `S` | Save screenshot to `screenshots/` |

---

## How it works

1. OpenCV captures frames from your webcam at 1280×720
2. Each frame is resized to 640px and JPEG-compressed before sending
3. A background thread sends the frame to Gemma 4 via Ollama
4. Gemma returns a JSON array of detected objects with their center coordinates
5. Labels are drawn at those positions with lerp smoothing between frames
6. The main thread displays at full 30fps — never waits for inference
# object-detect-using-gemma4
