import cv2
import threading
import ollama
import base64
import time
import os
import numpy as np
from ultralytics import YOLO
from datetime import datetime

yolo = YOLO("yolov8l.pt")

# ── Config ────────────────────────────────────────────────────────────────────
WINDOW_NAME = "Unravel AI — Live Vision"
MODEL      = "gemma:2b"
CAM_WIDTH  = 1280
CAM_HEIGHT = 720
INFER_SIZE = 640        
JPEG_QUAL  = 60
SMOOTH     = 0.3
FADE_IN    = 0.15
FADE_OUT   = 0.08
LABEL_TTL  = 2.0
FONT       = cv2.FONT_HERSHEY_DUPLEX

# ── Shared state ──────────────────────────────────────────────────────────────
latest_objects: list[dict] = []
smoothed_objects: list[dict] = []
label_state: dict = {}
lock = threading.Lock()
processing = False
last_infer_ms = 0.0
client = ollama.Client()

# ── Interaction & Zoom State ──────────────────────────────────────────────────
actual_w, actual_h = CAM_WIDTH, CAM_HEIGHT
zoom_level = 1.0
focused_track = None    

# ── AI Insight Fetcher ────────────────────────────────────────────────────────
def fetch_object_insight(label):
    """Fetches a quick description of the object using local Ollama model."""
    global focused_track
    if not focused_track or focused_track['label'] != label: return
    
    try:
        response = client.chat(model=MODEL, messages=[
            {"role": "system", "content": "You are a HUD AI system. Provide a factual, 1-sentence description (under 12 words) of the object type requested."},
            {"role": "user", "content": f"Object: {label}"}
        ])
        insight = response['message']['content'].strip()
        
        if focused_track and focused_track['label'] == label:
            focused_track['desc'] = insight
    except Exception as e:
        if focused_track and focused_track['label'] == label:
            focused_track['desc'] = f"AI Comm Link Error. (Is Ollama running?)"

def set_focus(obj):
    """Locks the HUD onto a specific object and triggers the AI Insight thread."""
    global focused_track
    focused_track = {
        "label": obj["label"],
        "cx": obj["cx"],
        "cy": obj["cy"],
        "bw": obj.get("bw", 0.15),
        "bh": obj.get("bh", 0.15),
        "conf": obj.get("conf", 0.0),
        "desc": "Fetching AI insights..."
    }
    threading.Thread(target=fetch_object_insight, args=(obj["label"],), daemon=True).start()

# ── Mouse Interaction ─────────────────────────────────────────────────────────
def mouse_callback(event, x, y, flags, param):
    """Handles Object Focusing and Resetting via Single Left-Click."""
    global focused_track, actual_w, actual_h

    if event == cv2.EVENT_LBUTTONDOWN:
        best_dist = float('inf')
        best_obj = None
        
        # Check if click is inside any object's bounding box
        for obj in smoothed_objects:
            ocx, ocy = int(obj["cx"] * actual_w), int(obj["cy"] * actual_h)
            
            # Use actual bounding box size (fallback to 15% of screen if missing)
            bw = obj.get("bw", 0.15) * actual_w
            bh = obj.get("bh", 0.15) * actual_h
            
            # Check if the mouse (x, y) falls inside the rectangle
            if (ocx - bw/2) <= x <= (ocx + bw/2) and (ocy - bh/2) <= y <= (ocy + bh/2):
                # If overlapping, pick the one whose center is closest to the click
                dist = ((ocx - x)**2 + (ocy - y)**2)**0.5
                if dist < best_dist:
                    best_dist = dist
                    best_obj = obj
        
        if best_obj:
            set_focus(best_obj)
        else:
            focused_track = None
            print("Focus Reset to All Objects")

# ── Inference ─────────────────────────────────────────────────────────────────
def infer(frame: np.ndarray):
    global latest_objects, processing, last_infer_ms
    t0 = time.time()
    try:
        results = yolo(frame, verbose=False)
        objs = []
        h, w = frame.shape[:2]
        for r in results:
            for box in r.boxes:
                conf = float(box.conf[0])
                if conf < 0.45: continue

                cls = int(box.cls[0])
                label = yolo.names[cls]
                x1, y1, x2, y2 = box.xyxy[0]
                
                # Extract Box Width and Box Height so we can click on them easily
                objs.append({
                    "label": label,
                    "cx": float(((x1 + x2) / 2) / w),
                    "cy": float(((y1 + y2) / 2) / h),
                    "bw": float((x2 - x1) / w),
                    "bh": float((y2 - y1) / h),
                    "conf": conf
                })
        with lock:
            latest_objects = objs
        last_infer_ms = (time.time() - t0) * 1000
    except Exception as e:
        print(f"Inference error: {e}")
    finally:
        processing = False

def update_smooth(new_objects):
    global smoothed_objects
    result = []
    for obj in new_objects:
        prev = next((s for s in smoothed_objects if s["label"] == obj["label"]), None)
        if prev is None:
            result.append(dict(obj))
        else:
            dist = abs(obj["cx"] - prev["cx"]) + abs(obj["cy"] - prev["cy"])
            f = 1.0 if dist > 0.25 else SMOOTH
            result.append({
                "label": obj["label"],
                "cx": prev["cx"] + (obj["cx"] - prev["cx"]) * f,
                "cy": prev["cy"] + (obj["cy"] - prev["cy"]) * f,
                "bw": obj.get("bw", 0.15),
                "bh": obj.get("bh", 0.15),
                "conf": obj["conf"] 
            })
    smoothed_objects = result
    return result

# ── Video Filters ─────────────────────────────────────────────────────────────
def apply_zoom(frame: np.ndarray, zoom_factor: float) -> np.ndarray:
    if zoom_factor <= 1.0: return frame
    h, w = frame.shape[:2]
    new_w, new_h = int(w / zoom_factor), int(h / zoom_factor)
    x1, y1 = (w - new_w) // 2, (h - new_h) // 2
    cropped = frame[y1:y1+new_h, x1:x1+new_w]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

def apply_grayscale(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

# ── cv2 drawing helpers ───────────────────────────────────────────────────────
def glow_text(frame, text, x, y, scale, color, thickness=2):
    for d in [3, 2]: cv2.putText(frame, text, (x, y), FONT, scale, color, thickness + d, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), FONT, scale, (255, 255, 255), thickness, cv2.LINE_AA)

def label_pill(frame, text, x, y, color, alpha=1.0):
    scale, thick = 0.65, 1
    (tw, th), bl = cv2.getTextSize(text, FONT, scale, thick)
    pad = 8
    x1, y1 = x - pad, y - th - pad
    x2, y2 = x + tw + pad, y + bl + pad // 2
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (int(color[0]*0.15), int(color[1]*0.15), int(color[2]*0.15)), -1)
    cv2.addWeighted(overlay, alpha * 0.75, frame, 1 - alpha * 0.75, 0, frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), tuple(int(v * alpha) for v in color), 1, cv2.LINE_AA)
    for d in [2, 1]: cv2.putText(frame, text, (x, y), FONT, scale, tuple(int(v * alpha * 0.4) for v in color), thick + d, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), FONT, scale, tuple(int(v * alpha) for v in (255, 255, 255)), thick, cv2.LINE_AA)

def reticle(frame, cx, cy, size, color, alpha, t):
    overlay = frame.copy()
    c = tuple(int(v) for v in color)
    arm = max(12, size // 3)
    lw = 2
    for dx, dy in [(-1,-1),(1,-1),(-1,1),(1,1)]:
        bx, by = cx + dx * size, cy + dy * size
        cv2.line(overlay, (bx, by), (bx - dx * arm, by), c, lw, cv2.LINE_AA)
        cv2.line(overlay, (bx, by), (bx, by - dy * arm), c, lw, cv2.LINE_AA)
    pulse = int(8 * abs(np.sin(t / 350)))
    cv2.circle(overlay, (cx, cy), size // 4 + pulse, c, 1, cv2.LINE_AA)
    cv2.circle(overlay, (cx, cy), 4, c, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

def draw_text_wrapped(frame, text, x, y, max_w, font, scale, color):
    words = text.split()
    lines, current_line = [], words[0] if words else ""
    for word in words[1:]:
        test_line = current_line + " " + word
        if cv2.getTextSize(test_line, font, scale, 1)[0][0] <= max_w: current_line = test_line
        else:
            lines.append(current_line)
            current_line = word
    if current_line: lines.append(current_line)
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (x, y + i * 22), font, scale, color, 1, cv2.LINE_AA)

# ── UI & Overlay ──────────────────────────────────────────────────────────────
def draw_object_details(frame, obj_info):
    if not obj_info: return
    x, y = 18, 110
    w, h = 380, 180

    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x+w, y+h), (10, 10, 15), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 200, 100), 1)
    
    cv2.putText(frame, f"TARGET LOCKED: {obj_info['label'].upper()}", (x+15, y+30), FONT, 0.65, (0, 255, 100), 1, cv2.LINE_AA)
    cv2.line(frame, (x+15, y+40), (x+w-15, y+40), (100, 100, 100), 1)
    
    conf_str = f"CONF: {obj_info.get('conf', 0.0):.2f}"
    pos_str = f"POS X: {obj_info['cx']:.2f}  Y: {obj_info['cy']:.2f}  {conf_str}"
    cv2.putText(frame, pos_str, (x+15, y+65), FONT, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, "STATUS: TRACKING", (x+15, y+85), FONT, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    
    cv2.putText(frame, "AI INSIGHT:", (x+15, y+115), FONT, 0.45, (0, 180, 255), 1, cv2.LINE_AA)
    desc = obj_info.get("desc", "Analyzing object signature...")
    draw_text_wrapped(frame, desc, x+15, y+140, max_w=w-30, font=FONT, scale=0.45, color=(160, 210, 255))

def draw_rgb_graph(frame):
    h, w = frame.shape[:2]
    graph_w, graph_h = 256, 80
    x, y = w - graph_w - 16, 95 
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + graph_w, y + graph_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    for i, col in enumerate(colors):
        hist = cv2.calcHist([frame], [i], None, [256], [0, 256])
        cv2.normalize(hist, hist, 0, graph_h - 5, cv2.NORM_MINMAX)
        pts = np.array([(x + j, y + graph_h - int(hist[j])) for j in range(256)], np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], False, col, 1, cv2.LINE_AA)
    cv2.rectangle(frame, (x, y), (x + graph_w, y + graph_h), (100, 100, 100), 1)
    cv2.putText(frame, "RGB HISTOGRAM", (x + 5, y + 15), FONT, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

def find_label_pos(px, py, lw, lh, placed, frame_w, frame_h):
    offsets = [
        (0, -70), (0, 70), (-lw - 50, 0), (lw + 10, 0),
        (-lw//2 - 30, -80), (lw//2 + 10, -80), (-lw//2 - 30,  80), (lw//2 + 10,  80),
        (0, -110), (0, 110),
    ]
    for ox, oy in offsets:
        lx = int(np.clip(px + ox - lw // 2, 6, frame_w - lw - 6))
        ly = int(np.clip(py + oy - lh // 2, 85, frame_h - lh - 6))
        r = (lx, ly, lx + lw, ly + lh)
        if not any(r[0] < p[2] and r[2] > p[0] and r[1] < p[3] and r[3] > p[1] for p in placed):
            return lx, ly, r
    lx = int(np.clip(px - lw // 2, 6, frame_w - lw - 6))
    ly = int(np.clip(py - 70 - len(placed) * (lh + 6), 85, frame_h - lh - 6))
    return lx, ly, (lx, ly, lx + lw, ly + lh)

def draw_objects(frame, objects, tick_ms):
    h, w = frame.shape[:2]
    now = time.time()
    seen = {o["label"] for o in objects}
    for label in seen:
        if label not in label_state: label_state[label] = {"alpha": 0.0, "alive": True, "last_seen": now}
        else: label_state[label]["alive"], label_state[label]["last_seen"] = True, now
    for label, state in label_state.items():
        if label not in seen and now - state["last_seen"] > LABEL_TTL: state["alive"] = False
    for state in label_state.values():
        state["alpha"] = min(1.0, state["alpha"] + FADE_IN) if state["alive"] else max(0.0, state["alpha"] - FADE_OUT)
    for label in [l for l, s in label_state.items() if s["alpha"] <= 0 and not s["alive"]]:
        del label_state[label]

    placements, placed_rects = [], []
    for i, obj in enumerate(objects):
        a = label_state.get(obj["label"], {}).get("alpha", 1.0)
        if a <= 0: continue
        
        conf = obj.get("conf", 0.0)
        if conf > 0.8: color = (0, 230, 118)
        elif conf > 0.5: color = (0, 215, 255)
        else: color = (50, 50, 255)

        px, py = int(np.clip(obj["cx"] * w, 60, w - 60)), int(np.clip(obj["cy"] * h, 80, h - 50))
        txt = f"{obj['label'].upper()} {conf:.2f}"
        
        (tw, th), _ = cv2.getTextSize(txt, FONT, 0.65, 1)
        lx, ly, rect = find_label_pos(px, py, tw + 16, th + 16, placed_rects, w, h)
        placed_rects.append(rect)
        placements.append((obj, i, px, py, lx, ly, txt, color, a))

    for obj, i, px, py, lx, ly, txt, color, a in placements:
        reticle(frame, px, py, 45, color, a, tick_ms)
        (tw, th), _ = cv2.getTextSize(txt, FONT, 0.65, 1)
        label_cx, label_cy = lx + 8 + tw // 2, ly + 8 + th // 2
        if ((label_cx - px)**2 + (label_cy - py)**2) ** 0.5 > 60:
            c = tuple(int(v * a * 0.6) for v in color)
            cv2.line(frame, (label_cx, label_cy), (px, py), c, 1, cv2.LINE_AA)
            cv2.circle(frame, (label_cx, label_cy), 2, c, -1)
        label_pill(frame, txt, lx + 8, ly + 8 + cv2.getTextSize(txt, FONT, 0.65, 1)[0][1], color, a)

def draw_ui(frame, fps, infer_ms, obj_count, paused, is_bw_mode, zoom_lvl):
    h, w = frame.shape[:2]

    frame[0:90] = (frame[0:90] * 0.25).astype(np.uint8)
    frame[h-32:h] = (frame[h-32:h] * 0.25).astype(np.uint8)

    glow_text(frame, "UNRAVEL AI  LIVE VISION", 18, 38, 0.85, (0, 180, 80))

    if is_bw_mode: cv2.putText(frame, "MODE: BLACK & WHITE", (18, 60), FONT, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    else: cv2.putText(frame, "MODE: COLOR", (18, 60), FONT, 0.5, (0, 215, 255), 1, cv2.LINE_AA)

    status = "PAUSED" if paused else f"LIVE  {fps:.0f} FPS"
    color = (80, 80, 255) if paused else (0, 220, 100)
    sw = cv2.getTextSize(status, FONT, 0.6, 1)[0][0]
    cv2.putText(frame, status, (w - sw - 16, 26), FONT, 0.6, color, 1, cv2.LINE_AA)

    lat = f"{infer_ms:.0f}ms  |  {obj_count} obj  |  ZOOM: {zoom_lvl:.1f}X"
    lw = cv2.getTextSize(lat, FONT, 0.5, 1)[0][0]
    cv2.putText(frame, lat, (w - lw - 16, 46), FONT, 0.5, (130, 130, 130), 1, cv2.LINE_AA)

    if not paused:
        cv2.circle(frame, (w - sw - 28, 20), int(5 + 3 * abs(np.sin(time.time() * 3))), (0, 220, 100), -1)

    cv2.putText(frame, "P=Pause S=Shot Q=Quit B=B&W C=Color I/O=Zoom | Click Obj=Focus, Click BG=Reset",
                (16, h - 10), FONT, 0.45, (90, 90, 90), 1, cv2.LINE_AA)

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    global processing, actual_w, actual_h, focused_track, zoom_level

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

    # WINDOWS FIX: Add cv2.CAP_DSHOW which stabilizes webcams on Windows
    if os.name == 'nt':
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(0)
        
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    os.makedirs("screenshots", exist_ok=True)

    paused = False
    is_bw_mode = False 
    fps, frame_count = 0.0, 0
    fps_timer = time.time()

    while True:
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)
        actual_h, actual_w = frame.shape[:2]

        frame = apply_zoom(frame, zoom_level)
        if is_bw_mode: frame = apply_grayscale(frame)

        frame_count += 1
        if time.time() - fps_timer >= 0.5:
            fps = frame_count / (time.time() - fps_timer)
            frame_count, fps_timer = 0, time.time()

        if not paused and not processing:
            processing = True
            threading.Thread(target=infer, args=(frame.copy(),), daemon=True).start()

        with lock:
            objs = list(latest_objects)

        smoothed = update_smooth(objs if not paused else smoothed_objects)
        
        objects_to_draw = []
        if focused_track:
            best_dist = float('inf')
            best_match = None
            for obj in smoothed:
                if obj["label"] == focused_track["label"]:
                    dist = ((obj["cx"] - focused_track["cx"])**2 + (obj["cy"] - focused_track["cy"])**2)**0.5
                    if dist < 0.3 and dist < best_dist:
                        best_dist = dist
                        best_match = obj
            
            if best_match:
                focused_track["cx"] = best_match["cx"]
                focused_track["cy"] = best_match["cy"]
                focused_track["bw"] = best_match.get("bw", 0.15)
                focused_track["bh"] = best_match.get("bh", 0.15)
                focused_track["conf"] = best_match["conf"] 
                objects_to_draw = [best_match]
        else:
            objects_to_draw = smoothed

        draw_objects(frame, objects_to_draw, int(time.time() * 1000))
        draw_rgb_graph(frame)
        draw_object_details(frame, focused_track)  
        draw_ui(frame, fps, last_infer_ms, len(objects_to_draw), paused, is_bw_mode, zoom_level)

        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF
        if key in [ord("q"), ord("Q")]: break
        elif key in [ord("p"), ord("P")]: paused = not paused
        elif key in [ord("s"), ord("S")]:
            fname = f"screenshots/{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(fname, frame)
        elif key in [ord("b"), ord("B")]: is_bw_mode = True
        elif key in [ord("c"), ord("C")]: is_bw_mode = False
        elif key in [ord("i"), ord("I")]: zoom_level = min(5.0, zoom_level + 0.2)
        elif key in [ord("o"), ord("O")]: zoom_level = max(1.0, zoom_level - 0.2)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
