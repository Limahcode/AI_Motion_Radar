import streamlit as st
import cv2
import math
from ultralytics import YOLO
import tempfile
import time
import os

# --- 1️⃣ Load YOLOv8 model ---
model = YOLO("best.pt")  # your trained model

st.title("🚦 AI Motion Radar App")
st.write("Upload a short video or use the demo video to detect, track, and estimate object speeds.")

# --- 2️⃣ File uploader + fallback ---
uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])
DEMO_VIDEO = "short_video.mp4"

if uploaded_file is not None:
    # Save uploaded video to a temp file
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    input_video = tfile.name
else:
    if os.path.exists(DEMO_VIDEO):
        st.info("Using bundled demo video...")
        input_video = DEMO_VIDEO
    else:
        st.warning("⚠️ Please upload a video to continue.")
        st.stop()

# --- 3️⃣ Open video to check validity ---
cap = cv2.VideoCapture(input_video)
if not cap.isOpened():
    st.error(f"❌ Could not open video: {input_video}")
    st.stop()

fps = cap.get(cv2.CAP_PROP_FPS) or 30
pixels_per_meter = 50  # adjust for your scene
tracker_data = {}  # {track_id: (x_center_prev, y_center_prev)}

stframe = st.empty()  # placeholder for frames

# --- 4️⃣ Process each frame with YOLO ---
for result in model.track(
    source=input_video, 
    persist=True, 
    tracker="bytetrack.yaml", 
    stream=True
):
    frame = result.orig_img
    if frame is None:
        continue

    if result.boxes is not None:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            conf = box.conf[0]
            cls = int(box.cls[0])
            track_id = int(box.id) if box.id is not None else None

            # Object center
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2

            # Speed calculation
            if track_id is not None and track_id in tracker_data:
                x_prev, y_prev = tracker_data[track_id]
                dist_pixels = math.hypot(x_center - x_prev, y_center - y_prev)
                speed_m_per_s = (dist_pixels / pixels_per_meter) * fps
                cv2.putText(frame, f"ID:{track_id} {speed_m_per_s:.2f} m/s",
                            (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255,0,0), 2)

            if track_id is not None:
                tracker_data[track_id] = (x_center, y_center)

            # Draw bounding box + class
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
            cv2.putText(frame, f"{model.names[cls]} {conf:.2f}",
                        (int(x1), int(y2)+15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    # Show frame in Streamlit
    stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

    # Slow down for UI
    time.sleep(1 / fps)

st.success("✅ Video processing complete!")
