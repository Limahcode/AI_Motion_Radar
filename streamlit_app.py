import streamlit as st
import cv2
import math
from ultralytics import YOLO
import tempfile
import os

# --- 1️⃣ Load YOLOv8 model ---
model = YOLO("best.pt")  # trained model

st.title("🚦 Object Tracking & Speed Estimation App")
st.write("Upload a short video to detect, track, and estimate object speeds.")

# --- 2️⃣ Upload video ---
uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])

if uploaded_file:
    # Save uploaded video to a temp file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    input_video = tfile.name
    output_video = input_video.replace(".mp4", "_out.mp4")

    cap = cv2.VideoCapture(input_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    tracker_data = {}  # {track_id: (x_center_prev, y_center_prev)}
    pixels_per_meter = 50  # adjust for your scene

    stframe = st.empty()  # placeholder for showing frames

    # --- 3️⃣ Process each frame ---
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(frame, persist=True, tracker="bytetrack.yaml", stream=True)

        for result in results:
            if result.boxes is not None:
                for i, box in enumerate(result.boxes):
                    x1, y1, x2, y2 = box.xyxy[0]
                    conf = box.conf[0]
                    cls = int(box.cls[0])
                    track_id = int(box.id) if box.id is not None else None

                    # Object center
                    x_center = (x1 + x2) / 2
                    y_center = (y1 + y2) / 2

                    # Speed calc
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
                    cv2.putText(frame, f"{model.names[cls]} {conf:.2f}", (int(x1), int(y2)+15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)


        out.write(frame)

        # Show live frame inside app
        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

    cap.release()
    out.release()

    st.success("✅ Video processing complete!")

    # --- 4️⃣ Display final video ---
    st.video(output_video)
