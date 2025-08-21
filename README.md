AI Motion Radar (YOLOv8 Object Tracking & Speed Estimation)

A computer vision project that detects, tracks, and estimates the speed of moving road users from traffic footage.
I annotated my own dataset (484 images) in CVAT, exported in YOLO format, trained a custom YOLOv8 model on Google Colab, tested locally in VSCode, and deployed with Streamlit Community Cloud.

The model is trained to recognize 5 traffic classes:

🚗 Car

🚌 Bus

🛺 Tricycle (Keke)

🚶 Person

🏍️ Bike

✨ Features

Upload a video (MP4/AVI/MOV/MKV)

YOLOv8 detection + tracking (ByteTrack) → stable IDs

Per-object speed estimate (m/s) using frame-to-frame displacement

Supports 5 custom classes: car, bus, tricycle, person, bike

Works on CPU (no GPU required)

One-click deployment on Streamlit Cloud

⚠️ Note: Speed is an estimate. Accuracy depends on camera angle, FPS, and your calibration (pixels_per_meter).


🧠 How it Works

Detect objects in each frame with my trained YOLOv8 model (best.pt).

Track detections across frames with ByteTrack so each object keeps a stable ID.

For every object ID, compute the distance moved between this frame and the previous frame (in pixels).

Convert pixels → meters using a simple scale: pixels_per_meter (set in the sidebar).

Convert per-frame distance to meters/second using the video FPS.

Draw bounding boxes with class, ID, and estimated speed.

Speed formula used:


📂 Dataset

Annotated 484 traffic frames in CVAT

Classes: car, bus, tricycle, person, bike

Exported in YOLO format (images + TXT labels)

~80/20 split for train/val


📦 Project Structure

capstonesfai_project
├── streamlit_app.py       # Streamlit UI + processing loop (tracking + speed)
├── best.pt                # Trained YOLOv8 weights (custom 5-class model)
├── requirements.txt       # Python dependencies (for local/Cloud)
├── packages.txt           # System dependencies for Cloud (ffmpeg, libGL)
└── README.md  


🚀 Quick Start (Local)

Requirements: Python 3.9–3.11 recommended.
# 1) Create and activate virtual environment (Windows PowerShell)
python -m venv venv
./venv/Scripts/Activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Run the app
streamlit run streamlit_app.py


☁️ Deploy to Streamlit Community Cloud

Push this project to a public GitHub repo (include best.pt).

Go to https://share.streamlit.io → New app.

Select repo/branch and entrypoint → streamlit_app.py.

Streamlit Cloud auto-installs packages and runs the app.

Your app will be live at:
https://<username>-ai-motion-radar.streamlit.app


🧩 Key Files
requirements.txt

CPU-friendly dependencies:
streamlit
ultralytics
opencv-python-headless
numpy
pillow
matplotlib
imageio
imageio-ffmpeg

--extra-index-url https://download.pytorch.org/whl/cpu
torch
torchvision


packages.txt

System dependencies:
ffmpeg
libgl1
libglib2.0-0

🧪 Training Process
1) Annotation (CVAT)

Imported traffic video → extracted frames

Labeled car, bus, tricycle, person, bike

Exported in YOLO format

2) Training (Colab)

!pip install ultralytics
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.train(data="/content/data.yaml", epochs=50, imgsz=640)


3) Deployment

Downloaded best.pt

Integrated into Streamlit app

Deployed to Streamlit Cloud


🧩 App Highlights

YOLO("best.pt") → loads trained model

model.track(..., persist=True, tracker="bytetrack.yaml") → stable IDs

Per-object speed estimated using bounding box centers


📊 Tips for Calibration

Use a known real-world distance (lane width ≈ 3.5m, zebra crossing, etc.)

Count its pixels in the frame

pixels_per_meter = pixels / meters

Use a region at similar depth to where motion is measured


✅ Results

Detects cars, buses, tricycles, persons, bikes in real-time

Tracks objects with unique IDs

Provides estimated speed overlay



👨‍💻 Author

Developed by Abimbola Alimat
Capstone Project — AI Motion Radar