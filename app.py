from ultralytics import YOLO
import cv2
import math

# --- 1️⃣ Load YOLOv8 model ---
model = YOLO("best.pt")  # your trained model

# --- 2️⃣ Open video ---
input_video = "short_video.mp4"
output_video = "tracked_output.mp4"
cap = cv2.VideoCapture(input_video)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# --- 3️⃣ Video writer ---
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

# --- 4️⃣ Tracker data ---
tracker_data = {}  # {track_id: (x_center_prev, y_center_prev)}

# --- 5️⃣ Scaling factor for speed ---
pixels_per_meter = 50  # adjust according to your scene

# --- 6️⃣ Process each frame ---
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Use tracker="bytetrack.yaml" to get track IDs
    results = model.predict(source=frame, tracker="bytetrack.yaml", stream=True)

    for result in results:
        if result.boxes is not None:
            for i, box in enumerate(result.boxes):
                x1, y1, x2, y2 = box.xyxy[0]  # bounding box
                conf = box.conf[0]            # confidence
                cls = int(box.cls[0])          # class
                track_id = int(box.id[0]) if hasattr(box, 'id') else None  # tracker ID

                # Compute object center
                x_center = (x1 + x2) / 2
                y_center = (y1 + y2) / 2

                # Compute speed if track_id exists
                if track_id is not None and track_id in tracker_data:
                    x_prev, y_prev = tracker_data[track_id]
                    dist_pixels = math.hypot(x_center - x_prev, y_center - y_prev)
                    speed_m_per_s = (dist_pixels / pixels_per_meter) * fps
                    cv2.putText(frame, f"ID:{track_id} Speed:{speed_m_per_s:.2f} m/s",
                                (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0,255,0), 3)

                # Update tracker data
                if track_id is not None:
                    tracker_data[track_id] = (x_center, y_center)

                # Draw bounding box and label
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
                cv2.putText(frame, f"{model.names[cls]} {conf:.2f}", (int(x1), int(y2)+15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)


    # Write frame to output video
    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"✅ Annotated video saved to {output_video}")
