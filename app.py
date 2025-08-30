from ultralytics import YOLO
import cv2
import math

# --- 1️⃣ Load YOLOv8 model ---
model = YOLO("best.pt")  # your trained model

# --- 2️⃣ Video input/output ---
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
pixels_per_meter = 50  # adjust according to your scene

# --- 5️⃣ Process each frame ---
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO with ByteTrack tracker
    results = model.predict(source=frame, tracker="bytetrack.yaml", stream=True)

    for result in results:
        if result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0]  # coordinates
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                track_id = int(box.id[0]) if box.id is not None else None

                # Object center
                x_center = (x1 + x2) / 2
                y_center = (y1 + y2) / 2

                # Compute speed (use 0.0 if no previous position)
                speed_m_per_s = 0.0
                if track_id is not None:
                    if track_id in tracker_data:
                        x_prev, y_prev = tracker_data[track_id]
                        dist_pixels = math.hypot(x_center - x_prev, y_center - y_prev)
                        speed_m_per_s = (dist_pixels / pixels_per_meter) * fps
                    # Update tracker data for next frame
                    tracker_data[track_id] = (x_center, y_center)

                # Prepare label
                label_text = f"{model.names[cls]} {conf:.2f}"
                if track_id is not None:
                    label_text += f" ID:{track_id} Speed:{speed_m_per_s:.2f} m/s"

                # Draw bounding box and label
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
                cv2.putText(frame, label_text, (int(x1), int(y1)-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    # Write frame to output
    out.write(frame)

    # Optional: show live
    cv2.imshow("AI Motion Radar", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"✅ Annotated video saved to {output_video}")
