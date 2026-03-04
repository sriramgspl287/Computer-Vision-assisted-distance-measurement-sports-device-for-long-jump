#low power identification and tracking of a person in a video stream, with distance estimation based on a single camera and known real-world width. 

import cv2
from ultralytics import YOLO

# ================= CONFIG =================
# Set VIDEO_SOURCE to an integer (webcam index) or a string (file or stream URL).
# Example: 0 for the default webcam, or "http://PHONE_IP:PORT/video" for an IP camera.
VIDEO_SOURCE = 0 # 0 = webcam | or "http://PHONE_IP:PORT/video"
REAL_WIDTH_M = 0.26   # your side-view body width in meters (26 cm)
CALIBRATION_FRAMES = 30
CONFIDENCE = 0.5
# ==========================================

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(VIDEO_SOURCE)

# verify the capture opened successfully
if not cap.isOpened():
    raise SystemExit(f"Cannot open video source: {VIDEO_SOURCE}")

# ----------- CALIBRATION ------------------
pixel_widths = []
print("Calibration: stand sideways at fixed distance (2 m)")

while len(pixel_widths) < CALIBRATION_FRAMES:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=CONFIDENCE, classes=[0], verbose=False)

    if results[0].boxes:
        box = results[0].boxes[0].xyxy[0].cpu().numpy()
        x1, _, x2, _ = map(int, box)
        width_px = x2 - x1

        # reject extreme noise
        if width_px > 20:
            pixel_widths.append(width_px)

        cv2.rectangle(frame, (x1, 50), (x2, 200), (0,255,0), 2)

    cv2.putText(frame,
                f"Calibrating {len(pixel_widths)}/{CALIBRATION_FRAMES}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Calibration", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

if len(pixel_widths) == 0:
    cap.release()
    cv2.destroyAllWindows()
    raise SystemExit("Calibration failed: no valid detections collected. Ensure a person was visible during calibration.")
REFERENCE_PIXEL_WIDTH = sum(pixel_widths) / len(pixel_widths)
meters_per_pixel = REAL_WIDTH_M / REFERENCE_PIXEL_WIDTH

print("Calibration complete")
print("Reference pixel width:", round(REFERENCE_PIXEL_WIDTH, 2))
print("Meters per pixel:", meters_per_pixel)

cv2.destroyWindow("Calibration")

# ----------- TRACKING ---------------------
prev_cx = None
distance_m = 0.0

# initialize tracking window size using a warm-up frame (fixes using w,h before definition)
ret, temp_frame = cap.read()
if not ret:
    cap.release()
    raise SystemExit("Cannot read frame to initialize tracker window")
h, w, _ = temp_frame.shape
cv2.namedWindow("Distance Tracker", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Distance Tracker", w, h)


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    A = int(round(w/3))
    B = int(round(2*w/3))

    # Draw zones
    cv2.line(frame, (A, 0), (A, h), (255,255,0), 2)
    cv2.line(frame, (B, 0), (B, h), (255,255,0), 2)

    results = model(frame, conf=CONFIDENCE, classes=[0], verbose=False)

    if results[0].boxes:
        box = results[0].boxes[0].xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = map(int, box)
        cx = (x1 + x2) // 2

        # distance accumulation
        if prev_cx is not None:
            delta_px = abs(cx - prev_cx)
            distance_m += delta_px * meters_per_pixel

        prev_cx = cx

        # zone logic
        if cx < A:
            zone = "LEFT"
        elif cx < B:
            zone = "MIDDLE"
        else:
            zone = "RIGHT"

        # draw
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.circle(frame, (cx,(y1+y2)//2), 5, (0,0,255), -1)

        cv2.putText(frame,
                    f"Distance: {distance_m:.2f} m",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        cv2.putText(frame,
                    f"Zone: {zone}",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    cv2.imshow("Distance Tracker", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()