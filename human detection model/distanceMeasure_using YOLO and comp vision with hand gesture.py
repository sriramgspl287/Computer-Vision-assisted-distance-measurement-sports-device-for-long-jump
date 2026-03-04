# working program of odometer + hand gesture recognition.


import cv2
from ultralytics import YOLO
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ================= CONFIG =================

VIDEO_SOURCE = 0 
REAL_WIDTH_M = 0.26   	
CALIBRATION_FRAMES = 30
CONFIDENCE = 0.45
# ==========================================

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(VIDEO_SOURCE)

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

    cv2.putText(frame, f"Calibrating {len(pixel_widths)}/{CALIBRATION_FRAMES}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow("Calibration", frame)
    if cv2.waitKey(1) & 0xFF == 27: break

REFERENCE_PIXEL_WIDTH = sum(pixel_widths) / len(pixel_widths)
meters_per_pixel = REAL_WIDTH_M / REFERENCE_PIXEL_WIDTH
cv2.destroyWindow("Calibration")

# ----------- HAND GESTURE RECOGNITION SETUP ------------------
# 1. Define the connections (Standard hand skeleton mapping)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),      # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),      # Index
    (5, 9), (9, 10), (10, 11), (11, 12), # Middle
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20) # Pinky
]

# 2. Setup Options
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    running_mode=vision.RunningMode.VIDEO
)



# ----------- TRACKING VARS ----------------
prev_cx = None
distance_m = 0.0
is_paused = False
final_record = 0.0

cv2.namedWindow('Distance Tracker Pro', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Distance Tracker Pro', 800, 600)

with vision.HandLandmarker.create_from_options(options) as detector:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        hand_result = detector.detect_for_video(mp_image, timestamp)

        # 1. GESTURE LOGIC & DRAWING
        if hand_result.hand_landmarks:
            for hand_landmarks in hand_result.hand_landmarks:
                # Detect Gestures
                thumb_open = hand_landmarks[4].y < hand_landmarks[2].y
                index_open = hand_landmarks[8].y < hand_landmarks[6].y
                middle_open = hand_landmarks[12].y < hand_landmarks[10].y
                ring_open = hand_landmarks[16].y < hand_landmarks[14].y
                pinky_open = hand_landmarks[20].y < hand_landmarks[18].y

                if thumb_open and index_open and middle_open and ring_open and pinky_open:
                    is_paused = True  # PALM PAUSES
                    label = "PAUSED (PALM)"
                elif thumb_open and not (index_open or middle_open or ring_open or pinky_open):
                    final_record = distance_m # THUMB RECORDS
                    label = f"RECORDED: {final_record:.2f}m"
                elif index_open and middle_open and not (thumb_open or ring_open or pinky_open):
                    # VICTORY SIGN RESETS
                    distance_m = 0.0
                    final_record = 0.0
                    is_paused = False
                    label = "RESET (V-SIGN)"
                else:
                    is_paused = False # RESTING HAND RESUMES
                    label = "TRACKING..."

                cv2.putText(frame, label, (int(hand_landmarks[0].x * w), int(hand_landmarks[0].y * h) - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                # Draw Hand Skeleton
                for connection in HAND_CONNECTIONS:
                    s = hand_landmarks[connection[0]]
                    e = hand_landmarks[connection[1]]
                    cv2.line(frame, (int(s.x*w), int(s.y*h)), (int(e.x*w), int(e.y*h)), (255, 0, 0), 2)
                for lm in hand_landmarks:
                    cv2.circle(frame, (int(lm.x*w), int(lm.y*h)), 3, (0, 255, 0), -1)

        # 2. YOLO DISTANCE LOGIC (Only runs if not paused)
        results = model(frame, conf=CONFIDENCE, classes=[0], verbose=False)
        if results[0].boxes and not is_paused:
            box = results[0].boxes[0].xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, box)
            cx = (x1 + x2) // 2
            if prev_cx is not None:
                distance_m += abs(cx - prev_cx) * meters_per_pixel
            prev_cx = cx
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

        # 3. DISPLAY OVERLAY
        cv2.putText(frame, f"Dist: {distance_m:.2f} m", (20, 50), 1, 2, (255,255,255), 2)
        if final_record > 0:
            cv2.putText(frame, f"Last Record: {final_record:.2f} m", (20, 90), 1, 1.5, (0, 255, 0), 2)

        cv2.imshow('Distance Tracker Pro', frame)
        if cv2.waitKey(5) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()