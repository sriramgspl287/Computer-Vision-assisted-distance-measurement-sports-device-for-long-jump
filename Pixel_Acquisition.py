#Step 1 — Pixel Acquisition
#**Goal:** Obtain accurate pixel coordinates from images/video/webcam.

#Implemented:

# - OpenCV feed (image / video / webcam)
# - Mouse click callback to print pixel coordinates `(u, v)`
# - Visual confirmation using overlay dots

#Validation performed:

# - Verified OpenCV coordinate system:
# - origin at top-left
# - x increases right
# - y increases downward
# - Confirmed pixel coordinates are consistent across clicks.


import cv2
import os
from ultralytics import YOLO
import numpy as np
import math


# ===== SELECT SOURCE =====
# 0  → webcam
# "video.mp4" → video file
# "image.jpg" → image file
SOURCE = "gettyimages-489091380-612x612.jpg"
# =========================



def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Clicked pixel: u={x}, v={y}")


def is_image(path):
    return isinstance(path, str) and os.path.splitext(path)[1].lower() in (
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp",
        ".tif",
        ".tiff",
    )


cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", mouse_callback)

if is_image(SOURCE):
    img = cv2.imread(SOURCE)
    if img is None:
        raise RuntimeError("Could not open image source")
    while True:
        cv2.imshow("Frame", img)
        key = cv2.waitKey(0)  # wait until a key is pressed
        if key == 27:  # ESC
            break
    cv2.destroyAllWindows()
else:
    cap = cv2.VideoCapture(SOURCE)
    if not cap.isOpened():
        raise RuntimeError("Could not open video source")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Frame", frame)

        key = cv2.waitKey(1)
        if key == 27:  # ESC to quit
            break

    cap.release()
    cv2.destroyAllWindows()