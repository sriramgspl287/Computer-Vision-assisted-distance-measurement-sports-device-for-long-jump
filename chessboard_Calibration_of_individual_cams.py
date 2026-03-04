import cv2
import numpy as np

# -------- SETTINGS --------
CHECKERBOARD = (11, 7)     # inner corners
SQUARE_SIZE = 0.025       # meters (25 mm squares)
MIN_IMAGES = 15
CAMERA_ID = 1 # change to 0 or 1 based on the camera to be calibrated
# --------------------------

criteria = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    30, 0.001
)

# Prepare 3D world points
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

objpoints = []
imgpoints = []

cap = cv2.VideoCapture(CAMERA_ID)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

print("Press SPACE to capture a valid checkerboard frame")
print("Press ENTER to calibrate once enough images collected")
print("Press ESC to exit")

# Make window resizable and medium sized
cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Calibration", 960, 540)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret_cb, corners = cv2.findChessboardCorners(gray, CHECKERBOARD)

    display = frame.copy()

    if ret_cb:
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        cv2.drawChessboardCorners(display, CHECKERBOARD, corners2, ret_cb)

    cv2.putText(display, f"Captured: {len(objpoints)}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0,255,0), 2)

    cv2.imshow("Calibration", display)

    key = cv2.waitKey(1) & 0xFF

    if ret_cb:
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        cv2.drawChessboardCorners(display, CHECKERBOARD, corners2, ret_cb)

    if key == 32:
        objpoints.append(objp)
        imgpoints.append(corners2)
        print(f"Captured frame {len(objpoints)}")

    elif key == 13:  # ENTER
        if len(objpoints) < MIN_IMAGES:
            print("Not enough images for calibration")
            continue

        print("Calibrating...")

        ret_calib, K, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, gray.shape[::-1], None, None
        )

        print("\n--- Calibration Results ---")
        print("RMS Reprojection Error:", ret_calib)
        print("\nCamera Matrix (K):\n", K)
        print("\nDistortion Coefficients:\n", dist)

        np.savez("mobilecam_intrinsics.npz",
                 K=K,
                 dist=dist,
                 image_size=gray.shape[::-1])

        print("\nSaved to mobilecam_intrinsics.npz")
        break

    elif key == 27:
        break

cap.release()
cv2.destroyAllWindows()