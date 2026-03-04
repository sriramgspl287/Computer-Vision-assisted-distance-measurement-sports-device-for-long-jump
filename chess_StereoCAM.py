import cv2
import numpy as np

# ---------- SETTINGS ----------
CHECKERBOARD = (11, 7)
SQUARE_SIZE = 0.025  # meters
MIN_PAIRS = 20
WEBCAM_ID = 0
PHONE_ID = 1
# ------------------------------

# Load intrinsics
webcam_data = np.load("webcam_intrinsics.npz")
phone_data  = np.load("mobilecam_intrinsics.npz")

K1 = webcam_data["K"]
dist1 = webcam_data["dist"]

K2 = phone_data["K"]
dist2 = phone_data["dist"]

# Prepare object points
objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1,2)
objp *= SQUARE_SIZE

objpoints = []
imgpoints1 = []
imgpoints2 = []

cap1 = cv2.VideoCapture(WEBCAM_ID)
cap2 = cv2.VideoCapture(PHONE_ID)

print("Press SPACE to capture valid stereo pair")
print("Press ENTER to run stereo calibration")
print("Press ESC to exit")

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if not ret1 or not ret2:
        break

    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    ret_cb1, corners1 = cv2.findChessboardCorners(gray1, CHECKERBOARD)
    ret_cb2, corners2 = cv2.findChessboardCorners(gray2, CHECKERBOARD)

    display1 = frame1.copy()
    display2 = frame2.copy()

    if ret_cb1 and ret_cb2:
        corners1 = cv2.cornerSubPix(gray1, corners1, (11,11), (-1,-1), criteria)
        corners2 = cv2.cornerSubPix(gray2, corners2, (11,11), (-1,-1), criteria)

        cv2.drawChessboardCorners(display1, CHECKERBOARD, corners1, ret_cb1)
        cv2.drawChessboardCorners(display2, CHECKERBOARD, corners2, ret_cb2)

    cv2.putText(display1, f"Pairs: {len(objpoints)}",
                (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Webcam", display1)
    cv2.imshow("Phone", display2)

    key = cv2.waitKey(1)

    if key == 32 and ret_cb1 and ret_cb2:
        objpoints.append(objp)
        imgpoints1.append(corners1)
        imgpoints2.append(corners2)
        print(f"Captured pair {len(objpoints)}")

    elif key == 13:
        if len(objpoints) < MIN_PAIRS:
            print("Not enough stereo pairs")
            continue

        print("Running stereo calibration...")

        flags = cv2.CALIB_FIX_INTRINSIC

        retStereo, K1, dist1, K2, dist2, R, T, E, F = cv2.stereoCalibrate(
            objpoints,
            imgpoints1,
            imgpoints2,
            K1,
            dist1,
            K2,
            dist2,
            gray1.shape[::-1],
            criteria=criteria,
            flags=flags
        )

        print("\n--- Stereo Calibration Results ---")
        print("Stereo RMS error:", retStereo)
        print("\nRotation Matrix (R):\n", R)
        print("\nTranslation Vector (T):\n", T)

        baseline = np.linalg.norm(T)
        print("\nBaseline (meters):", baseline)

        np.savez("stereo_params.npz",
                 R=R,
                 T=T,
                 E=E,
                 F=F,
                 baseline=baseline)

        print("\nSaved stereo_params.npz")
        break

    elif key == 27:
        break

cap1.release()
cap2.release()
cv2.destroyAllWindows()