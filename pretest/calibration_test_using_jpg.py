import numpy as np
import cv2 as cv
import glob
import os
import matplotlib.pyplot as plt


def calibrate(showPics=True):

    root = os.getcwd()
    calibrationDir = os.path.join(root, 'demoImages', 'calibration')
    imgPathList = glob.glob(os.path.join(calibrationDir, '*.jpg'))

    nRows, nCols = 9, 6
    termCriteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    worldPointsCur = np.zeros((nRows * nCols, 3), np.float32)
    worldPointsCur[:, :2] = np.mgrid[0:nRows, 0:nCols].T.reshape(-1, 2)

    worldPointsList = []
    imgPointsList = []

    for imgPath in imgPathList:
        img = cv.imread(imgPath)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        found, corners = cv.findChessboardCorners(gray, (nRows, nCols), None)

        if found:
            worldPointsList.append(worldPointsCur)
            refinedCorners = cv.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), termCriteria
            )
            imgPointsList.append(refinedCorners)

            if showPics:
                cv.drawChessboardCorners(img, (nRows, nCols), refinedCorners, found)
                cv.imshow("Chessboard", img)
                cv.waitKey(300)

    cv.destroyAllWindows()

    repError, camMatrix, distCoeff, rvecs, tvecs = cv.calibrateCamera(
        worldPointsList, imgPointsList, gray.shape[::-1], None, None
    )

    print(f"Reprojection error: {repError:.4f}")
    print("Camera matrix:\n", camMatrix)

    np.savez("calibration.npz",
             camMatrix=camMatrix,
             distCoeff=distCoeff)

    return camMatrix, distCoeff


def removeDistortion(camMatrix, distCoeff):

    img = cv.imread("demoImages/distortion2.jpg")
    h, w = img.shape[:2]

    newCamMatrix, roi = cv.getOptimalNewCameraMatrix(
        camMatrix, distCoeff, (w, h), 1, (w, h)
    )

    undistorted = cv.undistort(img, camMatrix, distCoeff, None, newCamMatrix)

    cv.line(img, (1769, 103), (1780, 922), (255, 255, 255), 2)
    cv.line(undistorted, (1769, 103), (1780, 922), (255, 255, 255), 2)

    plt.figure(figsize=(10,5))
    plt.subplot(121), plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB)), plt.title("Original")
    plt.subplot(122), plt.imshow(cv.cvtColor(undistorted, cv.COLOR_BGR2RGB)), plt.title("Undistorted")
    plt.show()


def runCalibration():
    calibrate(showPics=True)


def runRemoveDistortion():
    camMatrix, distCoeff = calibrate(showPics=False)
    removeDistortion(camMatrix, distCoeff)


if __name__ == "__main__":
    runCalibration()
    runRemoveDistortion()
