#not working yet, need to adjust the histogram matching algorithm and ensure proper video capture from both sources. This code attempts to match the color distribution of the mobile camera feed to that of the laptop webcam feed using LAB color space for better results.
#working on it right now, will update soon

#working better than pizel_acquisition.py but still not perfect. The histogram matching is a simple linear scaling of the LAB channels, which may not capture all the nuances of the color differences between the two cameras. Further adjustments and possibly more advanced techniques may be needed for a closer match.
import cv2
import numpy as np

# Adjust these to your detected indices
LAPTOP_INDEX = 0 
PHONE_INDEX = 1 # Or "http://localhost:4747/video" if using DroidCam URL

def match_mobile_to_webcam(mobile_frame, webcam_frame):
    """
    Forces the high-quality mobile frame to adopt the 
    color and brightness characteristics of the laptop webcam.
    """
    # Convert both to LAB color space
    # L = Lightness, A = Green-Red, B = Blue-Yellow
    mob_lab = cv2.cvtColor(mobile_frame, cv2.COLOR_BGR2LAB).astype("float32")
    web_lab = cv2.cvtColor(webcam_frame, cv2.COLOR_BGR2LAB).astype("float32")

    # Split channels
    m_l, m_a, m_b = cv2.split(mob_lab)
    w_l, w_a, w_b = cv2.split(web_lab)

    # Function to scale distribution of one channel to match another
    def scale_channel(src, target):
        src_mean, src_std = src.mean(), src.std()
        tgt_mean, tgt_std = target.mean(), target.std()
        
        # Avoid division by zero
        if src_std == 0: return src
        
        # Apply linear transformation: (x - mean) * (target_std / source_std) + target_mean
        result = (src - src_mean) * (tgt_std / src_std) + tgt_mean
        return np.clip(result, 0, 255).astype("uint8")

    # Match all three channels
    matched_l = scale_channel(m_l, w_l)
    matched_a = scale_channel(m_a, w_a)
    matched_b = scale_channel(m_b, w_b)

    # Merge and convert back to BGR
    matched_lab = cv2.merge([matched_l, matched_a, matched_b])
    return cv2.cvtColor(matched_lab, cv2.COLOR_LAB2BGR)

def main():
    # Use DSHOW for Windows stability
    cam_web = cv2.VideoCapture(LAPTOP_INDEX, cv2.CAP_DSHOW)
    cam_mob = cv2.VideoCapture(PHONE_INDEX, cv2.CAP_DSHOW)

    while True:
        ret_w, frame_web = cam_web.read()
        ret_m, frame_mob = cam_mob.read()

        if not ret_w or not ret_m:
            break

        # Standardize sizes (640, 480)
        frame_web = cv2.resize(frame_web, (640, 480))
        frame_mob = cv2.resize(frame_mob, (640, 480))

        # Match Mobile -> Webcam
        matched_mobile = match_mobile_to_webcam(frame_mob, frame_web)

        # Display side by side
        # Left: Original Webcam | Right: Matched Mobile
        combined = np.hstack((frame_web, matched_mobile))
        
        cv2.putText(combined, "Webcam (Target)", (10, 30), 1, 1.5, (0, 255, 0), 2)
        cv2.putText(combined, "Matched Mobile", (650, 30), 1, 1.5, (0, 255, 0), 2)
        

        cv2.imshow("Stereo Consistency", combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam_web.release()
    cam_mob.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()