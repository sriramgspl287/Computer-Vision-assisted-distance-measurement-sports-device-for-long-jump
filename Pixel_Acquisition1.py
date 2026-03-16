
#Stereo Vision Jump Distance Measurement System
#Cameras: Laptop Webcam (Camera A) + DroidCam Phone (Camera B)
#Features: Non-blocking threaded frame grabbers, frame sync, triangulation

#for stereo vision, measurement, or any real-world spatial analysis.
    
import cv2
import numpy as np
import threading
import time
import queue
from dataclasses import dataclass
from typing import Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# CONFIGURATION  (edit these)
# ─────────────────────────────────────────────
WEBCAM_INDEX       = 0          # laptop webcam (Camera A)
DROIDCAM_URL       = "http://192.168.1.100:4747/video"  # DroidCam URL (Camera B)
                                # Adjust IP to match your phone's DroidCam display
CAPTURE_WIDTH      = 640
CAPTURE_HEIGHT     = 480
TARGET_FPS         = 30
SYNC_TOLERANCE_MS  = 50         # max timestamp diff to consider frames "synced"
BASELINE_M         = 1.5        # distance between cameras in metres
FRAME_BUFFER_SIZE  = 2          # keep only latest N frames per camera

# ─── Camera intrinsics (replace with your calibration results) ────────────────
# Format: (fx, fy, cx, cy, dist_coeffs)
CAM_A_PARAMS = dict(
    fx=600.0, fy=600.0, cx=320.0, cy=240.0,
    dist=np.zeros(5)   # replace with actual distortion coefficients
)
CAM_B_PARAMS = dict(
    fx=580.0, fy=580.0, cx=320.0, cy=240.0,
    dist=np.zeros(5)
)


# ─────────────────────────────────────────────
# NON-BLOCKING FRAME GRABBER
# ─────────────────────────────────────────────
class FrameGrabber:
    """
    Dedicated thread that continuously drains the camera buffer.
    Only the LATEST frame is kept — no stale frame buildup.
    This is the key to eliminating lag with cv2.VideoCapture.
    """

    def __init__(self, source, name: str, width=640, height=480, fps=30):
        self.name   = name
        self.source = source
        self.width  = width
        self.height = height
        self.fps    = fps

        self._frame: Optional[np.ndarray] = None
        self._ts:    float = 0.0
        self._lock   = threading.Lock()
        self._stop   = threading.Event()
        self._ready  = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"grabber-{name}", daemon=True)

    def start(self):
        self._thread.start()
        log.info(f"[{self.name}] Grabber thread started, waiting for first frame...")
        if not self._ready.wait(timeout=10):
            raise RuntimeError(f"[{self.name}] Camera failed to open within 10 s")
        log.info(f"[{self.name}] Ready ✓")
        return self

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=3)

    def read(self) -> Tuple[Optional[np.ndarray], float]:
        """Return (frame, timestamp_s). Thread-safe. Non-blocking."""
        with self._lock:
            return self._frame, self._ts

    @property
    def is_alive(self):
        return self._thread.is_alive()

    def _open_cap(self):
        if isinstance(self.source, int):
            cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW if cv2.os.name == 'nt' else cv2.CAP_V4L2)
        else:
            cap = cv2.VideoCapture(self.source)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS,          self.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)   # minimize internal buffer → less lag
        return cap

    def _run(self):
        cap = self._open_cap()
        if not cap.isOpened():
            log.error(f"[{self.name}] Failed to open source: {self.source}")
            return

        first = True
        consecutive_failures = 0

        while not self._stop.is_set():
            ret, frame = cap.read()

            if not ret:
                consecutive_failures += 1
                if consecutive_failures > 30:
                    log.warning(f"[{self.name}] Too many read failures, reconnecting...")
                    cap.release()
                    time.sleep(1)
                    cap = self._open_cap()
                    consecutive_failures = 0
                continue

            consecutive_failures = 0
            ts = time.monotonic()

            with self._lock:
                self._frame = frame
                self._ts    = ts

            if first:
                self._ready.set()
                first = False

        cap.release()
        log.info(f"[{self.name}] Grabber stopped.")


# ─────────────────────────────────────────────
# FRAME SYNCHRONIZER
# ─────────────────────────────────────────────
class StereoSync:
    """
    Fetches latest frames from both grabbers and checks temporal sync.
    If timestamps differ by more than SYNC_TOLERANCE_MS it warns but still returns frames.
    """

    def __init__(self, grabber_a: FrameGrabber, grabber_b: FrameGrabber, tol_ms=50):
        self.ga  = grabber_a
        self.gb  = grabber_b
        self.tol = tol_ms / 1000.0

    def get_synced_pair(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float, bool]:
        """
        Returns (frame_a, frame_b, sync_delta_ms, is_synced)
        """
        fa, ta = self.ga.read()
        fb, tb = self.gb.read()

        if fa is None or fb is None:
            return None, None, 999.0, False

        delta = abs(ta - tb) * 1000.0
        synced = delta < (self.tol * 1000.0)
        return fa, fb, delta, synced


# ─────────────────────────────────────────────
# CAMERA CALIBRATION HELPERS
# ─────────────────────────────────────────────
@dataclass
class CameraModel:
    fx: float; fy: float; cx: float; cy: float
    dist: np.ndarray
    K: np.ndarray = None

    def __post_init__(self):
        self.K = np.array([[self.fx, 0, self.cx],
                           [0, self.fy, self.cy],
                           [0,       0,       1]], dtype=np.float64)

    def undistort_point(self, px, py) -> Tuple[float, float]:
        pt = np.array([[[px, py]]], dtype=np.float64)
        und = cv2.undistortPoints(pt, self.K, self.dist, P=self.K)
        return float(und[0,0,0]), float(und[0,0,1])

    def pixel_to_bearing_angle(self, px, py) -> Tuple[float, float]:
        """Return horizontal (azimuth) and vertical angles in radians."""
        ux, uy = self.undistort_point(px, py)
        theta_h = np.arctan2(ux - self.cx, self.fx)   # horizontal bearing
        theta_v = np.arctan2(uy - self.cy, self.fy)   # vertical bearing
        return theta_h, theta_v


cam_a = CameraModel(**CAM_A_PARAMS)
cam_b = CameraModel(**CAM_B_PARAMS)


# ─────────────────────────────────────────────
# TRIANGULATION
# ─────────────────────────────────────────────
def triangulate(alpha_rad: float, beta_rad: float, baseline: float) -> Tuple[float, float]:
    """
    Simple 2-ray intersection in the horizontal plane.
    Camera A at origin (0,0), Camera B at (baseline, 0).
    alpha: bearing angle from cam A (positive = toward cam B side)
    beta : bearing angle from cam B (positive = toward cam A side)

    Returns (x, y) real-world position relative to Camera A.
    """
    # tan(α) = x / y      →  x = y·tan(α)
    # tan(β) = (b-x) / y  →  b-x = y·tan(β)
    # → b = y(tan(α) + tan(β))
    tan_a = np.tan(alpha_rad)
    tan_b = np.tan(beta_rad)
    denom = tan_a + tan_b
    if abs(denom) < 1e-6:
        raise ValueError("Rays are parallel — cannot triangulate")
    y = baseline / denom
    x = y * tan_a
    return x, y


def compute_jump_distance(p_takeoff: Tuple[float, float],
                           p_landing: Tuple[float, float]) -> float:
    dx = p_landing[0] - p_takeoff[0]
    dy = p_landing[1] - p_takeoff[1]
    return np.hypot(dx, dy)


# ─────────────────────────────────────────────
# DISPLAY HELPERS
# ─────────────────────────────────────────────
def draw_crosshair(frame, cx, cy, color=(0,255,0), size=15):
    cv2.line(frame, (cx-size, cy), (cx+size, cy), color, 2)
    cv2.line(frame, (cx, cy-size), (cx, cy+size), color, 2)
    cv2.circle(frame, (cx, cy), size//2, color, 1)

def overlay_text(frame, lines, start_y=20, color=(255,255,255)):
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (10, start_y + i*22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3)
        cv2.putText(frame, line, (10, start_y + i*22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)


# ─────────────────────────────────────────────
# PHONE FEED ADJUSTMENT (brightness / contrast / resize)
# ─────────────────────────────────────────────
def match_phone_to_webcam(frame_phone: np.ndarray,
                           frame_webcam: np.ndarray) -> np.ndarray:
    """
    Simple histogram-based brightness/contrast matching of phone frame to webcam.
    Uses LAB color space for perceptual accuracy.
    """
    # Resize phone frame to match webcam if needed
    if frame_phone.shape != frame_webcam.shape:
        frame_phone = cv2.resize(frame_phone, (frame_webcam.shape[1], frame_webcam.shape[0]))

    # Convert to LAB
    lab_ref = cv2.cvtColor(frame_webcam, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_src = cv2.cvtColor(frame_phone,  cv2.COLOR_BGR2LAB).astype(np.float32)

    # Match mean and std of L channel only (luminance)
    for ch in range(3):  # match all 3 channels for color fidelity
        mean_r, std_r = lab_ref[:,:,ch].mean(), lab_ref[:,:,ch].std()
        mean_s, std_s = lab_src[:,:,ch].mean(), lab_src[:,:,ch].std()
        if std_s > 0:
            lab_src[:,:,ch] = (lab_src[:,:,ch] - mean_s) * (std_r / std_s) + mean_r

    lab_src = np.clip(lab_src, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab_src, cv2.COLOR_LAB2BGR)


# ─────────────────────────────────────────────
# MAIN APPLICATION
# ─────────────────────────────────────────────
class StereoJumpMeasurer:
    STATE_IDLE    = "IDLE — press T for Takeoff"
    STATE_TAKEOFF = "TAKEOFF set — press L for Landing"
    STATE_RESULT  = "RESULT — press R to reset"

    def __init__(self):
        self.grabber_a = FrameGrabber(WEBCAM_INDEX, "WebcamA",
                                       CAPTURE_WIDTH, CAPTURE_HEIGHT, TARGET_FPS)
        self.grabber_b = FrameGrabber(DROIDCAM_URL, "DroidCamB",
                                       CAPTURE_WIDTH, CAPTURE_HEIGHT, TARGET_FPS)
        self.sync      = StereoSync(self.grabber_a, self.grabber_b, SYNC_TOLERANCE_MS)

        self.state       = self.STATE_IDLE
        self.pt_takeoff  = None   # (x_world, y_world)
        self.pt_landing  = None
        self.distance_m  = None

        # Click points (pixel coords per camera)
        self.click_a: Optional[Tuple[int,int]] = None
        self.click_b: Optional[Tuple[int,int]] = None

        self.match_colors = True   # toggle with 'M'

    def _mouse_a(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.click_a = (x, y)

    def _mouse_b(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.click_b = (x, y)

    def _triangulate_clicks(self) -> Optional[Tuple[float, float]]:
        if self.click_a is None or self.click_b is None:
            return None
        alpha, _ = cam_a.pixel_to_bearing_angle(*self.click_a)
        beta,  _ = cam_b.pixel_to_bearing_angle(*self.click_b)
        try:
            return triangulate(alpha, beta, BASELINE_M)
        except ValueError as e:
            log.warning(f"Triangulation failed: {e}")
            return None

    def run(self):
        log.info("Starting grabbers...")
        self.grabber_a.start()
        self.grabber_b.start()

        cv2.namedWindow("Camera A — Webcam",  cv2.WINDOW_NORMAL)
        cv2.namedWindow("Camera B — DroidCam", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Camera A — Webcam",  self._mouse_a)
        cv2.setMouseCallback("Camera B — DroidCam", self._mouse_b)

        fps_counter, fps_start, fps_val = 0, time.monotonic(), 0.0

        print("\n═══════════════════════════════════════")
        print("  STEREO JUMP MEASURER — CONTROLS")
        print("  T  = record TAKEOFF position")
        print("  L  = record LANDING position")
        print("  R  = reset measurement")
        print("  M  = toggle color matching")
        print("  Q  = quit")
        print("  Click on BOTH frames to select a point")
        print("═══════════════════════════════════════\n")

        try:
            while True:
                fa, fb, delta_ms, synced = self.sync.get_synced_pair()
                if fa is None or fb is None:
                    time.sleep(0.01)
                    continue

                # ── Color match phone to webcam ───────────────────────────
                if self.match_colors:
                    fb = match_phone_to_webcam(fb, fa)

                # ── FPS counter ───────────────────────────────────────────
                fps_counter += 1
                now = time.monotonic()
                if now - fps_start >= 1.0:
                    fps_val     = fps_counter / (now - fps_start)
                    fps_counter = 0
                    fps_start   = now

                # ── Draw crosshairs on click points ───────────────────────
                disp_a, disp_b = fa.copy(), fb.copy()
                if self.click_a:
                    draw_crosshair(disp_a, *self.click_a, (0,255,0))
                if self.click_b:
                    draw_crosshair(disp_b, *self.click_b, (0,165,255))

                # ── Overlay info ──────────────────────────────────────────
                sync_color = (0,255,0) if synced else (0,0,255)
                sync_label = f"Sync: {delta_ms:.1f}ms {'OK' if synced else 'WARN'}"

                overlay_text(disp_a, [
                    f"FPS: {fps_val:.1f}",
                    sync_label,
                    f"State: {self.state}",
                    "Click to select point",
                ])
                overlay_text(disp_b, [
                    "Camera B — DroidCam",
                    f"Color match: {'ON' if self.match_colors else 'OFF'}",
                    "Click to select point",
                ])

                if self.distance_m is not None:
                    dist_text = f"JUMP DISTANCE: {self.distance_m:.3f} m"
                    cv2.putText(disp_a, dist_text, (10, CAPTURE_HEIGHT-20),
                                cv2.FONT_HERSHEY_DUPLEX, 0.8, (0,255,255), 2)

                cv2.imshow("Camera A — Webcam",  disp_a)
                cv2.imshow("Camera B — DroidCam", disp_b)

                # ── Keyboard ──────────────────────────────────────────────
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    break

                elif key == ord('m'):
                    self.match_colors = not self.match_colors
                    log.info(f"Color matching: {'ON' if self.match_colors else 'OFF'}")

                elif key == ord('t'):
                    pos = self._triangulate_clicks()
                    if pos:
                        self.pt_takeoff = pos
                        self.state = self.STATE_TAKEOFF
                        log.info(f"Takeoff recorded: {pos[0]:.3f}m, {pos[1]:.3f}m")
                    else:
                        log.warning("Click a point on BOTH camera views first!")

                elif key == ord('l'):
                    if self.pt_takeoff is None:
                        log.warning("Record takeoff first (press T)!")
                    else:
                        pos = self._triangulate_clicks()
                        if pos:
                            self.pt_landing = pos
                            self.distance_m = compute_jump_distance(
                                self.pt_takeoff, self.pt_landing)
                            self.state = self.STATE_RESULT
                            log.info(f"Landing recorded: {pos[0]:.3f}m, {pos[1]:.3f}m")
                            log.info(f"═══ JUMP DISTANCE: {self.distance_m:.4f} m ═══")
                        else:
                            log.warning("Click a point on BOTH camera views first!")

                elif key == ord('r'):
                    self.pt_takeoff  = None
                    self.pt_landing  = None
                    self.distance_m  = None
                    self.click_a     = None
                    self.click_b     = None
                    self.state       = self.STATE_IDLE
                    log.info("Reset.")

        finally:
            self.grabber_a.stop()
            self.grabber_b.stop()
            cv2.destroyAllWindows()
            log.info("Done.")


# ─────────────────────────────────────────────
# CALIBRATION UTILITY (run separately)
# ─────────────────────────────────────────────
def run_calibration(cam_source, cam_name: str,
                    checkerboard=(9,6), square_size_m=0.025):
    """
    Run this function separately for each camera to get intrinsics.
    Hold a checkerboard in front of the camera and press SPACE to capture.
    Press C to calibrate once you have 15+ good captures.
    """
    cap = cv2.VideoCapture(cam_source)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    objp = np.zeros((checkerboard[0]*checkerboard[1], 3), np.float32)
    objp[:,:2] = np.mgrid[0:checkerboard[0],
                           0:checkerboard[1]].T.reshape(-1,2) * square_size_m

    obj_pts, img_pts = [], []
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    print(f"\n[CALIBRATION] {cam_name}")
    print("  SPACE = capture frame  |  C = calibrate  |  Q = quit\n")

    while True:
        ret, frame = cap.read()
        if not ret: continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, checkerboard, None)
        disp = frame.copy()
        if found:
            cv2.drawChessboardCorners(disp, checkerboard, corners, found)
            cv2.putText(disp, "Board detected!", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        cv2.putText(disp, f"Captures: {len(obj_pts)}", (10, frame.shape[0]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
        cv2.imshow(f"Calibration — {cam_name}", disp)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' ') and found:
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            obj_pts.append(objp)
            img_pts.append(corners2)
            print(f"  Captured frame {len(obj_pts)}")
        elif key == ord('c') and len(obj_pts) >= 10:
            print("  Calibrating...")
            h, w = frame.shape[:2]
            ret, K, dist, _, _ = cv2.calibrateCamera(obj_pts, img_pts, (w,h), None, None)
            print(f"\n  ── Results for {cam_name} ──")
            print(f"  RMS error : {ret:.4f} px  (aim < 1.0)")
            print(f"  fx={K[0,0]:.2f}  fy={K[1,1]:.2f}")
            print(f"  cx={K[0,2]:.2f}  cy={K[1,2]:.2f}")
            print(f"  dist={dist.ravel().tolist()}\n")
            print("  → Paste these values into CAM_A_PARAMS / CAM_B_PARAMS in the main script")
            break

    cap.release()
    cv2.destroyAllWindows()


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "calibrate":
        # python stereo_capture.py calibrate a   → calibrate webcam
        # python stereo_capture.py calibrate b   → calibrate phone
        which = sys.argv[2].lower() if len(sys.argv) > 2 else "a"
        if which == "a":
            run_calibration(WEBCAM_INDEX, "Webcam (Camera A)")
        else:
            run_calibration(DROIDCAM_URL, "DroidCam (Camera B)")
    else:
        app = StereoJumpMeasurer()
        app.run()