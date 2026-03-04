"""
Stereo Jump Distance Measurer — Primitive / Manual Approach
────────────────────────────────────────────────────────────
WORKFLOW:
  1. Live dual-camera feed (webcam + DroidCam)
  2. Press SPACE to start recording, SPACE again to stop
  3. Scrub through recorded frames to find takeoff & landing
  4. Click landing point on Camera A view → vertical line drawn
  5. Click takeoff point  on Camera A view → parallelogram drawn
  6. Triangulate both points → real jump distance printed

Controls (Live mode):
  SPACE   → start / stop recording
  Q       → quit

Controls (Review mode):
  ← / →   → step through frames (hold for fast scroll)
  T        → mark current frame as TAKEOFF  (then click the point)
  L        → mark current frame as LANDING  (then click the point)
  ENTER    → compute distance from marked points
  R        → reset all marks
  Q        → back to live / quit
"""

import cv2
import numpy as np
import threading
import time
import os

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
WEBCAM_INDEX   = 0
DROIDCAM_URL   = 1   # ← change to your phone IP
CAPTURE_W      = 640
CAPTURE_H      = 480
TARGET_FPS     = 30
BASELINE_M     = 1.5          # metres between the two cameras  ← measure this
MAX_REC_SEC    = 3           # max recording buffer in seconds

# Camera intrinsics — paste your calibration results here
# (or leave defaults for a rough estimate)
CAM_A = dict(fx=700.0, fy=700.0, cx=320.0, cy=240.0, dist=np.zeros(5))
CAM_B = dict(fx=700.0, fy=700.0, cx=320.0, cy=240.0, dist=np.zeros(5))
# ──────────────────────────────────────────────────────────────────────────────


# ─── NON-BLOCKING FRAME GRABBER ───────────────────────────────────────────────
class FrameGrabber:
    def __init__(self, source, name):
        self.name   = name
        self.source = source
        self._frame = None
        self._ts    = 0.0
        self._lock  = threading.Lock()
        self._stop  = threading.Event()
        self._ready = threading.Event()
        self._t     = threading.Thread(target=self._run, daemon=True, name=name)

    def start(self):
        self._t.start()
        if not self._ready.wait(10):
            raise RuntimeError(f"{self.name}: camera did not open in 10 s")
        return self

    def stop(self):
        self._stop.set()

    def read(self):
        with self._lock:
            return self._frame, self._ts

    def _open(self):
        if isinstance(self.source, int):
            backend = cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY
            cap = cv2.VideoCapture(self.source, backend)
        else:
            cap = cv2.VideoCapture(self.source)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAPTURE_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_H)
        cap.set(cv2.CAP_PROP_FPS,          TARGET_FPS)
        cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
        return cap

    def _run(self):
        cap = self._open()
        if not cap.isOpened():
            print(f"[{self.name}] ERROR: cannot open {self.source}")
            return
        first = True
        fails = 0
        while not self._stop.is_set():
            ret, frame = cap.read()
            if not ret:
                fails += 1
                if fails > 20:
                    cap.release(); cap = self._open(); fails = 0
                continue
            fails = 0
            with self._lock:
                self._frame = frame
                self._ts    = time.monotonic()
            if first:
                self._ready.set(); first = False
        cap.release()


# ─── CAMERA MODEL ─────────────────────────────────────────────────────────────
class Camera:
    def __init__(self, params):
        self.fx   = params['fx'];  self.fy = params['fy']
        self.cx   = params['cx'];  self.cy = params['cy']
        self.dist = params['dist']
        self.K    = np.array([[self.fx,0,self.cx],[0,self.fy,self.cy],[0,0,1]], np.float64)

    def bearing(self, px, py):
        """Return horizontal bearing angle (radians) for a pixel."""
        pt  = np.array([[[float(px), float(py)]]], np.float64)
        und = cv2.undistortPoints(pt, self.K, self.dist, P=self.K)
        ux  = float(und[0,0,0])
        return np.arctan2(ux - self.cx, self.fx)


cam_a = Camera(CAM_A)
cam_b = Camera(CAM_B)


# ─── TRIANGULATION ────────────────────────────────────────────────────────────
def triangulate_h(alpha, beta, baseline):
    """Horizontal-plane triangulation. Returns (x, depth_y) in metres."""
    ta, tb = np.tan(alpha), np.tan(beta)
    denom  = ta + tb
    if abs(denom) < 1e-9:
        return None
    y = baseline / denom
    x = y * ta
    return x, y


# ─── DRAWING HELPERS ──────────────────────────────────────────────────────────
def put(img, text, pos, color=(255,255,255), scale=0.6, thickness=2):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0),   thickness+2)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)


def draw_scene(frame_a, frame_b,
               takeoff_px, landing_px,
               result_text=None):
    """
    On Camera A frame:
      • Vertical line from landing point to ground
      • Parallelogram from takeoff to landing
    Returns side-by-side composite.
    """
    fa = frame_a.copy()
    fb = frame_b.copy()

    h, w = fa.shape[:2]

    if landing_px:
        lx, ly = landing_px
        # Vertical line from landing point downward to bottom of frame
        cv2.line(fa, (lx, ly), (lx, h), (0, 255, 255), 2)
        # Dot at landing
        cv2.circle(fa, (lx, ly), 7, (0, 255, 255), -1)
        put(fa, "LANDING", (lx + 8, ly - 8), (0, 255, 255))

    if takeoff_px:
        tx, ty = takeoff_px
        cv2.circle(fa, (tx, ty), 7, (0, 165, 255), -1)
        put(fa, "TAKEOFF", (tx + 8, ty - 8), (0, 165, 255))

    if takeoff_px and landing_px:
        tx, ty = takeoff_px
        lx, ly = landing_px

        # Parallelogram vertices:
        #   bottom-left  = (tx, h-1)   takeoff ground
        #   top-left     = (tx, ty)    takeoff body point
        #   top-right    = (lx, ly)    landing body point
        #   bottom-right = (lx, h-1)  landing ground
        pts = np.array([
            [tx, h - 1],
            [tx, ty],
            [lx, ly],
            [lx, h - 1],
        ], np.int32)

        # Filled semi-transparent
        overlay = fa.copy()
        cv2.fillPoly(overlay, [pts], (255, 200, 0))
        cv2.addWeighted(overlay, 0.18, fa, 0.82, 0, fa)

        # Outline
        cv2.polylines(fa, [pts], isClosed=True, color=(255, 200, 0), thickness=2)

        # Horizontal distance arrow along the ground
        cv2.arrowedLine(fa, (tx, h - 5), (lx, h - 5), (255, 255, 255), 2, tipLength=0.03)
        cv2.arrowedLine(fa, (lx, h - 5), (tx, h - 5), (255, 255, 255), 2, tipLength=0.03)
        mid_x = (tx + lx) // 2
        put(fa, "jump distance", (mid_x - 50, h - 10), (255, 255, 255), 0.5)

    if result_text:
        # Large result banner
        (tw, th), _ = cv2.getTextSize(result_text, cv2.FONT_HERSHEY_DUPLEX, 1.1, 2)
        bx, by = w // 2 - tw // 2 - 10, 20
        cv2.rectangle(fa, (bx, by), (bx + tw + 20, by + th + 16), (20, 20, 20), -1)
        cv2.putText(fa, result_text, (bx + 10, by + th + 4),
                    cv2.FONT_HERSHEY_DUPLEX, 1.1, (0, 255, 100), 2)

    composite = np.hstack([fa, fb])
    return composite


# ─── REVIEW MODE ──────────────────────────────────────────────────────────────
class ReviewSession:
    def __init__(self, frames_a, frames_b):
        self.fa      = frames_a
        self.fb      = frames_b
        self.n       = len(frames_a)
        self.idx     = 0

        self.takeoff_frame = None
        self.landing_frame = None
        self.takeoff_px    = None   # (x,y) on cam A
        self.landing_px    = None
        self.takeoff_px_b  = None   # (x,y) on cam B
        self.landing_px_b  = None

        self.pending       = None   # 'takeoff' or 'landing' — waiting for click
        self.result_text   = None

        self._click_a      = None
        self._click_b      = None

    # ── mouse callbacks ────────────────────────────────────────────────────
    def _mouse_a(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self._click_a = (x, y)

    def _mouse_b(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and self.pending:
            self._click_b = (x, y)

    # ── process pending click ──────────────────────────────────────────────
    def _process_click(self):
        if self._click_a is None:
            return

        cx, cy = self._click_a
        self._click_a = None

        if self.pending == 'takeoff':
            self.takeoff_px    = (cx, cy)
            self.takeoff_frame = self.idx
            self.takeoff_px_b  = self._click_b
            self._click_b      = None
            self.result_text   = None
            print(f"  Takeoff set  → cam-A pixel ({cx},{cy})  frame {self.idx}")
            if self._click_b is None:
                print("  (click same point on Camera B then press ENTER to compute)")
        elif self.pending == 'landing':
            self.landing_px    = (cx, cy)
            self.landing_frame = self.idx
            self.landing_px_b  = self._click_b
            self._click_b      = None
            self.result_text   = None
            print(f"  Landing set  → cam-A pixel ({cx},{cy})  frame {self.idx}")

        self.pending = None

    # ── triangulate & compute ──────────────────────────────────────────────
    def _compute(self):
        if not (self.takeoff_px and self.landing_px):
            print("  Mark both TAKEOFF (T) and LANDING (L) first")
            return

        # Pixel-only distance on Camera A (fallback, no depth info)
        dx_px = abs(self.landing_px[0] - self.takeoff_px[0])

        # Try stereo triangulation if both B-clicks are available
        dist_3d = None
        if self.takeoff_px_b and self.landing_px_b:
            alpha_t = cam_a.bearing(*self.takeoff_px)
            beta_t  = cam_b.bearing(*self.takeoff_px_b)
            alpha_l = cam_a.bearing(*self.landing_px)
            beta_l  = cam_b.bearing(*self.landing_px_b)

            pt = triangulate_h(alpha_t, beta_t, BASELINE_M)
            pl = triangulate_h(alpha_l, beta_l, BASELINE_M)

            if pt and pl:
                dist_3d = np.hypot(pl[0]-pt[0], pl[1]-pt[1])

        if dist_3d is not None:
            self.result_text = f"Jump distance: {dist_3d:.3f} m  (stereo)"
        else:
            # Rough pixel estimate using cam_a focal length
            # At unknown depth this is only an angular estimate
            angle_diff = abs(cam_a.bearing(*self.landing_px) - cam_a.bearing(*self.takeoff_px))
            self.result_text = f"Horiz angle span: {np.degrees(angle_diff):.1f} deg  (click B-views for metric)"

        print(f"\n  ══ RESULT ═══════════════════════════════")
        print(f"  {self.result_text}")
        print(f"  Pixel distance (cam A): {dx_px} px")
        print(f"  ════════════════════════════════════════\n")

    # ── main review loop ───────────────────────────────────────────────────
    def run(self):
        WIN = "Review — ← → scrub | T=takeoff | L=landing | ENTER=compute | R=reset | Q=quit"
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN, CAPTURE_W * 2, CAPTURE_H + 60)
        cv2.setMouseCallback(WIN, self._split_mouse)  # single window, split routing

        print("\n── REVIEW MODE ─────────────────────────────────────────────────")
        print("  ← → scrub through frames")
        print("  T   = set TAKEOFF  (then click the point on BOTH camera halves)")
        print("  L   = set LANDING  (then click the point on BOTH camera halves)")
        print("  ENTER = compute jump distance")
        print("  R   = reset  |  Q = back\n")

        while True:
            frame_a = self.fa[self.idx]
            frame_b = self.fb[self.idx]

            # Handle pending click
            self._process_click()

            # Build display
            composite = draw_scene(frame_a, frame_b,
                                   self.takeoff_px if self.takeoff_frame == self.idx else None,
                                   self.landing_px if self.landing_frame == self.idx else None,
                                   self.result_text)

            # Always draw both markers even on other frames (faded)
            if self.takeoff_px and self.takeoff_frame != self.idx:
                put(composite, f"TAKEOFF @ frame {self.takeoff_frame}",
                    (10, CAPTURE_H - 40), (0, 165, 255), 0.5)
            if self.landing_px and self.landing_frame != self.idx:
                put(composite, f"LANDING @ frame {self.landing_frame}",
                    (10, CAPTURE_H - 20), (0, 255, 255), 0.5)

            # Draw both markers on their respective frames simultaneously
            if self.takeoff_px and self.landing_px:
                composite = draw_scene(self.fa[self.takeoff_frame],
                                       self.fb[self.takeoff_frame],
                                       self.takeoff_px, self.landing_px,
                                       self.result_text)

            # Status bar
            status = np.zeros((60, CAPTURE_W * 2, 3), np.uint8)
            mode_str = f"  Frame {self.idx+1}/{self.n}"
            if self.pending:
                mode_str += f"  |  Click {self.pending.upper()} point on BOTH halves"
            put(status, mode_str, (10, 22), (200, 200, 200))
            help_str = "T=takeoff  L=landing  ENTER=compute  R=reset  Q=quit"
            put(status, help_str, (10, 48), (120, 120, 120), 0.45, 1)

            display = np.vstack([composite, status])
            cv2.imshow(WIN, display)

            key = cv2.waitKey(30) & 0xFF

            if key == ord('q'):
                break
            elif key == 81 or key == ord('a'):   # left arrow
                self.idx = max(0, self.idx - 1)
            elif key == 83 or key == ord('d'):   # right arrow
                self.idx = min(self.n - 1, self.idx + 1)
            elif key == ord('t'):
                self.pending = 'takeoff'
                print("  Click TAKEOFF point on Camera A (left half), then Camera B (right half)")
            elif key == ord('l'):
                self.pending = 'landing'
                print("  Click LANDING point on Camera A (left half), then Camera B (right half)")
            elif key == 13:   # ENTER
                self._compute()
            elif key == ord('r'):
                self.takeoff_px = self.landing_px = None
                self.takeoff_px_b = self.landing_px_b = None
                self.takeoff_frame = self.landing_frame = None
                self.result_text = None
                print("  Reset.")

        cv2.destroyWindow(WIN)

    def _split_mouse(self, event, x, y, flags, param):
        """Route clicks to left (cam A) or right (cam B) half."""
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if x < CAPTURE_W:
            self._click_a = (x, y)
        else:
            self._click_b = (x - CAPTURE_W, y)


# ─── LIVE CAPTURE + RECORDING ─────────────────────────────────────────────────
def live_and_record():
    print("\nStarting cameras …")
    grabber_a = FrameGrabber(WEBCAM_INDEX, "WebcamA").start()
    grabber_b = FrameGrabber(DROIDCAM_URL, "DroidCamB").start()
    print("Both cameras ready.\n")

    WIN = "Live — SPACE=record/stop  R=review  Q=quit"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, CAPTURE_W * 2, CAPTURE_H + 50)

    recording  = False
    buf_a, buf_b = [], []
    max_frames   = MAX_REC_SEC * TARGET_FPS
    rec_start    = 0.0

    print("Controls:")
    print("  SPACE → start / stop recording")
    print("  R     → review last recording")
    print("  Q     → quit\n")

    while True:
        fa, _ = grabber_a.read()
        fb, _ = grabber_b.read()
        if fa is None or fb is None:
            time.sleep(0.02)
            continue

        if recording:
            buf_a.append(fa.copy())
            buf_b.append(fb.copy())
            if len(buf_a) >= max_frames:
                recording = False
                print(f"  Max recording length reached ({MAX_REC_SEC}s). Stopped.")

        # Display
        disp_a = fa.copy()
        disp_b = fb.copy()

        if recording:
            elapsed = time.monotonic() - rec_start
            cv2.circle(disp_a, (20, 20), 10, (0, 0, 255), -1)
            put(disp_a, f"REC  {elapsed:.1f}s  ({len(buf_a)} frames)", (38, 28), (0,0,255))

        put(disp_a, "Camera A — Webcam",  (10, CAPTURE_H - 10), (180,180,180), 0.5, 1)
        put(disp_b, "Camera B — DroidCam",(10, CAPTURE_H - 10), (180,180,180), 0.5, 1)

        composite = np.hstack([disp_a, disp_b])

        status = np.zeros((50, CAPTURE_W * 2, 3), np.uint8)
        if recording:
            msg = f"  ● RECORDING  {len(buf_a)} frames — SPACE to stop"
            col = (0, 80, 255)
        elif buf_a:
            msg = f"  {len(buf_a)} frames recorded — SPACE=new rec  R=review  Q=quit"
            col = (0, 200, 100)
        else:
            msg = "  SPACE to start recording the jump"
            col = (180, 180, 180)
        put(status, msg, (10, 30), col)
        cv2.imshow(WIN, np.vstack([composite, status]))

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == ord(' '):
            if not recording:
                buf_a.clear(); buf_b.clear()
                recording  = True
                rec_start  = time.monotonic()
                print(f"  ● Recording started … (max {MAX_REC_SEC}s, SPACE to stop)")
            else:
                recording = False
                print(f"  ■ Recording stopped. {len(buf_a)} frames captured.")

        elif key == ord('r'):
            if not buf_a:
                print("  Nothing recorded yet.")
            else:
                cv2.destroyWindow(WIN)
                ReviewSession(buf_a, buf_b).run()
                cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(WIN, CAPTURE_W * 2, CAPTURE_H + 50)

    grabber_a.stop()
    grabber_b.stop()
    cv2.destroyAllWindows()
    print("Done.")


# ─── ENTRY POINT ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    live_and_record()