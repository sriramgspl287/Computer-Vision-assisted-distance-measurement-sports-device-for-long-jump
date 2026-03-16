#MAIN_Primitive code that records and then we choose the points on the video to calculate the distance.





"""
Stereo Jump Distance Measurer
â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
Works with any two cameras (webcam + DroidCam, or two USB cams).
Resolution differences are handled automatically.

WORKFLOW:
  1. Live side-by-side feed from both cameras
  2. SPACE â†’ start recording, SPACE again â†’ stop
  3. R â†’ enter Review mode
  4. Scrub with â† â†’, find takeoff frame
  5. Press T â†’ click feet on LEFT half, then RIGHT half
  6. Scrub to landing frame
  7. Press L â†’ click feet on LEFT half, then RIGHT half
  8. ENTER â†’ draws vertical line + parallelogram + real distance

Controls â€” Live:
  SPACE  â†’ start / stop recording
  R      â†’ review recording
  Q      â†’ quit

Controls â€” Review:
  â† â†’    â†’ scrub through frames (hold for fast scroll)
  T      â†’ set takeoff  (click left half, then right half)
  L      â†’ set landing  (click left half, then right half)
  ENTER  â†’ compute & draw
  D      â†’ clear all marks
  Q      â†’ back to live
â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
"""

import cv2
import numpy as np
import threading
import time
import os

# â”€â”€â”€ SETTINGS â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
CAMERA_A       = 0                                    # webcam index
CAMERA_B       = 1                                    # second camera index
                                                      # or DroidCam URL:
                                                      # "http://192.168.x.x:4747/video"

DISPLAY_W      = 640          # width of each pane in the side-by-side window
DISPLAY_H      = 480          # height of each pane
MAX_REC_SEC    = 20           # max recording buffer length
MIRROR_CAM_A   = True         # True = mirror webcam left-right

# Stereo params â€” loaded automatically from stereo_params.npz if it exists.
# If not, set BASELINE_M manually and the system uses pixel-based estimates.
BASELINE_M     = 0.863        # metres between the two cameras



#LOAD STEREO PARAMS 
def load_stereo():
    K_default = np.array([[700., 0., 320.], [0., 700., 240.], [0., 0., 1.]], np.float64)
    dist_default = np.zeros(5, np.float64)
    if not os.path.exists("stereo_params.npz"):
        print("  stereo_params.npz not found - using fallback intrinsics")
        return K_default, dist_default, K_default.copy(), dist_default.copy(), BASELINE_M

    d = np.load("stereo_params.npz")
    keys = set(d.files)

    # Format A: full calibration archive with intrinsics and distortion.
    if {"K1", "dist1", "K2", "dist2"}.issubset(keys):
        baseline = float(d["baseline"]) if "baseline" in keys else BASELINE_M
        print("  stereo_params.npz loaded [OK] (full intrinsics)")
        return (d["K1"].astype(np.float64), d["dist1"].astype(np.float64),
                d["K2"].astype(np.float64), d["dist2"].astype(np.float64),
                baseline)

    # Format B: stereo extrinsics only (R/T/E/F). Keep baseline, fallback K/dist.
    if {"R", "T"}.issubset(keys) or "baseline" in keys:
        if "baseline" in keys:
            baseline = float(d["baseline"])
        elif "T" in keys:
            baseline = float(np.linalg.norm(d["T"]))
        else:
            baseline = BASELINE_M
        print("  stereo_params.npz loaded [OK] (extrinsics only; using fallback intrinsics)")
        return K_default, dist_default, K_default.copy(), dist_default.copy(), baseline

    print("  stereo_params.npz has unexpected keys - using fallback intrinsics")
    return K_default, dist_default, K_default.copy(), dist_default.copy(), BASELINE_M

K1, dist1, K2, dist2, BASELINE = load_stereo()


# TRIANGULATION
def bearing(px, py, K, dist):
    pt  = np.array([[[float(px), float(py)]]], np.float64)
    und = cv2.undistortPoints(pt, K, dist, P=K)
    ux  = float(und[0,0,0])
    return np.arctan2(ux - K[0,2], K[0,0])

def triangulate(ax, ay, bx, by):
    """Returns (x, depth) in metres, or None if rays are parallel."""
    a, b   = bearing(ax, ay, K1, dist1), bearing(bx, by, K2, dist2)
    ta, tb = np.tan(a), np.tan(b)
    denom  = ta + tb
    if abs(denom) < 1e-9:
        return None
    depth = BASELINE / denom
    return depth * ta, depth


# â”€â”€â”€ NON-BLOCKING FRAME GRABBER â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
class FrameGrabber:
    """
    One thread per camera.
    - Continuously drains the camera buffer (CAP_PROP_BUFFERSIZE=1)
    - Resizes to DISPLAY_W x DISPLAY_H inside the thread
    - Main loop reads the latest frame instantly, never waits
    """
    def __init__(self, source, name):
        self.name   = name
        self.source = source
        self._frame = None
        self._lock  = threading.Lock()
        self._stop  = threading.Event()
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name=f"grab-{name}")

    def start(self):
        self._thread.start()
        if not self._ready.wait(14):
            raise RuntimeError(f"[{self.name}] did not open in 14 s")
        print(f"  [{self.name}] ready [OK]")
        return self

    def stop(self):
        self._stop.set()

    def read(self):
        """Non-blocking. Returns latest frame (already resized) or None."""
        with self._lock:
            return self._frame

    def _open(self):
        if isinstance(self.source, int):
            cap = cv2.VideoCapture(
                self.source,
                cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY)
        else:
            cap = cv2.VideoCapture(self.source)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # â† flush stale frames fast
        return cap

    def _run(self):
        cap   = self._open()
        if not cap.isOpened():
            print(f"[{self.name}] ERROR: cannot open {self.source}")
            return
        first = True
        fails = 0
        while not self._stop.is_set():
            ret, frame = cap.read()
            if not ret:
                fails += 1
                if fails > 25:
                    cap.release(); time.sleep(0.8)
                    cap = self._open(); fails = 0
                continue
            fails = 0
            # Resize happens here in the grabber thread â€” never blocks main loop
            frame = cv2.resize(frame, (DISPLAY_W, DISPLAY_H))
            with self._lock:
                self._frame = frame
            if first:
                self._ready.set(); first = False
        cap.release()


# â”€â”€â”€ DRAWING â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def put(img, text, pos, color=(255,255,255), scale=0.62, thick=2):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0),   thick+2)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)

def make_bar(width, text, color=(150,150,150)):
    bar = np.zeros((46, width, 3), np.uint8)
    put(bar, text, (10, 30), color, 0.58, 1)
    return bar

def draw_measurement(fa, fb, t_px, l_px, result_str=None):
    """
    On Camera A frame draws:
      â€¢ Takeoff dot + label
      â€¢ Landing dot + label
      â€¢ Vertical line from landing down to ground
      â€¢ Parallelogram (takeoff â†’ landing â†’ ground)
      â€¢ Horizontal distance arrow at ground level
      â€¢ Result banner if provided
    """
    out = fa.copy()
    h, w = out.shape[:2]
    tx, ty = t_px
    lx, ly = l_px

    # Takeoff
    cv2.circle(out, (tx, ty), 8, (0, 140, 255), -1)
    put(out, "TAKEOFF", (tx+10, ty-8), (0,140,255))

    # Landing + vertical line
    cv2.circle(out, (lx, ly), 8, (0, 230, 255), -1)
    put(out, "LANDING", (lx+10, ly-8), (0,230,255))
    cv2.line(out, (lx, ly), (lx, h-1), (0,230,255), 2)

    # Parallelogram
    pts = np.array([[tx,h-1],[tx,ty],[lx,ly],[lx,h-1]], np.int32)
    overlay = out.copy()
    cv2.fillPoly(overlay, [pts], (255,200,0))
    cv2.addWeighted(overlay, 0.18, out, 0.82, 0, out)
    cv2.polylines(out, [pts], True, (255,200,0), 2)

    # Ground arrow
    gnd = h - 7
    cv2.arrowedLine(out, (tx, gnd), (lx, gnd), (255,255,255), 2, tipLength=0.04)
    cv2.arrowedLine(out, (lx, gnd), (tx, gnd), (255,255,255), 2, tipLength=0.04)
    mid = (tx+lx)//2
    put(out, "jump distance", (mid-52, gnd-8), (255,255,255), 0.45, 1)

    # Result banner
    if result_str:
        (tw, th), _ = cv2.getTextSize(result_str, cv2.FONT_HERSHEY_DUPLEX, 0.95, 2)
        bx = w//2 - tw//2 - 12
        cv2.rectangle(out, (bx,8), (bx+tw+24, 8+th+14), (20,20,20), -1)
        cv2.putText(out, result_str, (bx+12, 8+th+4),
                    cv2.FONT_HERSHEY_DUPLEX, 0.95, (0,255,120), 2)

    return out, fb.copy()


# â”€â”€â”€ REVIEW SESSION â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
class ReviewSession:
    def __init__(self, frames_a, frames_b):
        self.fa  = frames_a
        self.fb  = frames_b
        self.n   = len(frames_a)
        self.idx = 0

        self.t_a = self.t_b = None      # takeoff pixel (cam A, cam B)
        self.l_a = self.l_b = None      # landing pixel  (cam A, cam B)
        self.t_f = self.l_f = None      # frame indices

        self.pending     = None          # 'takeoff' or 'landing'
        self._cl         = None          # pending left click
        self._cr         = None          # pending right click
        self.result_str  = None

    def _mouse(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN or not self.pending:
            return
        if x < DISPLAY_W:                        # left half = cam A
            self._cl = (x, y)
            print(f"    Cam-A ({x},{y}) â€” now click the same point on the RIGHT half")
        else:                                     # right half = cam B
            self._cr = (x - DISPLAY_W, y)
            print(f"    Cam-B ({x - DISPLAY_W},{y})")

    def _flush(self):
        """Commit once both halves have been clicked."""
        if self._cl is None or self._cr is None:
            return
        if self.pending == 'takeoff':
            self.t_a, self.t_b, self.t_f = self._cl, self._cr, self.idx
            print(f"  [OK] Takeoff set  (frame {self.idx})")
        elif self.pending == 'landing':
            self.l_a, self.l_b, self.l_f = self._cl, self._cr, self.idx
            print(f"  [OK] Landing set  (frame {self.idx})")
        self._cl = self._cr = None
        self.pending = None
        self.result_str = None

    def _compute(self):
        if not (self.t_a and self.l_a):
            print("  Set both TAKEOFF (T) and LANDING (L) first"); return

        dx_px  = abs(self.l_a[0] - self.t_a[0])

        dist_3d = None
        if self.t_b and self.l_b:
            pt = triangulate(*self.t_a, *self.t_b)
            pl = triangulate(*self.l_a, *self.l_b)
            if pt and pl:
                dist_3d = np.hypot(pl[0]-pt[0], pl[1]-pt[1])

        if dist_3d is not None:
            self.result_str = f"Jump: {dist_3d:.3f} m"
        else:
            self.result_str = f"Pixel span: {dx_px} px  (need both cam clicks for metres)"

        print(f"\n  â•â• RESULT â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•")
        print(f"  {self.result_str}")
        print(f"  Pixel Î”x: {dx_px} px")
        print(f"  â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•\n")

    def run(self):
        WIN = "Review  |  â†â†’ scrub  T=takeoff  L=landing  ENTER=compute  D=clear  Q=back"
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN, DISPLAY_W*2, DISPLAY_H+80)
        cv2.setMouseCallback(WIN, self._mouse)

        print("\nâ”€â”€ REVIEW â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€")
        print("  â† â†’   scrub frames")
        print("  T      set TAKEOFF  â†’ click LEFT half then RIGHT half")
        print("  L      set LANDING  â†’ click LEFT half then RIGHT half")
        print("  ENTER  compute distance")
        print("  D      clear marks")
        print("  Q      back to live\n")

        while True:
            self._flush()

            # If both marks set, freeze on the drawing (takeoff frame)
            if self.t_a and self.l_a:
                fa = self.fa[self.t_f].copy()
                fb = self.fb[self.t_f].copy()
                fa, fb = draw_measurement(fa, fb, self.t_a, self.l_a, self.result_str)
            else:
                fa = self.fa[self.idx].copy()
                fb = self.fb[self.idx].copy()
                # Show individual markers on their frames
                if self.t_a and self.t_f == self.idx:
                    cv2.circle(fa, self.t_a, 8, (0,140,255), -1)
                    put(fa, "TAKEOFF", (self.t_a[0]+8, self.t_a[1]-8), (0,140,255))
                if self.l_a and self.l_f == self.idx:
                    cv2.circle(fa, self.l_a, 8, (0,230,255), -1)
                    put(fa, "LANDING", (self.l_a[0]+8, self.l_a[1]-8), (0,230,255))

            # Pending click guide
            if self.pending:
                guide = (f"Click {self.pending.upper()} on LEFT (Cam A)"
                         if self._cl is None else
                         f"Now click {self.pending.upper()} on RIGHT (Cam B)")
                put(fa, guide, (10, DISPLAY_H-14), (0,255,180))

            composite = np.hstack([fa, fb])

            marks = ""
            if self.t_a: marks += f"  T@{self.t_f}"
            if self.l_a: marks += f"  L@{self.l_f}"
            ready = self.t_a and self.l_a

            bar = make_bar(
                composite.shape[1],
                f"  Frame {self.idx+1}/{self.n}{marks}"
                f"  |  T=takeoff  L=landing  ENTER=compute  D=clear  Q=back",
                (0,190,100) if ready else (140,140,140))

            cv2.imshow(WIN, np.vstack([composite, bar]))
            key = cv2.waitKey(30) & 0xFF

            if key in (ord('q'), 27):
                break
            elif key in (81, ord('a'), 2):    # â† left arrow
                self.idx = max(0, self.idx-1)
            elif key in (83, ord('d'), 3):    # â†’ right arrow
                self.idx = min(self.n-1, self.idx+1)
            elif key == ord('t'):
                self.pending = 'takeoff'
                self._cl = self._cr = None
                print("  Click TAKEOFF feet on LEFT half (Cam A), then RIGHT half (Cam B)")
            elif key == ord('l'):
                self.pending = 'landing'
                self._cl = self._cr = None
                print("  Click LANDING feet on LEFT half (Cam A), then RIGHT half (Cam B)")
            elif key == 13:
                self._compute()
            elif key == ord('d'):
                self.t_a = self.t_b = self.l_a = self.l_b = None
                self.t_f = self.l_f = self.result_str = self.pending = None
                print("  Marks cleared.")

        cv2.destroyWindow(WIN)


# â”€â”€â”€ LIVE CAPTURE â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def live():
    print(f"\nOpening cameras  A={CAMERA_A}  B={CAMERA_B} â€¦")
    ga = FrameGrabber(CAMERA_A, "CamA").start()
    gb = FrameGrabber(CAMERA_B, "CamB").start()

    WIN = "Live  |  SPACE=record/stop  R=review  Q=quit"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, DISPLAY_W*2, DISPLAY_H+50)

    buf_a, buf_b = [], []
    recording    = False
    rec_start    = 0.0
    max_frames   = MAX_REC_SEC * 30

    print("\n  SPACE  â†’ start / stop recording")
    print("  R      â†’ review last recording")
    print("  Q      â†’ quit\n")

    while True:
        fa = ga.read()
        fb = gb.read()

        if fa is None or fb is None:
            time.sleep(0.02)
            continue

        if MIRROR_CAM_A:
            fa = cv2.flip(fa, 1)

        if recording:
            buf_a.append(fa.copy())
            buf_b.append(fb.copy())
            if len(buf_a) >= max_frames:
                recording = False
                print(f"  Max length hit. {len(buf_a)} frames captured.")

        d1, d2 = fa.copy(), fb.copy()

        if recording:
            elapsed = time.monotonic() - rec_start
            cv2.circle(d1, (22,22), 11, (0,0,210), -1)
            put(d1, f"REC  {elapsed:.1f}s", (40,30), (0,0,255))

        put(d1, "Cam A", (10, DISPLAY_H-10), (110,110,110), 0.5, 1)
        put(d2, "Cam B", (10, DISPLAY_H-10), (110,110,110), 0.5, 1)

        composite = np.hstack([d1, d2])

        if recording:
            bt  = f"  â— RECORDING {len(buf_a)} frames â€” SPACE to stop"
            bc  = (0,50,200)
        elif buf_a:
            bt  = f"  {len(buf_a)} frames â€” SPACE=new recording  R=review  Q=quit"
            bc  = (0,160,70)
        else:
            bt  = "  SPACE to start recording"
            bc  = (140,140,140)

        cv2.imshow(WIN, np.vstack([composite, make_bar(composite.shape[1], bt, bc)]))
        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), 27):
            break
        elif key == 32:
            if not recording:
                buf_a.clear(); buf_b.clear()
                recording = True; rec_start = time.monotonic()
                print("  â— Recording started â€¦ (SPACE to stop)")
            else:
                recording = False
                print(f"  â–  Stopped. {len(buf_a)} frames captured.")
        elif key == ord('r'):
            if not buf_a:
                print("  Nothing recorded yet.")
            else:
                cv2.destroyWindow(WIN)
                ReviewSession(buf_a, buf_b).run()
                cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(WIN, DISPLAY_W*2, DISPLAY_H+50)

    ga.stop(); gb.stop()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    live()

