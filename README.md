# CompVision

Computer vision prototypes for stereo jump-distance measurement, camera calibration, and human tracking using OpenCV, YOLOv8, and MediaPipe.

## What This Project Does

CompVision explores two main tracks:
- Stereo geometry workflows for measuring jump distance from two cameras
- Person-tracking experiments (with optional hand-gesture control)

The repository currently contains multiple prototype scripts. This README highlights the **recommended entrypoints** for new contributors.

## Recommended Entrypoints (Renamed for Clarity)

Use these as the public-facing script names in docs/issues/PR discussions.

| Friendly name (recommended) | Current file in repo | Purpose |
|---|---|---|
| `calibrate_single_camera.py` | `chessboard_Calibration_of_individual_cams.py` | Intrinsic calibration for one camera using checkerboard images from live feed |
| `calibrate_stereo_pair.py` | `chess_StereoCAM.py` | Stereo calibration using two cameras + previously saved intrinsics |
| `measure_jump_stereo.py` | `MAIN_Primitive code that records and then we choose the point.py` | Main workflow: dual feed, record, review, click takeoff/landing, triangulate distance |
| `track_distance_yolo.py` | `human detection model/DistanceMeasure_using YOLO and comp vision.py` | Single-camera distance accumulation based on YOLO person detection |
| `track_distance_yolo_gesture.py` | `human detection model/distanceMeasure_using YOLO and comp vision with hand gesture.py` | YOLO + MediaPipe hand gestures (pause/reset/record controls) |

Note: these are **documentation aliases** for now. The physical file names remain unchanged in the repository.

## Project Layout

```text
.
|-- human detection model/
|-- pretest/
|-- runs/
|-- chessboard_Calibration_of_individual_cams.py
|-- chess_StereoCAM.py
|-- MAIN_Primitive code that records and then we choose the point.py
|-- Pixel_Acquisition.py
|-- Pixel_Acquisition1.py
|-- new_prog_claude.py
|-- matching different camera color grading code.py
|-- PROBLEMS.md
|-- webcam_intrinsics.npz
|-- mobilecam_intrinsics.npz
|-- stereo_params.npz
|-- yolov8n.pt
|-- hand_landmarker.task
```

## Requirements

- Python 3.10+
- Windows/Linux/macOS with OpenCV camera access
- Two camera sources for stereo flows (e.g., webcam + DroidCam)

Install dependencies:

```bash
pip install opencv-python numpy ultralytics mediapipe matplotlib
```

## Quick Start

1. Calibrate camera A and camera B (run once per camera).

```bash
python chessboard_Calibration_of_individual_cams.py
```

2. Run stereo calibration.

```bash
python chess_StereoCAM.py
```

3. Run the main stereo jump measurement flow.

```bash
python "MAIN_Primitive code that records and then we choose the point.py"
```

## Outputs and Artifacts

- Calibration files:
  - `webcam_intrinsics.npz`
  - `mobilecam_intrinsics.npz`
  - `stereo_params.npz`
- Model assets:
  - `yolov8n.pt`
  - `hand_landmarker.task`
- YOLO output folders:
  - `runs/detect/predict*`

## Known Issues

- MediaPipe package/API compatibility can vary by version (`solutions` vs `tasks`).
- See `PROBLEMS.md` for current findings and diagnostic commands.

## Roadmap (Open-Source Ready)

### 1) Repository Hygiene
- [ ] Rename long/space-containing script filenames to clean snake_case names
- [ ] Move experiments into `experiments/` and stable apps into `apps/`
- [ ] Add `requirements.txt` with pinned versions
- [ ] Add sample `.env.example` or config module for camera IDs/URLs

### 2) Reliability and Reproducibility
- [ ] Standardize calibration file schema (`K1`, `dist1`, `K2`, `dist2`, `R`, `T`, `baseline`)
- [ ] Add startup validation for missing model/calibration assets
- [ ] Improve error messages for camera connection failures
- [ ] Add deterministic demo mode with prerecorded video

### 3) Developer Experience
- [ ] Add CLI interface (`python -m compvision ...`) for common tasks
- [ ] Add unit tests for triangulation math and geometry helpers
- [ ] Add lint/format checks (ruff + black)
- [ ] Add GitHub Actions for tests/lint

### 4) Documentation and Demos
- [ ] Add architecture diagram (capture -> calibration -> triangulation -> distance)
- [ ] Add GIF/video walkthrough of main stereo workflow
- [ ] Add benchmark notes (latency/FPS on reference hardware)
- [ ] Add troubleshooting matrix (camera backend, MediaPipe version, model path)

### 5) Productization
- [ ] Build modular package layout (`compvision/`)
- [ ] Introduce real-time synchronized point selection UX
- [ ] Add optional auto keypoint selection (pose/foot landmarks)
- [ ] Publish first tagged release (`v0.1.0`)

## Contributing (Suggested)

Until the repo is fully reorganized:
- Open issues with script name + reproduction steps
- Include camera setup (indices/URL), model files used, and OS details
- Keep PRs scoped to one subsystem (calibration, stereo measure, tracking)

## License

Add a license file before public release (MIT recommended for broad adoption).
# Computer-Vision-assisted-distance-measurement-sports-device-for-long-jump
Computer Vision assisted distance measurement sports device for long jump.
