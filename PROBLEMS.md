**Summary**
- This document records problems encountered while running and editing `DepthCal.py` in this workspace, what I did to reproduce/diagnose them, and recommended fixes.

**Reproduced Errors**
- ImportError when running the original script:

  ImportError: cannot import name 'text' from 'mediapipe.tasks.python.vision' (…site-packages/mediapipe/tasks/python/vision/__init__.py)

- AttributeError / missing API surface when trying to use the Solutions API:

  AttributeError: module 'mediapipe' has no attribute 'solutions'

  (Observed during quick checks: `mediacheck.py` printed `mp.__version__ = 0.10.31` and the package contained `tasks` but `mediapipe.solutions` was not importable.)

**Root Causes**
- Two different MediaPipe API surfaces (Solutions vs Tasks) exist and their package layout changed across releases / builds.
- The script imported `mediapipe.tasks.python.vision.text` (Tasks API) but later code expected `mp.solutions.hands` (Solutions API). Mixing these caused ImportError or AttributeError depending on which subpackages are present in the installed mediapipe wheel.
- The installed mediapipe build on this system exposes `tasks` but not `solutions`, so code using `mp.solutions` fails.

**What I changed / actions taken**
- Removed incorrect/unused `mediapipe.tasks.python` imports from `DepthCal.py` and restored reliance on the Solutions API where appropriate. See [DepthCal.py](DepthCal.py).
- Added a runtime compatibility check to `DepthCal.py` that raises a clear ImportError suggesting a compatible MediaPipe install if `mp.solutions` is missing.
- Created and ran a small diagnostic script `mediacheck.py` to inspect what subpackages the installed MediaPipe exposes. See [mediacheck.py](mediacheck.py).

**Commands to reproduce diagnostics / checks**
- Show mediapipe package info and version:

```powershell
python -m pip show mediapipe
```

- Try importing and printing version (errors if not installed):

```powershell
python -c "import mediapipe as mp; print(getattr(mp,'__version__',None))"
```

- Check whether `solutions` or `tasks` API is present:

```powershell
python -c "import mediapipe as mp; print('has solutions=', hasattr(mp,'solutions')); print('has tasks=', hasattr(mp,'tasks'))"
```

- Run the diagnostic script I added:

```powershell
python mediacheck.py
```

**Recommended fixes**
- Option A — Install a MediaPipe release that provides the Solutions API (quickest to get current `DepthCal.py` working):

```powershell
python -m pip uninstall mediapipe -y
python -m pip install mediapipe==0.10.0
```

- Option B — Convert the script to use the MediaPipe Tasks API (e.g., `HandLandmarker` from `mediapipe.tasks`) so it works with the currently installed mediapipe wheel. This is a code rewrite but avoids changing the environment.

- Always run the diagnosis commands above in the same Python environment you use to run your scripts (use `python -m pip` to ensure the same interpreter).

**Prevention / Best practices**
- Pin dependencies in `requirements.txt` (e.g., `mediapipe==0.10.0`) so CI/dev machines use the same package layout.
- After installing a package, run a minimal import test (one-liner) to verify the API you depend on is available.
- Avoid importing large subpackages you don't need; import precisely (e.g., `from mediapipe import solutions` vs `from mediapipe.tasks...`) and consult the package docs for your installed version.

**Next steps (pick one)**
- I can change this workspace to use a compatible MediaPipe version and run `DepthCal.py` to confirm it works.
- Or I can rewrite `DepthCal.py` to target the Tasks API so it runs with the currently installed mediapipe wheel.

Feel free to pick which option you prefer and I’ll proceed.
