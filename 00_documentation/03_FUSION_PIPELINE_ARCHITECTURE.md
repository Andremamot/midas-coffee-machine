# ArUco-MiDaS-YOLO Fusion Pipeline Architecture

This document outlines the core concepts, paradigm shifts, and operational mechanics of the new `07_midas_aruco_fusion` module. This pipeline represents a massive upgrade over the legacy offline-calibration methods described in previous documentation.

---

## 1. Paradigm Shift: Live In-Session Calibration
The legacy workflow (saving hundreds of manual snapshots to `01_calibration` and running offline Python scripts to train a Multivariate Regression model) is **officially obsolete for production**. 

The new Fusion pipeline utilizes **Live In-Session Auto-Calibration**:
- **Zero Dataset Collection**: You no longer need to collect datasets of photos.
- **In-Memory Computation**: By running `run_fusion.py` with a `--calibrate` flag, the system automatically sucks in data arrays (ArUco distances, YOLO bounding boxes, MiDaS depths) frame-by-frame directly into RAM.
- **Instant Deployment**: Once enough samples are gathered (which takes about 5 to 30 seconds), the system instantly computes the algebraic or polynomial constants via `numpy.polyfit`, saves them to a lightweight `calibration.json` file, and transitions directly into live height-estimation mode without needing a restart.

### What are "Cup Profiles"?
Passing the `--cup-profile` flag (e.g., `--cup-profile americano`) **does not** mean you are training an AI model on a dataset of Americano cups. It simply changes the name of the output JSON file (e.g., `calibration_fisheye_americano.json`). This allows the factory machine to instantly load the correct pre-calculated geometric constants when the user selects a specific menu item, ensuring the perspective geometry matches the physical cup being used.

---

## 2. Fisheye Lens & Moildev Integration
Pinhole Camera geometry relies heavily on straight lines. Raw fisheye lenses introduce extreme barrel distortion that mathematically destroys these calculations.

If the `--fisheye` flag is provided:
1. **Undistortion**: `MoilUndistorter` perfectly flattens the image before it reaches the AI or ArUco detectors.
2. **Dynamic Camera Matrix**: If the user utilizes the "Anypoint" feature (panning, pitching, or digitally zooming via mouse scroll), the effective focal length changes. The Fusion pipeline dynamically recalculates and updates the `aruco.camera_matrix` on every single frame. This ensures absolute distance ($Z_{tray}$) remains perfectly accurate regardless of digital zoom.

---

## 3. Moving Beyond MiDaS (The 7 Calibration Modes)
While legacy systems relied entirely on the ambiguous relative depth ratio of MiDaS (`m_rim / m_tray`), the new pipeline introduces **7 distinct modes**, allowing us to bypass MiDaS entirely for height calculation if desired.

### MiDaS-Dependent Modes (Modes 1, 2, 3, 4, 6)
These modes still calculate cup height by feeding the YOLO bounding box and ArUco ROI into MiDaS to extract relative depth metrics. They are highly resilient to YOLO bounding box "jitter", but suffer from MiDaS's sensitivity to scene lighting and contrast flicker.

### Pure Geometry Modes (Modes 5 & 7)
These modes **do not use MiDaS depth predictions to calculate height**. Instead, they rely purely on Pinhole Camera Physics:
1. The absolute distance to the tray ($Z_{tray}$) is provided by ArUco.
2. The pixel height of the cup is provided by YOLO (`bbox_h`).
3. Using the known camera focal length, the physical height is extracted geometrically.

---

## 4. Solving "Scale Ambiguity"
A common concern is: *If a cup is much wider (larger diameter) but has the same physical height, won't the system think it's taller because the bounding box is bigger?*

The system prevents this scale ambiguity through two mechanisms:
1. **Strictly Vertical Pixels**: The math in `height_math.py` explicitly uses `bbox_h = y2 - y1`. It entirely ignores the width (`x2 - x1`). Therefore, a cup expanding horizontally does not artificially inflate the height calculation.
2. **3D Perspective Correction**: Because cups are 3D cylinders (or frustums), taller cups have rims that physically bulge closer to the camera lens, causing non-linear pixel growth. Mode 7 handles this perspective distortion by solving a set of algebraic constants (`A` and `B`) derived from two cups of different heights. 

---

## 5. Production Recommendation: Mode 7 (Analytic)
Out of all 7 available modes, **Mode 7 (Universal Analytic Geometry)** is the gold standard and the most "production-ready".

**Why Mode 7 is the best:**
1. **Stationary Calibration**: Unlike Z-Grid modes (where a technician must manually slide the robot nozzle up and down to build a curve), Mode 7 requires zero movement. You simply place Cup 1, swap it for Cup 2, and the system is fully calibrated.
2. **MiDaS-Independent**: It relies entirely on ArUco distance and YOLO bounding box height. It is immune to MiDaS contrast flicker and gray-scale hallucination errors.
3. **Perfect Perspective Math**: It uses the formula `cup_height = (bbox_h * z_tray - A) / (bbox_h + B)`. By taking two known cup heights, it perfectly solves the algebraic constants `A` and `B`, cancelling out the 3D perspective distortion (the "bulge" effect) down to the millimeter.

> **When NOT to use Mode 7:** If your YOLO detection model is unstable (the bounding box boundaries jitter rapidly up and down), Mode 7's output will jitter as well. In this specific scenario, fallback to **Mode 6 (Bilateral Z-Grid)**, which leverages MiDaS to smooth out bounding box inconsistencies.
