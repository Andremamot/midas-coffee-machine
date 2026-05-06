# Fusion Pipeline Calibration & Usage Guide

This guide covers the operational procedures for calibrating and running the new `07_midas_aruco_fusion` module on the production machine. 

## 1. Hardware Prerequisites
Unlike legacy methods that required offline datasets, you only need:
1. **The physical camera & robot rig.**
2. **Exactly 2 calibration cups** with a distinct difference in physical height (e.g., Cup A = 7.6 cm, Cup B = 11.4 cm).

> **Important:** Always input the *true physical height* of the cups using the `--true-height` and `--true-height-2` flags.

---

## 2. Mode 6: Bilateral Z-Grid (Universal MiDaS Mode)

**Best Used For:** Open-world scenarios where the machine will encounter dozens of unlisted cup shapes and varied diameters. Mode 6 utilizes MiDaS to generalize depth universally.

**Command:**
```bash
python run_fusion.py --calibrate 6 --true-height 7.6 --true-height-2 11.4 --n-positions 3
```

### The Calibration Flow
*Rule of Thumb: Never move the robot nozzle while swapping Cup 1 and Cup 2. The nozzle must only move when transitioning to a new Z-Position.*

* **Position 1 (Low Altitude):**
  1. Place **Cup 1 (7.6cm)** on the tray. Wait for UI to finish sampling (approx. 3-4 seconds).
  2. The UI will prompt a Swap. Remove Cup 1 and place **Cup 2 (11.4cm)**. Wait for sampling.
* **Position 2 (Medium Altitude):**
  3. The UI will prompt to "MOVE NOZZLE". Slide the camera up.
  4. Press `SPACE` to confirm the new height.
  5. Place **Cup 1**, wait for sampling. Swap to **Cup 2**, wait for sampling.
* **Position 3 (High Altitude):**
  6. Slide the camera to its highest position. Press `SPACE`.
  7. Place **Cup 1**, wait for sampling. Swap to **Cup 2**, wait for sampling.

**Why this specific flow?** 
To solve the linear equations mathematically (`H1/Z = m*R1+c` and `H2/Z = m*R2+c`), the absolute distance to the tray ($Z_{tray}$) must remain completely identical for both cups. By swapping the cups *before* moving the nozzle, we guarantee that the $Z_{tray}$ variable is isolated.

---

## 3. Mode 7: Analytic Geometry (Fast & Perfect)

**Best Used For:** Fixed-menu machines (e.g., standard vending machines) where the cup diameter is known and constant. It bypasses MiDaS entirely for millimeter-perfect geometric calculation.

**Command:**
```bash
python run_fusion.py --calibrate 7 --true-height 7.6 --true-height-2 11.4
```

### The Calibration Flow
*Rule of Thumb: The robot nozzle remains completely stationary. No Z-Grid sliding required.*

1. Leave the robot nozzle at its default/working altitude.
2. Place **Cup 1 (7.6cm)** on the tray. Wait 3 seconds for sampling.
3. The UI will prompt a Swap. Place **Cup 2 (11.4cm)**. Wait 3 seconds.
4. Done. The system instantly computes the perspective curve (Constants A and B) and switches to Live Mode.

---

## 4. Running LIVE Mode (Production)

Once calibration is completed via any of the modes above, a `calibration.json` file is generated in the directory. You no longer need to pass the `--calibrate` flags to run the machine.

**Command to run the machine in standard production:**
```bash
python run_fusion.py --headless
```

If you calibrated a specific cup profile using the `--cup-profile` flag during calibration, you must explicitly call that profile during live mode:
```bash
python run_fusion.py --cup-profile americano --headless
```

### Keyboard Shortcuts (UI Mode)
If `--headless` is not used, the debug interface will show up:
* `R` : Start/Stop MP4 Video Recording (saved to `/results/video`)
* `S` : Save Snapshot Image (saved to `/results/live_cam`)
* `Q` or `ESC` : Quit pipeline and automatically generate a PDF Report.
