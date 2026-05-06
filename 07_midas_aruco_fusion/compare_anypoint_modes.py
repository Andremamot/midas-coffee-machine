"""
compare_anypoint_modes.py — Perbandingan interaktif Moildev Mode 1 vs Mode 2
=============================================================================
Menampilkan dua jendela secara bersamaan:
  - KIRI : Mode 1 (alpha / beta / zoom)  → sistem koordinat spherical/polar
  - KANAN: Mode 2 (pitch / yaw / roll / zoom) → sistem koordinat Euler

Kontrol via OpenCV Trackbar (slider interaktif):

  Mode 1 Controls:
    alpha  : 0–110° (zenith distance, seberapa jauh dari pusat)
    beta   : 0–360° (arah azimuth, rotasi melingkar)
    zoom   : 1–20×

  Mode 2 Controls:
    pitch  : -110–+110° (tilt up/down)
    yaw    : -110–+110° (pan left/right)
    roll   : -110–+110° (rotation around optical axis)
    zoom   : 1–20×

Keyboard:
  [S] / [s]  → Simpan screenshot kedua mode (tersimpan di ./results/compare_modes/)
  [Q] / [ESC]→ Keluar

Usage:
  python compare_anypoint_modes.py --camera 0
  python compare_anypoint_modes.py --camera 0 --cap-width 2592 --cap-height 1944 --camera-name syue_7730v1_6
"""

import os
import sys
import argparse
import time
from datetime import datetime

import cv2
import numpy as np

# ── Path setup ─────────────────────────────────────────────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_MOIL_DIR = os.path.join(_THIS_DIR, "moildev")
if _MOIL_DIR not in sys.path:
    sys.path.insert(0, _MOIL_DIR)

try:
    from Moildev import Moildev as MoildevLib
except ImportError as e:
    print(f"[ERROR] Gagal import Moildev dari {_MOIL_DIR}: {e}")
    sys.exit(1)

# ── Output directory ────────────────────────────────────────────────────────
SAVE_DIR = os.path.join(_THIS_DIR, "results", "compare_modes")
os.makedirs(SAVE_DIR, exist_ok=True)

# ── Window names ────────────────────────────────────────────────────────────
WIN_M1   = "MODE 1 — Alpha / Beta / Zoom  (Spherical)"
WIN_M2   = "MODE 2 — Pitch / Yaw / Roll / Zoom  (Euler)"
WIN_CTRL = "[ CONTROLS ]  S=Screenshot  Q=Quit"

# ── Trackbar helper ─────────────────────────────────────────────────────────
def nothing(x):
    pass

def make_trackbar(win, name, default, lo, hi):
    """Buat trackbar dengan offset agar bisa handle nilai negatif."""
    cv2.createTrackbar(name, win, default - lo, hi - lo, nothing)

def get_trackbar(win, name, lo):
    return cv2.getTrackbarPos(name, win) + lo


def add_overlay(frame, mode_label, params: dict, color=(0, 220, 255)):
    """Gambar panel info di pojok kiri atas frame."""
    h, w = frame.shape[:2]
    S = max(1.0, w / 640)  # scale factor supaya teks proporsional

    # Panel background
    panel_h = int((len(params) + 2) * 28 * S)
    panel_w = int(310 * S)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), (10, 10, 30), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.rectangle(frame, (0, 0), (panel_w, panel_h), color, 2)

    # Mode label
    cv2.putText(frame, mode_label, (int(10 * S), int(25 * S)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65 * S, color, 2, cv2.LINE_AA)

    # Parameter values
    for i, (k, v) in enumerate(params.items()):
        cv2.putText(frame, f"  {k}: {v}", (int(10 * S), int((25 + (i + 1) * 28) * S)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52 * S, (210, 230, 255), 2, cv2.LINE_AA)

    # Keyboard hint
    cv2.putText(frame, "S=Screenshot  Q=Quit", (int(10 * S), int((25 + (len(params) + 1) * 28) * S)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45 * S, (120, 200, 120), 1, cv2.LINE_AA)
    return frame


def run(camera_idx, cap_width, cap_height, camera_name, json_path):
    # ── Kamera ──────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        print(f"[ERROR] Kamera {camera_idx} tidak bisa dibuka.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  cap_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_height)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[CAM] Resolusi: {actual_w}x{actual_h}")

    # Warmup
    print("[CAM] Warmup...", end="", flush=True)
    for _ in range(30):
        ret, frame = cap.read()
        if ret and frame is not None:
            if cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean() > 10:
                break
        time.sleep(0.05)
    print(" siap.")

    # ── Moildev init ────────────────────────────────────────────────────────
    print(f"[MOIL] Loading '{camera_name}' dari {json_path} ...")
    try:
        moil1 = MoildevLib(json_path, camera_name)
        moil2 = MoildevLib(json_path, camera_name)
    except Exception as e:
        print(f"[ERROR] Gagal inisiasi Moildev: {e}")
        cap.release()
        sys.exit(1)
    print("[MOIL] ✅ Siap.")

    # ── Windows ─────────────────────────────────────────────────────────────
    cv2.namedWindow(WIN_M1, cv2.WINDOW_NORMAL)
    cv2.namedWindow(WIN_M2, cv2.WINDOW_NORMAL)
    cv2.namedWindow(WIN_CTRL, cv2.WINDOW_NORMAL)

    # Posisikan jendela berdampingan
    cv2.moveWindow(WIN_M1,   0,   0)
    cv2.moveWindow(WIN_M2,   700, 0)
    cv2.moveWindow(WIN_CTRL, 0,   600)

    # ── Trackbars ───────────────────────────────────────────────────────────
    # Mode 1 — alpha: 0..110,  beta: 0..360,  zoom: 1..20
    cv2.createTrackbar("Alpha  (0..110)",   WIN_CTRL, 0,   110, nothing)
    cv2.createTrackbar("Beta   (0..360)",   WIN_CTRL, 0,   360, nothing)
    cv2.createTrackbar("Zoom M1 (1..20)x",  WIN_CTRL, 1,   20,  nothing)

    # Mode 2 — pitch/yaw/roll: -110..110 (offset +110 agar slider mulai 0)
    #           zoom: 1..20
    cv2.createTrackbar("Pitch (-110..110)", WIN_CTRL, 110, 220, nothing)
    cv2.createTrackbar("Yaw   (-110..110)", WIN_CTRL, 110, 220, nothing)
    cv2.createTrackbar("Roll  (-110..110)", WIN_CTRL, 110, 220, nothing)
    cv2.createTrackbar("Zoom M2 (1..20)x",  WIN_CTRL, 1,   20,  nothing)

    # Dummy image untuk jendela kontrol agar tidak kosong
    ctrl_img = np.zeros((120, 600, 3), dtype=np.uint8)
    cv2.putText(ctrl_img, "Geser slider di bawah ini", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2)
    cv2.putText(ctrl_img, "Mode 1: Alpha/Beta/Zoom  |  Mode 2: Pitch/Yaw/Roll/Zoom",
                (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(ctrl_img, "[S] Screenshot   [Q/ESC] Keluar",
                (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 200, 120), 1)
    cv2.imshow(WIN_CTRL, ctrl_img)

    print("\n[INFO] Tekan [S] untuk screenshot, [Q] atau ESC untuk keluar.\n")

    prev_m1_params = None
    prev_m2_params = None
    map_x1 = map_y1 = map_x2 = map_y2 = None

    while True:
        ret, raw = cap.read()
        if not ret or raw is None:
            continue

        # ── Baca slider ──────────────────────────────────────────────────
        alpha  = cv2.getTrackbarPos("Alpha  (0..110)",   WIN_CTRL)
        beta   = cv2.getTrackbarPos("Beta   (0..360)",   WIN_CTRL)
        zoom_m1 = max(1, cv2.getTrackbarPos("Zoom M1 (1..20)x", WIN_CTRL))

        pitch  = cv2.getTrackbarPos("Pitch (-110..110)", WIN_CTRL) - 110
        yaw    = cv2.getTrackbarPos("Yaw   (-110..110)", WIN_CTRL) - 110
        roll   = cv2.getTrackbarPos("Roll  (-110..110)", WIN_CTRL) - 110
        zoom_m2 = max(1, cv2.getTrackbarPos("Zoom M2 (1..20)x", WIN_CTRL))

        m1_params = (alpha, beta, zoom_m1)
        m2_params = (pitch, yaw, roll, zoom_m2)

        # ── Regenerate maps hanya jika parameter berubah (efisiensi) ─────
        if m1_params != prev_m1_params:
            map_x1, map_y1 = moil1.maps_anypoint_mode1(alpha, beta, zoom_m1)
            prev_m1_params = m1_params

        if m2_params != prev_m2_params:
            map_x2, map_y2 = moil2.maps_anypoint_mode2(pitch, yaw, roll, zoom_m2)
            prev_m2_params = m2_params

        # ── Remap ────────────────────────────────────────────────────────
        frame_m1 = cv2.remap(raw, map_x1, map_y1, cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        frame_m2 = cv2.remap(raw, map_x2, map_y2, cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        # ── Overlay info ─────────────────────────────────────────────────
        add_overlay(frame_m1, "MODE 1 — Spherical (alpha/beta)", {
            "alpha": f"{alpha}°  (zenith distance)",
            "beta ": f"{beta}°  (azimuth 0-360°)",
            "zoom ": f"{zoom_m1}×",
        }, color=(0, 200, 255))

        add_overlay(frame_m2, "MODE 2 — Euler (pitch/yaw/roll)", {
            "pitch": f"{pitch:+}°  (tilt up/down)",
            "yaw  ": f"{yaw:+}°  (pan left/right)",
            "roll ": f"{roll:+}°  (rotate axis)",
            "zoom ": f"{zoom_m2}×",
        }, color=(0, 255, 160))

        # ── Tampilkan ─────────────────────────────────────────────────────
        cv2.imshow(WIN_M1, frame_m1)
        cv2.imshow(WIN_M2, frame_m2)

        # ── Key handler ──────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), ord('Q'), 27):
            print("[INFO] Keluar.")
            break

        elif key in (ord('s'), ord('S')):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path_m1 = os.path.join(SAVE_DIR, f"{ts}_mode1_a{alpha}_b{beta}_z{zoom_m1}.jpg")
            path_m2 = os.path.join(SAVE_DIR, f"{ts}_mode2_p{pitch}_y{yaw}_r{roll}_z{zoom_m2}.jpg")
            cv2.imwrite(path_m1, frame_m1)
            cv2.imwrite(path_m2, frame_m2)
            print(f"[SAVE] Mode 1 → {path_m1}")
            print(f"[SAVE] Mode 2 → {path_m2}")

            # Juga simpan side-by-side comparison
            # Resize ke tinggi yang sama dulu jika perlu
            h1, w1 = frame_m1.shape[:2]
            h2, w2 = frame_m2.shape[:2]
            if h1 != h2:
                frame_m2_rs = cv2.resize(frame_m2, (w2, h1))
            else:
                frame_m2_rs = frame_m2
            side_by_side = np.hstack([frame_m1, frame_m2_rs])
            path_compare = os.path.join(SAVE_DIR, f"{ts}_compare.jpg")
            cv2.imwrite(path_compare, side_by_side)
            print(f"[SAVE] Side-by-side → {path_compare}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Perbandingan Moildev Mode 1 vs Mode 2")
    ap.add_argument("--camera",       type=int,   default=0,
                    help="Index kamera (default: 0)")
    ap.add_argument("--cap-width",    type=int,   default=2592,
                    help="Lebar resolusi (default: 2592)")
    ap.add_argument("--cap-height",   type=int,   default=1944,
                    help="Tinggi resolusi (default: 1944)")
    ap.add_argument("--camera-name",  type=str,   default="syue_7730v1_6",
                    help="Nama profil kamera di camera_parameters.json (default: syue_7730v1_6)")
    ap.add_argument("--json",         type=str,
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "camera_parameters.json"),
                    help="Path ke camera_parameters.json")
    args = ap.parse_args()

    run(
        camera_idx  = args.camera,
        cap_width   = args.cap_width,
        cap_height  = args.cap_height,
        camera_name = args.camera_name,
        json_path   = args.json,
    )
