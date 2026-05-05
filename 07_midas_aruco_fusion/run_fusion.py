"""
run_fusion.py — ArUco + MiDaS + YOLO Cup Height Estimator
=============================================================
Sistem estimasi tinggi gelas dengan ArUco sebagai jangkar jarak absolut
dan MiDaS + YOLO untuk pemetaan kedalaman relatif bibir gelas.

Fitur:
  - Auto-calibration 1-Point: --calibrate 1 --true-height 7.6
  - Auto-calibration 2-Point: --calibrate 2 --true-height 7.6 --true-height-2 10.2
  - Mode live otomatis membaca calibration.json (format bebas 1-pt / 2-pt)
  - Recording (R), Screenshot (S), Report (Q/ESC)
"""

import os
import sys
import argparse
import time
import json
import shutil
from datetime import datetime

import core.calibration_storage as cs
import core.height_math as hm
import core.session_reporter as sr
from core.moil_undistorter import MoilUndistorter
from core.anypoint_controller import AnypointController
from core.image_preprocess import normalize_lighting

import cv2
import numpy as np
import matplotlib.pyplot as plt

# ── Root project path ───────────────────────────────────────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR  = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# ── Import dari sub-modul proyek (via ROOT_DIR) ─────────────────────────────
try:
    from midas_volumecup.depth    import MidasDepthEstimator
    from midas_volumecup.detector import YoloDetector
    from midas_volumecup.volume_math import calculate_z_rim_alpha
except ImportError as e:
    print(f"[ERROR] Gagal import midas_volumecup: {e}")
    sys.exit(1)

try:
    _ARUCO_DIR = os.path.join(ROOT_DIR, "06_aruco_marker")
    if _ARUCO_DIR not in sys.path:
        sys.path.insert(0, _ARUCO_DIR)
    from aruco_detector import ArucoDetector
except ImportError as e:
    print(f"[ERROR] Gagal import ArucoDetector: {e}")
    sys.exit(1)


# ── Persiapan Direktori Hasil ─────────────────────────────────────────────
RESULT_DIR     = os.path.join(_THIS_DIR, "results")
REPORT_DIR     = os.path.join(RESULT_DIR, "report")
# For module
sr.REPORT_DIR = REPORT_DIR
VIDEO_DIR      = os.path.join(RESULT_DIR, "video")
SCREENSHOT_DIR = os.path.join(RESULT_DIR, "live_cam")
CALIB_PATH     = os.path.join(_THIS_DIR, "calibration.json")

for d in [REPORT_DIR, VIDEO_DIR, SCREENSHOT_DIR]:
    os.makedirs(d, exist_ok=True)


# ╔═════════════════════════════════════════════════════════════════════════╗
# ║  PIPELINE UTAMA                                                         ║
# ╚═════════════════════════════════════════════════════════════════════════╝

def run_pipeline(camera_idx: int, headless: bool, calib_data: dict,
                 marker_size: float, calibrate_mode: int, true_height: float, true_height_2: float,
                 n_positions: int = 3, cap_width: int = 1280, cap_height: int = 720,
                 no_anypoint: bool = False):
    print("=" * 55)
    print("  🚀  ArUco + MiDaS + YOLO  |  Cup Height Estimator")
    print("=" * 55)

    print("[INIT] Loading ArucoDetector...")
    aruco = ArucoDetector(marker_size_cm=marker_size)
    print("[INIT] Loading YoloDetector...")
    yolo_weights = os.path.join(ROOT_DIR, "weights", "cup_detection_v3_12_s_best.pt")
    yolo  = YoloDetector(weights_path=yolo_weights)
    print("[INIT] Loading MidasDepthEstimator (this may take a while)...")
    midas_weights = os.path.join(ROOT_DIR, "weights", "midas_v21_small_256.pt")
    midas = MidasDepthEstimator(weights_path=midas_weights)
    print("[INIT] ✅ All detectors ready.\n")

    cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        print(f"[ERROR] Tidak bisa membuka kamera index {camera_idx}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  cap_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_height)
    if args.manual_exposure > 0:
        print(f"[CAM] Menggunakan Manual Exposure: {args.manual_exposure}")
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1) # 1 = Manual Mode di V4L2
        cap.set(cv2.CAP_PROP_EXPOSURE, args.manual_exposure)
    else:
        # Aktifkan auto-exposure agar sensor bisa menyesuaikan pencahayaan
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)   # 3 = aperture priority (auto)

    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)       # matikan autofocus (fisheye fixed-focus)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[CAM] Resolusi: {actual_w}x{actual_h} (diminta {cap_width}x{cap_height})")

    # ── Warmup: tunggu sensor auto-expose (gambar hitam = belum siap) ─────────
    time.sleep(1.5)
    print("[CAM] Warmup kamera...", end="", flush=True)
    tmp_frame = None
    for attempt in range(90):
        ret, tmp_frame = cap.read()
        if ret and tmp_frame is not None:
            brightness = cv2.cvtColor(tmp_frame, cv2.COLOR_BGR2GRAY).mean()
            if brightness > 15:  # frame cukup terang → sensor siap
                print(f" siap (brightness={brightness:.0f}, {attempt+1} frame)")
                break
        time.sleep(0.1)
    else:
        print(f" timeout (brightness masih gelap)")
        # Tetap lanjutkan, mungkin lingkungan memang gelap
    if tmp_frame is None:
        print("[ERROR] Kamera tidak menghasilkan frame.")
        cap.release()
        return

    # Baca exposure aktual dari kamera setelah warmup
    actual_exposure_raw = int(cap.get(cv2.CAP_PROP_EXPOSURE))
    if args.manual_exposure <= 0:
        args.manual_exposure = actual_exposure_raw
        print(f"[CAM] Exposure aktual terdeteksi: {actual_exposure_raw} (raw) = {actual_exposure_raw/1000:.1f} detik")


    # ── Inisiasi Moildev undistorter (hanya jika --fisheye aktif) ─────────────
    moil_undistorter = None
    anypoint_ctrl    = None
    if getattr(args, "fisheye", False):
        try:
            json_path    = os.path.join(_THIS_DIR, "camera_parameters.json")
            camera_name  = getattr(args, "moil_camera_name", "syue_7730v1_6")
            moil_pitch   = float(getattr(args, "moil_pitch",  0.0))
            moil_yaw     = float(getattr(args, "moil_yaw",    0.0))
            moil_roll    = float(getattr(args, "moil_roll",   0.0))
            moil_zoom    = float(getattr(args, "moil_zoom",   1.4))

            h, w = tmp_frame.shape[:2]
            moil_undistorter = MoilUndistorter(
                json_path    = json_path,
                camera_name  = camera_name,
                pitch        = moil_pitch,
                yaw          = moil_yaw,
                roll         = moil_roll,
                zoom         = moil_zoom,
                mode         = getattr(args, "moil_mode", 2),
                use_opencl   = True,
                frame_width  = w,
                frame_height = h,
            )

            # Override camera matrix ArUco dengan focal length Moildev
            new_K = moil_undistorter.build_aruco_camera_matrix(w, h)
            aruco.camera_matrix = new_K
            print(f"[MOIL] ArUco camera matrix overridden: "
                  f"fx={new_K[0,0]:.1f}, fy={new_K[1,1]:.1f}, "
                  f"cx={new_K[0,2]:.1f}, cy={new_K[1,2]:.1f}")

            # Inisiasi anypoint controller (mouse drag)
            anypoint_ctrl = AnypointController(moil_undistorter)

        except Exception as e:
            print(f"[MOIL ERROR] Gagal inisiasi MoilUndistorter: {e}")
            print("[MOIL] Melanjutkan tanpa fisheye undistortion.")
            moil_undistorter = None
            anypoint_ctrl    = None

    WIN_NAME = "ArUco + MiDaS | Cup Height Estimator"

    # Shared state untuk status LED — dibaca oleh live_pipeline untuk UI
    _led_state = {"detected": False}
    _last_exposure_change_t = [0.0]  # track waktu terakhir exposure diubah

    def get_frame():
        if gui_desired_exposure[0] != args.manual_exposure:
            val = gui_desired_exposure[0]
            if cap and cap.isOpened():
                cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # Force manual mode sebelum set exposure
                cap.set(cv2.CAP_PROP_EXPOSURE, val)
                args.manual_exposure = val
                _last_exposure_change_t[0] = time.time()
                print(f"[CAM] Hardware exposure applied: {val}")

        ret, f = cap.read()
        if not ret:
            return False, None

        # Force resize if camera hardware ignores our requested resolution
        if f.shape[1] != cap_width or f.shape[0] != cap_height:
            f = cv2.resize(f, (cap_width, cap_height), interpolation=cv2.INTER_LINEAR)

        # ── Software lighting normalization ────────────────────────────────
        # Hanya aktif di mode HEADLESS (tanpa GUI).
        # Saat GUI aktif, user mengontrol hardware exposure langsung,
        # normalisasi software TIDAK dijalankan sama sekali.
        if args.manual_exposure > 0 and gui is None:
            f, led_on = normalize_lighting(f)
            _led_state["detected"] = led_on

        if moil_undistorter is not None and not no_anypoint:
            f = moil_undistorter.undistort(f)
            if anypoint_ctrl is not None and not headless:
                anypoint_ctrl.draw_overlay(f)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('r') or key == ord('R'):
                    anypoint_ctrl.reset()
                    print(f"[MOIL] Reset anypoint → pitch={moil_undistorter.pitch}, yaw={moil_undistorter.yaw}, zoom={moil_undistorter.zoom}")
                elif key == ord('s') or key == ord('S'):
                    print(f"[MOIL] Current params: --moil-pitch {moil_undistorter.pitch:.1f} "
                          f"--moil-yaw {moil_undistorter.yaw:.1f} "
                          f"--moil-roll {moil_undistorter.roll:.1f} "
                          f"--moil-zoom {moil_undistorter.zoom:.2f}")

                # SANGAT PENTING: Update camera matrix ArUco secara dinamis setiap frame!
                # Jika user melakukan zoom in/out, focal length ekuivalen berubah.
                # Ini mencegah jarak mendadak salah saat user melakukan scroll.
                aruco.camera_matrix = moil_undistorter.build_aruco_camera_matrix(f.shape[1], f.shape[0])

        return True, f


    import threading
    from core.gui_fusion import FusionGUI
    gui = None
    gui_desired_exposure = [args.manual_exposure]
    
    def set_exposure_cb(val):
        gui_desired_exposure[0] = val
        print(f"[GUI] Requested exposure change to: {val}")

    if not headless:
        import gi
        gi.require_version('Gtk', '3.0')
        from gi.repository import Gtk
        gui = FusionGUI(moil_undistorter=moil_undistorter, headless=False,
                        initial_exposure=args.manual_exposure,
                        exposure_callback=set_exposure_cb)
        print("[GUI] GTK3 Interface Active")
    else:
        gui = FusionGUI(moil_undistorter=moil_undistorter, headless=True,
                        initial_exposure=args.manual_exposure,
                        exposure_callback=set_exposure_cb)
        
    # Delegate to calibration routines in a background thread
    def worker_thread():
        import core.calibration_routines as calib_rt
        
        # ── Setup Mode (Preview sebelum kalibrasi dimulai) ──────────────────────
        if calibrate_mode != 0 and gui and not headless:
            calib_names = {
                1: "1-Point", 2: "2-Point", 3: "Z-Grid", 
                4: "BBox", 5: "Geometric", 6: "Bilateral", 7: "Analytic"
            }
            cname = calib_names.get(calibrate_mode, "Unknown")
            gui.enter_setup_mode(cname)
            print(f"[SETUP] Waiting for user to configure camera and start {cname} calibration...")
            
            while not gui.calibration_ready_event.is_set():
                ret, frame = get_frame()
                if not ret:
                    break
                
                gui.update_image(frame)
                key = gui.get_key()
                if key == 27:  # ESC pressed during setup
                    print("[SETUP] Aborted by user.")
                    return
        
        calib_data = None
        if calibrate_mode in (1, 2):
            calib_data = calib_rt.run_calib_1p_2p(get_frame, cap, aruco, yolo, midas, headless, true_height, true_height_2, calibrate_mode, gui)
        elif calibrate_mode == 3:
            calib_data = calib_rt.run_calib_zgrid(get_frame, cap, aruco, yolo, midas, headless, true_height, n_positions, gui)
        elif calibrate_mode == 4:
            calib_data = calib_rt.run_calib_bbox(get_frame, cap, aruco, yolo, midas, headless, true_height, gui)
        elif calibrate_mode == 5:
            calib_data = calib_rt.run_calib_geom(get_frame, cap, aruco, yolo, midas, headless, true_height, n_positions, gui)
        elif calibrate_mode == 6:
            calib_data = calib_rt.run_calib_bilateral(get_frame, cap, aruco, yolo, midas, headless, true_height, true_height_2, n_positions, gui)
        elif calibrate_mode == 7:
            calib_data = calib_rt.run_calib_analytic(get_frame, cap, aruco, yolo, midas, headless, true_height, true_height_2, gui)
        elif calibrate_mode != 0:
            print("[ERROR] Unknown calibration mode")
            if gui and not headless: gui.queue_key(27)
            return

        if calib_data is None and calibrate_mode != 0:
            print("[CALIB] Error or Aborted. Exiting.")
            if gui and not headless: gui.queue_key(27)
            return

        active_poly_Kgeom = [1.0]
        active_cup_str = "LEGACY (1 Profile)"
        if calib_data and calib_data.get("type") == 5:
            if "profiles" in calib_data:
                if getattr(args, "target_cup", None):
                    target_str = str(args.target_cup)
                    if target_str in calib_data["profiles"]:
                        active_poly_Kgeom = calib_data["profiles"][target_str]["poly_Kgeom"]
                        active_cup_str = target_str
                    else:
                        keys = list(calib_data["profiles"].keys())
                        active_cup_str = keys[0] if keys else "Unknown"
                        active_poly_Kgeom = calib_data["profiles"][active_cup_str].get("poly_Kgeom", [1.0]) if keys else [1.0]
                else:
                    keys = list(calib_data["profiles"].keys())
                    active_cup_str = keys[0] if keys else "Unknown"
                    active_poly_Kgeom = calib_data["profiles"][active_cup_str].get("poly_Kgeom", [1.0]) if keys else [1.0]
            else:
                active_poly_Kgeom = calib_data.get("poly_Kgeom", [1.0])
                
        # Pasang _led_state ke args agar live_pipeline bisa membaca status LED via UI
        args._led_state = _led_state if args.manual_exposure > 0 else None

        # Delegate to live pipeline
        import core.live_pipeline as live_pipe
        
        if gui and not headless:
            import gi
            gi.require_version('Gtk', '3.0')
            from gi.repository import GLib
            # Update Mode label
            GLib.idle_add(gui.lbl_status_calib.set_text, f"Mode: Live ({active_cup_str})")
            
        live_pipe.run_live_pipeline(get_frame, cap, aruco, yolo, midas, headless, calib_data, marker_size, active_poly_Kgeom, active_cup_str, args, SCREENSHOT_DIR, VIDEO_DIR, gui)
        
        if gui and not headless:
            gui.queue_key(27) # Trigger quit when done

    thread = threading.Thread(target=worker_thread, daemon=True)
    thread.start()

    if not headless:
        gui.show_all()
        Gtk.main()
    else:
        # If headless, just wait for thread to finish
        thread.join()

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="ArUco + MiDaS Cup Height Estimator")
    ap.add_argument("--camera",       type=int,   default=0,     help="Index kamera (default: 0)")
    ap.add_argument("--cap-width",    type=int,   default=2592,  help="Lebar resolusi USB stream (default: 2592)")
    ap.add_argument("--cap-height",   type=int,   default=1944,  help="Tinggi resolusi USB stream (default: 1944)")
    ap.add_argument("--headless",     action="store_true",        help="Tanpa UI — mode terminal")
    ap.add_argument("--marker-size",  type=float, default=5.0,   help="Ukuran Fisik ArUco di meja (cm)")
    ap.add_argument("--calibrate",    type=int,   default=0, choices=[0,1,2,3,4,5,6,7],
                    help="0:Live 1:1Pt 2:2Pt 3:ZGrid 4:BBox 5:Geom 6:Bilateral 7:Analytic")
    ap.add_argument("--true-height",  type=float, default=None,  help="Reference cup height in cm.")
    ap.add_argument("--true-height-2",type=float, default=None,  help="Second cup height for 2-point calibration.")
    ap.add_argument("--target-cup",   type=float, default=None,  help="LIVE MODE: specify the target menu cup height to monitor (e.g. 7.6 or 11.4)")
    ap.add_argument("--n-positions",  type=int,   default=3,     help="Number of Z positions for Z-Grid calibration. Default: 3")
    ap.add_argument("--cup-profile",  type=str,   default="default", help="Nama profil gelas (misal: short, tall) untuk membedakan file kalibrasi.")
    ap.add_argument("--fisheye",           action="store_true",
                    help="Enable fisheye undistortion via Moildev (gunakan bersama --moil-camera-name)")
    ap.add_argument("--moil-camera-name",  type=str,   default="syue_7730v1_6",
                    help="Nama profil kamera di camera_parameters.json (default: syue_7730v1_6)")
    ap.add_argument("--moil-pitch",        type=float, default=0.0,
                    help="Anypoint pitch dalam derajat (default: 0.0, kamera menatap lurus)")
    ap.add_argument("--moil-yaw",          type=float, default=0.0,
                    help="Anypoint yaw dalam derajat (default: 0)")
    ap.add_argument("--moil-roll",         type=float, default=0.0,
                    help="Anypoint roll dalam derajat (default: 0)")
    ap.add_argument("--moil-zoom",         type=float, default=1.4,
                    help="Zoom factor anypoint Moildev (default: 1.4)")
    ap.add_argument("--moil-mode",         type=int, default=2,
                    help="Mode anypoint: 1 (Alpha/Beta) atau 2 (Pitch/Yaw/Roll) (default: 2)")
    ap.add_argument("--no-anypoint",       action="store_true",
                    help="Gunakan fisheye mode tapi TANPA remap anypoint (frame raw fisheye)")
    ap.add_argument("--manual-exposure",   type=int,   default=0,
                    help="Setel nilai manual exposure kamera (misal: 156). Default=0 (Auto-brightness)")

    args = ap.parse_args()

    if args.fisheye:
        CALIB_PATH = os.path.join(_THIS_DIR, f"calibration_fisheye_{args.cup_profile}.json")
    elif args.cup_profile != "default":
        CALIB_PATH = os.path.join(_THIS_DIR, f"calibration_{args.cup_profile}.json")

    # Sync CALIB_PATH to storage module
    cs.CALIB_PATH = CALIB_PATH

    calib_data = {}
    if args.calibrate > 0:
        if args.calibrate in (1, 3, 4, 5) and args.true_height is None:
            print(f"[ERROR] Calibration mode {args.calibrate} requires --true-height")
            sys.exit(1)
        if args.calibrate in (2, 6, 7) and (args.true_height is None or args.true_height_2 is None):
            print(f"[ERROR] Calibration mode {args.calibrate} requires --true-height AND --true-height-2")
            print("Example: --calibrate 7 --true-height 7.6 --true-height-2 11.4")
            sys.exit(1)
    else:
        cs.CALIB_PATH = CALIB_PATH
        calib_data = cs.load_calibration()
        if not calib_data:
            print("[ERROR] calibration.json not found! Calibrate first, e.g.:")
            print("  python run_fusion.py --calibrate 5 --true-height 7.6")
            sys.exit(1)

    run_pipeline(
        camera_idx=args.camera,
        headless=args.headless,
        calib_data=calib_data,
        marker_size=args.marker_size,
        calibrate_mode=args.calibrate,
        true_height=args.true_height or 0.0,
        true_height_2=args.true_height_2 or 0.0,
        n_positions=args.n_positions,
        cap_width=args.cap_width,
        cap_height=args.cap_height,
        no_anypoint=args.no_anypoint,
    )
