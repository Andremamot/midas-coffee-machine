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
# KRITIS: Matikan OpenCL sepenuhnya.
# OpenCL context TIDAK thread-safe saat diakses dari multiple pipeline
# (Moildev remap + YOLO + MiDaS + GTK rendering) → menyebabkan
# "terminate called without an active exception" (SIGABRT) dan SIGSEGV.
# CPU-only mode lebih lambat tapi 100% stabil.
cv2.ocl.setUseOpenCL(False)
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


def load_aruco_calibration(args) -> float:
    """
    Mencoba memuat file kalibrasi ArUco (JSON) berdasarkan args.cup_profile
    dan args.fisheye untuk mendapatkan focal_length_px.
    """
    if args.fisheye:
        aruco_calib_name = f"calibration_aruco_fisheye_{args.cup_profile}.json"
    else:
        aruco_calib_name = f"calibration_aruco_{args.cup_profile}.json"
    aruco_calib_path = os.path.join(_THIS_DIR, aruco_calib_name)

    if os.path.exists(aruco_calib_path):
        try:
            with open(aruco_calib_path, "r") as f:
                data = json.load(f)
            fl = data.get("focal_length_px")
            if fl is not None:
                print(f"[ARUCO CALIB] ✅ Memuat focal length hasil kalibrasi: {fl:.1f} px (dari {aruco_calib_name})")
                return float(fl)
        except Exception as e:
            print(f"[ARUCO CALIB] ⚠ Gagal membaca file kalibrasi ArUco: {e}")
    return None



# ╔═════════════════════════════════════════════════════════════════════════╗
# ║  PIPELINE UTAMA                                                         ║
# ╚═════════════════════════════════════════════════════════════════════════╝

def run_pipeline(camera_idx: int, headless: bool, calib_data: dict,
                 marker_size: float, calibrate_mode: int, true_height: float, true_height_2: float,
                 n_positions: int = 3, cap_width: int = 2592, cap_height: int = 1944,
                 no_anypoint: bool = False):
    print("=" * 55)
    print("  🚀  ArUco + MiDaS + YOLO  |  Cup Height Estimator")
    print("=" * 55)

    print("[INIT] Loading ArucoDetector...")
    aruco = ArucoDetector(marker_size_cm=marker_size)
    # Coba muat focal length dari manual parameter atau JSON kalibrasi ArUco
    focal_override = getattr(args, "focal_length", None)
    if focal_override is None:
        focal_override = load_aruco_calibration(args)
    if focal_override is not None:
        aruco.camera_matrix[0, 0] = focal_override
        aruco.camera_matrix[1, 1] = focal_override
        args.focal_length = focal_override
        print(f"[ARUCO] focal length di-override: fx={focal_override:.1f}px")
    print("[INIT] Loading YoloDetector...")
    yolo_weights = os.path.join(ROOT_DIR, "weights", "best.pt")
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

    # Sanity check: jika nilai terlalu rendah (< 100), kamera mungkin mewarisi state buruk
    # dari sesi sebelumnya (mis. test diagnostik yang meninggalkan exposure=1).
    # Reset ke auto sebentar lalu baca ulang.
    if actual_exposure_raw < 100:
        print(f"[CAM] Exposure terdeteksi tidak valid ({actual_exposure_raw}), reset ke auto...")
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)  # auto mode
        time.sleep(2.0)
        for _ in range(10):
            cap.grab()
        actual_exposure_raw = int(cap.get(cv2.CAP_PROP_EXPOSURE))
        print(f"[CAM] Setelah reset auto: exposure = {actual_exposure_raw}")

    if args.manual_exposure <= 0:
        args.manual_exposure = actual_exposure_raw
        print(f"[CAM] Exposure aktual terdeteksi: {actual_exposure_raw} (raw) = {actual_exposure_raw/1000:.1f} detik")


    # ── KRITIS: Paksa kamera ke Manual Mode dengan nilai yang sama ──────────────
    # Saat warmup, kamera dalam mode Auto-Exposure. Jika dibiarkan, gambar di startup
    # akan terlihat berbeda dari setelah user pertama kali mengubah exposure (yang
    # memaksa masuk ke manual mode). Dengan mengunci ke manual mode sekarang,
    # kondisi startup konsisten dengan kondisi setelah penyesuaian exposure.
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)          # 1 = Manual Mode V4L2
    cap.set(cv2.CAP_PROP_EXPOSURE, args.manual_exposure)
    # Flush buffer agar frame dengan exposure baru langsung tampil
    for _ in range(4):
        cap.grab()
    print(f"[CAM] Manual mode dikunci: exposure={args.manual_exposure} raw ({args.manual_exposure/1000:.1f})")


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
            moil_zoom    = float(getattr(args, "moil_zoom",   2.0))

            h, w = tmp_frame.shape[:2]
            moil_undistorter = MoilUndistorter(
                json_path    = json_path,
                camera_name  = camera_name,
                pitch        = moil_pitch,
                yaw          = moil_yaw,
                roll         = moil_roll,
                zoom         = moil_zoom,
                mode         = getattr(args, "moil_mode", 0),
                use_opencl   = False,   # OFF: thread-safety (sama seperti Kakip CPU-only)
                frame_width  = w,
                frame_height = h,
            )

            # Override camera matrix ArUco dengan focal length Moildev
            new_K = moil_undistorter.build_aruco_camera_matrix(w, h)
            if getattr(args, "focal_length", None) is not None:
                new_K[0, 0] = args.focal_length
                new_K[1, 1] = args.focal_length
                print(f"[ARUCO] focal length di-override manual: fx={args.focal_length:.1f}px")
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

    _fps_prev_t = [time.time()]  # FPS tracker untuk overlay

    # GUI shared state — harus didefinisikan SEBELUM _camera_reader
    gui_desired_exposure = [args.manual_exposure]
    gui_desired_gain     = [128]
    gui_desired_bri      = [0]

    # ── Threaded Camera Reader ──────────────────────────────────────────────
    # cap.read() memblokir ~300-500ms per frame di resolusi 2592x1944.
    # Threaded reader membaca terus di background → get_frame() langsung ambil
    # frame terbaru tanpa menunggu kamera.
    import threading as _thr
    _cam_frame = [None]       # frame terbaru dari kamera
    _cam_ret   = [False]
    _cam_lock  = _thr.Lock()
    _cam_alive = [True]

    def _camera_reader():
        """Background thread: baca frame terus-menerus dari kamera."""
        while _cam_alive[0]:
            # Terapkan perubahan hardware sekaligus (atomik)
            if (gui_desired_exposure[0] != args.manual_exposure or
                gui_desired_gain[0] != getattr(args, '_current_gain', 128) or
                gui_desired_bri[0] != getattr(args, '_current_bri', 0)):
                if cap and cap.isOpened():
                    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
                    cap.set(cv2.CAP_PROP_EXPOSURE, gui_desired_exposure[0])
                    cap.set(cv2.CAP_PROP_GAIN, gui_desired_gain[0])
                    cap.set(cv2.CAP_PROP_BRIGHTNESS, gui_desired_bri[0])
                    args.manual_exposure = gui_desired_exposure[0]
                    args._current_gain = gui_desired_gain[0]
                    args._current_bri = gui_desired_bri[0]
                    _last_exposure_change_t[0] = time.time()
                    print(f"[CAM] HW applied: exp={args.manual_exposure} gain={args._current_gain} bri={args._current_bri}")

            ret, frame = cap.read()
            with _cam_lock:
                _cam_ret[0] = ret
                _cam_frame[0] = frame

    # NOTE: Thread distart SETELAH GUI init — lihat bawah

    def get_frame():
        # Ambil frame terbaru dari threaded reader (non-blocking)
        with _cam_lock:
            ret = _cam_ret[0]
            f = _cam_frame[0]
            _cam_frame[0] = None  # tandai sudah diambil

        if not ret or f is None:
            return False, None


        # Force resize if camera hardware ignores our requested resolution
        if f.shape[1] != cap_width or f.shape[0] != cap_height:
            f = cv2.resize(f, (cap_width, cap_height), interpolation=cv2.INTER_LINEAR)

        # ── Software lighting normalization ────────────────────────────────
        normalize_active = True
        if gui is not None:
            normalize_active = gui.is_normalize_enabled()

        if args.manual_exposure > 0 and normalize_active:
            f, led_on = normalize_lighting(f)
            _led_state["detected"] = led_on

        if moil_undistorter is not None and not no_anypoint:
            f = moil_undistorter.undistort(f)
            if anypoint_ctrl is not None and not headless:
                anypoint_ctrl.draw_overlay(f)
                
                # Hanya panggil cv2.waitKey jika TIDAK pakai GUI GTK
                # Memanggil cv2.waitKey di background thread saat GTK aktif akan menyebabkan SIGABRT!
                if gui is None:
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

        # ── FPS overlay (pojok kiri bawah, skala proporsional ke resolusi) ─
        now_t = time.time()
        dt = now_t - _fps_prev_t[0]
        _fps_prev_t[0] = now_t
        fps_val = 1.0 / dt if dt > 0 else 0
        h_f, w_f = f.shape[:2]
        _S = max(1.0, w_f / 1280.0)  # skala teks proporsional ke lebar frame
        cv2.putText(f, f"FPS: {fps_val:.1f}", (int(10*_S), h_f - int(20*_S)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7 * _S, (0, 255, 0), max(2, int(2*_S)))

        return True, f


    import threading
    from core.gui_fusion import FusionGUI
    gui = None

    def set_smart_exposure_cb(exp_val, gain_val, bri_val):
        gui_desired_exposure[0] = exp_val
        gui_desired_gain[0] = gain_val
        gui_desired_bri[0] = bri_val
        print(f"[GUI] Smart Exp applied -> Shutter: {exp_val}, ISO: {gain_val}, EV: {bri_val}")

    if not headless:
        import gi
        gi.require_version('Gtk', '3.0')
        from gi.repository import Gtk
        gui = FusionGUI(moil_undistorter=moil_undistorter, headless=False,
                        initial_exposure=args.manual_exposure,
                        exposure_callback=set_smart_exposure_cb)
        gui_desired_exposure[0] = args.manual_exposure
        # Sinkronkan gain/brightness aktual dari hardware
        actual_gain = int(cap.get(cv2.CAP_PROP_GAIN))
        actual_bri = int(cap.get(cv2.CAP_PROP_BRIGHTNESS))
        gui_desired_gain[0] = actual_gain
        gui_desired_bri[0] = actual_bri
        print("[GUI] GTK3 Interface Active")
    else:
        gui = FusionGUI(moil_undistorter=moil_undistorter, headless=True,
                        initial_exposure=args.manual_exposure,
                        exposure_callback=set_smart_exposure_cb)

    # Start camera reader thread SETELAH gui_desired_* sudah siap
    _cam_thread = _thr.Thread(target=_camera_reader, daemon=True)
    _cam_thread.start()
        
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
        
        # PENTING: Untuk mode kalibrasi, hasil kalibrasi disimpan di _calib_result.
        # Untuk mode Live (calibrate_mode=0), gunakan calib_data dari luar (closure)
        # yang sudah di-load dari JSON.
        # JANGAN pakai nama 'calib_data' di sini karena Python akan membuat
        # local variable baru yang menyembunyikan (shadow) outer variable!
        _calib_result = None
        if calibrate_mode in (1, 2):
            _calib_result = calib_rt.run_calib_1p_2p(get_frame, cap, aruco, yolo, midas, headless, true_height, true_height_2, calibrate_mode, gui)
        elif calibrate_mode == 3:
            _calib_result = calib_rt.run_calib_zgrid(get_frame, cap, aruco, yolo, midas, headless, true_height, n_positions, gui)
        elif calibrate_mode == 4:
            _calib_result = calib_rt.run_calib_bbox(get_frame, cap, aruco, yolo, midas, headless, true_height, gui)
        elif calibrate_mode == 5:
            _calib_result = calib_rt.run_calib_geom(get_frame, cap, aruco, yolo, midas, headless, true_height, n_positions, gui)
        elif calibrate_mode == 6:
            _calib_result = calib_rt.run_calib_bilateral(get_frame, cap, aruco, yolo, midas, headless, true_height, true_height_2, n_positions, gui)
        elif calibrate_mode == 7:
            _calib_result = calib_rt.run_calib_analytic(get_frame, cap, aruco, yolo, midas, headless, true_height, true_height_2, gui)
        elif calibrate_mode != 0:
            print("[ERROR] Unknown calibration mode")
            if gui and not headless: gui.queue_key(27)
            return

        # Mode Live: gunakan calib_data dari JSON (outer closure)
        # Mode Kalibrasi: gunakan hasil kalibrasi baru
        active_calib = _calib_result if calibrate_mode != 0 else calib_data

        if active_calib is None and calibrate_mode != 0:
            print("[CALIB] Error or Aborted. Exiting.")
            if gui and not headless: gui.queue_key(27)
            return

        print(f"[LIVE] calib_data type={type(active_calib)}, value={active_calib}")

        active_poly_Kgeom = [1.0]
        active_cup_str = "LEGACY (1 Profile)"
        if active_calib and active_calib.get("type") == 5:
            if "profiles" in active_calib:
                if getattr(args, "target_cup", None):
                    target_str = str(args.target_cup)
                    if target_str in active_calib["profiles"]:
                        active_poly_Kgeom = active_calib["profiles"][target_str]["poly_Kgeom"]
                        active_cup_str = target_str
                    else:
                        keys = list(active_calib["profiles"].keys())
                        active_cup_str = keys[0] if keys else "Unknown"
                        active_poly_Kgeom = active_calib["profiles"][active_cup_str].get("poly_Kgeom", [1.0]) if keys else [1.0]
                else:
                    keys = list(active_calib["profiles"].keys())
                    active_cup_str = keys[0] if keys else "Unknown"
                    active_poly_Kgeom = active_calib["profiles"][active_cup_str].get("poly_Kgeom", [1.0]) if keys else [1.0]
            else:
                active_poly_Kgeom = active_calib.get("poly_Kgeom", [1.0])

        print(f"[LIVE] active_cup_str={active_cup_str}, poly_Kgeom={active_poly_Kgeom}")

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

        live_pipe.run_live_pipeline(get_frame, cap, aruco, yolo, midas, headless, active_calib, marker_size, active_poly_Kgeom, active_cup_str, args, SCREENSHOT_DIR, VIDEO_DIR, gui)
        
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

    # ── Cleanup yang benar untuk menghindari SIGABRT ──
    # Beritahu background thread untuk berhenti membaca frame
    _cam_alive[0] = False
    if _cam_thread.is_alive():
        _cam_thread.join(timeout=1.0)
    
    # Release hardware kamera setelah thread reader dipastikan mati
    if cap and cap.isOpened():
        cap.release()

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
    ap.add_argument("--moil-zoom",         type=float, default=2.0,
                    help="Zoom factor anypoint Moildev (default: 2.0, sama dengan Kakip)")
    ap.add_argument("--moil-mode",         type=int, default=0,
                    help="Mode anypoint: 0 (mode1 Alpha/Beta), 1 (mode1), 2 (Pitch/Yaw/Roll) (default: 0)")
    ap.add_argument("--no-anypoint",       action="store_true",
                    help="Gunakan fisheye mode tapi TANPA remap anypoint (frame raw fisheye)")
    ap.add_argument("--manual-exposure",   type=int,   default=0,
                    help="Setel nilai manual exposure kamera (misal: 156). Default=0 (Auto-brightness)")
    ap.add_argument("--focal-length",      type=float, default=None,
                    help="Manual override focal length ArUco (px) untuk kalibrasi jarak presisi.")

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
