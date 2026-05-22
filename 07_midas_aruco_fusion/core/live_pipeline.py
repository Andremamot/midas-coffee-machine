import time
import cv2
import numpy as np
import threading
import core.height_math as hm
import core.session_reporter as sr
import core.volume_math as vm
import os
from datetime import datetime

# Lock global untuk melindungi MiDaS inference dari re-entrancy di multi-thread GTK
_midas_lock = threading.Lock()

try:
    import gi
    gi.require_version('Gtk', '3.0')
    from gi.repository import GLib as _GLib
    _HAS_GTK = True
except Exception:
    _GLib = None
    _HAS_GTK = False

def run_live_pipeline(get_frame, cap, aruco, yolo, midas, headless, calib_data, marker_size, active_poly_Kgeom, active_cup_str, args, SCREENSHOT_DIR, VIDEO_DIR, gui=None):
    MIDAS_FPS_LIMIT = 5.0
    midas_interval  = 1.0 / MIDAS_FPS_LIMIT
    last_midas_t    = 0.0

    z_tray_live:    float       = 0.0
    aruco_roi:      tuple | None = None
    cup_bboxes: list = []
    cup_heights_ema: list = [None, None]
    last_depth_norm = None
    EMA_ALPHA = 0.35

    # ctype 5 (Geometric) dan 7 (Analytic) tidak butuh MiDaS untuk height
    _ctype_no_midas = {5, 7}

    is_recording = False
    video_writer = None
    screenshot_paths = []

    stats_total_frames = 0
    stats_midas_runs = 0

    history_z_tray = []
    history_cup_h = {0: [], 1: []}
    history_frames = []

    _prev_frame_t = time.time()
    fps_val = 0.0

    try:
        while True:
            ret, frame = get_frame()
            if not ret:
                time.sleep(0.1)
                continue

            now = time.time()
            dt = now - _prev_frame_t
            _prev_frame_t = now
            fps_val = 1.0 / dt if dt > 0 else 0
            stats_total_frames += 1
            h_frame, w_frame = frame.shape[:2]

            aruco_results = aruco.detect(frame)
            if aruco_results:
                best = aruco.get_best_distance(aruco_results)
                if best:
                    z_tray_live = best["distance_cm"]
                    corners = aruco_results[0].get("corners")
                    if corners is not None:
                        pts  = np.array(corners, dtype=np.float32)
                        x1, y1 = np.min(pts, axis=0).astype(int)
                        x2, y2 = np.max(pts, axis=0).astype(int)
                        pad_x  = max(2, (x2 - x1) // 10)
                        pad_y  = max(2, (y2 - y1) // 10)
                        aruco_roi = (x1 + pad_x, y1 + pad_y, x2 - pad_x, y2 - pad_y)

            # ── Deteksi YOLO setiap frame ──────────────────────────────────────
            boxes = yolo.detect(frame)
            ctype_now = (calib_data or {}).get("type", 1)

            # ── Fast-path: hitung height SETIAP frame untuk Geometric/Analytic ─
            # Tidak perlu menunggu MiDaS — cukup bbox YOLO + jarak ArUco
            if ctype_now in _ctype_no_midas and z_tray_live > 0 and boxes:
                boxes_sorted = sorted(boxes, key=lambda b: b["bbox"][0])[:2]
                cup_bboxes = [b["bbox"] for b in boxes_sorted]
                for i, bbox in enumerate(cup_bboxes):
                    height_raw = 0.0
                    if calib_data is not None:
                        if ctype_now == 5:
                            focal_px = aruco.camera_matrix[0, 0]
                            height_raw = hm.calc_height_geom(z_tray_live, bbox, focal_px, active_poly_Kgeom)
                        else:  # ctype 7
                            height_raw = hm.calc_height_analytic(z_tray_live, bbox, calib_data.get("A", 0.0), calib_data.get("B", 0.0))
                    if i == 0 and stats_total_frames % 30 == 1:
                        x1d, y1d, x2d, y2d = bbox
                        print(f"[DBG] ctype={ctype_now} | z={z_tray_live:.1f}cm | bbox_h={y2d-y1d}px | focal={aruco.camera_matrix[0,0]:.0f} | h={height_raw:.2f}cm")
                    if height_raw > 0:
                        if cup_heights_ema[i] is None:
                            cup_heights_ema[i] = height_raw
                        else:
                            cup_heights_ema[i] = (EMA_ALPHA * height_raw) + ((1.0 - EMA_ALPHA) * cup_heights_ema[i])
                    else:
                        cup_heights_ema[i] = None
                # Reset slot yang tidak terdeteksi
                for i in range(len(cup_bboxes), 2):
                    cup_heights_ema[i] = None
            elif ctype_now in _ctype_no_midas and not boxes:
                # YOLO tidak deteksi apa-apa: reset
                cup_bboxes = []
                for i in range(2):
                    cup_heights_ema[i] = None

            # ── MiDaS rate-limited block (untuk SEMUA ctype) ──────────────────
            # - Semua ctype: update depth PiP visualization
            # - Hanya ctype depth-based (1-4, 6): hitung height dari depth data
            if (now - last_midas_t) >= midas_interval and z_tray_live > 0 and aruco_roi:
                _boxes_now = boxes if boxes else []

                # Update cup_bboxes untuk depth-based modes
                if ctype_now not in _ctype_no_midas and _boxes_now:
                    bs = sorted(_boxes_now, key=lambda b: b["bbox"][0])[:2]
                    cup_bboxes = [b["bbox"] for b in bs]

                # Jalankan MiDaS (dengan lock untuk thread safety)
                try:
                    with _midas_lock:
                        depth_map = midas.process(frame)
                    stats_midas_runs += 1
                    depth_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    last_depth_norm = depth_norm
                except Exception as _e:
                    print(f"[MIDAS] Inference error (skipped): {_e}")
                    last_midas_t = now
                    continue

                # Hitung height dari depth — HANYA untuk depth-based modes
                if ctype_now not in _ctype_no_midas:
                    if _boxes_now:
                        m_tray = midas.get_tray_depth(depth_map, aruco_roi)
                        for i in range(2):
                            if i < len(cup_bboxes):
                                bbox = cup_bboxes[i]
                                m_rim = midas.get_rim_depth(depth_map, bbox)
                                height_raw = 0.0

                                if calib_data is not None and m_tray > 0 and m_rim > 0:
                                    if ctype_now == 2:
                                        height_raw = hm.calc_height_2point(m_rim, m_tray, z_tray_live, calib_data.get("m", 0.1), calib_data.get("c", 0.0))
                                    elif ctype_now == 3:
                                        height_raw = hm.calc_height_zgrid(m_rim, m_tray, z_tray_live, calib_data.get("poly_K", [0.8]))
                                    elif ctype_now == 4:
                                        height_raw = hm.calc_height_bbox(m_rim, m_tray, z_tray_live, bbox, calib_data.get("m_ref", 0.15), calib_data.get("c_ref", 0.0), calib_data.get("ref_bbox_area_px", 10000.0))
                                    elif ctype_now == 6:
                                        height_raw = hm.calc_height_bilateral_zgrid(m_rim, m_tray, z_tray_live, calib_data.get("poly_m", [0.1, 0]), calib_data.get("poly_c", [0.0, 0]))
                                    else:
                                        height_raw = hm.calc_height_1point(m_rim, m_tray, z_tray_live, calib_data.get("K", 0.8))

                                if height_raw > 0:
                                    if cup_heights_ema[i] is None:
                                        cup_heights_ema[i] = height_raw
                                    else:
                                        cup_heights_ema[i] = (EMA_ALPHA * height_raw) + ((1.0 - EMA_ALPHA) * cup_heights_ema[i])
                                    history_cup_h[i].append(cup_heights_ema[i])
                                else:
                                    cup_heights_ema[i] = None
                                    history_cup_h[i].append(0.0)
                            else:
                                cup_heights_ema[i] = None
                                history_cup_h[i].append(0.0)

                        history_z_tray.append(z_tray_live)
                        history_frames.append(stats_total_frames)
                    else:
                        cup_bboxes = []
                        for i in range(2):
                            cup_heights_ema[i] = None
                            history_cup_h[i].append(0.0)
                        history_z_tray.append(z_tray_live)
                        history_frames.append(stats_total_frames)

                last_midas_t = now

            disp = frame.copy()
            if aruco_results:
                disp = aruco.annotate_frame(disp, aruco_results)
            if aruco_roi:
                cv2.rectangle(disp, aruco_roi[:2], aruco_roi[2:], (255, 140, 0), 3)
            for bbox in cup_bboxes:
                x1c, y1c, x2c, y2c = bbox
                cv2.rectangle(disp, (x1c, y1c), (x2c, y2c), (0, 255, 80), 5)

            # UI Scaling factor (dinamis berdasarkan lebar frame)
            # Baseline: frame 2592x1944 -> S ≈ 2.5
            S = max(0.5, w_frame / 1000.0)
            panel_w, panel_h = int(560 * S), int(155 * S)
            cv2.rectangle(disp, (20, 20), (20 + panel_w, 20 + panel_h), (25, 25, 25), -1)
            cv2.rectangle(disp, (20, 20), (20 + panel_w, 20 + panel_h), (90, 90, 90), 2)

            if last_depth_norm is not None:
                depth_color = cv2.applyColorMap(last_depth_norm, cv2.COLORMAP_JET)
                pip_h, pip_w = int(h_frame / 3.2), int(w_frame / 3.2)
                depth_resized = cv2.resize(depth_color, (pip_w, pip_h))
                pip_margin = int(40 * S)
                pip_y1 = h_frame - pip_h - pip_margin
                pip_y2 = h_frame - pip_margin
                pip_x1 = w_frame - pip_w - int(10 * S)
                pip_x2 = w_frame - int(10 * S)
                disp[pip_y1:pip_y2, pip_x1:pip_x2] = depth_resized
                cv2.rectangle(disp, (pip_x1, pip_y1), (pip_x2, pip_y2), (200, 200, 200), 3)
                cv2.putText(disp, "MiDaS Depth", (pip_x1 + int(12*S), pip_y1 + int(28*S)), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * S, (255, 255, 255), 3)

            # Panel Text
            cv2.putText(disp, "CUP HEIGHTS", (int(40*S), int(45*S)), cv2.FONT_HERSHEY_SIMPLEX, 0.55 * S, (170, 170, 170), 3)
            cv2.putText(disp, f"Z_tray: {z_tray_live:.1f} cm", (int(230*S), int(40*S)), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * S, (255, 160, 60), 3)

            base_y = int(95 * S)
            for i in range(2):
                h_val = cup_heights_ema[i]
                y_pos = base_y + (i * int(50 * S))
                lbl = f"CUP {i+1}:"
                if h_val and h_val > 0:
                    cv2.putText(disp, lbl, (int(40*S), y_pos-int(15*S)), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * S, (200, 200, 200), 2)
                    cv2.putText(disp, f"{h_val:.1f} cm", (int(115*S), y_pos+int(5*S)), cv2.FONT_HERSHEY_DUPLEX, 1.3 * S, (0, 255, 100), 4)
                    z_rim_val = max(0.0, z_tray_live - h_val)
                    cv2.putText(disp, f"Z_rim: {z_rim_val:.1f} cm", (int(280*S), y_pos-int(4*S)), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * S, (100, 255, 100), 2)

                    if i < len(cup_bboxes):
                        focal_px = aruco.camera_matrix[0, 0]
                        # Ukur lebar di level rim (strip atas bbox), bukan bbox_w penuh.
                        # Ini mencegah overestimasi diameter pada gelas yang body > rim.
                        rim_w_px = vm.measure_rim_width_px(frame, cup_bboxes[i])
                        diameter = vm.calc_diameter(rim_w_px, z_rim_val, focal_px)
                        volume_ml = vm.calc_volume(h_val, diameter)
                        cv2.putText(disp, f"D:{diameter:.1f}cm", (int(395*S), y_pos-int(4*S)), cv2.FONT_HERSHEY_SIMPLEX, 0.45 * S, (200, 200, 255), 2)
                        if volume_ml > 0:
                            cv2.putText(disp, f"V:{volume_ml:.0f}mL", (int(480*S), y_pos-int(4*S)), cv2.FONT_HERSHEY_SIMPLEX, 0.45 * S, (255, 220, 80), 2)
                        else:
                            cv2.putText(disp, "V:--mL", (int(480*S), y_pos-int(4*S)), cv2.FONT_HERSHEY_SIMPLEX, 0.45 * S, (100, 100, 100), 2)
                    else:
                        cv2.putText(disp, f"D:-- cm", (int(395*S), y_pos-int(4*S)), cv2.FONT_HERSHEY_SIMPLEX, 0.45 * S, (100, 100, 100), 2)
                        cv2.putText(disp, f"V:--mL", (int(480*S), y_pos-int(4*S)), cv2.FONT_HERSHEY_SIMPLEX, 0.45 * S, (100, 100, 100), 2)
                else:
                    cv2.putText(disp, lbl, (int(40*S), y_pos-int(15*S)), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * S, (100, 100, 100), 2)
                    cv2.putText(disp, "-- cm", (int(115*S), y_pos+int(5*S)), cv2.FONT_HERSHEY_DUPLEX, 1.3 * S, (70, 70, 70), 4)
                    cv2.putText(disp, f"Z_rim: -- cm", (int(280*S), y_pos-int(4*S)), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * S, (100, 100, 100), 2)
                    cv2.putText(disp, f"D:-- cm", (int(395*S), y_pos-int(4*S)), cv2.FONT_HERSHEY_SIMPLEX, 0.45 * S, (100, 100, 100), 2)
                    cv2.putText(disp, f"V:--mL", (int(480*S), y_pos-int(4*S)), cv2.FONT_HERSHEY_SIMPLEX, 0.45 * S, (100, 100, 100), 2)

            if is_recording:
                if int(time.time() * 2) % 2 == 0:
                    cv2.circle(disp, (w_frame - int(140*S), int(45*S)), int(15*S), (0, 0, 255), -1)
                    cv2.putText(disp, "REC", (w_frame - int(115*S), int(58*S)), cv2.FONT_HERSHEY_SIMPLEX, 0.9 * S, (0, 0, 255), 4)
                if video_writer is not None:
                    video_writer.write(disp)

            bar_h = int(45 * S)
            cv2.rectangle(disp, (0, h_frame - bar_h), (w_frame, h_frame), (15, 15, 15), -1)

            # Status LED supplement (hanya tampil saat --manual-exposure aktif)
            _led_state = getattr(args, "_led_state", None)
            led_txt = ""
            if _led_state is not None:
                led_on = _led_state.get("detected", False)
                led_txt = f" | LED: {'ON' if led_on else 'OFF'}"
                led_color = (0, 220, 255) if led_on else (100, 100, 100)
                cv2.putText(disp, f"LED: {'ON' if led_on else 'OFF'}",
                            (w_frame - int(200*S), h_frame - int(15*S)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5 * S, led_color, 3)

            bar_txt = f"FPS: {fps_val:.1f} | ArUco: {'OK' if z_tray_live else 'X'} | YOLO: {'OK' if cup_bboxes else 'X'} | [R] Record  [S] Screen  [Q] Quit"
            cv2.putText(disp, bar_txt, (int(20*S), h_frame - int(15*S)), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * S, (130, 200, 130), 3)

            if getattr(calib_data, "get", lambda x: 0)("type") == 5 or (isinstance(calib_data, dict) and calib_data.get("type") == 5):
                cv2.putText(disp, f"TARGET MENU: {active_cup_str} cm", (int(40*S), int(145*S)), cv2.FONT_HERSHEY_SIMPLEX, 0.65 * S, (0, 255, 255), 3)

            if gui and not headless:
                aruco_ok = 'OK' if z_tray_live else 'X'
                yolo_ok = 'OK' if cup_bboxes else 'X'
                led_state_obj = getattr(args, "_led_state", None) or {}
                led_st = "ON" if led_state_obj.get("detected", False) else "OFF"
                status_text = f"ArUco: {aruco_ok} | YOLO: {yolo_ok} | LED: {led_st}"
                if _HAS_GTK:
                    _GLib.idle_add(lambda t=status_text, g=gui: (g._alive and g.lbl_status_ai.set_text(t)) or False)
                gui.update_image(disp)
                key = gui.get_key()
            elif not headless:
                cv2.imshow("ArUco + MiDaS | Cup Height Estimator", disp)
                key = cv2.waitKey(1) & 0xFF
            else:
                key = -1

            if not headless:
                if key in (27, ord('q')):
                    break
                elif key == ord('s'):
                    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
                    ss_path = os.path.join(SCREENSHOT_DIR, f"fusion_{ts}.jpg")
                    cv2.imwrite(ss_path, disp)
                    screenshot_paths.append(ss_path)
                    print(f"[SHOT] Screenshot tersimpan: {ss_path}")
                elif key == ord('r'):
                    if not is_recording:
                        ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
                        vid_path = os.path.join(VIDEO_DIR, f"fusion_{ts}.mp4")
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        # Gunakan 10 fps minimum — cap.get(CAP_PROP_FPS) mengembalikan 1.0
                        # di resolusi 2592x1944 yang akan menghasilkan video patah-patah
                        rec_fps = max(10.0, cap.get(cv2.CAP_PROP_FPS))
                        video_writer = cv2.VideoWriter(vid_path, fourcc, rec_fps, (w_frame, h_frame))
                        is_recording = True
                        print(f"🔴 [REC] Mulai merekam video: {vid_path}")
                    else:
                        is_recording = False
                        if video_writer: video_writer.release()
                        video_writer = None
                        print("⏹ [REC] Merekam dihentikan.")
            else:
                if (now - last_midas_t) < 0.05:
                    h1 = cup_heights_ema[0]
                    h2 = cup_heights_ema[1]
                    s1 = f"C1:{h1:.2f}cm" if h1 and h1 > 0 else "C1:--"
                    s2 = f"C2:{h2:.2f}cm" if h2 and h2 > 0 else "C2:--"
                    print(f"[DATA] z_tray={z_tray_live:.2f}cm | {s1} | {s2}")


    except KeyboardInterrupt:
        print("\n[INFO] Execution stopped by user (Ctrl+C).")
    finally:
            if video_writer: video_writer.release()
            if not headless and not gui: cv2.destroyAllWindows()

            print("\n[DONE] Pipeline closed. Generating Final Report...")
            sr._generate_session_report(
                calib_data=calib_data,
                marker_size_cm=marker_size,
                focal_len=aruco.camera_matrix[0,0],
                total_frames=stats_total_frames,
                midas_runs=stats_midas_runs,
                history_z_tray=history_z_tray,
                history_cup_h=history_cup_h,
                history_frames=history_frames,
                screenshots=screenshot_paths
            )
