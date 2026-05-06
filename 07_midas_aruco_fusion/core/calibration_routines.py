import time
import cv2
import numpy as np
import core.calibration_storage as cs
import core.height_math as hm
import sys

def run_calib_1p_2p(get_frame, cap, aruco, yolo, midas, headless, true_height, true_height_2, calibrate_mode, gui=None):
    CALIB_WARMUP_SEC  = 5.0
    CALIB_SAMPLE_SEC  = 8.0

    if not headless:
        cv2.namedWindow("ArUco + MiDaS | Cup Height Estimator", cv2.WINDOW_NORMAL)

    calib_ratios_1, calib_z_trays_1 = [], []
    calib_ratios_2, calib_z_trays_2 = [], []

    phase = "warmup_1"
    calib_start = time.time()

    print(f"━" * 55)
    print(f"  ⚙  KALIBRASI BERSYARAT IN-SESSION ({calibrate_mode}-Point)")
    print(f"━" * 55)

    last_midas_calib = 0.0
    boxes = None

    while phase != "done":
        ret, frame = get_frame()
        if not ret:
            time.sleep(0.05)
            continue

        elapsed = time.time() - calib_start
        h_f, w_f = frame.shape[:2]

        if last_midas_calib == 0.0 or (time.time() - last_midas_calib) > 1.0:
            # If YOLO didn't run recently, clear old boxes
            boxes = None

        aruco_results = aruco.detect(frame)
        z_calib = 0.0
        aruco_roi_c = None
        if aruco_results:
            best = aruco.get_best_distance(aruco_results)
            if best:
                z_calib = best["distance_cm"]
                corners = aruco_results[0].get("corners")
                if corners is not None:
                    pts = np.array(corners, dtype=np.float32)
                    x1c, y1c = np.min(pts, axis=0).astype(int)
                    x2c, y2c = np.max(pts, axis=0).astype(int)
                    aruco_roi_c = (x1c + 2, y1c + 2, x2c - 2, y2c - 2)

        if (time.time() - last_midas_calib) > 0.2 and z_calib > 0 and aruco_roi_c:
            boxes = yolo.detect(frame)
            if boxes:
                bbox_c = boxes[0]["bbox"]
                dm = midas.process(frame)
                last_midas_calib = time.time()

                m_rim  = midas.get_rim_depth(dm, bbox_c)
                m_tray = midas.get_tray_depth(dm, aruco_roi_c)

                if m_rim > 0 and m_tray > 0:
                    if phase == "sampling_1":
                        calib_ratios_1.append(m_rim / m_tray)
                        calib_z_trays_1.append(z_calib)
                    elif phase == "sampling_2":
                        calib_ratios_2.append(m_rim / m_tray)
                        calib_z_trays_2.append(z_calib)
            else:
                last_midas_calib = time.time()

        # Transisi fase otomatis per timer (dan sample count)
        if phase == "warmup_1" and elapsed >= CALIB_WARMUP_SEC:
            phase = "sampling_1"
            calib_start = time.time()  # Reset timer untuk sampling
            elapsed = 0.0
        elif phase == "sampling_1":
            if len(calib_ratios_1) >= 5:
                if calibrate_mode == 1:
                    phase = "done"
                else:
                    phase = "swap_wait"
            elif elapsed > 30.0:
                print("[CALIB] Timeout: Cannot properly detect cup 1 (YOLO/ArUco failed).")
                phase = "done"

        elif phase == "warmup_2" and elapsed >= CALIB_WARMUP_SEC:
            phase = "sampling_2"
            calib_start = time.time()
            elapsed = 0.0
        elif phase == "sampling_2":
            if len(calib_ratios_2) >= 5:
                phase = "done"
            elif elapsed > 30.0:
                print("[CALIB] Timeout: Cannot properly detect cup 2 (YOLO/ArUco failed).")
                phase = "done"

        # UI Overlay Kalibrasi
        disp_c = frame.copy()
        if aruco_results:
            disp_c = aruco.annotate_frame(disp_c, aruco_results)

        # Gambar bounding box Yolo saat kalibrasi agar user tahu Yolo melihat gelasnya
        if boxes:
            for b in boxes:
                x1c, y1c, x2c, y2c = b["bbox"]
                cv2.rectangle(disp_c, (x1c, y1c), (x2c, y2c), (0, 255, 80), 4)

        S = 2.5
        panel_w, panel_h = int(530 * S), int(105 * S)
        cv2.rectangle(disp_c, (25, 25), (25 + panel_w, 25 + panel_h), (20, 20, 40), -1)
        cv2.rectangle(disp_c, (25, 25), (25 + panel_w, 25 + panel_h), (0, 200, 255), 3)

        # Indikator Hardware
        status_aruco = "OK" if z_calib > 0 else "NOT FOUND"
        status_yolo  = "OK" if boxes else "NOT FOUND"
        cv2.putText(disp_c, f"[ArUco: {status_aruco}]  [YOLO: {status_yolo}]", (int(45*S), int(115*S)), cv2.FONT_HERSHEY_SIMPLEX, 0.45 * S, (200, 200, 200), 3)

        if phase.startswith("warmup_"):
            idx = phase[-1]
            pct = min(100, int((elapsed / CALIB_WARMUP_SEC) * 100))
            H_t = true_height if idx == "1" else true_height_2
            cv2.putText(disp_c, f"WARMING UP CUP {idx} (H={H_t}cm)", (int(45*S), int(60*S)), cv2.FONT_HERSHEY_SIMPLEX, 0.55 * S, (0, 200, 255), 3)
            cv2.putText(disp_c, f"Keep cup still. Prog: {pct}%", (int(45*S), int(90*S)), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * S, (150, 200, 255), 2)

        elif phase.startswith("sampling_"):
            idx = phase[-1]
            n = len(calib_ratios_1) if idx == "1" else len(calib_ratios_2)
            cv2.putText(disp_c, f"SAMPLING DATA CUP {idx} (Count: {n}/5)", (int(45*S), int(60*S)), cv2.FONT_HERSHEY_SIMPLEX, 0.55 * S, (0, 255, 180), 3)
            cv2.putText(disp_c, "Ensure camera and cup are visible...", (int(45*S), int(90*S)), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * S, (150, 255, 200), 2)

        elif phase == "swap_wait":
            cv2.rectangle(disp_c, (25, 25), (25 + panel_w, 25 + panel_h), (200, 50, 50), -1)
            cv2.putText(disp_c, "SWAP THE CUP NOW", (int(45*S), int(60*S)), cv2.FONT_HERSHEY_SIMPLEX, 0.65 * S, (255, 255, 255), 4)
            cv2.putText(disp_c, f"Place a cup with height {true_height_2} cm on the tray.", (int(45*S), int(85*S)), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * S, (200, 220, 255), 2)
            cv2.putText(disp_c, "Then PRESS 'SPACE' to continue.", (int(45*S), int(110*S)), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * S, (100, 255, 100), 2)

        if not headless:
            if gui:
                gui.update_image(disp_c)
                key = gui.get_key()
            else:
                cv2.imshow("ArUco + MiDaS | Cup Height Estimator", disp_c)
                key = cv2.waitKey(1) & 0xFF
        else:
            key = -1

        if key == 27:
            cap.release()
            cv2.destroyAllWindows()
            sys.exit(0)
        if phase == "swap_wait" and key == ord(' '):
            calib_start = time.time()
            phase = "warmup_2"

    # Hitung hasil kalibrasi
    if len(calib_ratios_1) < 3:
        print("[CALIB] Error: Insufficient data for cup 1.")
        return

    R1 = float(np.mean(calib_ratios_1))
    Z1 = float(np.mean(calib_z_trays_1))

    if calibrate_mode == 1:
        K_factor = R1 * (1.0 - true_height / Z1)
        cs.save_calibration_1p(K_factor, Z1, R1, true_height)
        calib_data = {"type": 1, "K": K_factor}
    else:
        if len(calib_ratios_2) < 3:
            print("[CALIB] Error: Insufficient data for cup 2.")
            return
        R2 = float(np.mean(calib_ratios_2))
        Z2 = float(np.mean(calib_z_trays_2))

        # Linear Fit: H/Z = m * R + c
        # Titik 1: Y1 = H1/Z1, Titik 2: Y2 = H2/Z2
        Y1 = true_height / Z1
        Y2 = true_height_2 / Z2

        if abs(R2 - R1) < 0.05:
            print("[CALIB] Error: Both cups have nearly identical rim/tray ratio in MiDaS.")
            print("       Use cups with a wider height difference (e.g. 7.6cm and 11cm).")
            return

        m = (Y2 - Y1) / (R2 - R1)
        c = Y1 - m * R1
        cs.save_calibration_2p(m, c, 
                            {"R": R1, "Z": Z1, "H": true_height}, 
                            {"R": R2, "Z": Z2, "H": true_height_2})
        calib_data = {"type": 2, "m": m, "c": c}

    print("[CALIB] ✅ Success. Entering LIVE mode!\n")
    return calib_data


def run_calib_zgrid(get_frame, cap, aruco, yolo, midas, headless, true_height, n_positions, gui=None):
    print("━" * 55)
    print(f"  ⚙  Z-GRID CALIBRATION ({n_positions} positions)")
    print(f"     Cup reference height: {true_height} cm")
    print("━" * 55)

    CALIB_WARMUP_SEC = 4.0
    CALIB_SAMPLE_SEC = 6.0
    grid_data = []   # list of (z_avg, R_avg) per position
    pos_idx   = 0
    phase     = "warmup"
    calib_start = time.time()
    boxes = None
    last_midas_calib = 0.0
    pos_ratios, pos_z_trays = [], []

    while pos_idx < n_positions or phase not in ("done", "warmup"):
        ret, frame = get_frame()
        if not ret:
            time.sleep(0.05)
            continue

        elapsed = time.time() - calib_start
        if last_midas_calib == 0.0 or (time.time() - last_midas_calib) > 1.0:
            boxes = None

        aruco_results = aruco.detect(frame)
        z_calib   = 0.0
        aruco_roi_c = None
        if aruco_results:
            best = aruco.get_best_distance(aruco_results)
            if best:
                z_calib = best["distance_cm"]
                corners = aruco_results[0].get("corners")
                if corners is not None:
                    pts = np.array(corners, dtype=np.float32)
                    x1c, y1c = np.min(pts, axis=0).astype(int)
                    x2c, y2c = np.max(pts, axis=0).astype(int)
                    aruco_roi_c = (x1c+2, y1c+2, x2c-2, y2c-2)

        if (time.time() - last_midas_calib) > 0.2 and z_calib > 0 and aruco_roi_c and phase == "sampling":
            boxes = yolo.detect(frame)
            if boxes:
                bbox_c = boxes[0]["bbox"]
                dm = midas.process(frame)
                last_midas_calib = time.time()
                m_rim  = midas.get_rim_depth(dm, bbox_c)
                m_tray = midas.get_tray_depth(dm, aruco_roi_c)
                if m_rim > 0 and m_tray > 0:
                    pos_ratios.append(m_rim / m_tray)
                    pos_z_trays.append(z_calib)
            else:
                last_midas_calib = time.time()

        if phase == "warmup" and elapsed >= CALIB_WARMUP_SEC:
            phase = "sampling"
            calib_start = time.time()
            elapsed = 0.0
        elif phase == "sampling":
            if len(pos_ratios) >= 5 or elapsed > 30.0:
                # Commit this position
                if len(pos_ratios) >= 3:
                    R_avg = float(np.mean(pos_ratios))
                    Z_avg = float(np.mean(pos_z_trays))
                    grid_data.append({"R": R_avg, "Z": Z_avg})
                    print(f"[CALIB] Position {pos_idx+1}/{n_positions} committed: Z={Z_avg:.2f}cm, R={R_avg:.4f}")
                pos_ratios.clear()
                pos_z_trays.clear()
                pos_idx += 1
                if pos_idx >= n_positions:
                    phase = "done"
                else:
                    phase = "swap_wait"

        # UI
        disp_c = frame.copy()
        if aruco_results:
            disp_c = aruco.annotate_frame(disp_c, aruco_results)
        if boxes:
            for b in boxes:
                x1b, y1b, x2b, y2b = b["bbox"]
                cv2.rectangle(disp_c, (x1b, y1b), (x2b, y2b), (0, 255, 80), 4)

        S = 2.5
        panel_w, panel_h = int(535 * S), int(105 * S)
        cv2.rectangle(disp_c, (25, 25), (25 + panel_w, 25 + panel_h), (20, 20, 40), -1)
        cv2.rectangle(disp_c, (25, 25), (25 + panel_w, 25 + panel_h), (0, 200, 255), 3)
        
        status_a = "OK" if z_calib > 0 else "NOT FOUND"
        status_y = "OK" if boxes else "NOT FOUND"
        cv2.putText(disp_c, f"[ArUco: {status_a}]  [YOLO: {status_y}]", (int(45*S), int(115*S)), cv2.FONT_HERSHEY_SIMPLEX, 0.45 * S, (200,200,200), 3)

        if phase == "warmup":
            pct = min(100, int((elapsed/CALIB_WARMUP_SEC)*100))
            cv2.putText(disp_c, f"Z-GRID: Warming up position {pos_idx+1}/{n_positions}", (int(45*S), int(60*S)), cv2.FONT_HERSHEY_SIMPLEX, 0.55 * S, (0, 200, 255), 3)
            cv2.putText(disp_c, f"Keep cup + nozzle still. Prog: {pct}%", (int(45*S), int(90*S)), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * S, (150, 200, 255), 2)
        elif phase == "sampling":
            cv2.putText(disp_c, f"Z-GRID: Sampling pos {pos_idx+1}/{n_positions} (Count: {len(pos_ratios)}/5)", (int(45*S), int(60*S)), cv2.FONT_HERSHEY_SIMPLEX, 0.55 * S, (0, 255, 180), 3)
            cv2.putText(disp_c, f"Z_tray = {z_calib:.1f} cm", (int(45*S), int(90*S)), cv2.FONT_HERSHEY_SIMPLEX, 0.55 * S, (255, 180, 60), 3)
        elif phase == "swap_wait":
            cv2.rectangle(disp_c, (25, 25), (25 + panel_w, 25 + panel_h), (160, 50, 30), -1)
            cv2.putText(disp_c, f"MOVE NOZZLE TO NEXT POSITION", (int(45*S), int(62*S)), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * S, (255, 255, 255), 4)
            cv2.putText(disp_c, f"({pos_idx+1}/{n_positions} done)  Keep same cup visible.", (int(45*S), int(88*S)), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * S, (200, 220, 255), 2)
            cv2.putText(disp_c, "Press SPACE when ready.", (int(45*S), int(110*S)), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * S, (100, 255, 100), 2)

        if not headless:
            if gui:
                gui.update_image(disp_c)
                key = gui.get_key()
            else:
                cv2.imshow("ArUco + MiDaS | Cup Height Estimator", disp_c)
                key = cv2.waitKey(1) & 0xFF
        else:
            key = -1
            
        if key == 27:
            cap.release(); cv2.destroyAllWindows(); sys.exit(0)
        if phase == "swap_wait" and key == ord(' '):
            calib_start = time.time()
            phase = "warmup"

        if phase == "done":
            break

    # Compute polynomial fit across grid points
    if len(grid_data) < 2:
        print("[CALIB] Not enough position data. Aborting.")
        return

    Z_pts  = np.array([p["Z"] for p in grid_data])
    R_pts  = np.array([p["R"] for p in grid_data])

    # K_i = R_i * (1.0 - H / Z_i)
    K_pts = R_pts * (1.0 - true_height / Z_pts)

    deg = min(len(grid_data) - 1, 2)
    poly_K = np.polyfit(Z_pts, K_pts, deg=deg).tolist()

    cs.save_calibration_3p(poly_K, Z_pts.tolist(), true_height)
    calib_data = {"type": 3, "poly_K": poly_K}
    print("[CALIB] ✅ Z-Grid calibration done! Entering LIVE mode.\n")

    return calib_data


def run_calib_bbox(get_frame, cap, aruco, yolo, midas, headless, true_height, gui=None):
    print("━" * 55)
    print("  ⚙  BBOX AREA COMPENSATION CALIBRATION (Type 4)")
    print(f"     Cup reference height: {true_height} cm")
    print("━" * 55)

    CALIB_WARMUP_SEC = 4.0
    positions_4 = []  # list of {R, Z, area}
    phase = "warmup"
    calib_start = time.time()
    boxes = None
    last_midas_calib = 0.0
    pos_ratios4, pos_z4, pos_areas4 = [], [], []
    pos_idx4 = 0
    N_POS_4  = 2  # low Z and high Z

    while pos_idx4 < N_POS_4:
        ret, frame = get_frame()
        if not ret:
            time.sleep(0.05); continue

        elapsed = time.time() - calib_start
        if last_midas_calib == 0.0 or (time.time() - last_midas_calib) > 1.0:
            boxes = None

        aruco_results = aruco.detect(frame)
        z_calib = 0.0; aruco_roi_c = None
        if aruco_results:
            best = aruco.get_best_distance(aruco_results)
            if best:
                z_calib = best["distance_cm"]
                corners = aruco_results[0].get("corners")
                if corners is not None:
                    pts = np.array(corners, dtype=np.float32)
                    x1c, y1c = np.min(pts, axis=0).astype(int)
                    x2c, y2c = np.max(pts, axis=0).astype(int)
                    aruco_roi_c = (x1c+2, y1c+2, x2c-2, y2c-2)

        if (time.time() - last_midas_calib) > 0.2 and z_calib > 0 and aruco_roi_c and phase == "sampling":
            boxes = yolo.detect(frame)
            if boxes:
                bbox_c = boxes[0]["bbox"]
                dm = midas.process(frame)
                last_midas_calib = time.time()
                m_rim4  = midas.get_rim_depth(dm, bbox_c)
                m_tray4 = midas.get_tray_depth(dm, aruco_roi_c)
                if m_rim4 > 0 and m_tray4 > 0:
                    x1b, y1b, x2b, y2b = bbox_c
                    area = float((x2b - x1b) * (y2b - y1b))
                    pos_ratios4.append(m_rim4 / m_tray4)
                    pos_z4.append(z_calib)
                    pos_areas4.append(area)
            else:
                last_midas_calib = time.time()

        if phase == "warmup" and elapsed >= CALIB_WARMUP_SEC:
            phase = "sampling"; calib_start = time.time(); elapsed = 0.0
        elif phase == "sampling":
            if len(pos_ratios4) >= 5 or elapsed > 30.0:
                if len(pos_ratios4) >= 3:
                    R_avg4 = float(np.mean(pos_ratios4))
                    Z_avg4 = float(np.mean(pos_z4))
                    A_avg4 = float(np.mean(pos_areas4))
                    positions_4.append({"R": R_avg4, "Z": Z_avg4, "area": A_avg4})
                    print(f"[CALIB] BBox pos {pos_idx4+1}/2 committed: Z={Z_avg4:.2f}cm, area={A_avg4:.0f}px²")
                pos_ratios4.clear(); pos_z4.clear(); pos_areas4.clear()
                pos_idx4 += 1
                if pos_idx4 < N_POS_4:
                    phase = "swap_wait"

        # UI
        disp_c = frame.copy()
        if aruco_results: disp_c = aruco.annotate_frame(disp_c, aruco_results)
        if boxes:
            for b in boxes:
                x1b,y1b,x2b,y2b = b["bbox"]
                cv2.rectangle(disp_c,(x1b,y1b),(x2b,y2b),(0,255,80),4)

        S = 2.5
        panel_w, panel_h = int(535 * S), int(105 * S)
        cv2.rectangle(disp_c, (25, 25), (25 + panel_w, 25 + panel_h), (20, 20, 40), -1)
        cv2.rectangle(disp_c, (25, 25), (25 + panel_w, 25 + panel_h), (0, 200, 255), 3)
        
        status_a = "OK" if z_calib > 0 else "NOT FOUND"
        status_y = "OK" if boxes else "NOT FOUND"
        cv2.putText(disp_c,f"[ArUco: {status_a}]  [YOLO: {status_y}]",(int(45*S), int(115*S)),cv2.FONT_HERSHEY_SIMPLEX,0.45 * S,(200,200,200),3)
        if phase == "warmup":
            pct = min(100, int((elapsed/CALIB_WARMUP_SEC)*100))
            cv2.putText(disp_c,f"BBOX-AREA: Warming up position {pos_idx4+1}/2",(int(45*S), int(60*S)),cv2.FONT_HERSHEY_SIMPLEX,0.55 * S,(0,200,255),3)
            cv2.putText(disp_c,f"Keep cup still. Prog: {pct}%",(int(45*S), int(90*S)),cv2.FONT_HERSHEY_SIMPLEX,0.5 * S,(150,200,255),2)
        elif phase == "sampling":
            n4 = len(pos_ratios4)
            cv2.putText(disp_c,f"BBOX-AREA: Sampling pos {pos_idx4+1}/2 (Count: {n4}/5)",(int(45*S), int(60*S)),cv2.FONT_HERSHEY_SIMPLEX,0.55 * S,(0,255,180),3)
            cv2.putText(disp_c,f"Z_tray = {z_calib:.1f} cm",(int(45*S), int(90*S)),cv2.FONT_HERSHEY_SIMPLEX,0.55 * S,(255,180,60),3)
        elif phase == "swap_wait":
            cv2.rectangle(disp_c,(25, 25), (25 + panel_w, 25 + panel_h),(160, 50, 30),-1)
            cv2.putText(disp_c,"MOVE NOZZLE TO DIFFERENT HEIGHT",(int(45*S), int(62*S)),cv2.FONT_HERSHEY_SIMPLEX,0.6 * S,(255,255,255),4)
            cv2.putText(disp_c,"(1/2 done)  Keep same cup visible.",(int(45*S), int(88*S)),cv2.FONT_HERSHEY_SIMPLEX,0.5 * S,(200,220,255),2)
            cv2.putText(disp_c,"Press SPACE when ready.",(int(45*S), int(110*S)),cv2.FONT_HERSHEY_SIMPLEX,0.5 * S,(100,255,100),2)

        if not headless:
            if gui:
                gui.update_image(disp_c)
                key = gui.get_key()
            else:
                cv2.imshow("ArUco + MiDaS | Cup Height Estimator", disp_c)
                key = cv2.waitKey(1) & 0xFF
        else:
            key = -1

        if key == 27: cap.release(); cv2.destroyAllWindows(); sys.exit(0)
        if phase == "swap_wait" and key == ord(' '):
            calib_start = time.time(); phase = "warmup"

    if len(positions_4) < 2:
        print("[CALIB] Not enough BBox position data. Aborting.")
        return

    # Reference is position with largest bbox (closest camera = most detail = ref)
    ref = max(positions_4, key=lambda p: p["area"])
    # Compute m and c from reference point: H/Z = m_ref * R_ref + c_ref
    # Use 2-point linear from both measured points for a proper m_ref/c_ref
    p1, p2 = positions_4[0], positions_4[1]
    Y1_4  = true_height / p1["Z"]
    Y2_4  = true_height / p2["Z"]
    dR_4  = p2["R"] - p1["R"]
    m_ref4 = (Y2_4 - Y1_4) / dR_4 if abs(dR_4) > 0.02 else 0.15
    c_ref4 = Y1_4 - m_ref4 * p1["R"]

    cs.save_calibration_4p(m_ref4, c_ref4, ref["area"], p1["Z"], p2["Z"], true_height)
    calib_data = {"type": 4, "m_ref": m_ref4, "c_ref": c_ref4, "ref_bbox_area_px": ref["area"]}
    print("[CALIB] ✅ BBox Area calibration done! Entering LIVE mode.\n")
    return calib_data


def run_calib_geom(get_frame, cap, aruco, yolo, midas, headless, true_height, n_positions, gui=None):
    focal_length_px = aruco.camera_matrix[0, 0]
    print("━" * 55)
    print(f"  ⚙  GEOMETRIC Z-GRID CALIBRATION ({n_positions} positions)")
    print(f"     Cup reference height : {true_height} cm")
    print(f"     Camera focal length  : {focal_length_px:.1f} px")
    print("━" * 55)

    CALIB_WARMUP_SEC = 5.0
    CALIB_SAMPLE_SEC = 5.0
    phase = "warmup"
    calib_start = time.time()
    boxes = None
    last_det = 0.0

    pos_idx = 0
    grid_data = []  # will store {"Z": z, "H_px": bbox_h_px}
    current_g_z, current_g_h = [], []

    while phase != "done":
        ret, frame = get_frame()
        if not ret:
            time.sleep(0.05); continue

        elapsed = time.time() - calib_start
        if (time.time() - last_det) > 1.0:
            boxes = None

        aruco_results = aruco.detect(frame)
        z_calib = 0.0
        if aruco_results:
            best = aruco.get_best_distance(aruco_results)
            if best: z_calib = best["distance_cm"]

        # Run YOLO continuously for visual feedback
        boxes = yolo.detect(frame)
        
        if boxes and z_calib > 0 and phase == "sampling":
            if (time.time() - last_det) > 0.15:
                x1b, y1b, x2b, y2b = boxes[0]["bbox"]
                bbox_h = float(y2b - y1b)
                if bbox_h > 5:
                    current_g_z.append(z_calib)
                    current_g_h.append(bbox_h)
                    last_det = time.time()

        if phase == "warmup" and elapsed >= CALIB_WARMUP_SEC:
            phase = "sampling"; calib_start = time.time(); elapsed = 0.0
        elif phase == "sampling":
            if len(current_g_z) >= 30:
                avg_z = float(np.median(current_g_z))
                avg_h_px = float(np.median(current_g_h))
                grid_data.append({"Z": avg_z, "H_px": avg_h_px})
                print(f"[CALIB] Position {pos_idx+1}/{n_positions} committed: Z={avg_z:.2f}cm, bbox_H={avg_h_px:.1f}px")

                pos_idx += 1
                if pos_idx >= n_positions:
                    phase = "done"
                else:
                    phase = "swap_wait"

        # UI
        disp_c = frame.copy()
        if aruco_results: disp_c = aruco.annotate_frame(disp_c, aruco_results)
        if boxes:
            for b in boxes:
                x1b, y1b, x2b, y2b = b["bbox"]
                cv2.rectangle(disp_c, (x1b, y1b), (x2b, y2b), (0, 255, 80), 4)

        S = 2.5
        panel_w, panel_h = int(535 * S), int(105 * S)
        cv2.rectangle(disp_c, (25, 25), (25 + panel_w, 25 + panel_h), (20, 20, 40), -1)
        cv2.rectangle(disp_c, (25, 25), (25 + panel_w, 25 + panel_h), (0, 220, 120), 3)
        
        status_a = "OK" if z_calib > 0 else "NOT FOUND"
        status_y = "OK" if boxes else "NOT FOUND"
        cv2.putText(disp_c, f"[ArUco: {status_a}]  [YOLO: {status_y}]", (int(45*S), int(115*S)), cv2.FONT_HERSHEY_SIMPLEX, 0.45 * S, (200, 200, 200), 3)

        if phase == "warmup":
            pct = min(100, int((elapsed / CALIB_WARMUP_SEC) * 100))
            cv2.putText(disp_c, f"GEO-GRID ({pos_idx+1}/{n_positions}): Warming up... {pct}%", (int(45*S), int(60*S)), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * S, (0, 220, 255), 3)
            cv2.putText(disp_c, f"Keep cup {true_height}cm still.", (int(45*S), int(90*S)), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * S, (150, 220, 255), 2)
        elif phase == "sampling":
            n5 = len(current_g_z)
            cv2.putText(disp_c, f"GEO-GRID ({pos_idx+1}/{n_positions}): Sampling ({n5}/30)", (int(45*S), int(60*S)), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * S, (0, 255, 180), 3)
            cv2.putText(disp_c, f"Z_tray = {z_calib:.1f} cm", (int(45*S), int(90*S)), cv2.FONT_HERSHEY_SIMPLEX, 0.55 * S, (150, 255, 200), 3)
        elif phase == "swap_wait":
            cv2.rectangle(disp_c, (25, 25), (25 + panel_w, 25 + panel_h), (40, 40, 150), -1)
            cv2.putText(disp_c, "MOVE NOZZLE TO NEW HEIGHT", (int(45*S), int(60*S)), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * S, (255, 255, 255), 4)
            cv2.putText(disp_c, "Wait for focus, then press SPACE.", (int(45*S), int(90*S)), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * S, (200, 220, 255), 2)

        if not headless:
            if gui:
                gui.update_image(disp_c)
                key = gui.get_key()
            else:
                cv2.imshow("ArUco + MiDaS | Cup Height Estimator", disp_c)
                key = cv2.waitKey(1) & 0xFF
        else:
            key = -1

        if key == 27: cap.release(); cv2.destroyAllWindows(); sys.exit(0)
        if phase == "swap_wait" and key == ord(' '):
            calib_start = time.time(); phase = "warmup"
            current_g_z, current_g_h = [], []

    # Polynomial Fit
    Z_pts = np.array([p["Z"] for p in grid_data])
    H_px_pts = np.array([p["H_px"] for p in grid_data])

    # K_geom = H_true / (Z_tray * H_px / F)
    K_pts = true_height / (Z_pts * H_px_pts / focal_length_px)
    deg = min(len(grid_data) - 1, 2)
    poly_Kgeom = np.polyfit(Z_pts, K_pts, deg=deg).tolist()

    cs.save_calibration_5p(poly_Kgeom, Z_pts.tolist(), true_height)
    calib_data = {"type": 5, "poly_Kgeom": poly_Kgeom}
    print("[CALIB] ✅ Geometric Z-Grid calibration done! Entering LIVE mode.\n")
    return calib_data


def run_calib_bilateral(get_frame, cap, aruco, yolo, midas, headless, true_height, true_height_2, n_positions, gui=None):
    print("━" * 55)
    print(f"  ⚙  BILATERAL Z-GRID CALIBRATION ({n_positions} positions X 2 cups)")
    print(f"     Cup 1 height : {true_height} cm")
    print(f"     Cup 2 height : {true_height_2} cm")
    print("━" * 55)

    CALIB_WARMUP = 4.0
    CALIB_SAMPLE = 4.0
    phase = "warmup_c1"
    calib_start = time.time()
    pos_idx = 0

    last_midas_calib = 0.0
    aruco_roi_c = None
    boxes = None

    m_pts, c_pts, Z_pts = [], [], []
    r1_samples, r2_samples, z_samples = [], [], []

    while phase != "done":
        ret, frame = get_frame()
        if not ret: time.sleep(0.05); continue

        elapsed = time.time() - calib_start

        if (time.time() - last_midas_calib) > 1.0: boxes = None

        aruco_results = aruco.detect(frame)
        z_calib = 0.0
        aruco_roi_c = None
        if aruco_results:
            best = aruco.get_best_distance(aruco_results)
            if best:
                z_calib = best["distance_cm"]
                corners = aruco_results[0].get("corners")
                if corners is not None:
                    pts = np.array(corners, dtype=np.float32)
                    x1c, y1c = np.min(pts, axis=0).astype(int)
                    x2c, y2c = np.max(pts, axis=0).astype(int)
                    aruco_roi_c = (x1c + 2, y1c + 2, x2c - 2, y2c - 2)

        if (time.time() - last_midas_calib) > 0.2 and z_calib > 0 and aruco_roi_c and phase.startswith("sample_"):
            boxes = yolo.detect(frame)
            if boxes:
                bbox_c = boxes[0]["bbox"]
                dm = midas.process(frame)
                last_midas_calib = time.time()

                m_rim  = midas.get_rim_depth(dm, bbox_c)
                m_tray = midas.get_tray_depth(dm, aruco_roi_c)

                if m_rim > 0 and m_tray > 0:
                    if phase == "sample_c1":
                        r1_samples.append(m_rim / m_tray)
                        z_samples.append(z_calib)
                    elif phase == "sample_c2":
                        r2_samples.append(m_rim / m_tray)
            else:
                last_midas_calib = time.time()

        if phase == "warmup_c1" and elapsed >= CALIB_WARMUP:
            phase = "sample_c1"; calib_start = time.time(); elapsed = 0.0
        elif phase == "sample_c1":
            if len(r1_samples) >= 8:
                phase = "swap_c2"; calib_start = time.time()
        elif phase == "warmup_c2" and elapsed >= CALIB_WARMUP:
            phase = "sample_c2"; calib_start = time.time(); elapsed = 0.0
        elif phase == "sample_c2":
            if len(r2_samples) >= 8:
                if len(r1_samples) < 3 or len(r2_samples) < 3:
                    print(f"[CALIB] Failed sampling at Z-pos {pos_idx+1}. Aborting.")
                    return

                R1_avg = float(np.median(r1_samples))
                R2_avg = float(np.median(r2_samples))
                Z_avg = float(np.median(z_samples))

                # Calculate m_i and c_i
                # H1/Z = m*R1 + c  and  H2/Z = m*R2 + c
                Y1 = true_height / Z_avg
                Y2 = true_height_2 / Z_avg

                dR = R2_avg - R1_avg
                if abs(dR) < 0.005:
                    print("[CALIB] ERROR: Both cups registered identical depth ratios.")
                    return

                m_i = (Y2 - Y1) / dR
                c_i = Y1 - m_i * R1_avg

                m_pts.append(m_i); c_pts.append(c_i); Z_pts.append(Z_avg)
                print(f"[CALIB] Z-Pos {pos_idx+1}/{n_positions} committed: Z={Z_avg:.1f}cm, m={m_i:.4f}, c={c_i:.4f}")

                pos_idx += 1
                r1_samples.clear(); r2_samples.clear(); z_samples.clear()

                if pos_idx >= n_positions:
                    phase = "done"
                else:
                    phase = "swap_z"

        disp_c = frame.copy()
        if aruco_results: disp_c = aruco.annotate_frame(disp_c, aruco_results)
        if boxes:
            for b in boxes:
                x1b, y1b, x2b, y2b = b["bbox"]
                cv2.rectangle(disp_c, (x1b, y1b), (x2b, y2b), (0, 255, 80), 4)

        S = 2.5
        panel_w, panel_h = int(535 * S), int(105 * S)
        cv2.rectangle(disp_c, (25, 25), (25 + panel_w, 25 + panel_h), (20, 20, 40), -1)
        cv2.rectangle(disp_c, (25, 25), (25 + panel_w, 25 + panel_h), (0, 220, 120), 3)

        cv2.putText(disp_c, f"[ArUco: {'OK' if z_calib>0 else 'NO'}] [pos: {pos_idx+1}/{n_positions}]", (int(45*S), int(115*S)), cv2.FONT_HERSHEY_SIMPLEX, 0.45 * S, (200, 200, 200), 3)

        if phase.startswith("warmup_c"):
            idx = 1 if "c1" in phase else 2
            H_t = true_height if idx == 1 else true_height_2
            pct = min(100, int((elapsed / CALIB_WARMUP) * 100))
            cv2.putText(disp_c, f"BILATERAL GRID: Warming Up Cup {idx} ... {pct}%", (int(45*S), int(60*S)), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * S, (0, 220, 255), 3)
            cv2.putText(disp_c, f"Place {H_t}cm cup. Keep still.", (int(45*S), int(90*S)), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * S, (150, 220, 255), 2)
        elif phase.startswith("sample_c"):
            idx = 1 if "c1" in phase else 2
            cnt = len(r1_samples) if idx == 1 else len(r2_samples)
            cv2.putText(disp_c, f"BILATERAL GRID: Sampling Cup {idx} ({cnt}/8)", (int(45*S), int(60*S)), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * S, (0, 255, 180), 3)
            cv2.putText(disp_c, f"Z_tray = {z_calib:.1f} cm", (int(45*S), int(90*S)), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * S, (150, 255, 200), 2)
        elif phase == "swap_c2":
            cv2.rectangle(disp_c, (25, 25), (25 + panel_w, 25 + panel_h), (80, 50, 150), -1)
            cv2.putText(disp_c, f"SWAP TO TALL CUP ({true_height_2} cm)", (int(45*S), int(60*S)), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * S, (255, 255, 255), 4)
            cv2.putText(disp_c, "Press SPACE to scan cup 2.", (int(45*S), int(95*S)), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * S, (200, 220, 255), 2)
        elif phase == "swap_z":
            cv2.rectangle(disp_c, (25, 25), (25 + panel_w, 25 + panel_h), (40, 40, 150), -1)
            cv2.putText(disp_c, "MOVE NOZZLE TO DIFFERENT HEIGHT", (int(45*S), int(60*S)), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * S, (255, 255, 255), 4)
            cv2.putText(disp_c, f"Wait for focus. PLACE {true_height}cm CUP. SPACE.", (int(45*S), int(95*S)), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * S, (200, 220, 255), 2)

        if not headless:
            if gui:
                gui.update_image(disp_c)
                key = gui.get_key()
            else:
                cv2.imshow("ArUco + MiDaS | Cup Height Estimator", disp_c)
                key = cv2.waitKey(1) & 0xFF
        else:
            key = -1
            
        if key == 27: cap.release(); cv2.destroyAllWindows(); sys.exit(0)
        if key == ord(' '):
            if phase == "swap_c2":
                phase = "warmup_c2"; calib_start = time.time()
            elif phase == "swap_z":
                phase = "warmup_c1"; calib_start = time.time()

    # Fit m(Z) and c(Z)
    deg = min(len(Z_pts)-1, 2)
    poly_m = np.polyfit(Z_pts, m_pts, deg=deg).tolist()
    poly_c = np.polyfit(Z_pts, c_pts, deg=deg).tolist()

    cs.save_calibration_6p(poly_m, poly_c, Z_pts, true_height, true_height_2)
    calib_data = {"type": 6, "poly_m": poly_m, "poly_c": poly_c}
    print("[CALIB] ✅ Bilateral Z-Grid calibration done! Entering LIVE mode.\n")

    return calib_data


def run_calib_analytic(get_frame, cap, aruco, yolo, midas, headless, true_height, true_height_2, gui=None):
    print("━" * 55)
    print("  ⚙  UNIVERSAL ANALYTIC GEOMETRY (Type 7 - Fast & Perfect)")
    print(f"     Cup 1 height : {true_height} cm")
    print(f"     Cup 2 height : {true_height_2} cm")
    print("━" * 55)

    CALIB_SEC = 5.0
    phase = "warmup_1"
    calib_start = time.time()

    y_samples_1, y_samples_2 = [], []  # Storing Y = bbox_h * (Z - H_true)
    last_det = 0.0
    boxes = None

    while phase != "done":
        ret, frame = get_frame()
        if not ret: time.sleep(0.05); continue
        elapsed = time.time() - calib_start

        if (time.time() - last_det) > 1.0: boxes = None

        aruco_results = aruco.detect(frame)
        z_calib = 0.0
        if aruco_results:
            best = aruco.get_best_distance(aruco_results)
            if best: z_calib = best["distance_cm"]

        if (time.time() - last_det) > 0.1 and z_calib > 0 and phase.startswith("sample_"):
            boxes = yolo.detect(frame)
            last_det = time.time()
            if boxes:
                x1b, y1b, x2b, y2b = boxes[0]["bbox"]
                bbox_h = float(y2b - y1b)
                if bbox_h > 5:
                    if phase == "sample_1":
                        # Y = bbox_h * (Z - H_true)
                        Y = bbox_h * (z_calib - true_height)
                        y_samples_1.append(Y)
                    elif phase == "sample_2":
                        Y = bbox_h * (z_calib - true_height_2)
                        y_samples_2.append(Y)

        if phase == "warmup_1" and elapsed >= 3.0:
            phase = "sample_1"; calib_start = time.time(); elapsed = 0.0
        elif phase == "sample_1" and len(y_samples_1) >= 30:
            phase = "swap"; calib_start = time.time()
        elif phase == "warmup_2" and elapsed >= 3.0:
            phase = "sample_2"; calib_start = time.time(); elapsed = 0.0
        elif phase == "sample_2" and len(y_samples_2) >= 30:
            Y1_avg = float(np.median(y_samples_1))
            Y2_avg = float(np.median(y_samples_2))

            # We have Y1 = A + B * H1
            #         Y2 = A + B * H2
            # B = (Y2 - Y1) / (H2 - H1)
            dH = true_height_2 - true_height
            if abs(dH) < 0.1:
                print("[CALIB] ERROR: Cups must have different heights!")
                return
            B = (Y2_avg - Y1_avg) / dH
            A = Y1_avg - B * true_height

            cs.save_calibration_7(A, B, true_height, true_height_2)
            calib_data = {"type": 7, "A": A, "B": B}
            print(f"[CALIB] A = {A:.2f}, B = {B:.2f}")
            print("[CALIB] ✅ Analytic Geometry Done! Entering LIVE mode.\n")
            phase = "done"

        disp_c = frame.copy()
        if aruco_results: disp_c = aruco.annotate_frame(disp_c, aruco_results)
        if boxes:
            for b in boxes:
                x1b, y1b, x2b, y2b = b["bbox"]
                cv2.rectangle(disp_c, (x1b, y1b), (x2b, y2b), (0, 255, 80), 4)

        S = 2.5
        panel_w, panel_h = int(535 * S), int(105 * S)
        cv2.rectangle(disp_c, (25, 25), (25 + panel_w, 25 + panel_h), (20, 20, 60), -1)
        cv2.rectangle(disp_c, (25, 25), (25 + panel_w, 25 + panel_h), (90, 90, 90), 3)

        if phase == "warmup_1" or phase == "sample_1":
            cv2.putText(disp_c, f"YOLO ANALYTIC: Cup 1 ({true_height}cm)", (int(45*S), int(60*S)), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * S, (0, 220, 255), 3)
            if phase == "sample_1":
                cv2.putText(disp_c, f"Sampling [{len(y_samples_1)}/30]", (int(45*S), int(95*S)), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * S, (0, 255, 150), 3)
        elif phase == "swap":
            cv2.rectangle(disp_c, (25, 25), (25 + panel_w, 25 + panel_h), (80, 50, 150), -1)
            cv2.putText(disp_c, f"SWAP TO CUP 2 ({true_height_2} cm)", (int(45*S), int(60*S)), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * S, (255, 255, 255), 4)
            cv2.putText(disp_c, "Press SPACE when ready.", (int(45*S), int(95*S)), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * S, (200, 220, 255), 2)
        elif phase == "warmup_2" or phase == "sample_2":
            cv2.putText(disp_c, f"YOLO ANALYTIC: Cup 2 ({true_height_2}cm)", (int(45*S), int(60*S)), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * S, (0, 220, 255), 3)
            if phase == "sample_2":
                cv2.putText(disp_c, f"Sampling [{len(y_samples_2)}/30]", (int(45*S), int(95*S)), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * S, (0, 255, 150), 3)
                cv2.putText(disp_c, f"Sampling [{len(y_samples_2)}/30]", (18, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 150), 1)

        if not headless:
            if gui:
                gui.update_image(disp_c)
                key = gui.get_key()
            else:
                cv2.imshow("ArUco + MiDaS | Cup Height Estimator", disp_c)
                key = cv2.waitKey(1) & 0xFF
        else:
            key = -1
            
        if key == 27: cap.release(); sys.exit(0)
        if phase == "swap" and key == ord(' '):
            phase = "warmup_2"; calib_start = time.time()

    return calib_data
