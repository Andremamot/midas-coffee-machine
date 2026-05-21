"""
run_aruco.py — CLI Entry Point untuk ArUco Marker Detection Experiment

3 mode operasi:
  --image PATH          → proses satu gambar
  --camera [INDEX]      → live webcam dengan overlay real-time
  --generate-marker     → generate gambar marker untuk di-print

Mode Fisheye (Moildev):
  --fisheye             → aktifkan undistortion lensa fisheye via Moildev
  --moil-camera-name    → profil kamera di camera_parameters.json
  --moil-zoom           → faktor zoom hybrid (default 2.0)
  --moil-mode           → mode Moildev: 0/1=Alpha/Beta, 2=Pitch/Yaw
  --moil-pitch, --moil-yaw → arah pandang kamera (derajat)

Contoh:
  python run_aruco.py --generate-marker
  python run_aruco.py --image foto_marker.jpg --marker-size 5.0
  python run_aruco.py --camera 0 --marker-size 5.0 --lock-focus

  # Fisheye (kamera syue_7730v1_6, resolusi 2592x1944):
  python run_aruco.py --camera 0 --marker-size 2.5 --fisheye \\
      --moil-camera-name syue_7730v1_6 --moil-zoom 2.0 --moil-mode 0

Kalibrasi focal length untuk ArUco fisheye:
  Saat --fisheye aktif, focal length ArUco secara otomatis di-override
  menggunakan adjusted focal length dari Moildev (parameter5 * calibRatio).
  Jika hasil jarak tidak akurat, sesuaikan --moil-zoom dan ukur ulang.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

import cv2

# Pastikan path bisa import lokal
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from aruco_detector import ArucoDetector

# ── Import Moildev (opsional, hanya jika --fisheye) ────────────────────────
_FUSION_DIR = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "07_midas_aruco_fusion"))
if _FUSION_DIR not in sys.path:
    sys.path.insert(0, _FUSION_DIR)

try:
    from core.moil_undistorter import MoilUndistorter
    _HAS_MOIL = True
except ImportError:
    _HAS_MOIL = False


def _print_result(results, source=""):
    """Print hasil deteksi dalam format JSON."""
    prefix = f"[{source}] " if source else ""
    print(f"\n{prefix}{'='*50}")
    for r in results:
        output = {
            "id": r["id"],
            "distance_cm": r["distance_cm"],
            "center": [round(r["center"][0], 1), round(r["center"][1], 1)],
            "euler_deg": r["euler_deg"],
            "reprojection_error": r["reprojection_error"],
        }
        print(json.dumps(output, indent=2, ensure_ascii=False))
    if not results:
        print("  No ArUco marker detected")
    print(f"{'='*50}\n")


# ── Mode: Single Image ─────────────────────────────────────────────────
def process_single_image(detector, image_path, output_dir, moil_undistorter=None):
    """Proses satu gambar dan simpan hasil."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Tidak bisa membaca gambar: {image_path}")
        return

    # Fisheye undistortion jika aktif
    if moil_undistorter is not None:
        img = moil_undistorter.undistort(img)
        print(f"[MOIL] Undistortion applied (zoom={moil_undistorter.zoom:.1f}x)")

    print(f"Memproses: {image_path}")
    results = detector.detect(img)
    _print_result(results, source=os.path.basename(image_path))

    annotated = detector.annotate_frame(img, results)
    basename = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join(output_dir, f"aruco_{basename}.jpg")
    cv2.imwrite(out_path, annotated)
    print(f"Visualisasi disimpan: {out_path}")


# ── Mode: Live Camera ───────────────────────────────────────────────────
def run_live_camera(detector, camera_index=0, lock_focus=False, focus_value=0,
                    moil_undistorter=None, cap_width=640, cap_height=480):
    """Live camera mode dengan overlay real-time."""

    SCREENSHOT_DIR = os.path.join(_SCRIPT_DIR, "results", "live_cam")
    os.makedirs(SCREENSHOT_DIR, exist_ok=True)

    # Import camera_utils dari tray_detector jika ada, fallback ke dasar
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Error: Tidak bisa membuka kamera index {camera_index}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_height)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[CAMERA] Resolusi: {actual_w}x{actual_h}")
    if moil_undistorter is not None:
        print(f"[MOIL] Fisheye undistortion AKTIF — zoom={moil_undistorter.zoom:.1f}x")

    if lock_focus:
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        cap.set(cv2.CAP_PROP_FOCUS, focus_value)
        actual_focus = cap.get(cv2.CAP_PROP_FOCUS)
        print(f"[CAMERA] ✅ Focus LOCKED: auto-focus OFF, focus={actual_focus}")

    # Warm-up: drain V4L2 buffer agar frame pertama tidak timeout
    print("[CAMERA] Warming up...")
    warmed = False
    for _ in range(30):
        ret, _ = cap.read()
        if ret:
            warmed = True
            break
        time.sleep(0.05)
    if not warmed:
        print(f"Error: Kamera index {camera_index} tidak bisa memberikan frame. "
              f"Pastikan kamera terhubung dan tidak dipakai aplikasi lain.")
        cap.release()
        return

    print(f"\nLive camera started (index={camera_index})")
    print("Press 'q' to exit | 's' screenshot")
    if lock_focus:
        print(f"🔒 Focus locked at value={focus_value}")

    fps_counter = 0
    fps_time = time.time()
    fps_display = 0.0

    # Statistics & Reporting
    stats_total = 0
    stats_detected = 0
    stats_d_sum = 0.0
    stats_d_min = float('inf')
    stats_d_max = float('-inf')
    
    # History for reporting
    distance_history = []
    frame_indices = []
    screenshot_paths = []
    
    # Video Recording State
    VIDEO_DIR = os.path.join(_SCRIPT_DIR, "results", "video")
    os.makedirs(VIDEO_DIR, exist_ok=True)
    is_recording = False
    video_writer = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame from camera")
            break

        # ── Fisheye Undistortion (jika Moil aktif) ────────────────────────
        if moil_undistorter is not None:
            frame = moil_undistorter.undistort(frame)
        elif frame.shape[1] > 640:
            frame = cv2.resize(frame, (640, 480))

        t_start = time.time()
        results = detector.detect(frame)
        t_process = time.time() - t_start

        # Update statistics
        stats_total += 1
        if results:
            stats_detected += 1
            # Gunakan median dari semua marker (filtered by reproj error)
            best = detector.get_best_distance(results)
            if best:
                d_val = best["distance_cm"]
                stats_d_sum += d_val
                stats_d_min = min(stats_d_min, d_val)
                stats_d_max = max(stats_d_max, d_val)
                
                # Record history
                distance_history.append(d_val)
                frame_indices.append(stats_total)

        # Annotate markers
        annotated = detector.annotate_frame(frame, results)

        # FPS counter
        fps_counter += 1
        elapsed = time.time() - fps_time
        if elapsed >= 1.0:
            fps_display = fps_counter / elapsed
            fps_counter = 0
            fps_time = time.time()

        # ── UI Overlay ──────────────────────────────────────────────────
        h, w = annotated.shape[:2]

        # Get best distance for this frame
        best = detector.get_best_distance(results) if results else None
        current_dist = best["distance_cm"] if best else None

        # Update running average for display
        session_avg = round(stats_d_sum / stats_detected, 2) if stats_detected > 0 else 0.0

        # ── Top-left info panel (semi-transparent) ───────────────────────
        panel_w, panel_h = 220, 130
        overlay = annotated.copy()
        cv2.rectangle(overlay, (8, 8), (8 + panel_w, 8 + panel_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.55, annotated, 0.45, 0, annotated)

        # Garis border panel
        cv2.rectangle(annotated, (8, 8), (8 + panel_w, 8 + panel_h), (80, 80, 80), 1)

        # ── Live Distance (large) ─────────────────────────────────────────
        if current_dist is not None:
            # Warna berdasarkan keyakinan: hijau jika banyak marker, kuning jika sedikit
            n_valid = best["used_count"] if best else 0
            if n_valid >= 4:
                dist_color = (0, 255, 128)   # Hijau terang
            elif n_valid >= 2:
                dist_color = (0, 220, 255)   # Kuning-cyan
            else:
                dist_color = (0, 150, 255)   # Oranye (hati-hati)

            cv2.putText(annotated, "DISTANCE",
                        (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 160), 1)
            cv2.putText(annotated, f"{current_dist:.2f} cm",
                        (18, 62), cv2.FONT_HERSHEY_DUPLEX, 0.95, dist_color, 2)
        else:
            cv2.putText(annotated, "DISTANCE",
                        (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 100, 100), 1)
            cv2.putText(annotated, "-- cm",
                        (18, 62), cv2.FONT_HERSHEY_DUPLEX, 0.95, (80, 80, 80), 2)

        # Divider
        cv2.line(annotated, (14, 72), (220, 72), (70, 70, 70), 1)

        # ── Stats row ─────────────────────────────────────────────────────
        n_markers = len(results)
        markers_color = (0, 255, 0) if n_markers > 0 else (80, 80, 255)
        
        cv2.putText(annotated, f"Markers : {n_markers}",
                    (18, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        cv2.putText(annotated, f"Avg     : {session_avg:.2f} cm",
                    (18, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
        cv2.putText(annotated, f"Frames  : {stats_total}",
                    (18, 126), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)

        # ── Bottom status bar ─────────────────────────────────────────────
        bar_y = h - 28
        cv2.rectangle(annotated, (0, bar_y), (w, h), (15, 15, 15), -1)
        fps_txt = f"FPS: {fps_display:.1f}   |   Process: {t_process*1000:.0f} ms   |   [S] Screenshot   [R] Record   [Q] Quit"
        cv2.putText(annotated, fps_txt, (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (140, 200, 140), 1)

        # ── Recording Indicator ───────────────────────────────────────────
        if is_recording:
            # Efek blinking: Tampil 0.5s, sembunyi 0.5s
            if int(time.time() * 2) % 2 == 0:
                cv2.circle(annotated, (w - 65, 25), 6, (0, 0, 255), -1)
                cv2.putText(annotated, "REC", (w - 50, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            if video_writer is not None:
                video_writer.write(annotated)

        cv2.imshow("ArUco Distance Detector", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Live camera dihentikan.")
            if is_recording and video_writer is not None:
                video_writer.release()
                print("Recording stopped automatically.")
            break
        elif key == ord('s'):
            timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
            ss_filename = f"aruco_capture_{timestamp}.jpg"
            ss_path = os.path.join(SCREENSHOT_DIR, ss_filename)
            cv2.imwrite(ss_path, annotated)
            screenshot_paths.append(ss_path)
            _print_result(results, source="screenshot")
            print(f"Screenshot saved: {ss_path}")
        elif key == ord('r'):
            if not is_recording:
                # Start recording
                timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
                vid_filename = f"aruco_record_{timestamp}.mp4"
                vid_path = os.path.join(VIDEO_DIR, vid_filename)
                
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                fps_cap = cap.get(cv2.CAP_PROP_FPS)
                if fps_cap <= 0: 
                    fps_cap = 30.0
                video_writer = cv2.VideoWriter(vid_path, fourcc, fps_cap, (w, h))
                is_recording = True
                print(f"🔴 Memulai rekaman video: {vid_path}")
            else:
                # Stop recording
                is_recording = False
                if video_writer is not None:
                    video_writer.release()
                    video_writer = None
                    print("⏹ Berhenti merekam.")

    if is_recording and video_writer is not None:
        video_writer.release()
    cap.release()
    cv2.destroyAllWindows()

    print("\n" + "="*50)
    try:
        gt_input = input("Enter ground truth distance in cm [Press Enter to skip]: ").strip()
    except (EOFError, KeyboardInterrupt):
        gt_input = ""

    # Calculate final stats
    avg_d = stats_d_sum / stats_detected if stats_detected > 0 else 0.0
    
    import re
    # Clean up non-numeric characters (e.g. arrow key escape sequences like ^[[A)
    gt_input_clean = re.sub(r"[^\d.,-]", "", gt_input)
    ground_truth = None
    if gt_input_clean:
        try:
            ground_truth = float(gt_input_clean.replace(',', '.'))
        except ValueError:
            print(f"⚠ Invalid input ('{gt_input}'), ignoring ground truth.")


    # Generate report data
    report_stats = {
        "total_frames": stats_total,
        "detected_frames": stats_detected,
        "avg_distance": round(avg_d, 2),
        "min_distance": round(stats_d_min, 2) if stats_detected > 0 else 0.0,
        "max_distance": round(stats_d_max, 2) if stats_detected > 0 else 0.0,
        "spread": round(stats_d_max - stats_d_min, 2) if stats_detected > 0 else 0.0,
        "detection_rate": round(stats_detected / max(1, stats_total) * 100, 1),
        "dictionary": detector.dictionary_name,
        "marker_size_cm": detector.marker_size_cm,
        "focal_length": round(detector.camera_matrix[0, 0], 1),
        "lock_focus": lock_focus,
        "ground_truth": ground_truth
    }

    # Print to terminal
    print(f"\n{'='*50}")
    print("📐 ARUCO MARKER SESSION REPORT 📐")
    print(f"{'='*50}")
    print(f"Total frames processed  : {report_stats['total_frames']}")
    if stats_detected > 0:
        print(f"Frames with marker      : {report_stats['detected_frames']} ({report_stats['detection_rate']}%)")
        print(f"Average Distance        : {report_stats['avg_distance']} cm")
        print(f"Minimum Distance        : {report_stats['min_distance']} cm")
        print(f"Maximum Distance        : {report_stats['max_distance']} cm")
        print(f"Distance Range (spread) : {report_stats['spread']} cm")
        print("-" * 50)
        print(f"🎯 FINAL ESTIMATED DISTANCE: {report_stats['avg_distance']} cm 🎯")
    else:
        print("Frames with marker      : 0 (No marker detected)")
    print(f"{'='*50}\n")
    
    # Generate Markdown Report
    _generate_markdown_report(report_stats, distance_history, frame_indices, screenshot_paths)


def _generate_markdown_report(stats, distances, frames, screenshots):
    """Generate Markdown report with matplotlib chart."""
    import matplotlib
    matplotlib.use('Agg')  # Bypass GTK/Wayland GUI rendering
    import matplotlib.pyplot as plt
    import shutil

    # 1. Create timestamped folder
    # Format: YYYY-MM-DD_HH-mm-ss
    timestamp_folder = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    report_dir = os.path.join(_SCRIPT_DIR, "results", "report", timestamp_folder)
    os.makedirs(report_dir, exist_ok=True)
    
    print(f"Generating report in: {report_dir}")

    # 2. Generate distance chart
    chart_filename = "distance_chart.png"
    chart_path = os.path.join(report_dir, chart_filename)
    
    if distances:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
        
        # Line plot
        ax1.plot(frames, distances, label='Distance (cm)', color='#1f77b4', linewidth=1.5)
        ax1.axhline(y=stats['avg_distance'], color='r', linestyle='--', label=f"Avg: {stats['avg_distance']}cm")
        ax1.set_title(f"ArUco Distance Tracking Session\n({timestamp_folder})")
        ax1.set_ylabel("Distance (cm)")
        ax1.grid(True, linestyle=':', alpha=0.7)
        ax1.legend()
        
        # Scatter plot
        ax2.scatter(frames, distances, s=10, alpha=0.6, color='#2ca02c', label='Distance per frame')
        ax2.axhline(y=stats['avg_distance'], color='r', linestyle='--', label=f"Avg: {stats['avg_distance']}cm")
        ax2.set_title("Scatter Plot: Distance per Detected Frame")
        ax2.set_xlabel("Frame Number")
        ax2.set_ylabel("Distance (cm)")
        ax2.grid(True, linestyle=':', alpha=0.7)
        ax2.legend()
        
        # Adjust Y-axis to see variations better
        if stats['spread'] > 0:
            margin = max(1.0, stats['spread'] * 0.5)
            ax1.set_ylim(stats['min_distance'] - margin, stats['max_distance'] + margin)
            ax2.set_ylim(stats['min_distance'] - margin, stats['max_distance'] + margin)
            
        plt.tight_layout()
        plt.savefig(chart_path)
        plt.close()
        print(f"✅ Chart saved: {chart_filename}")


    # 3. Copy screenshots to report folder for local linking
    local_screenshots = []
    if screenshots:
        ss_subfolder = os.path.join(report_dir, "screenshots")
        os.makedirs(ss_subfolder, exist_ok=True)
        for i, src in enumerate(screenshots):
            fname = os.path.basename(src)
            dst = os.path.join(ss_subfolder, fname)
            shutil.copy2(src, dst)
            local_screenshots.append(os.path.join("screenshots", fname))
        print(f"✅ {len(screenshots)} screenshots copied to report folder")

    # 4. Write Markdown file
    md_path = os.path.join(report_dir, "report.md")
    
    # Calculate Std Dev and precision metrics
    import numpy as np
    arr = np.array(distances) if distances else np.array([0.0])
    std_dev = round(float(np.std(arr)), 2) if distances else 0.0
    median_dist = round(float(np.median(arr)), 2) if distances else 0.0
    p5  = round(float(np.percentile(arr, 5)),  2) if distances else 0.0
    p95 = round(float(np.percentile(arr, 95)), 2) if distances else 0.0
    precision_error = round(p95 - p5, 2)
    
    with open(md_path, "w") as f:
        f.write(f"# ArUco Depth Verification: Session Report\n")
        f.write(f"Generated on: {timestamp_folder.replace('_', ' ')}\n\n")
        
        f.write(f"## 1. Session Parameters\n")
        f.write(f"Parameters used during this ArUco detection session:\n\n")
        f.write(f"| Parameter | Value |\n")
        f.write(f"| :--- | :--- |\n")
        f.write(f"| **ArUco Dictionary** | {stats.get('dictionary', 'N/A')} |\n")
        f.write(f"| **Physical Marker Size** | {stats.get('marker_size_cm', 'N/A')} cm |\n")
        f.write(f"| **Focal Length (K)** | {stats.get('focal_length', 'N/A')} px |\n")
        f.write(f"| **Lock Focus** | {'ON' if stats.get('lock_focus') else 'OFF'} |\n\n")

        f.write(f"## 2. Global Stability Summary\n")
        f.write(f"Statistical summary of distance measurements gathered over {stats['total_frames']} frames:\n\n")
        f.write(f"| Metric | Value | Description |\n")
        f.write(f"| :--- | :--- | :--- |\n")
        f.write(f"| **Average Distance** | **{stats['avg_distance']} cm** | Mean of all valid detections. |\n")
        f.write(f"| **Median Distance (P50)** | **{median_dist} cm** | Most representative single value. |\n")
        f.write(f"| **Precision Error (P95−P5)** | **{precision_error} cm** | 90% of readings fall within this range. |\n")
        
        gt = stats.get('ground_truth')
        if gt is not None and gt > 0:
            error_val = abs(median_dist - gt)
            error_pct = (error_val / gt) * 100
            f.write(f"| **Absolute Error** | **{error_pct:.2f}%** | Variance from true distance ({gt} cm, err: {error_val:.2f} cm). |\n")
            
        f.write(f"| **Standard Deviation ($\\sigma$)** | {std_dev} cm | Consistency of the detection. |\n")
        f.write(f"| **Minimum Distance** | {stats['min_distance']} cm | Closest measured point. |\n")
        f.write(f"| **Maximum Distance** | {stats['max_distance']} cm | Furthest measured point. |\n")
        f.write(f"| **Distance Spread** | {stats['spread']} cm | Range between min and max. |\n")
        f.write(f"| **Detection Rate** | {stats['detection_rate']}% | Percentage of frames with marker. |\n\n")
        
        if distances:
            f.write(f"## 3. Visual Evidence\n")
            f.write(f"### Distance Tracking Chart\n")
            f.write(f"![Distance Chart]({chart_filename})\n\n")
        
        if local_screenshots:
            f.write(f"### Captured Screenshots\n")
            for ss_rel_path in local_screenshots:
                f.write(f"![Screenshot]({ss_rel_path})\n\n")

        f.write(f"## 4. Conclusion\n")
        if stats['detection_rate'] > 80:
            conclusion = "The session shows high detection stability and consistent distance reporting."
        elif stats['detection_rate'] > 50:
            conclusion = "The session shows moderate detection reliability. Consider adjusting lighting or focus."
        else:
            conclusion = "Low detection rate observed. Accuracy may be compromised."
        
        f.write(f"{conclusion}\n")

    print(f"✅ Enhanced Markdown report generated: {md_path}")

    # 5. Write JSON report
    _generate_json_report(report_dir, timestamp_folder, stats, distances, frames, local_screenshots, std_dev)


def _generate_json_report(report_dir, timestamp_folder, stats, distances, frames, local_screenshots, std_dev):
    """Simpan data sesi dalam format JSON terstruktur (compact)."""
    import json
    import numpy as np

    MAX_SAMPLES = 50  # Maksimum titik yang disimpan untuk grafik

    # Downsampling: ambil N titik merata dari seluruh data
    if distances:
        n = len(distances)
        if n <= MAX_SAMPLES:
            samples = [round(v, 2) for v in distances]
            stride = 1
        else:
            stride = n // MAX_SAMPLES
            indices = list(range(0, n, stride))[:MAX_SAMPLES]
            samples = [round(distances[i], 2) for i in indices]

        # Percentiles untuk analisis distribusi
        arr = np.array(distances)
        percentiles = {
            "p5":  round(float(np.percentile(arr, 5)),  2),
            "p25": round(float(np.percentile(arr, 25)), 2),
            "p50": round(float(np.percentile(arr, 50)), 2),
            "p75": round(float(np.percentile(arr, 75)), 2),
            "p95": round(float(np.percentile(arr, 95)), 2),
        }
    else:
        samples, stride = [], 1
        percentiles = {"p5": 0, "p25": 0, "p50": 0, "p75": 0, "p95": 0}

    json_data = {
        "session_timestamp": timestamp_folder,
        "parameters": {
            "dictionary": stats.get("dictionary", "N/A"),
            "marker_size_cm": stats.get("marker_size_cm", "N/A"),
            "focal_length_px": stats.get("focal_length", "N/A"),
            "lock_focus": stats.get("lock_focus", False),
        },
        "summary": {
            "total_frames": stats["total_frames"],
            "detected_frames": stats["detected_frames"],
            "detection_rate_pct": stats["detection_rate"],
            "ground_truth_cm": stats.get("ground_truth"),
            "avg_distance_cm": stats["avg_distance"],
            "std_dev_cm": std_dev,
            "min_distance_cm": stats["min_distance"],
            "max_distance_cm": stats["max_distance"],
            "spread_cm": stats["spread"],
            "percentiles": percentiles,
        },
        "distance_samples": samples,
        "sample_stride": stride,
        "screenshots": local_screenshots,
    }

    json_path = os.path.join(report_dir, "session_data.json")
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

    size_kb = os.path.getsize(json_path) / 1024
    print(f"✅ JSON report generated: {json_path} ({size_kb:.1f} KB)")



# ── Helper: Kalibrasi Focal Length Empiris ─────────────────────────────────
def _run_focal_calibration(detector, args, moil_undistorter):
    """
    Mode kalibrasi focal length.

    Cara pakai:
      1. Taruh marker ArUco di jarak TEPAT (ukur dengan penggaris)
      2. Jalankan dengan --calibrate-focal <jarak_cm>
      3. Program membuka kamera, deteksi marker, lalu hitung fx yang benar
      4. Salin nilai fx yang dicetak → pakai --focal-length di sesi berikutnya
    """
    import numpy as np

    true_dist_cm = args.calibrate_focal
    print(f"\n{'='*55}")
    print(f"  🎯  MODE KALIBRASI FOCAL LENGTH")
    print(f"  Jarak referensi: {true_dist_cm} cm")
    print(f"  Pastikan marker ArUco {args.marker_size} cm ada di jarak TEPAT {true_dist_cm} cm")
    print(f"  Tekan SPACE untuk capture, Q untuk keluar")
    print(f"{'='*55}\n")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("Error: Tidak bisa membuka kamera")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.cap_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.cap_height)

    fx_samples = []
    current_fx = detector.camera_matrix[0, 0]

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if moil_undistorter is not None:
            frame = moil_undistorter.undistort(frame)

        results = detector.detect(frame)
        annotated = detector.annotate_frame(frame, results)

        # Overlay info
        h, w = annotated.shape[:2]
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, 0), (w, 60), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.7, annotated, 0.3, 0, annotated)

        if results:
            best = detector.get_best_distance(results)
            if best:
                measured = best["distance_cm"]
                # Hitung fx yang seharusnya: fx_benar = fx × (true / measured)
                fx_correct = current_fx * (true_dist_cm / measured)
                info = (f"Measured={measured:.1f}cm  "
                        f"True={true_dist_cm:.1f}cm  "
                        f"fx_benar={fx_correct:.1f}px  "
                        f"[SPACE] simpan")
                cv2.putText(annotated, info, (10, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 120), 2)
        else:
            cv2.putText(annotated, "Marker tidak terdeteksi — arahkan ke marker",
                        (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 2)

        n_samples = len(fx_samples)
        cv2.putText(annotated, f"Samples: {n_samples}  [Q] selesai",
                    (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        cv2.imshow("Kalibrasi Focal Length", annotated)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' ') and results:
            best = detector.get_best_distance(results)
            if best:
                measured = best["distance_cm"]
                fx_correct = current_fx * (true_dist_cm / measured)
                fx_samples.append(fx_correct)
                print(f"  [Sample {len(fx_samples)}] measured={measured:.2f}cm → fx_benar={fx_correct:.1f}px")
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if fx_samples:
        fx_median = float(np.median(fx_samples))
        print(f"\n{'='*55}")
        print(f"  ✅  HASIL KALIBRASI FOCAL LENGTH")
        print(f"  Samples   : {len(fx_samples)}")
        print(f"  fx median : {fx_median:.1f} px")
        print(f"\n  Gunakan perintah ini di sesi berikutnya:")
        base_cmd = (
            f"python run_aruco.py --camera {args.camera} "
            f"--marker-size {args.marker_size}"
        )
        if args.fisheye:
            base_cmd += (
                f" --fisheye --moil-camera-name {args.moil_camera_name} "
                f"--moil-zoom {args.moil_zoom} --moil-mode {args.moil_mode} "
                f"--cap-width {args.cap_width} --cap-height {args.cap_height}"
            )
        base_cmd += f" --focal-length {fx_median:.1f}"
        print(f"\n  {base_cmd}")
        print(f"{'='*55}\n")
        # Simpan hasil kalibrasi ke file JSON agar bisa di-load otomatis oleh 07_midas_aruco_fusion
        import json
        from datetime import datetime
        if args.fisheye:
            aruco_calib_name = f"calibration_aruco_fisheye_{args.cup_profile}.json"
        else:
            aruco_calib_name = f"calibration_aruco_{args.cup_profile}.json"
        aruco_calib_path = os.path.join(_FUSION_DIR, aruco_calib_name)

        calib_data = {
            "type": "aruco_calibration",
            "focal_length_px": fx_median,
            "camera_name": args.moil_camera_name if args.fisheye else "standard",
            "zoom": args.moil_zoom if args.fisheye else 1.0,
            "mode": args.moil_mode if args.fisheye else 0,
            "calibrated_at": datetime.now().isoformat()
        }
        try:
            with open(aruco_calib_path, "w") as f:
                json.dump(calib_data, f, indent=2)
            print(f"[CALIB] 💾 Tersimpan file kalibrasi ArUco → {aruco_calib_path}")
        except Exception as e:
            print(f"[ERROR] Gagal menyimpan kalibrasi ArUco: {e}")
    else:
        print("Tidak ada sample yang diambil.")


# ══════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="ArUco Marker Detection Experiment — Distance Estimation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Contoh penggunaan:
  # 1. Generate marker (print ini di kertas)
  python run_aruco.py --generate-marker
  python run_aruco.py --generate-marker --marker-id 5 --dictionary DICT_5X5_50

  # 2. Test pada gambar
  python run_aruco.py --image foto_marker.jpg --marker-size 5.0

  # 3. Live camera
  python run_aruco.py --camera 0 --marker-size 5.0 --lock-focus
        """,
    )

    # Mode operasi
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--image", type=str,
                      help="Path ke satu file gambar")
    mode.add_argument("--camera", nargs="?", const=0, type=int,
                      help="Index kamera untuk live mode (default: 0)")
    mode.add_argument("--generate-marker", action="store_true",
                      help="Generate gambar ArUco marker untuk di-print")

    # ArUco options
    parser.add_argument("--marker-size", type=float, default=5.0,
                        help="Ukuran fisik sisi marker dalam cm (default: 5.0)")
    parser.add_argument("--dictionary", type=str, default="DICT_4X4_50",
                        help="ArUco dictionary (default: DICT_4X4_50)")
    parser.add_argument("--marker-id", type=int, default=0,
                        help="Marker ID untuk generate (default: 0)")
    parser.add_argument("--marker-px", type=int, default=400,
                        help="Ukuran marker dalam piksel untuk generate (default: 400)")

    # Camera options
    parser.add_argument("--lock-focus", action="store_true",
                        help="Lock camera focus (matikan auto-focus)")
    parser.add_argument("--focus-value", type=int, default=0,
                        help="Nilai fokus tetap saat --lock-focus aktif (default: 0)")
    parser.add_argument("--cap-width", type=int, default=640,
                        help="Lebar resolusi capture USB (default: 640; pakai 2592 untuk fisheye 5MP)")
    parser.add_argument("--cap-height", type=int, default=480,
                        help="Tinggi resolusi capture USB (default: 480; pakai 1944 untuk fisheye 5MP)")

    # Calibration
    parser.add_argument("--params", type=str, default=None,
                        help="Path ke calibration_params.yml")

    # Output
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Direktori output untuk visualisasi")

    # ── Fisheye / Moildev options ─────────────────────────────────────────
    parser.add_argument("--fisheye", action="store_true",
                        help="Aktifkan undistortion lensa fisheye via Moildev")
    parser.add_argument("--moil-camera-name", type=str, default="syue_7730v1_6",
                        help="Nama profil kamera di camera_parameters.json (default: syue_7730v1_6)")
    parser.add_argument("--moil-zoom", type=float, default=2.0,
                        help="Faktor zoom hybrid Moildev (default: 2.0)")
    parser.add_argument("--moil-mode", type=int, default=0,
                        help="Mode Moildev: 0/1=Alpha/Beta (default), 2=Pitch/Yaw")
    parser.add_argument("--moil-pitch", type=float, default=0.0,
                        help="Pitch/Alpha arah pandang dalam derajat (default: 0.0)")
    parser.add_argument("--moil-yaw", type=float, default=0.0,
                        help="Yaw/Beta arah pandang dalam derajat (default: 0.0)")
    parser.add_argument("--moil-params-json", type=str, default=None,
                        help="Path ke camera_parameters.json Moildev (default: auto-detect)")
    parser.add_argument("--focal-length", type=float, default=None,
                        help="Override focal length ArUco secara manual (px). "
                             "Gunakan ini setelah kalibrasi empiris. "
                             "Jika tidak diset, diambil dari Moildev adjusted focal length.")
    parser.add_argument("--calibrate-focal", type=float, default=None, metavar="TRUE_DIST_CM",
                        help="Mode kalibrasi focal length: masukkan jarak sebenarnya (cm) ke marker. "
                             "Program akan menghitung fx yang tepat lalu mencetak hasilnya untuk disimpan.")
    parser.add_argument("--cup-profile", type=str, default="default",
                        help="Nama profil gelas (misal: short, tall) untuk membedakan file kalibrasi.")

    args = parser.parse_args()

    # ── Generate Marker Mode ─────────────────────────────────────────────
    if args.generate_marker:
        from generate_marker import generate_marker
        generate_marker(
            marker_id=args.marker_id,
            size_px=args.marker_px,
            dictionary_name=args.dictionary,
        )
        return

    # ── Detection Modes ──────────────────────────────────────────────────
    output_dir = args.output_dir or os.path.join(_SCRIPT_DIR, "results")
    os.makedirs(output_dir, exist_ok=True)

    print("Initializing ArUco detector...")
    detector = ArucoDetector(
        marker_size_cm=args.marker_size,
        dictionary_name=args.dictionary,
        params_path=args.params,
    )
    print("Detector siap.\n")

    # ── Inisialisasi Moildev (jika --fisheye) ────────────────────────────
    moil_undistorter = None
    if args.fisheye:
        if not _HAS_MOIL:
            print("[ERROR] --fisheye diberikan tapi MoilUndistorter tidak bisa diimport.")
            print("        Pastikan conda env midas-py310 aktif dan folder moildev/ ada.")
            sys.exit(1)

        # Auto-detect path camera_parameters.json
        if args.moil_params_json:
            cam_json = args.moil_params_json
        else:
            cam_json = os.path.join(_FUSION_DIR, "camera_parameters.json")
        if not os.path.isfile(cam_json):
            print(f"[ERROR] camera_parameters.json tidak ditemukan: {cam_json}")
            sys.exit(1)

        cap_w = args.cap_width
        cap_h = args.cap_height

        print(f"[MOIL] Inisialisasi MoilUndistorter...")
        print(f"[MOIL]   Camera : {args.moil_camera_name}")
        print(f"[MOIL]   Zoom   : {args.moil_zoom}x  Mode: {args.moil_mode}")
        print(f"[MOIL]   Pitch  : {args.moil_pitch}°  Yaw: {args.moil_yaw}°")
        print(f"[MOIL]   Stream : {cap_w}x{cap_h}")
        try:
            import numpy as np  # pastikan np tersedia di scope ini
            moil_undistorter = MoilUndistorter(
                json_path      = cam_json,
                camera_name    = args.moil_camera_name,
                pitch          = args.moil_pitch,
                yaw            = args.moil_yaw,
                zoom           = args.moil_zoom,
                mode           = args.moil_mode,
                use_opencl     = False,  # CPU-safe untuk portabilitas
                frame_width    = cap_w,
                frame_height   = cap_h,
            )
            # Override camera matrix ArUco dengan focal length hasil Moildev yang sudah memperhitungkan zoom dan resolusi
            detector.camera_matrix = moil_undistorter.build_aruco_camera_matrix(cap_w, cap_h)
            if args.focal_length is not None:
                detector.camera_matrix[0, 0] = args.focal_length
                detector.camera_matrix[1, 1] = args.focal_length
                print(f"[ARUCO] focal length di-override manual: fx={args.focal_length:.1f}px")
            detector.dist_coeffs = np.zeros(5, dtype=np.float64)  # sudah undistorted
            adj_f = detector.camera_matrix[0, 0]
            cx = detector.camera_matrix[0, 2]
            cy = detector.camera_matrix[1, 2]
            print(f"[MOIL] ✅ ArUco camera matrix di-override: fx={adj_f:.1f}px, cx={cx:.0f}, cy={cy:.0f}")
        except Exception as e:
            print(f"[ERROR] Gagal inisialisasi MoilUndistorter: {e}")
            sys.exit(1)
    elif args.focal_length is not None:
        # Fisheye tidak aktif, tapi user ingin override focal length biasa
        import numpy as np
        K = detector.camera_matrix.copy()
        K[0, 0] = args.focal_length
        K[1, 1] = args.focal_length
        detector.camera_matrix = K
        print(f"[ARUCO] focal length di-override manual: fx={args.focal_length:.1f}px")

    # ── Mode Kalibrasi Focal Length ────────────────────────────────────────
    if args.calibrate_focal is not None:
        _run_focal_calibration(detector, args, moil_undistorter)
        return

    if args.image:
        process_single_image(detector, args.image, output_dir,
                             moil_undistorter=moil_undistorter)
    elif args.camera is not None:
        run_live_camera(detector, camera_index=args.camera,
                        lock_focus=args.lock_focus,
                        focus_value=args.focus_value,
                        moil_undistorter=moil_undistorter,
                        cap_width=args.cap_width,
                        cap_height=args.cap_height)


if __name__ == "__main__":
    main()
