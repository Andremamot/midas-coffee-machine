import os
import json
import shutil
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPORT_DIR = "results/report"

def _generate_session_report(calib_data, marker_size_cm, focal_len, total_frames, midas_runs,
                             history_z_tray, history_cup_h, history_frames, screenshots):
    
    # Backward compatibility checking
    if isinstance(history_cup_h, list):
        history_cup_h = {0: history_cup_h}

    all_cups = []
    for h_list in history_cup_h.values():
        all_cups.extend(h_list)

    valid_cups = [h for h in all_cups if h > 0]
    valid_trays = [z for z in history_z_tray if z > 0]
    
    avg_h = float(np.mean(valid_cups)) if valid_cups else 0.0
    avg_z = float(np.mean(valid_trays)) if valid_trays else 0.0
    std_h = float(np.std(valid_cups)) if valid_cups else 0.0
    max_h = float(np.max(valid_cups)) if valid_cups else 0.0
    min_h = float(np.min(valid_cups)) if valid_cups else 0.0

    print("="*50)
    print("📊 ARUCO+MIDAS FUSION SESSION REPORT")
    print("="*50)
    print(f"Total Camera Frames : {total_frames}")
    print(f"Total MiDaS Inferences: {midas_runs}")
    print(f"Average Tray Z-dist : {avg_z:.2f} cm")
    print(f"Average Cup Height  : {avg_h:.2f} cm")
    print(f"Cup Height Variance : ± {std_h:.2f} cm")
    print("="*50 + "\n")
    
    if not history_frames:
        return

    ts_folder = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    report_folder = os.path.join(REPORT_DIR, ts_folder)
    os.makedirs(report_folder, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.plot(history_frames, history_z_tray, label='Z_tray (ArUco)', color='blue', alpha=0.6)
    
    colors = ['green', 'orange', 'purple']
    for c_id, h_list in history_cup_h.items():
        if any(h > 0 for h in h_list):
            plt.plot(history_frames, h_list, label=f'Cup {c_id+1} Height', color=colors[c_id % len(colors)], linewidth=2)
            
    if avg_h > 0:
        plt.axhline(y=avg_h, color='red', linestyle='--', label=f'Global Avg Height: {avg_h:.2f}cm')
    plt.title("ArUco + MiDaS Fusion Pipeline Tracker (Multi-Cup)")
    plt.xlabel("Frame Number (MiDaS executions)")
    plt.ylabel("Centimeters (cm)")
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()
    chart_path = os.path.join(report_folder, "session_chart.png")
    plt.savefig(chart_path)
    plt.close()
    
    ss_target = []
    if screenshots:
        ss_sub = os.path.join(report_folder, "screenshots")
        os.makedirs(ss_sub, exist_ok=True)
        for s in screenshots:
            basename = os.path.basename(s)
            dst = os.path.join(ss_sub, basename)
            shutil.copy2(s, dst)
            ss_target.append(f"screenshots/{basename}")
            
    md_path = os.path.join(report_folder, "report.md")
    with open(md_path, "w") as f:
        f.write("# ArUco + MiDaS Fusion Session Report\n\n")
        f.write(f"**Date/Time:** {ts_folder.replace('_', ' ')}\n\n")

        f.write("## 1. Parameters\n")
        f.write("Parameters used during this AI depth fusion session:\n\n")
        f.write("| Parameter | Value |\n")
        f.write("| :--- | :--- |\n")
        f.write(f"| **Physical Marker Size** | {marker_size_cm} cm |\n")
        
        ctype = (calib_data or {}).get("type", 1)
        if ctype == 2: calib_str = f"2-Point Linear (m={(calib_data or {}).get('m',0):.5f}, c={(calib_data or {}).get('c',0):.5f})"
        elif ctype == 3: calib_str = "Z-Grid Polynomial"
        elif ctype == 4: calib_str = "BBox Area Scaling"
        elif ctype == 5: calib_str = "5-Geometric Z-Grid"
        elif ctype == 6: calib_str = "Bilateral Z-Grid"
        elif ctype == 7: calib_str = "Analytic Projection"
        else: calib_str = f"1-Point K-Factor (K={(calib_data or {}).get('K',0):.5f})"
            
        f.write(f"| **Calibration Model** | {calib_str} |\n")
        f.write(f"| **Camera Focal Length** | {focal_len:.1f} px |\n\n")

        f.write("## 2. Global Stability Summary\n")
        f.write("Statistical summary of cup height predictions gathered over the running frames:\n\n")
        
        f.write("| Metric | Value | Description |\n")
        f.write("| :--- | :--- | :--- |\n")
        f.write(f"| **Average Cup Height** | **{avg_h:.2f} cm** | Mean of all valid predictions. |\n")
        if valid_cups:
            p50 = float(np.median(valid_cups))
            p95 = float(np.percentile(valid_cups, 95))
            p05 = float(np.percentile(valid_cups, 5))
            f.write(f"| **Median Height (P50)** | **{p50:.2f} cm** | Most representative single value. |\n")
            f.write(f"| **Precision Error (P95−P5)** | **{p95 - p05:.2f} cm** | 90% of readings fall within this range. |\n")
        f.write(f"| **Standard Deviation ($\\sigma$)** | {std_h:.2f} cm | Consistency / jitter of the AI model. |\n")
        f.write(f"| **Tray Anchor Depth (Z)** | {avg_z:.2f} cm | Average physical depth of the tray. |\n")
        f.write(f"| **Minimum / Maximum Height** | {min_h:.2f} / {max_h:.2f} cm | Extremes recorded. |\n")
        f.write(f"| **Total Frames / Inferences** | {total_frames} / {midas_runs} | Pipeline tracking efficiency. |\n\n")

        f.write("## 3. Visual Evidence\n")
        f.write("### Depth Tracking Chart\n")
        f.write("![Session Chart](session_chart.png)\n\n")
        
        if ss_target:
            f.write("## 4. Screenshots\n")
            for ss in ss_target:
                f.write(f"- ![{ss}]({ss})\n")

    json_path = os.path.join(report_folder, "session_data.json")
    json_data = {
        "session_timestamp": ts_folder,
        "parameters": {
            "marker_size_cm": marker_size_cm,
            "calibration_model": calib_data,
            "focal_length_px": float(focal_len)
        },
        "summary": {
            "total_frames": total_frames,
            "midas_inferences": midas_runs,
            "avg_cup_height_cm": avg_h,
            "min_cup_height_cm": min_h,
            "max_cup_height_cm": max_h,
            "std_dev_cup_height_cm": std_h,
            "avg_z_tray_cm": avg_z
        },
        "frame_metrics_history": {
            "frame_indices": [int(x) for x in history_frames],
            "z_tray_history": [float(x) for x in history_z_tray],
            "cup_height_history": {str(k): [float(x) if x is not None else 0.0 for x in v] for k, v in history_cup_h.items()}
        },
        "screenshots": ss_target
    }
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

    size_kb = os.path.getsize(json_path) / 1024
    print(f"Report saved to: {report_folder} (Including session_data.json {size_kb:.1f} KB)")
