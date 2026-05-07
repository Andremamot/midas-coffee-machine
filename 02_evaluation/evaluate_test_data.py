import os
import sys
import cv2
import yaml
import numpy as np
import json
import math
from datetime import datetime

# Ensure project root is in path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from midas_volumecup.depth import MidasDepthEstimator
from midas_volumecup.detector import YoloDetector
from midas_volumecup.volume_math import calculate_z_rim, calculate_z_rim_alpha

def generate_report():
    # Look for snapshots in the dataset folder
    snapshot_dir = os.path.join(root_dir, "04_dataset", "test_snapshots")
    # snapshot_dir = os.path.join(root_dir, "01_calibration", "calibration_snapshots")
    
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(os.path.dirname(__file__), "evaluation_results", f"eval_{timestamp_str}")
    config_file = os.path.join(root_dir, 'midas_calibration.yaml')
    report_path = os.path.join(output_dir, "validation_report.md")
    
    if not os.path.exists(snapshot_dir):
        print(f"Error: Directory '{snapshot_dir}' not found. Please run calibration first.")
        return
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. Load Calibration Configuration
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
            alpha = float(config.get('alpha', 1.0))
            a = float(config.get('a', 0.0)) # purely for fallback
            tray_roi = tuple(map(int, config.get('tray_roi', [10,400,100,470])))
            print(f"Loaded Calibration: alpha={alpha:.4f}")
    except Exception as e:
        print(f"Failed to load midas_calibration.yaml: {e}")
        return

    images = sorted([f for f in os.listdir(snapshot_dir) if f.endswith('.jpg')])
    if not images:
        print(f"No snapshots found in '{snapshot_dir}'.")
        return

    # 2. Initialize Models
    print("Initializing AI Models...")
    depth_estimator = MidasDepthEstimator()
    detector = YoloDetector()

    report_data = []
    total_error = 0.0
    valid_count = 0

    print("Processing snapshots...")
    for img_name in images:
        filepath = os.path.join(snapshot_dir, img_name)
        
        # Parse True Z from filename
        try:
            parts = img_name.split('_')
            true_z_tray = float(parts[1].replace('tray', '').replace('cm', ''))
            true_z_rim = float(parts[2].replace('rim', '').replace('cm', ''))
            true_z = true_z_rim
        except:
            continue

        frame = cv2.imread(filepath)
        if frame is None: continue

        # Run Inference
        depth_map = depth_estimator.process(frame)
        boxes = detector.detect(frame)
        depth_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        m_tray = depth_estimator.get_tray_depth(depth_norm, tray_roi)
        
        if boxes and m_tray > 0:
            m_rim = depth_estimator.get_rim_depth(depth_norm, boxes[0]['bbox'])
            ratio = m_rim / m_tray
            pred_z = calculate_z_rim_alpha(m_rim, m_tray, true_z_tray, alpha)
            
            # Distance to floor (where ratio would be 1.0)
            pred_floor_z = calculate_z_rim_alpha(m_tray, m_tray, true_z_tray, alpha)

            error = abs(pred_z - true_z)
            error_percent = (error / true_z) * 100.0 if true_z > 0 else 0
            
            # Save Debug Image
            debug_frame = frame.copy()
            bpx = boxes[0]['bbox']
            cv2.rectangle(debug_frame, (bpx[0], bpx[1]), (bpx[2], bpx[3]), (0,255,0), 2)
            cv2.putText(debug_frame, f"Ratio: {ratio:.2f}", (bpx[0], bpx[1]-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            cv2.putText(debug_frame, f"Pred: {pred_z:.1f}cm (Err:{error_percent:.1f}%)", (bpx[0], bpx[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
            
            # Create Heatmap
            heatmap = cv2.applyColorMap(depth_norm, cv2.COLORMAP_INFERNO)
            cv2.rectangle(heatmap, (bpx[0], bpx[1]), (bpx[2], bpx[3]), (0,255,0), 2)
            cv2.rectangle(heatmap, (tray_roi[0], tray_roi[1]), (tray_roi[2], tray_roi[3]), (255,0,0), 2)
            
            # Merge Side-by-Side
            combined_debug = np.hstack((debug_frame, heatmap))
            
            # Add Detailed Info Panel at bottom left
            h, w = combined_debug.shape[:2]
            y_offset = h - 170
            
            info_lines = [
                f"True Height: {true_z:.2f} cm",
                f"Pred Height: {pred_z:.2f} cm",
                f"Error: {error:.2f} cm ({error_percent:.1f}%)",
                f"Floor Z Pred: {pred_floor_z:.1f} cm",
                f"M_Rim: {m_rim:.1f}  |  M_Tray: {m_tray:.1f}"
            ]
            
            for i, line in enumerate(info_lines):
                # Draw black outline for readability
                cv2.putText(combined_debug, line, (20, y_offset + (i*30)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
                # Draw white text
                cv2.putText(combined_debug, line, (20, y_offset + (i*30)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            debug_name = f"debug_{img_name}"
            cv2.imwrite(os.path.join(output_dir, debug_name), combined_debug)

            # Store Data
            report_data.append({
                "filename": img_name,
                "debug_image": debug_name,
                "m_rim": m_rim,
                "m_tray": m_tray,
                "ratio": ratio,
                "true_z_tray": true_z_tray,
                "true_z": true_z,
                "pred_z": pred_z,
                "error": error,
                "error_pct": error_percent,
                "pred_floor_z": pred_floor_z
            })
            total_error += error
            valid_count += 1

    # 3. Write Markdown Report
    mae = total_error / valid_count if valid_count > 0 else 0
    avg_floor_z = np.mean([d['pred_floor_z'] for d in report_data]) if report_data else 0
    
    if valid_count > 0:
        errors = [d['error'] for d in report_data]
        error_pcts = [d['error_pct'] for d in report_data]
        rmse = math.sqrt(sum(e**2 for e in errors) / valid_count)
        
        mean_error_signed = sum((d['pred_z'] - d['true_z']) for d in report_data) / valid_count
        std_dev = math.sqrt(sum(((d['pred_z'] - d['true_z']) - mean_error_signed)**2 for d in report_data) / valid_count)
        
        mape = sum(error_pcts) / valid_count
        
        d_5mm = sum(1 for e in errors if e <= 0.5) / valid_count * 100
        d_1cm = sum(1 for e in errors if e <= 1.0) / valid_count * 100
        d_2cm = sum(1 for e in errors if e <= 2.0) / valid_count * 100
    else:
        rmse = std_dev = mape = d_5mm = d_1cm = d_2cm = 0.0
    
    with open(report_path, 'w') as f:
        f.write(f"# MiDaS Depth Calibration: Final Validation Report\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write(f"## 1. Calibration Parameters\n")
        f.write(f"The system is currently using the **Pure Physics Multiplier Model**:\n")
        f.write(f"$$ Z = (\\frac{{Z_{{tray}}}}{{R}}) \\times \\alpha $$\n\n")
        f.write(f"| Parameter | Value |\n| :--- | :--- |\n")
        f.write(f"| **Alpha Multiplier** | {alpha:.4f} |\n| **Tray ROI** | {tray_roi} |\n")
        f.write(f"| **Predicted Floor Z** | **{avg_floor_z:.2f} cm** |\n\n")
        
        f.write(f"## 2. Global Accuracy Summary\n")
        f.write(f"| Metric | Value | Description |\n| :--- | :--- | :--- |\n")
        f.write(f"| **Mean Absolute Error (MAE)** | **{mae:.2f} cm** | Average absolute distance off target. |\n")
        f.write(f"| **Root Mean Sq Error (RMSE)** | **{rmse:.2f} cm** | Punishes severe outliers heavily. |\n")
        f.write(f"| **Standard Deviation ($\sigma$)** | **{std_dev:.2f} cm** | Consistency of the error spread. |\n")
        f.write(f"| **Mean Abs Pct Error (MAPE)** | **{mape:.1f}%** | Average percentage distance off target. |\n")
        f.write(f"| **Strict ($\delta < 5mm$)** | **{d_5mm:.1f}%** | Predictions within 5mm of True Z. |\n")
        f.write(f"| **Standard ($\delta < 1cm$)** | **{d_1cm:.1f}%** | Predictions within 10mm of True Z. |\n")
        f.write(f"| **Loose ($\delta < 2cm$)** | **{d_2cm:.1f}%** | Predictions within 20mm of True Z. |\n")
        f.write(f"| **Valid Test Set Frames** | **{valid_count}** | Total snapshots successfully evaluated. |\n\n")
        
        f.write(f"## 3. Individual Breakdown\n")
        f.write(f"| Snapshot | M_rim | M_tray | Ratio | True Z | Pred Z | Error % |\n")
        f.write(f"| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for d in report_data:
            f.write(f"| {d['filename']} | {d['m_rim']:.1f} | {d['m_tray']:.1f} | **{d['ratio']:.2f}** | {d['true_z']:.2f}cm | {d['pred_z']:.2f}cm | {d['error_pct']:.1f}% |\n")

        f.write(f"\n## 4. Visual Evidence\n")
        for d in report_data:
            f.write(f"### Sample: {d['filename']}\n")
            f.write(f"![Debug Image]({d['debug_image']})\n\n")
            f.write(f"**Math Trace**:\n")
            f.write(f"- Absolute Floor Distance (Predicted): **{d['pred_floor_z']:.2f} cm**\n")
            f.write(f"- $R = {d['m_rim']:.1f} / {d['m_tray']:.1f} = {d['ratio']:.3f}$\n")
            f.write(f"- $Z = ({d['true_z_tray']:.1f} / {d['ratio']:.3f}) \\times {alpha:.4f} = {d['pred_z']:.1f} cm$\n")
            f.write(f"- **Result**: {d['pred_z']:.2f} cm\n\n")
            f.write(f"---\n\n")

    print(f"\nSuccess! Detailed report written to: {report_path}")
    print(f"Debug images saved in: {output_dir}/")

if __name__ == "__main__":
    generate_report()
