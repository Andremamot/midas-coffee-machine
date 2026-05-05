# ArUco + MiDaS Fusion Session Report

**Date/Time:** 2026-04-29 14-24-07

## 1. Parameters
Parameters used during this AI depth fusion session:

| Parameter | Value |
| :--- | :--- |
| **Physical Marker Size** | 2.5 cm |
| **Calibration Model** | 5-Geometric Z-Grid |
| **Camera Focal Length** | 757.0 px |

## 2. Global Stability Summary
Statistical summary of cup height predictions gathered over the running frames:

| Metric | Value | Description |
| :--- | :--- | :--- |
| **Average Cup Height** | **9.67 cm** | Mean of all valid predictions. |
| **Median Height (P50)** | **9.92 cm** | Most representative single value. |
| **Precision Error (P95−P5)** | **4.60 cm** | 90% of readings fall within this range. |
| **Standard Deviation ($\sigma$)** | 1.66 cm | Consistency / jitter of the AI model. |
| **Tray Anchor Depth (Z)** | 14.93 cm | Average physical depth of the tray. |
| **Minimum / Maximum Height** | 2.85 / 12.52 cm | Extremes recorded. |
| **Total Frames / Inferences** | 101 / 69 | Pipeline tracking efficiency. |

## 3. Visual Evidence
### Depth Tracking Chart
![Session Chart](session_chart.png)

## 4. Screenshots
- ![screenshots/fusion_2026-04-29_142406.jpg](screenshots/fusion_2026-04-29_142406.jpg)
