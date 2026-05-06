# ArUco + MiDaS Fusion Session Report

**Date/Time:** 2026-04-29 14-17-47

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
| **Average Cup Height** | **10.98 cm** | Mean of all valid predictions. |
| **Median Height (P50)** | **10.60 cm** | Most representative single value. |
| **Precision Error (P95−P5)** | **8.16 cm** | 90% of readings fall within this range. |
| **Standard Deviation ($\sigma$)** | 3.28 cm | Consistency / jitter of the AI model. |
| **Tray Anchor Depth (Z)** | 16.71 cm | Average physical depth of the tray. |
| **Minimum / Maximum Height** | 7.17 / 15.45 cm | Extremes recorded. |
| **Total Frames / Inferences** | 125 / 97 | Pipeline tracking efficiency. |

## 3. Visual Evidence
### Depth Tracking Chart
![Session Chart](session_chart.png)

