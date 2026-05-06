# ArUco + MiDaS Fusion Session Report

**Date/Time:** 2026-05-04 13-37-18

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
| **Average Cup Height** | **4.97 cm** | Mean of all valid predictions. |
| **Median Height (P50)** | **4.40 cm** | Most representative single value. |
| **Precision Error (P95−P5)** | **7.52 cm** | 90% of readings fall within this range. |
| **Standard Deviation ($\sigma$)** | 2.83 cm | Consistency / jitter of the AI model. |
| **Tray Anchor Depth (Z)** | 50.24 cm | Average physical depth of the tray. |
| **Minimum / Maximum Height** | 1.13 / 12.26 cm | Extremes recorded. |
| **Total Frames / Inferences** | 248 / 130 | Pipeline tracking efficiency. |

## 3. Visual Evidence
### Depth Tracking Chart
![Session Chart](session_chart.png)

