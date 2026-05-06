# ArUco + MiDaS Fusion Session Report

**Date/Time:** 2026-04-29 14-01-11

## 1. Parameters
Parameters used during this AI depth fusion session:

| Parameter | Value |
| :--- | :--- |
| **Physical Marker Size** | 2.5 cm |
| **Calibration Model** | 1-Point K-Factor |
| **Camera Focal Length** | 757.0 px |

## 2. Global Stability Summary
Statistical summary of cup height predictions gathered over the running frames:

| Metric | Value | Description |
| :--- | :--- | :--- |
| **Average Cup Height** | **103.13 cm** | Mean of all valid predictions. |
| **Median Height (P50)** | **108.55 cm** | Most representative single value. |
| **Precision Error (P95−P5)** | **77.09 cm** | 90% of readings fall within this range. |
| **Standard Deviation ($\sigma$)** | 25.68 cm | Consistency / jitter of the AI model. |
| **Tray Anchor Depth (Z)** | 13.35 cm | Average physical depth of the tray. |
| **Minimum / Maximum Height** | 54.13 / 153.66 cm | Extremes recorded. |
| **Total Frames / Inferences** | 96 / 80 | Pipeline tracking efficiency. |

## 3. Visual Evidence
### Depth Tracking Chart
![Session Chart](session_chart.png)

