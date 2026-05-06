# ArUco + MiDaS Fusion Session Report

**Date/Time:** 2026-04-29 13-40-43

## 1. Parameters
Parameters used during this AI depth fusion session:

| Parameter | Value |
| :--- | :--- |
| **Physical Marker Size** | 2.0 cm |
| **Calibration Model** | 1-Point K-Factor |
| **Camera Focal Length** | 757.0 px |

## 2. Global Stability Summary
Statistical summary of cup height predictions gathered over the running frames:

| Metric | Value | Description |
| :--- | :--- | :--- |
| **Average Cup Height** | **60.99 cm** | Mean of all valid predictions. |
| **Median Height (P50)** | **41.29 cm** | Most representative single value. |
| **Precision Error (P95−P5)** | **147.68 cm** | 90% of readings fall within this range. |
| **Standard Deviation ($\sigma$)** | 51.91 cm | Consistency / jitter of the AI model. |
| **Tray Anchor Depth (Z)** | 12.03 cm | Average physical depth of the tray. |
| **Minimum / Maximum Height** | 6.45 / 159.56 cm | Extremes recorded. |
| **Total Frames / Inferences** | 20 / 18 | Pipeline tracking efficiency. |

## 3. Visual Evidence
### Depth Tracking Chart
![Session Chart](session_chart.png)

