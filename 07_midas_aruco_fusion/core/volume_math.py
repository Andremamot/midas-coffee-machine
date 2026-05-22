"""
core/volume_math.py — Fungsi kalkulasi volume gelas
=====================================================
Modul ini berisi fungsi pure-math untuk estimasi volume gelas
dari data sensor (frame + bbox + z_tray + focal_px).

Pipeline:
  1. measure_rim_width_px(frame, bbox) → lebar rim dalam piksel
  2. calc_diameter(rim_w_px, z_rim, focal_px) → diameter fisik (cm)
  3. calc_volume(h_cup, diameter) → volume (mL ≈ cm³)

Prinsip Kunci:
  - Diameter diukur di level **rim** (bibir gelas), bukan seluruh bbox.
  - Ini mencegah overestimasi pada gelas yang body-nya lebih lebar dari rim.
  - Fungsi get_rim_depth() di MiDaS juga men-sample di strip yang sama,
    sehingga h_cup dan diameter konsisten secara spasial.
"""

import math
import cv2
import numpy as np


def measure_rim_width_px(frame: np.ndarray, bbox: tuple) -> float:
    """
    Ukur lebar gelas di level rim (strip atas bounding box).

    Menggunakan thresholding pada strip horizontal tipis di bagian atas bbox
    untuk menemukan tepi kiri dan kanan gelas di level bibir. Ini lebih akurat
    dari bbox_w penuh karena tidak terpengaruh body yang lebih lebar atau handle.

    Parameters
    ----------
    frame : np.ndarray
        Frame BGR yang sudah di-undistort (post-Moildev).
    bbox : tuple (x1, y1, x2, y2)
        Bounding box gelas dari YOLO.

    Returns
    -------
    float
        Lebar rim dalam piksel. Selalu > 0 (fallback ke bbox center width).
    """
    x1, y1, x2, y2 = bbox
    bbox_w = x2 - x1
    bbox_h = y2 - y1

    if bbox_w < 2 or bbox_h < 2:
        return max(1.0, float(bbox_w))

    h_frame, w_frame = frame.shape[:2]

    # Ketebalan strip rim: 10% tinggi bbox, minimal 4 piksel
    rim_thickness = max(4, bbox_h // 10)

    # Clamp ke batas frame
    ry1 = max(0, y1)
    ry2 = min(h_frame, y1 + rim_thickness)
    rx1 = max(0, x1)
    rx2 = min(w_frame, x2)

    if ry2 <= ry1 or rx2 <= rx1:
        return max(1.0, float(bbox_w))

    # Ambil strip rim dari frame
    rim_strip = frame[ry1:ry2, rx1:rx2]
    if rim_strip.size == 0:
        return max(1.0, float(bbox_w))

    # Konversi ke grayscale dan threshold
    if len(rim_strip.shape) == 3:
        gray_strip = cv2.cvtColor(rim_strip, cv2.COLOR_BGR2GRAY)
    else:
        gray_strip = rim_strip

    # Otsu threshold untuk memisahkan gelas dari background secara otomatis.
    # Lebih robust daripada threshold tetap karena beradaptasi dengan
    # kondisi pencahayaan yang bervariasi.
    _, mask = cv2.threshold(gray_strip, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Cari kolom yang memiliki piksel gelas
    cols_with_cup = np.where(mask.any(axis=0))[0]

    if len(cols_with_cup) >= 2:
        rim_w = float(cols_with_cup[-1] - cols_with_cup[0])
        return max(1.0, rim_w)

    # Fallback: gunakan 50% tengah bbox sebagai estimasi konservatif
    fallback_w = float(bbox_w) * 0.5
    return max(1.0, fallback_w)


def calc_diameter(rim_w_px: float, z_rim_cm: float, focal_px: float) -> float:
    """
    Hitung diameter fisik gelas (cm) via model pinhole kamera.

    Formula: diameter = (rim_w_px × z_rim) / focal_px

    Self-compensating terhadap pergerakan kamera naik/turun:
    - Kamera naik → z_rim membesar, rim_w_px mengecil → diameter tetap
    - Kamera turun → z_rim mengecil, rim_w_px membesar → diameter tetap

    Parameters
    ----------
    rim_w_px : float
        Lebar rim dalam piksel (dari measure_rim_width_px).
    z_rim_cm : float
        Jarak kamera ke bibir gelas dalam cm (= z_tray - h_cup).
    focal_px : float
        Focal length efektif dalam piksel (dari aruco.camera_matrix[0,0]).
        Sudah termasuk koreksi Moildev zoom.

    Returns
    -------
    float
        Diameter dalam cm. Return 0.0 jika input tidak valid.
    """
    if z_rim_cm <= 0 or focal_px <= 0 or rim_w_px <= 0:
        return 0.0
    return float((rim_w_px * z_rim_cm) / focal_px)


def calc_volume(h_cup_cm: float, diameter_cm: float) -> float:
    """
    Hitung volume gelas (mL) menggunakan model silinder.

    Formula: V = π × (d/2)² × h

    1 cm³ = 1 mL, jadi output langsung dalam mL.

    Parameters
    ----------
    h_cup_cm : float
        Tinggi gelas dalam cm (dari h_cup = z_tray - z_rim).
    diameter_cm : float
        Diameter gelas dalam cm (dari calc_diameter).

    Returns
    -------
    float
        Volume dalam mL (cm³). Return 0.0 jika input tidak valid.
    """
    if h_cup_cm <= 0 or diameter_cm <= 0:
        return 0.0
    radius = diameter_cm / 2.0
    return float(math.pi * (radius ** 2) * h_cup_cm)
