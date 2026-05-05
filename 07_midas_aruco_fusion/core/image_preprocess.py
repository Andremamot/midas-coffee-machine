"""
core/image_preprocess.py — Preprocessing pipeline untuk meningkatkan kualitas
gambar fisheye sebelum deteksi ArUco dan rim gelas.

Modul ini menyediakan:
  - apply_clahe()           : CLAHE untuk kontrast lokal
  - apply_unsharp_mask()    : Sharpening dengan unsharp masking
  - detect_fisheye_circle() : Deteksi lingkaran fisheye dengan HoughCircles
  - crop_fisheye_to_rect()  : Crop area fisheye ke persegi panjang
  - enhance_for_detection() : Pipeline lengkap (CLAHE + unsharp + normalisasi)
  - detect_led_state()      : Auto-deteksi status LED supplement dari statistik frame
  - normalize_lighting()    : Normalisasi pencahayaan per-frame (pengganti auto-exposure)
"""

import cv2
import numpy as np


def apply_clahe(
    frame: np.ndarray,
    clip_limit: float = 3.0,
    tile_grid_size: tuple = (8, 8),
) -> np.ndarray:
    """
    Terapkan CLAHE (Contrast Limited Adaptive Histogram Equalization) pada
    channel L dari LAB colorspace untuk meningkatkan kontrast lokal tanpa
    mengubah saturasi warna.

    Args:
        frame      : BGR image
        clip_limit : Batas amplifikasi kontras (default 3.0; semakin tinggi
                     semakin kontras tapi semakin noisy)
        tile_grid_size : Ukuran tile untuk histogram lokal

    Returns:
        BGR image yang sudah di-enhance
    """
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def apply_unsharp_mask(
    frame: np.ndarray,
    sigma: float = 1.0,
    strength: float = 1.5,
) -> np.ndarray:
    """
    Terapkan unsharp masking untuk mempertajam gambar.

    Formula: output = frame * (1 + strength) - blurred * strength

    Args:
        frame    : BGR image
        sigma    : Standar deviasi Gaussian blur kernel
        strength : Kekuatan sharpening (default 1.5; lebih tinggi = lebih tajam)

    Returns:
        BGR image yang sudah di-sharpen
    """
    blurred = cv2.GaussianBlur(frame, (0, 0), sigma)
    sharpened = cv2.addWeighted(frame, 1.0 + strength, blurred, -strength, 0)
    return sharpened


def detect_fisheye_circle(
    frame: np.ndarray,
    dp: float = 1.2,
    min_dist_ratio: float = 0.5,
    param1: float = 100,
    param2: float = 30,
) -> tuple:
    """
    Deteksi lingkaran fisheye menggunakan HoughCircles.

    Untuk kamera fisheye yang dipasang pointing-down, gambar akan memiliki
    lingkaran besar di tengah dengan area hitam di sekeliling (vignetting).

    Returns:
        (cx, cy, radius) dalam pixel.
        Fallback ke pusat + min(H,W)/2 jika tidak ada lingkaran terdeteksi.
    """
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 2)

    min_r = int(min(h, w) * 0.25)
    max_r = int(min(h, w) * 0.55)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=dp,
        minDist=int(min(h, w) * min_dist_ratio),
        param1=param1,
        param2=param2,
        minRadius=min_r,
        maxRadius=max_r,
    )

    if circles is not None:
        # Ambil lingkaran dengan radius terbesar (= fisheye circle utama)
        circles = np.round(circles[0, :]).astype(int)
        best = max(circles, key=lambda c: c[2])
        cx, cy, r = int(best[0]), int(best[1]), int(best[2])
    else:
        # Fallback: estimasi dari vignetting (area tengah lebih terang)
        cx, cy = w // 2, h // 2
        # Radius dari threshold otomatis
        _, thresh = cv2.threshold(blurred, 15, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            (cx, cy), r = cv2.minEnclosingCircle(largest)
            cx, cy, r = int(cx), int(cy), int(r)
        else:
            r = int(min(h, w) * 0.4)

    return cx, cy, r


def crop_fisheye_to_rect(
    frame: np.ndarray,
    margin: float = 0.05,
    output_size: tuple = None,
) -> np.ndarray:
    """
    Crop frame fisheye ke kotak persegi panjang yang mencakup area valid
    (dalam lingkaran fisheye), lalu resize ke output_size.

    Area crop adalah bounding box dari lingkaran fisheye dikurangi margin.

    Args:
        frame       : BGR fisheye image
        margin      : Margin tambahan (0.05 = 5% dari radius) untuk menghindari
                      tepi lingkaran yang blur
        output_size : (width, height) output. None = gunakan ukuran crop asli.

    Returns:
        BGR image hasil crop
    """
    cx, cy, r = detect_fisheye_circle(frame)
    h, w = frame.shape[:2]

    # Inscribed square dalam lingkaran (diagonal = 2r → sisi = r*sqrt(2))
    half_side = int(r * (1.0 - margin) / np.sqrt(2))

    x1 = max(0, cx - half_side)
    y1 = max(0, cy - half_side)
    x2 = min(w, cx + half_side)
    y2 = min(h, cy + half_side)

    cropped = frame[y1:y2, x1:x2]

    if output_size is not None:
        cropped = cv2.resize(cropped, output_size, interpolation=cv2.INTER_LINEAR)

    return cropped


def detect_led_state(
    frame: np.ndarray,
    brightness_threshold: float = 140.0,
    std_threshold: float = 45.0,
) -> bool:
    """
    Deteksi otomatis apakah LED supplement aktif berdasarkan statistik frame.

    LED ON  → mean brightness channel L tinggi (> brightness_threshold) DAN
               distribusi sempit (std < std_threshold) karena pencahayaan merata.
    LED OFF → mean rendah ATAU distribusi lebar (shadow/highlight tidak merata).

    Args:
        frame                : BGR image dari kamera
        brightness_threshold : Threshold mean brightness channel L (default 140.0)
        std_threshold        : Threshold std dev channel L (default 45.0)

    Returns:
        True jika LED terdeteksi aktif, False jika ambient-only.
    """
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    mean_val, std_val = cv2.meanStdDev(lab[:, :, 0])
    mean_L = float(mean_val[0][0])
    std_L  = float(std_val[0][0])
    return (mean_L > brightness_threshold) and (std_L < std_threshold)


def normalize_lighting(
    frame: np.ndarray,
    brightness_threshold: float = 140.0,
    std_threshold: float = 45.0,
    clahe_clip_led: float = 1.8,
    clahe_clip_ambient: float = 3.2,
    percentile_low: float = 2.0,
    percentile_high_led: float = 95.0,
    percentile_high_ambient: float = 98.0,
) -> tuple:
    """
    Normalisasi pencahayaan per-frame sebagai pengganti auto-exposure hardware.
    Dijalankan saat kamera beroperasi dalam mode manual exposure (--manual-exposure).

    Status LED supplement dideteksi **secara otomatis** setiap frame dari
    statistik brightness — tidak memerlukan flag atau input manual.

    Profil LED ON  : percentile_high ketat (95%), CLAHE lunak (1.8)
                     → cegah highlight clipping dari LED langsung ke lensa
    Profil LED OFF : percentile_high longgar (98%), CLAHE agresif (3.2)
                     → tarik detail dari shadow area nozzle mesin kopi

    Pipeline per-frame:
      1. detect_led_state()            — analisis mean + std channel L
      2. Pilih profil (LED / ambient)  — sesuai kondisi terdeteksi
      3. Percentile stretch channel L  — normalkan rentang intensitas global
      4. Adaptive CLAHE channel L      — pulihkan detail lokal
      5. Convert LAB → BGR

    Args:
        frame                  : BGR image dari kamera (mode manual exposure)
        brightness_threshold   : Threshold mean L untuk deteksi LED (default 140.0)
        std_threshold          : Threshold std L untuk deteksi LED (default 45.0)
        clahe_clip_led         : Clip limit CLAHE saat LED ON (default 1.8)
        clahe_clip_ambient     : Clip limit CLAHE saat LED OFF (default 3.2)
        percentile_low         : Persentil bawah untuk stretch (default 2.0)
        percentile_high_led    : Persentil atas saat LED ON (default 95.0)
        percentile_high_ambient: Persentil atas saat LED OFF (default 98.0)

    Returns:
        Tuple (normalized_frame: np.ndarray, led_detected: bool)
        - normalized_frame : BGR uint8 dengan ukuran sama dengan input
        - led_detected     : True jika LED terdeteksi aktif pada frame ini
    """
    # ── Kerja di LAB colorspace (channel L = luminance, bebas warna) ──────────
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    L   = lab[:, :, 0]

    # Hitung statistik dengan OpenCV (sangat cepat)
    mean_val, std_val = cv2.meanStdDev(L)
    mean_L = float(mean_val[0][0])
    std_L_orig = float(std_val[0][0])

    # Inlining detect_led_state untuk menghindari cvtColor ganda
    led_on = (mean_L > brightness_threshold) and (std_L_orig < std_threshold)

    p_high    = percentile_high_led    if led_on else percentile_high_ambient
    clip_val  = clahe_clip_led         if led_on else clahe_clip_ambient

    # ── Step 1: Percentile stretch pada channel L ─────────────────────────────
    # Subsample image 16x (4x4) untuk komputasi np.percentile yang super cepat
    L_sub = L[::4, ::4]
    p_lo_val = float(np.percentile(L_sub, percentile_low))
    p_hi_val = float(np.percentile(L_sub, p_high))

    if p_hi_val > p_lo_val:
        # Gunakan Lookup Table (LUT) alih-alih numpy array math untuk performa instan
        lut = np.arange(256, dtype=np.float32)
        lut = (lut - p_lo_val) / (p_hi_val - p_lo_val) * 255.0
        lut = np.clip(lut, 0, 255).astype(np.uint8)
        lab[:, :, 0] = cv2.LUT(L, lut)

    # ── Step 2: Adaptive CLAHE pada channel L ─────────────────────────────────
    # clip_limit dikalikan faktor adaptif dari std dev frame asli (sudah dihitung)
    adapt_clip = clip_val * max(0.5, min(2.0, 40.0 / (std_L_orig + 1e-6)))
    clahe      = cv2.createCLAHE(clipLimit=adapt_clip, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])

    normalized = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return normalized, led_on


def enhance_for_detection(
    frame: np.ndarray,
    clahe_clip: float = 3.0,
    unsharp_sigma: float = 1.0,
    unsharp_strength: float = 1.5,
    manual_exposure_mode: bool = False,
) -> np.ndarray:
    """
    Pipeline lengkap preprocessing untuk meningkatkan deteksi ArUco dan rim gelas.

    Urutan (manual_exposure_mode=False):
      1. CLAHE  — perbaiki kontrast lokal (marker terlalu gelap/terang)
      2. Unsharp masking — pertajam tepi untuk deteksi ArUco lebih akurat

    Urutan (manual_exposure_mode=True — kamera dalam mode manual exposure):
      0. normalize_lighting() — normalisasi pencahayaan sebagai pengganti
                                auto-exposure hardware yang dimatikan
      1. CLAHE
      2. Unsharp masking

    Args:
        frame                : BGR image (fisheye yang sudah di-undistort)
        clahe_clip           : CLAHE clip limit
        unsharp_sigma        : Sigma Gaussian untuk unsharp mask
        unsharp_strength     : Kekuatan sharpening
        manual_exposure_mode : Jika True, jalankan normalize_lighting() lebih dulu

    Returns:
        BGR image yang sudah di-enhance, shape sama dengan input.
    """
    if manual_exposure_mode:
        frame, _ = normalize_lighting(frame)
    enhanced = apply_clahe(frame, clip_limit=clahe_clip)
    enhanced = apply_unsharp_mask(enhanced, sigma=unsharp_sigma,
                                  strength=unsharp_strength)
    return enhanced
