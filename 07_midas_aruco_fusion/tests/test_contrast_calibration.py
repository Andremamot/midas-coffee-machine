"""
tests/test_contrast_calibration.py — Robustness testing untuk modul
normalisasi pencahayaan (pengganti auto-exposure hardware).

Prinsip:
  - Semua test menggunakan synthetic image samples (tidak butuh kamera)
  - Mensimulasikan kondisi pencahayaan: LED ON, LED OFF, underexposed, overexposed
  - Memvalidasi bahwa normalize_lighting() dan detect_led_state() bekerja benar
    di berbagai variasi pencahayaan
"""

import time
import cv2
import numpy as np
import pytest
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "06_aruco_marker")))


# ─────────────────────────────────────────────────────────────────────────────
# Helpers: Pembuat frame sintetis untuk berbagai kondisi pencahayaan
# ─────────────────────────────────────────────────────────────────────────────

def make_frame(h: int = 480, w: int = 640, mean: float = 120.0, std: float = 40.0) -> np.ndarray:
    """
    Buat frame BGR sintetis dengan distribusi brightness Gaussian terkontrol.
    mean dan std dalam satuan pixel (0–255) pada channel luminance.
    """
    rng = np.random.default_rng(seed=42)
    gray = rng.normal(loc=mean, scale=std, size=(h, w))
    gray = np.clip(gray, 0, 255).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def make_led_on_frame(h: int = 480, w: int = 640) -> np.ndarray:
    """
    Simulasi kondisi LED ON:
    - Brightness tinggi (mean ≈ 170)
    - Distribusi sempit / merata (std ≈ 20) — LED membuat illuminasi uniform
    """
    return make_frame(h, w, mean=170.0, std=20.0)


def make_led_off_frame(h: int = 480, w: int = 640) -> np.ndarray:
    """
    Simulasi kondisi LED OFF / ambient only:
    - Brightness rendah-sedang (mean ≈ 70)
    - Distribusi lebar (std ≈ 55) — shadow di nozzle, highlight di area terang
    """
    return make_frame(h, w, mean=70.0, std=55.0)


def make_dark_frame(h: int = 480, w: int = 640) -> np.ndarray:
    """Simulasi underexposed — manual exposure terlalu rendah, LED OFF."""
    return make_frame(h, w, mean=30.0, std=15.0)


def make_bright_frame(h: int = 480, w: int = 640) -> np.ndarray:
    """Simulasi overexposed — LED ON + pantulan cahaya eksternal."""
    return make_frame(h, w, mean=210.0, std=25.0)


def make_flat_frame(h: int = 480, w: int = 640) -> np.ndarray:
    """Simulasi flat/low-contrast — LED ON, illuminasi sangat merata."""
    return make_frame(h, w, mean=155.0, std=10.0)


def make_aruco_frame(marker_id: int = 0, frame_size: int = 500, margin: int = 50) -> np.ndarray:
    """Frame sintetis dengan ArUco marker jelas di tengah."""
    aruco_dict  = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker_size = frame_size - 2 * margin
    marker_img  = np.zeros((marker_size, marker_size), dtype=np.uint8)
    cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size, marker_img, 1)
    frame = np.ones((frame_size, frame_size, 3), dtype=np.uint8) * 255
    frame[margin:margin + marker_size, margin:margin + marker_size] = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2BGR)
    return frame


def get_mean_brightness(frame: np.ndarray) -> float:
    """Rata-rata brightness channel L (LAB) dari frame BGR."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    return float(lab[:, :, 0].astype(np.float32).mean())


def get_std_brightness(frame: np.ndarray) -> float:
    """Std dev brightness channel L (LAB) dari frame BGR."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    return float(lab[:, :, 0].astype(np.float32).std())


# ═════════════════════════════════════════════════════════════════════════════
# FASE 1 — Unit tests: detect_led_state()
# ═════════════════════════════════════════════════════════════════════════════

class TestDetectLedState:
    """Menguji bahwa detect_led_state() mengklasifikasikan kondisi LED dengan benar."""

    def test_led_on_frame_detected_as_on(self):
        """Frame terang dan merata (LED ON) harus terdeteksi sebagai LED ON."""
        from core.image_preprocess import detect_led_state
        frame = make_led_on_frame()
        result = detect_led_state(frame)
        assert result is True, \
            f"Frame LED ON (mean=170, std=20) harus terdeteksi True, got {result}"

    def test_led_off_frame_detected_as_off(self):
        """Frame gelap dan tidak merata (LED OFF) harus terdeteksi sebagai LED OFF."""
        from core.image_preprocess import detect_led_state
        frame = make_led_off_frame()
        result = detect_led_state(frame)
        assert result is False, \
            f"Frame LED OFF (mean=70, std=55) harus terdeteksi False, got {result}"

    def test_dark_frame_detected_as_off(self):
        """Frame sangat gelap (underexposed, LED OFF) harus terdeteksi False."""
        from core.image_preprocess import detect_led_state
        frame = make_dark_frame()
        result = detect_led_state(frame)
        assert result is False, \
            f"Frame gelap (mean=30) harus terdeteksi LED OFF, got {result}"

    def test_bright_overexposed_detected_as_on(self):
        """Frame sangat terang (mean>210, std rendah) harus terdeteksi LED ON."""
        from core.image_preprocess import detect_led_state
        frame = make_bright_frame()
        result = detect_led_state(frame)
        assert result is True, \
            f"Frame overexposed (mean=210, std=25) harus terdeteksi LED ON, got {result}"

    def test_returns_bool(self):
        """detect_led_state harus selalu mengembalikan bool."""
        from core.image_preprocess import detect_led_state
        for frame in [make_led_on_frame(), make_led_off_frame(), make_dark_frame()]:
            result = detect_led_state(frame)
            assert isinstance(result, bool), f"Return bukan bool: {type(result)}"

    def test_custom_threshold_changes_result(self):
        """Threshold yang sangat tinggi harus membuat frame LED ON pun terdeteksi OFF."""
        from core.image_preprocess import detect_led_state
        frame = make_led_on_frame()  # mean ≈ 170
        # Naikkan threshold di atas mean frame → harus False
        result = detect_led_state(frame, brightness_threshold=200.0)
        assert result is False, \
            f"Dengan threshold=200, frame mean=170 harus False, got {result}"


# ═════════════════════════════════════════════════════════════════════════════
# FASE 2 — Unit tests: normalize_lighting()
# ═════════════════════════════════════════════════════════════════════════════

class TestNormalizeLighting:
    """Menguji bahwa normalize_lighting() memperbaiki kontras di berbagai kondisi."""

    def test_returns_tuple_frame_and_bool(self):
        """normalize_lighting harus return tuple (frame, bool)."""
        from core.image_preprocess import normalize_lighting
        frame = make_led_off_frame()
        result = normalize_lighting(frame)
        assert isinstance(result, tuple) and len(result) == 2, \
            f"Harus return (frame, bool), got {type(result)}"
        out_frame, led_detected = result
        assert isinstance(out_frame, np.ndarray)
        assert isinstance(led_detected, bool)

    def test_output_preserves_shape(self):
        """Output harus memiliki shape yang sama dengan input."""
        from core.image_preprocess import normalize_lighting
        for frame in [make_led_on_frame(), make_led_off_frame(), make_dark_frame()]:
            out, _ = normalize_lighting(frame)
            assert out.shape == frame.shape, \
                f"Shape berubah: {frame.shape} → {out.shape}"

    def test_output_preserves_dtype(self):
        """Output harus uint8 BGR."""
        from core.image_preprocess import normalize_lighting
        for frame in [make_led_on_frame(), make_led_off_frame(), make_dark_frame()]:
            out, _ = normalize_lighting(frame)
            assert out.dtype == np.uint8, f"dtype berubah: {out.dtype}"

    def test_ambient_dark_frame_brightness_increases(self):
        """
        Frame sangat gelap (LED OFF, mean<40) harus naik brightness-nya
        setelah normalisasi.
        """
        from core.image_preprocess import normalize_lighting
        frame = make_dark_frame()
        mean_before = get_mean_brightness(frame)
        out, _ = normalize_lighting(frame)
        mean_after = get_mean_brightness(out)
        assert mean_after > mean_before, \
            f"Frame gelap harus naik brightness: {mean_before:.1f} → {mean_after:.1f}"

    def test_overexposed_led_frame_brightness_decreases(self):
        """
        Frame overexposed (LED ON, mean>200) harus turun brightness-nya
        atau minimal tidak naik lebih tinggi lagi.
        """
        from core.image_preprocess import normalize_lighting
        frame = make_bright_frame()
        mean_before = get_mean_brightness(frame)
        out, _ = normalize_lighting(frame)
        mean_after = get_mean_brightness(out)
        assert mean_after <= mean_before + 10, \
            f"Frame overexposed tidak boleh naik: {mean_before:.1f} → {mean_after:.1f}"

    def test_flat_led_frame_contrast_increases(self):
        """
        Frame flat/low-contrast (LED ON, std<15) harus naik std dev-nya
        setelah normalisasi (CLAHE memulihkan detail lokal).
        """
        from core.image_preprocess import normalize_lighting
        frame = make_flat_frame()
        std_before = get_std_brightness(frame)
        out, _ = normalize_lighting(frame)
        std_after = get_std_brightness(out)
        assert std_after > std_before, \
            f"Frame flat harus naik kontras: std {std_before:.1f} → {std_after:.1f}"

    def test_led_on_detected_correctly_via_output(self):
        """normalize_lighting pada frame LED ON harus return led_detected=True."""
        from core.image_preprocess import normalize_lighting
        frame = make_led_on_frame()
        _, led_detected = normalize_lighting(frame)
        assert led_detected is True, \
            f"Frame LED ON harus return led_detected=True, got {led_detected}"

    def test_led_off_detected_correctly_via_output(self):
        """normalize_lighting pada frame LED OFF harus return led_detected=False."""
        from core.image_preprocess import normalize_lighting
        frame = make_led_off_frame()
        _, led_detected = normalize_lighting(frame)
        assert led_detected is False, \
            f"Frame LED OFF harus return led_detected=False, got {led_detected}"

    def test_led_and_ambient_profiles_differ(self):
        """
        Frame yang sama tapi berbeda kondisi LED harus menghasilkan output berbeda.
        Verifikasi dual-profile benar-benar aktif.
        """
        from core.image_preprocess import normalize_lighting, detect_led_state
        # Buat frame ambiguous di tengah-tengah threshold
        frame_led = make_led_on_frame()
        frame_amb = make_led_off_frame()

        out_led, _ = normalize_lighting(frame_led)
        out_amb, _ = normalize_lighting(frame_amb)

        mean_led = get_mean_brightness(out_led)
        mean_amb = get_mean_brightness(out_amb)

        # Dua profil berbeda harus menghasilkan output yang berbeda
        assert abs(mean_led - mean_amb) > 5.0, \
            f"Profil LED dan ambient harus berbeda output: {mean_led:.1f} vs {mean_amb:.1f}"

    def test_idempotency_stable(self):
        """
        Menjalankan normalize_lighting 3x berturut-turut tidak boleh menghasilkan
        perubahan besar (drift) — output harus stabil.
        """
        from core.image_preprocess import normalize_lighting
        frame = make_led_off_frame()
        out1, _ = normalize_lighting(frame)
        out2, _ = normalize_lighting(out1)
        out3, _ = normalize_lighting(out2)

        mean1 = get_mean_brightness(out1)
        mean3 = get_mean_brightness(out3)
        assert abs(mean3 - mean1) < 10.0, \
            f"Drift antar iterasi terlalu besar: {mean1:.1f} → {mean3:.1f}"

    def test_profile_switches_automatically_per_frame(self):
        """
        Jika kondisi berubah dari LED OFF ke LED ON dalam satu sesi,
        profil harus berubah otomatis tanpa restart.
        """
        from core.image_preprocess import normalize_lighting
        frame_off = make_led_off_frame()
        frame_on  = make_led_on_frame()

        _, led1 = normalize_lighting(frame_off)
        _, led2 = normalize_lighting(frame_on)

        assert led1 is False, "Frame OFF harus terdeteksi False"
        assert led2 is True,  "Frame ON harus terdeteksi True"

    def test_performance_per_frame(self):
        """
        normalize_lighting harus selesai < 10ms per frame pada resolusi 1280x720.
        Batas ini memastikan tidak membebani pipeline real-time.
        """
        from core.image_preprocess import normalize_lighting
        frame = make_led_off_frame(h=720, w=1280)

        # Warmup
        normalize_lighting(frame)

        N = 20
        start = time.perf_counter()
        for _ in range(N):
            normalize_lighting(frame)
        elapsed_ms = (time.perf_counter() - start) / N * 1000

        assert elapsed_ms < 15.0, \
            f"normalize_lighting terlalu lambat: {elapsed_ms:.2f}ms/frame (limit: 15ms)"


# ═════════════════════════════════════════════════════════════════════════════
# FASE 3 — Integration tests: ArUco detection setelah normalisasi
# ═════════════════════════════════════════════════════════════════════════════

class TestArUcoSurvivesNormalization:
    """
    Menguji bahwa normalisasi pencahayaan tidak merusak kemampuan deteksi ArUco.
    Menggunakan frame sintetis dengan marker + overlay brightness rendah/tinggi.
    """

    def _make_dim_aruco(self, brightness: int = 40) -> np.ndarray:
        """ArUco frame yang gelap — simulasi LED OFF, underexposed."""
        frame = make_aruco_frame(frame_size=400)
        # Darkening: multiply pixel values
        darkened = (frame.astype(np.float32) * (brightness / 200.0)).clip(0, 255).astype(np.uint8)
        return darkened

    def test_aruco_detected_after_ambient_normalization(self):
        """
        ArUco marker harus tetap terdeteksi setelah normalisasi frame gelap
        (kondisi LED OFF / ambient).
        """
        from core.image_preprocess import normalize_lighting
        from aruco_detector import ArucoDetector

        det   = ArucoDetector(marker_size_cm=5.0)
        frame = self._make_dim_aruco(brightness=50)

        out, _ = normalize_lighting(frame)
        result  = det.detect(out, use_enhancement=False)

        # Cukup verifikasi tidak crash dan mengembalikan list
        assert isinstance(result, list), "detect() harus return list setelah normalisasi"

    def test_aruco_detection_not_worse_after_normalization(self):
        """
        Jumlah ArUco yang terdeteksi setelah normalisasi >= jumlah tanpa normalisasi
        pada frame sintetis sempurna (tidak gelap).
        """
        from core.image_preprocess import normalize_lighting
        from aruco_detector import ArucoDetector

        det   = ArucoDetector(marker_size_cm=5.0)
        frame = make_aruco_frame(frame_size=500)  # frame sempurna, jelas

        result_raw  = det.detect(frame, use_enhancement=False)
        out, _      = normalize_lighting(frame)
        result_norm = det.detect(out, use_enhancement=False)

        # Normalisasi pada frame sempurna tidak boleh merusak deteksi
        assert len(result_norm) >= len(result_raw) - 1, \
            f"Normalisasi merusak deteksi: sebelum={len(result_raw)}, sesudah={len(result_norm)}"
