"""
tests/test_fisheye_pipeline.py — TDD untuk pipeline fisheye: menguji FUNGSI, bukan data nyata.

Prinsip:
  - Setiap test membuat data sintetis sendiri (bukan bergantung file kamera)
  - Test memvalidasi bahwa fungsi bekerja sesuai kontrak
  - Test tidak bergantung pada kondisi kamera, pencahayaan, atau posisi marker
"""
import cv2
import numpy as np
import pytest
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '06_aruco_marker')))


# ─────────────────────────────────────────────────────────────────────────────
# Helpers: Pembuat data sintetis
# ─────────────────────────────────────────────────────────────────────────────

def make_dark_frame(h=480, w=640, brightness=35):
    """Frame gelap simulasi kondisi raw fisheye."""
    img = np.full((h, w, 3), brightness, dtype=np.uint8)
    return img

def make_fisheye_frame(h=480, w=640):
    """Frame dengan lingkaran fisheye di tengah dan area hitam di luar."""
    img = np.full((h, w, 3), 50, dtype=np.uint8)
    cx, cy = w // 2, h // 2
    r = min(h, w) // 2 - 10
    # Isi area dalam lingkaran dengan brightness lebih tinggi
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (cx, cy), r, 255, -1)
    img[mask > 0] = 120
    # Area luar lingkaran = hitam (simulasi vignetting fisheye)
    img[mask == 0] = 0
    return img

def make_aruco_frame(marker_id=0, frame_size=500, margin=50):
    """
    Frame putih dengan satu marker ArUco 4x4 yang jelas di tengah.
    Ini adalah gambar sintetis sempurna untuk validasi fungsi deteksi.
    """
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker_size = frame_size - 2 * margin
    marker_img = np.zeros((marker_size, marker_size), dtype=np.uint8)
    cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size, marker_img, 1)
    frame = np.ones((frame_size, frame_size, 3), dtype=np.uint8) * 255
    frame[margin:margin+marker_size, margin:margin+marker_size] = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2BGR)
    return frame

def make_blurred_aruco_frame(sigma=2.0, **kwargs):
    """ArUco frame yang disengaja diblur untuk test ketahanan deteksi."""
    frame = make_aruco_frame(**kwargs)
    return cv2.GaussianBlur(frame, (0, 0), sigma)


# ═════════════════════════════════════════════════════════════════════════════
# FASE 1 — Unit tests: core/image_preprocess.py
# ═════════════════════════════════════════════════════════════════════════════

class TestApplyClahe:
    """Menguji bahwa apply_clahe() meningkatkan kontras gambar gelap."""

    def test_clahe_increases_mean_brightness_on_dark_frame(self):
        """apply_clahe pada frame gelap harus meningkatkan brightness rata-rata."""
        from core.image_preprocess import apply_clahe
        dark = make_dark_frame(brightness=30)
        enhanced = apply_clahe(dark, clip_limit=4.0)
        gray_orig = cv2.cvtColor(dark, cv2.COLOR_BGR2GRAY).mean()
        gray_enh = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY).mean()
        assert gray_enh > gray_orig, \
            f"CLAHE tidak meningkatkan brightness: {gray_orig:.1f} → {gray_enh:.1f}"

    def test_clahe_preserves_image_shape(self):
        """apply_clahe tidak boleh mengubah dimensi gambar."""
        from core.image_preprocess import apply_clahe
        frame = make_dark_frame(h=360, w=640)
        result = apply_clahe(frame)
        assert result.shape == frame.shape, \
            f"Shape berubah: {frame.shape} → {result.shape}"

    def test_clahe_preserves_dtype(self):
        """apply_clahe harus mengembalikan uint8."""
        from core.image_preprocess import apply_clahe
        frame = make_dark_frame()
        result = apply_clahe(frame)
        assert result.dtype == np.uint8, f"dtype berubah: {result.dtype}"

    def test_clahe_higher_clip_more_contrast(self):
        """clip_limit lebih tinggi harus menghasilkan contrast lebih tinggi."""
        from core.image_preprocess import apply_clahe
        frame = make_dark_frame(brightness=30)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_low = cv2.cvtColor(apply_clahe(frame, clip_limit=1.0), cv2.COLOR_BGR2GRAY)
        gray_high = cv2.cvtColor(apply_clahe(frame, clip_limit=8.0), cv2.COLOR_BGR2GRAY)
        std_low = gray_low.std()
        std_high = gray_high.std()
        assert std_high >= std_low, \
            f"clip_limit tinggi harus contrast >= clip rendah: {std_high:.1f} vs {std_low:.1f}"


class TestApplyUnsharpMask:
    """Menguji bahwa apply_unsharp_mask() mempertajam tepi."""

    def test_unsharp_increases_laplacian_variance(self):
        """Sharpening harus meningkatkan Laplacian variance."""
        from core.image_preprocess import apply_unsharp_mask
        # Buat gambar dengan tepi jelas lalu blur sedikit
        frame = make_aruco_frame()
        blurred = cv2.GaussianBlur(frame, (0, 0), 1.5)
        sharpened = apply_unsharp_mask(blurred, sigma=1.0, strength=1.5)
        lap_blur = cv2.Laplacian(cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
        lap_sharp = cv2.Laplacian(cv2.cvtColor(sharpened, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
        assert lap_sharp > lap_blur, \
            f"Unsharp mask tidak meningkatkan sharpness: {lap_blur:.1f} → {lap_sharp:.1f}"

    def test_unsharp_preserves_shape(self):
        """apply_unsharp_mask tidak boleh mengubah dimensi."""
        from core.image_preprocess import apply_unsharp_mask
        frame = make_dark_frame(h=480, w=640)
        result = apply_unsharp_mask(frame)
        assert result.shape == frame.shape

    def test_unsharp_strength_zero_returns_similar_image(self):
        """strength=0 harus menghasilkan gambar mendekati original."""
        from core.image_preprocess import apply_unsharp_mask
        frame = make_aruco_frame()
        result = apply_unsharp_mask(frame, strength=0.0)
        diff = np.abs(frame.astype(float) - result.astype(float)).mean()
        assert diff < 5.0, f"strength=0 harus mendekati original, diff={diff:.2f}"


class TestDetectFisheyeCircle:
    """Menguji bahwa detect_fisheye_circle() menemukan lingkaran di tengah."""

    def test_detects_circle_center_correctly(self):
        """Pusat lingkaran fisheye sintetis harus terdeteksi dekat tengah frame."""
        from core.image_preprocess import detect_fisheye_circle
        frame = make_fisheye_frame(h=480, w=640)
        cx, cy, r = detect_fisheye_circle(frame)
        assert abs(cx - 320) < 640 * 0.15, f"cx={cx} terlalu jauh dari tengah"
        assert abs(cy - 240) < 480 * 0.15, f"cy={cy} terlalu jauh dari tengah"

    def test_detects_circle_radius_is_reasonable(self):
        """Radius harus antara 25% dan 55% dari lebar frame."""
        from core.image_preprocess import detect_fisheye_circle
        frame = make_fisheye_frame(h=480, w=640)
        cx, cy, r = detect_fisheye_circle(frame)
        assert 640 * 0.25 < r < 640 * 0.55, \
            f"Radius {r} di luar range [160, 352]"

    def test_returns_three_values(self):
        """detect_fisheye_circle harus return (cx, cy, r)."""
        from core.image_preprocess import detect_fisheye_circle
        frame = make_fisheye_frame()
        result = detect_fisheye_circle(frame)
        assert len(result) == 3, "Harus return 3 nilai: (cx, cy, r)"


class TestCropFisheyeToRect:
    """Menguji bahwa crop_fisheye_to_rect() menghapus area hitam."""

    def test_cropped_has_less_black_pixels(self):
        """Setelah crop, pixel hitam harus kurang dari frame asli."""
        from core.image_preprocess import crop_fisheye_to_rect
        frame = make_fisheye_frame(h=480, w=640)
        gray_orig = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        black_orig = np.sum(gray_orig < 5) / gray_orig.size

        cropped = crop_fisheye_to_rect(frame)
        gray_crop = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        black_crop = np.sum(gray_crop < 5) / gray_crop.size

        assert black_crop < black_orig, \
            f"Crop tidak mengurangi pixel hitam: {black_orig:.2%} → {black_crop:.2%}"

    def test_cropped_is_3channel(self):
        """Hasil crop harus tetap 3 channel (BGR)."""
        from core.image_preprocess import crop_fisheye_to_rect
        frame = make_fisheye_frame()
        result = crop_fisheye_to_rect(frame)
        assert len(result.shape) == 3 and result.shape[2] == 3

    def test_custom_output_size(self):
        """output_size parameter harus menghasilkan gambar dengan ukuran yang diminta."""
        from core.image_preprocess import crop_fisheye_to_rect
        frame = make_fisheye_frame(h=480, w=640)
        result = crop_fisheye_to_rect(frame, output_size=(320, 240))
        assert result.shape[:2] == (240, 320), \
            f"Output size salah: {result.shape[:2]} bukan (240, 320)"


class TestEnhanceForDetection:
    """Menguji bahwa enhance_for_detection() merupakan wrapper yang benar."""

    def test_returns_same_shape(self):
        """enhance_for_detection harus mengembalikan shape yang sama."""
        from core.image_preprocess import enhance_for_detection
        frame = make_dark_frame(h=480, w=640)
        result = enhance_for_detection(frame)
        assert result.shape == frame.shape

    def test_returns_uint8(self):
        """enhance_for_detection harus mengembalikan uint8."""
        from core.image_preprocess import enhance_for_detection
        frame = make_dark_frame()
        result = enhance_for_detection(frame)
        assert result.dtype == np.uint8

    def test_improves_brightness_on_dark_frame(self):
        """enhance_for_detection harus meningkatkan brightness frame gelap."""
        from core.image_preprocess import enhance_for_detection
        frame = make_dark_frame(brightness=30)
        result = enhance_for_detection(frame)
        b_before = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean()
        b_after = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY).mean()
        assert b_after > b_before, f"Brightness tidak meningkat: {b_before:.1f} → {b_after:.1f}"


# ═════════════════════════════════════════════════════════════════════════════
# FASE 2 — Unit tests: ArucoDetector params & methods
# ═════════════════════════════════════════════════════════════════════════════

class TestArucoDetectorParams:
    """
    Menguji bahwa ArucoDetector diinisialisasi dengan parameter yang benar
    untuk kondisi fisheye blur.
    """

    def test_winsizemax_tolerant_for_blur(self):
        """adaptiveThreshWinSizeMax >= 71 diperlukan untuk gambar blur berat."""
        from aruco_detector import ArucoDetector
        det = ArucoDetector(marker_size_cm=5.0)
        assert det.aruco_params.adaptiveThreshWinSizeMax >= 71, \
            f"winSizeMax={det.aruco_params.adaptiveThreshWinSizeMax} harus >= 71 untuk blur"

    def test_min_perimeter_small_for_edge_markers(self):
        """minMarkerPerimeterRate <= 0.01 agar marker kecil/terpotong bisa dideteksi."""
        from aruco_detector import ArucoDetector
        det = ArucoDetector(marker_size_cm=5.0)
        assert det.aruco_params.minMarkerPerimeterRate <= 0.01, \
            f"minPerim={det.aruco_params.minMarkerPerimeterRate:.3f} harus <= 0.01"

    def test_corner_refine_none_for_blur(self):
        """CORNER_REFINE_NONE harus dipakai karena SUBPIX gagal di gambar blur."""
        from aruco_detector import ArucoDetector
        det = ArucoDetector(marker_size_cm=5.0)
        assert det.aruco_params.cornerRefinementMethod == cv2.aruco.CORNER_REFINE_NONE, \
            "cornerRefinementMethod harus CORNER_REFINE_NONE"

    def test_error_correction_high(self):
        """errorCorrectionRate >= 0.8 untuk toleran noise di pattern marker."""
        from aruco_detector import ArucoDetector
        det = ArucoDetector(marker_size_cm=5.0)
        assert det.aruco_params.errorCorrectionRate >= 0.8, \
            f"errorCorrectionRate={det.aruco_params.errorCorrectionRate} harus >= 0.8"

    def test_preprocessing_is_active(self):
        """_HAS_PREPROCESS harus True — CLAHE harus tersedia."""
        from aruco_detector import _HAS_PREPROCESS
        assert _HAS_PREPROCESS is True, \
            "_HAS_PREPROCESS=False: core.image_preprocess tidak bisa diimport"


class TestArucoDetectorOnSyntheticData:
    """
    Menguji fungsi detect() pada gambar sintetis yang terkontrol.
    Gambar sintetis SEMPURNA (tajam, kontras tinggi) = deteksi pasti berhasil.
    """

    def test_detect_perfect_marker_returns_list(self):
        """detect() pada marker sempurna harus return list (bukan crash)."""
        from aruco_detector import ArucoDetector
        det = ArucoDetector(marker_size_cm=5.0)
        frame = make_aruco_frame(marker_id=0, frame_size=400)
        result = det.detect(frame, use_enhancement=False)
        assert isinstance(result, list)

    def test_detect_perfect_marker_finds_at_least_one(self):
        """detect() pada marker sempurna HARUS menemukan minimal 1 marker."""
        from aruco_detector import ArucoDetector
        det = ArucoDetector(marker_size_cm=5.0)
        frame = make_aruco_frame(marker_id=0, frame_size=400)
        result = det.detect(frame, use_enhancement=False)
        assert len(result) >= 1, \
            "detect() tidak menemukan marker pada gambar sintetis sempurna. Bug di pipeline!"

    def test_detect_returns_correct_marker_id(self):
        """Marker ID yang terdeteksi harus sesuai dengan yang dibuat."""
        from aruco_detector import ArucoDetector
        det = ArucoDetector(marker_size_cm=5.0)
        for target_id in [0, 1, 2, 3]:
            frame = make_aruco_frame(marker_id=target_id, frame_size=400)
            result = det.detect(frame, use_enhancement=False)
            assert len(result) >= 1, f"Marker ID={target_id} tidak terdeteksi"
            detected_id = result[0]['id']
            assert detected_id == target_id, \
                f"ID salah: dibuat {target_id}, terdeteksi {detected_id}"

    def test_detect_result_has_required_keys(self):
        """Setiap item di result harus memiliki key yang diperlukan pipeline."""
        from aruco_detector import ArucoDetector
        det = ArucoDetector(marker_size_cm=5.0)
        frame = make_aruco_frame(marker_id=0, frame_size=400)
        result = det.detect(frame, use_enhancement=False)
        required_keys = {'id', 'corners', 'distance_cm', 'center'}
        for r in result:
            missing = required_keys - set(r.keys())
            assert not missing, f"Key yang hilang di result: {missing}"

    def test_detect_corners_within_frame_bounds(self):
        """Corner koordinat harus dalam batas frame."""
        from aruco_detector import ArucoDetector
        det = ArucoDetector(marker_size_cm=5.0)
        frame = make_aruco_frame(frame_size=400)
        h, w = frame.shape[:2]
        result = det.detect(frame, use_enhancement=False)
        for r in result:
            for x, y in r['corners']:
                assert 0 <= x <= w, f"Corner x={x:.1f} di luar [0, {w}]"
                assert 0 <= y <= h, f"Corner y={y:.1f} di luar [0, {h}]"

    def test_detect_empty_frame_returns_empty_list(self):
        """Frame kosong (tidak ada marker) harus return list kosong."""
        from aruco_detector import ArucoDetector
        det = ArucoDetector(marker_size_cm=5.0)
        frame = make_dark_frame()
        result = det.detect(frame, use_enhancement=False)
        assert result == [], f"Frame kosong harus return [], bukan {result}"


class TestDetectWithFallback:
    """Menguji fungsi detect_with_fallback() — mechanism fallback multi-skala."""

    def test_method_exists(self):
        """detect_with_fallback harus ada di ArucoDetector."""
        from aruco_detector import ArucoDetector
        det = ArucoDetector(marker_size_cm=5.0)
        assert hasattr(det, 'detect_with_fallback'), \
            "ArucoDetector harus punya method detect_with_fallback()"

    def test_returns_list_always(self):
        """detect_with_fallback selalu return list, tidak crash."""
        from aruco_detector import ArucoDetector
        det = ArucoDetector(marker_size_cm=5.0)
        frame = make_dark_frame()
        result = det.detect_with_fallback(frame, max_scale=2)
        assert isinstance(result, list)

    def test_tries_multiple_scales_when_fails(self):
        """
        Jika deteksi gagal di resolusi asli, fallback harus mencoba upscale.
        Diverifikasi dengan SpyDetector yang mencatat ukuran frame yang di-detect.
        """
        from aruco_detector import ArucoDetector

        class SpyDetector(ArucoDetector):
            def __init__(self):
                super().__init__(marker_size_cm=5.0)
                self.tried_sizes = []
            def detect(self, frame, use_enhancement=True):
                self.tried_sizes.append(frame.shape[:2])
                return []  # selalu gagal → paksa fallback mencoba semua skala

        spy = SpyDetector()
        spy.detect_with_fallback(make_dark_frame(h=240, w=320), max_scale=3)

        tried_heights = [s[0] for s in spy.tried_sizes]
        assert any(h >= 480 for h in tried_heights), \
            f"Fallback harus mencoba 2x upscale (>=480), tapi hanya: {tried_heights}"

    def test_finds_marker_on_perfect_frame(self):
        """detect_with_fallback pada marker sempurna harus return hasil."""
        from aruco_detector import ArucoDetector
        det = ArucoDetector(marker_size_cm=5.0)
        frame = make_aruco_frame(frame_size=400)
        result = det.detect_with_fallback(frame, max_scale=2)
        assert len(result) >= 1, "detect_with_fallback gagal pada frame sempurna"

    def test_corners_rescaled_to_original_space(self):
        """
        Jika marker ditemukan via upscale 2x, koordinat corner harus
        dikembalikan ke ruang koordinat frame asli (dibagi 2).
        """
        from aruco_detector import ArucoDetector

        class FirstFailThenFindDetector(ArucoDetector):
            """Gagal di skala asli, berhasil di 2x upscale."""
            def __init__(self):
                super().__init__(marker_size_cm=5.0)
                self.call_count = 0
            def detect(self, frame, use_enhancement=True):
                self.call_count += 1
                if self.call_count == 1:
                    return []  # gagal di skala 1x
                # Berhasil di 2x: return corner di ruang 2x
                fh, fw = frame.shape[:2]
                return [{
                    'id': 0,
                    'corners': np.array([[fw//4, fh//4], [3*fw//4, fh//4],
                                         [3*fw//4, 3*fh//4], [fw//4, 3*fh//4]], dtype=float),
                    'center': (fw//2, fh//2),
                    'distance_cm': 30.0,
                    'tvec': np.array([0.0, 0.0, 30.0]),
                    'rvec': np.array([0.0, 0.0, 0.0]),
                    'euler_deg': {}, 'reprojection_error': 0.0
                }]

        det = FirstFailThenFindDetector()
        orig_frame = make_dark_frame(h=200, w=300)
        result = det.detect_with_fallback(orig_frame, max_scale=2)
        assert len(result) == 1
        # Corner harus dalam batas frame ASLI (300x200), bukan frame 2x (600x400)
        fh, fw = orig_frame.shape[:2]
        for x, y in result[0]['corners']:
            assert x <= fw, f"Corner x={x:.1f} melebihi frame asli width={fw}"
            assert y <= fh, f"Corner y={y:.1f} melebihi frame asli height={fh}"


# ═════════════════════════════════════════════════════════════════════════════
# FASE 3 — Unit tests: MoilUndistorter.build_aruco_camera_matrix()
# ═════════════════════════════════════════════════════════════════════════════

class TestMoilArucoCameraMatrix:
    """
    Menguji bahwa build_aruco_camera_matrix() menghasilkan focal length yang
    reasonable untuk output MoildevApplicator (unicorn-solution port).

    Diupdate dari MoilUndistorter → MoildevApplicator (nama sesuai unicorn-solution).
    """

    def _make_mock_moil(self, param5=1000.0, calib_ratio=1.0,
                         img_w=1280, img_h=720):
        """
        Buat MoildevApplicator minimal tanpa hardware via direct attribute injection.
        Nama atribut menyesuaikan dengan implementasi baru (_parameter5, _calib_ratio).
        """
        from core.moildev_applicator import MoildevApplicator
        moil = object.__new__(MoildevApplicator)
        moil._parameter5    = param5
        moil._calib_ratio   = calib_ratio
        moil._sensor_width  = img_w
        moil._sensor_height = img_h
        moil.frame_width    = img_w
        moil.frame_height   = img_h
        return moil

    def test_focal_length_not_double_scaled(self):
        """
        Focal length di camera matrix harus == adjusted_focal_length saja
        (tidak dikali scale lagi), karena adjusted_focal_length sudah dalam
        unit piksel output.

        Formula benar (unicorn-solution): fl = param5 / calibRatio
        """
        from core.moildev_applicator import MoildevApplicator
        moil = self._make_mock_moil(param5=1000.0, calib_ratio=1.0,
                                     img_w=1280, img_h=720)
        K = moil.build_aruco_camera_matrix(frame_width=1280, frame_height=720)
        fl = K[0, 0]
        adj_fl = moil.adjusted_focal_length  # = param5/calib_ratio = 1000.0
        # Jika double-scaled: fl = 1000 * (1280/1280) = 1000 → OK
        # Jika wrongly scaled ke sensor asli (misal 2592): fl = 1000 * (1280/2592) = 386 → SALAH
        assert fl == pytest.approx(adj_fl, rel=0.05), \
            f"Focal length double-scaled: expected≈{adj_fl:.1f}, got {fl:.1f}"

    def test_focal_length_reasonable_for_live_conditions(self):
        """
        Dengan parameter5≈1000 (tipikal syue_7730v1), focal length output
        harus dalam range 300–1500 px untuk resolusi 1280x720.
        Nilai 218 terlalu kecil (menyebabkan Z_tray = 47777 cm).
        """
        from core.moildev_applicator import MoildevApplicator
        moil = self._make_mock_moil(param5=1000.0, calib_ratio=1.0,
                                     img_w=1280, img_h=720)
        K = moil.build_aruco_camera_matrix(frame_width=1280, frame_height=720)
        fl = K[0, 0]
        assert 300 <= fl <= 1500, \
            f"Focal length {fl:.1f} tidak reasonable untuk 1280x720. " \
            f"Terlalu kecil → Z estimation meleset jauh (Z=47777cm)."

    def test_principal_point_at_frame_center(self):
        """Principal point (cx, cy) harus di tengah frame."""
        from core.moildev_applicator import MoildevApplicator
        moil = self._make_mock_moil(param5=1000.0, calib_ratio=1.0,
                                     img_w=1280, img_h=720)
        K = moil.build_aruco_camera_matrix(frame_width=1280, frame_height=720)
        assert K[0, 2] == pytest.approx(640.0), f"cx={K[0,2]} bukan 640"
        assert K[1, 2] == pytest.approx(360.0), f"cy={K[1,2]} bukan 360"

    def test_matrix_shape_3x3(self):
        """build_aruco_camera_matrix harus return ndarray 3x3."""
        from core.moildev_applicator import MoildevApplicator
        moil = self._make_mock_moil()
        K = moil.build_aruco_camera_matrix(frame_width=1280, frame_height=720)
        assert K.shape == (3, 3), f"Shape salah: {K.shape}"

    def test_focal_length_scales_with_resolution(self):
        """
        Focal length harus proporsional dengan resolusi output:
        jika resolusi 1.5x lebih besar, focal length juga 1.5x.
        """
        from core.moildev_applicator import MoildevApplicator
        moil = self._make_mock_moil(param5=1000.0, calib_ratio=1.0,
                                     img_w=1280, img_h=720)
        K_720  = moil.build_aruco_camera_matrix(frame_width=1280, frame_height=720)
        K_1080 = moil.build_aruco_camera_matrix(frame_width=1920, frame_height=1080)
        ratio  = K_1080[0, 0] / K_720[0, 0]
        assert ratio == pytest.approx(1920/1280, rel=0.05), \
            f"Focal length tidak proporsional dengan resolusi: ratio={ratio:.3f}, expected={1920/1280:.3f}"


# ═════════════════════════════════════════════════════════════════════════════
# FASE 4 — Unit tests: MoildevApplicator (unicorn-solution port)
# ═════════════════════════════════════════════════════════════════════════════

class TestMoildevApplicatorLUT:
    """
    Menguji LUT Alpha-Rho yang dibangun via Horner's Method.
    Diport dari MoildevApplicator::initializeAlphaRhoTables() unicorn-solution.
    """

    def _make_lut(self, p0=0.0, p1=0.0, p2=0.0, p3=0.0, p4=0.0, p5=1.0, calib=1.0):
        """Build LUT menggunakan implementasi baru."""
        from core.moildev_applicator import MoildevApplicator
        return MoildevApplicator._init_lut(p0, p1, p2, p3, p4, p5, calib)

    def test_lut_alpha_rho_size(self):
        """LUT alpha→rho harus memiliki 1800 entri."""
        a2r, _ = self._make_lut()
        assert len(a2r) == 1800, f"alpha_to_rho size={len(a2r)}, expected 1800"

    def test_lut_rho_alpha_min_size(self):
        """LUT rho→alpha harus memiliki minimal 1 entri."""
        _, r2a = self._make_lut()
        assert len(r2a) >= 1, "rho_to_alpha kosong!"

    def test_lut_alpha_zero_gives_rho_zero(self):
        """Alpha=0 (index 0) harus menghasilkan rho=0."""
        a2r, _ = self._make_lut(p5=1.0)
        assert a2r[0] == pytest.approx(0.0, abs=1e-9), \
            f"Alpha=0 harus rho=0, got {a2r[0]}"

    def test_lut_alpha_rho_monotonic_increasing(self):
        """alpha_to_rho harus monoton meningkat (semakin jauh dari pusat = semakin besar rho)."""
        a2r, _ = self._make_lut(p5=1.0)
        for i in range(1, len(a2r)):
            assert a2r[i] >= a2r[i-1], \
                f"LUT tidak monoton di index {i}: a2r[{i}]={a2r[i]:.4f} < a2r[{i-1}]={a2r[i-1]:.4f}"

    def test_horner_vs_direct_evaluation(self):
        """
        Horner's Method harus menghasilkan nilai yang sama dengan evaluasi langsung.
        """
        import math
        p0, p1, p2, p3, p4, p5, calib = 0.001, -0.01, 0.05, 0.0, 0.0, 1.0, 1.0
        a2r, _ = self._make_lut(p0, p1, p2, p3, p4, p5, calib)
        DEG_TO_RAD = math.pi / 180.0

        # Verifikasi untuk beberapa titik
        for deg_tenth in [100, 300, 600, 900]:
            alpha = (deg_tenth / 10.0) * DEG_TO_RAD
            # Direct evaluation (formula lama)
            rho_direct = (p0*alpha**6 + p1*alpha**5 + p2*alpha**4
                          + p3*alpha**3 + p4*alpha**2 + p5*alpha) * calib
            assert a2r[deg_tenth] == pytest.approx(rho_direct, rel=1e-9), \
                f"Horner vs direct mismatch di alpha={deg_tenth/10:.1f}deg"


class TestMoildevApplicatorOutput:
    """
    Menguji output undistort() dan build_aruco_camera_matrix()
    dari MoildevApplicator tanpa hardware kamera nyata.
    """

    def _make_mock_applicator(self, param5=1000.0, calib_ratio=1.0,
                               sensor_w=1280, sensor_h=720,
                               frame_w=640, frame_h=480):
        """Inject atribut langsung tanpa konstruktor (skip hardware init)."""
        from core.moildev_applicator import MoildevApplicator
        import threading
        import numpy as np

        moil = object.__new__(MoildevApplicator)
        moil._parameter5    = param5
        moil._calib_ratio   = calib_ratio
        moil._sensor_width  = sensor_w
        moil._sensor_height = sensor_h
        moil._icx           = sensor_w / 2.0
        moil._icy           = sensor_h / 2.0
        moil.frame_width    = frame_w
        moil.frame_height   = frame_h
        moil.pitch          = 0.0
        moil.yaw            = 0.0
        moil.roll           = 0.0
        moil.zoom           = 1.4
        moil.moil_zoom      = 1.4
        moil.digital_zoom   = 1.0
        moil.mode           = 2
        moil.opencl_active  = False
        moil.target_size    = None
        moil._maps_lock     = threading.Lock()

        # Buat identity maps (map_x[y,x]=x, map_y[y,x]=y)
        map_x = np.tile(np.arange(frame_w, dtype=np.float32), (frame_h, 1))
        map_y = np.tile(np.arange(frame_h, dtype=np.float32).reshape(-1, 1), (1, frame_w))
        moil._map_x_cpu = map_x
        moil._map_y_cpu = map_y
        moil._map_x     = map_x
        moil._map_y     = map_y

        # Build LUT (pakai parameter dummy sederhana)
        moil._alpha_to_rho, moil._rho_to_alpha = MoildevApplicator._init_lut(
            0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0
        )
        return moil

    def test_undistort_preserves_shape(self):
        """undistort() harus mengembalikan frame dengan ukuran yang sama."""
        moil = self._make_mock_applicator()
        import numpy as np
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = moil.undistort(frame)
        assert result.shape == frame.shape, \
            f"Shape berubah: {frame.shape} → {result.shape}"

    def test_undistort_returns_uint8(self):
        """undistort() harus mengembalikan uint8."""
        moil = self._make_mock_applicator()
        import numpy as np
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = moil.undistort(frame)
        assert result.dtype == np.uint8

    def test_undistort_none_input_returns_none(self):
        """undistort(None) harus return None tanpa crash."""
        moil = self._make_mock_applicator()
        result = moil.undistort(None)
        assert result is None

    def test_aruco_matrix_shape(self):
        """build_aruco_camera_matrix harus return 3x3."""
        moil = self._make_mock_applicator()
        K = moil.build_aruco_camera_matrix(640, 480)
        assert K.shape == (3, 3)

    def test_aruco_matrix_fx_equals_fy(self):
        """fx harus sama dengan fy (square pixels)."""
        moil = self._make_mock_applicator()
        K = moil.build_aruco_camera_matrix(640, 480)
        assert K[0, 0] == pytest.approx(K[1, 1], rel=1e-9), \
            f"fx={K[0,0]:.2f} ≠ fy={K[1,1]:.2f}"

    def test_get_alpha_beta_returns_tuple(self):
        """get_alpha_beta harus return tuple (alpha, beta)."""
        moil = self._make_mock_applicator()
        result = moil.get_alpha_beta(320, 240)
        assert isinstance(result, tuple) and len(result) == 2

    def test_get_alpha_beta_center_approximately_zero(self):
        """
        Koordinat tengah harus menghasilkan alpha≈0
        (titik pusat = tidak ada defleksi vertikal).
        """
        moil = self._make_mock_applicator(sensor_w=640, sensor_h=480,
                                           frame_w=640, frame_h=480)
        alpha, beta = moil.get_alpha_beta(320, 240)
        assert abs(alpha) <= 1.0, \
            f"Alpha di tengah harus ≈0, got {alpha:.2f}"

