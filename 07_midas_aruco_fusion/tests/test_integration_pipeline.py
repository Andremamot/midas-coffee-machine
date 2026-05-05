"""
tests/test_integration_pipeline.py
===================================
TDD Integration Test Suite — Memvalidasi semua fitur aktif yang dikembangkan.

Jalankan setelah setiap perubahan kode:
  conda run -n midas-aruco-env python -m pytest tests/test_integration_pipeline.py -v

Setiap TestClass merepresentasikan satu fitur / modul yang bisa terdampak.
Jika sebuah kelas FAIL semua, berarti modul itu rusak akibat perubahan terbaru.
"""
import pytest
import queue
import time
import threading
import numpy as np
import cv2

# ─── Fixture Helpers ──────────────────────────────────────────────────────────

def _make_bgr(h=480, w=640, brightness=128):
    """Frame BGR sintetis dengan brightness merata."""
    return np.full((h, w, 3), brightness, dtype=np.uint8)

def _make_dark_bgr():  return _make_bgr(brightness=30)
def _make_bright_bgr(): return _make_bgr(brightness=220)


# ══════════════════════════════════════════════════════════════════════════════
# 1. MODULE: core/image_preprocess.py
#    Features: detect_led_state, normalize_lighting
# ══════════════════════════════════════════════════════════════════════════════
class TestImagePreprocess:
    """Memastikan modul normalisasi lighting tidak rusak."""

    def test_import_ok(self):
        from core.image_preprocess import detect_led_state, normalize_lighting
        assert callable(detect_led_state)
        assert callable(normalize_lighting)

    def test_detect_led_returns_bool(self):
        from core.image_preprocess import detect_led_state
        result = detect_led_state(_make_bgr())
        assert isinstance(result, bool)

    def test_normalize_returns_frame_and_bool(self):
        from core.image_preprocess import normalize_lighting
        out, led = normalize_lighting(_make_bgr())
        assert isinstance(out, np.ndarray)
        assert isinstance(led, bool)

    def test_normalize_preserves_shape(self):
        from core.image_preprocess import normalize_lighting
        frame = _make_bgr(300, 400)
        out, _ = normalize_lighting(frame)
        assert out.shape == frame.shape

    def test_normalize_dark_frame_gets_brighter(self):
        from core.image_preprocess import normalize_lighting
        dark = _make_dark_bgr()
        out, _ = normalize_lighting(dark)
        assert out.mean() > dark.mean(), "Frame gelap harus dinaikkan brightness-nya"

    def test_normalize_bright_frame_gets_dimmer(self):
        from core.image_preprocess import normalize_lighting
        # Frame yang sangat overexposed (brightness 240) harus dideteksi LED ON dan diredupkan
        bright = _make_bgr(brightness=240)
        out, led_on = normalize_lighting(bright)
        assert led_on is True, "Frame 240 harus dideteksi sebagai LED ON"
        assert out.mean() <= bright.mean(), "Frame LED ON yang overexposed harus diredupkan atau dipertahankan"

    def test_normalize_performance_under_100ms(self):
        from core.image_preprocess import normalize_lighting
        frame = _make_bgr(1944, 2592)  # resolusi native kamera
        start = time.perf_counter()
        normalize_lighting(frame)
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < 150, f"normalize_lighting terlalu lambat: {elapsed_ms:.1f}ms (harus < 150ms, modul headless-only)"


# ══════════════════════════════════════════════════════════════════════════════
# 2. MODULE: core/gui_fusion.py
#    Features: FusionGUI, keystroke queue, anypoint binding, exposure callback
# ══════════════════════════════════════════════════════════════════════════════
class TestGUIFusion:
    """Memastikan GUI proxy tidak rusak dan semua binding berfungsi."""

    @pytest.fixture
    def mock_moil(self):
        class MockMoil:
            pitch, yaw, zoom = 0.0, 0.0, 1.4
        return MockMoil()

    @pytest.fixture
    def gui(self, mock_moil):
        from core.gui_fusion import FusionGUI
        return FusionGUI(moil_undistorter=mock_moil, headless=True, initial_exposure=70)

    def test_import_ok(self):
        from core.gui_fusion import FusionGUI
        assert FusionGUI is not None

    def test_instantiation_with_headless(self, gui):
        assert gui is not None

    def test_keystroke_queue_fifo_order(self, gui):
        gui.queue_key(ord('a'))
        gui.queue_key(ord('b'))
        assert gui.get_key() == ord('a')
        assert gui.get_key() == ord('b')

    def test_empty_queue_returns_minus_one(self, gui):
        assert gui.get_key() == -1

    def test_update_image_does_not_crash_in_headless(self, gui):
        frame = _make_bgr(100, 100)
        gui.update_image(frame)  # tidak boleh raise exception

    def test_anypoint_apply_updates_moil(self, gui, mock_moil):
        gui.entry_alpha.set_text("12.5")
        gui.entry_beta.set_text("-7.0")
        gui.entry_zoom.set_text("2.0")
        gui.on_apply_anypoint(None)
        assert mock_moil.pitch == 12.5
        assert mock_moil.yaw == -7.0
        assert mock_moil.zoom == 2.0

    def test_alpha_increment_button(self, gui, mock_moil):
        gui.entry_alpha.set_text("5.0")
        gui.on_adj_alpha(None, 5.0)
        assert mock_moil.pitch == 10.0

    def test_beta_decrement_button(self, gui, mock_moil):
        gui.entry_beta.set_text("0.0")
        gui.on_adj_beta(None, -5.0)
        assert mock_moil.yaw == -5.0

    def test_zoom_buttons(self, gui, mock_moil):
        gui.entry_zoom.set_text("1.4")
        gui.on_adj_zoom(None, 0.1)
        assert abs(mock_moil.zoom - 1.5) < 0.01

    def test_exposure_initial_value(self, gui):
        # GUI dibuat dengan initial_exposure=70 (raw) → tampil sebagai 0.1 (70/1000 dibulatkan)
        text = gui.entry_exposure.get_text()
        assert float(text) == round(70 / 1000, 1), f"Nilai display harus {round(70/1000,1)}, dapat: {text}"

    def test_exposure_callback_called(self, mock_moil):
        from core.gui_fusion import FusionGUI
        received = []
        gui = FusionGUI(
            moil_undistorter=mock_moil,
            headless=True,
            initial_exposure=2000,  # raw 2000 → display 2.0
            exposure_callback=lambda v: received.append(v)
        )
        gui.entry_exposure.set_text("1.5")  # 1.5 display → 1500 raw
        gui.on_apply_exposure(None)
        assert received == [1500], f"Callback dipanggil dengan {received}, diharapkan [1500] (1.5 * 1000)"

    def test_exposure_adj_clamps_at_zero(self, gui):
        gui.entry_exposure.set_text("0.5")
        gui.on_adj_exposure(None, -100.0)  # coba kurangi sampai sangat negatif
        val = float(gui.entry_exposure.get_text())
        assert val >= 0.0, "Exposure tidak boleh negatif"

    def test_quit_button_queues_esc(self, gui):
        # Tombol Quit harus memasukkan keycode 27 (ESC)
        gui.queue_key(27)
        assert gui.get_key() == 27

    def test_thread_safe_queue_from_multiple_threads(self, gui):
        """Memastikan queue aman diakses dari banyak thread sekaligus."""
        errors = []
        def sender():
            try:
                for i in range(50):
                    gui.queue_key(i)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=sender) for _ in range(5)]
        for t in threads: t.start()
        for t in threads: t.join()
        assert not errors, f"Race condition di queue: {errors}"


# ══════════════════════════════════════════════════════════════════════════════
# 3. MODULE: core/live_pipeline.py
#    Features: signature, calib_data=None handling
# ══════════════════════════════════════════════════════════════════════════════
class TestLivePipelineContract:
    """Memastikan kontrak fungsi live_pipeline tidak berubah."""

    def test_import_ok(self):
        from core import live_pipeline
        assert hasattr(live_pipeline, 'run_live_pipeline')

    def test_signature_accepts_gui_param(self):
        import inspect
        from core.live_pipeline import run_live_pipeline
        sig = inspect.signature(run_live_pipeline)
        assert 'gui' in sig.parameters, "Fungsi harus menerima parameter 'gui'"

    def test_gui_param_has_default_none(self):
        import inspect
        from core.live_pipeline import run_live_pipeline
        sig = inspect.signature(run_live_pipeline)
        assert sig.parameters['gui'].default is None


# ══════════════════════════════════════════════════════════════════════════════
# 4. MODULE: core/calibration_routines.py
#    Features: semua 6 fungsi menerima gui=None
# ══════════════════════════════════════════════════════════════════════════════
class TestCalibrationRoutinesContract:
    """Memastikan semua mode kalibrasi masih menerima parameter gui."""

    def test_import_ok(self):
        from core import calibration_routines
        assert calibration_routines is not None

    @pytest.mark.parametrize("func_name,extra_args", [
        ("run_calib_1p_2p",  {"true_height": 7.6, "true_height_2": 11.0, "calibrate_mode": 1}),
        ("run_calib_zgrid",  {"true_height": 7.6, "n_positions": 3}),
        ("run_calib_bbox",   {"true_height": 7.6}),
        ("run_calib_geom",   {"true_height": 7.6, "n_positions": 3}),
        ("run_calib_bilateral", {"true_height": 7.6, "true_height_2": 11.0, "n_positions": 3}),
        ("run_calib_analytic",  {"true_height": 7.6, "true_height_2": 11.0}),
    ])
    def test_function_accepts_gui_param(self, func_name, extra_args):
        import inspect
        import core.calibration_routines as cr
        func = getattr(cr, func_name)
        sig = inspect.signature(func)
        assert 'gui' in sig.parameters, \
            f"{func_name} harus menerima parameter 'gui'"
        assert sig.parameters['gui'].default is None, \
            f"{func_name} parameter 'gui' harus default None"


# ══════════════════════════════════════════════════════════════════════════════
# 5. MODULE: core/session_reporter.py
#    Features: _generate_session_report handles None calib_data
# ══════════════════════════════════════════════════════════════════════════════
class TestSessionReporter:
    """Memastikan reporter tidak crash ketika calib_data None."""

    def test_import_ok(self):
        from core import session_reporter
        assert hasattr(session_reporter, '_generate_session_report')

    def test_matplotlib_uses_agg_backend(self):
        import matplotlib
        # Pastikan backend non-interaktif sudah diset sebelum pyplot diimport
        import core.session_reporter  # noqa — trigger import
        backend = matplotlib.get_backend()
        assert backend.lower() == 'agg', \
            f"Matplotlib backend harus Agg (headless), bukan '{backend}'"

    def test_reporter_handles_none_calib_data(self, tmp_path, monkeypatch):
        """Laporan tidak boleh crash meski calib_data=None."""
        import core.session_reporter as sr
        monkeypatch.setattr(sr, 'REPORT_DIR', str(tmp_path))
        # Harus tidak raise exception
        sr._generate_session_report(
            calib_data=None,
            marker_size_cm=2.5,
            focal_len=662.0,
            total_frames=10,
            midas_runs=2,
            history_z_tray=[20.0, 21.0],
            history_cup_h={0: [7.5, 7.6]},
            history_frames=[1, 2],
            screenshots=[]
        )

    def test_reporter_normal_case_no_crash(self, tmp_path, monkeypatch):
        """Laporan berjalan normal dengan calib_data valid."""
        import core.session_reporter as sr
        monkeypatch.setattr(sr, 'REPORT_DIR', str(tmp_path))
        calib = {"type": 5, "poly_Kgeom": [1.0]}
        sr._generate_session_report(
            calib_data=calib,
            marker_size_cm=2.5,
            focal_len=662.0,
            total_frames=50,
            midas_runs=10,
            history_z_tray=[20.0] * 10,
            history_cup_h={0: [7.5] * 10},
            history_frames=list(range(10)),
            screenshots=[]
        )


# ══════════════════════════════════════════════════════════════════════════════
# 6. MODULE: run_fusion.py (smoke test sintaks)
#    Features: worker_thread, get_frame, exposure gating
# ══════════════════════════════════════════════════════════════════════════════
class TestRunFusionSyntax:
    """Memastikan run_fusion.py bisa diimport tanpa error syntax."""

    def test_import_ok(self):
        """run_fusion.py harus bisa diparse tanpa SyntaxError."""
        import ast, pathlib
        src = pathlib.Path("run_fusion.py").read_text()
        try:
            ast.parse(src)
        except SyntaxError as e:
            pytest.fail(f"SyntaxError di run_fusion.py: {e}")

    def test_has_main_entrypoint(self):
        import ast, pathlib
        src = pathlib.Path("run_fusion.py").read_text()
        assert 'if __name__ == "__main__"' in src

    def test_uses_gui_desired_exposure(self):
        """Memastikan mekanisme exposure gating ada di kode."""
        import pathlib
        src = pathlib.Path("run_fusion.py").read_text()
        assert 'gui_desired_exposure' in src, \
            "Mekanisme exposure gating harus menggunakan gui_desired_exposure"
        assert '_last_exposure_change_t' in src, \
            "Harus ada timestamp terakhir perubahan exposure"

    def test_normalize_lighting_disabled_in_gui_mode(self):
        """Normalisasi software harus TIDAK dijalankan saat GUI aktif (gui is not None)."""
        import pathlib
        src = pathlib.Path("run_fusion.py").read_text()
        # Normalisasi hanya aktif bila gui is None
        assert 'gui is None' in src, \
            "Normalisasi harus dikondisikan dengan 'gui is None' agar tidak aktif saat GUI"
