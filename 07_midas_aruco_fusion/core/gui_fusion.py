import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, GdkPixbuf, GLib, Gdk
import cv2
import queue

class FusionGUI(Gtk.Window):
    def __init__(self, moil_undistorter=None, headless=False, initial_exposure=0,
                 exposure_callback=None, gain_callback=None, brightness_callback=None):
        super().__init__(title="ArUco + MiDaS | Fusion Interface")
        self.set_default_size(1280, 720)
        
        self.moil_undistorter = moil_undistorter
        self.headless = headless
        self.smart_exposure_callback = exposure_callback
        self.initial_exposure = initial_exposure
        self._alive = True
        self._last_ui_frame_t = 0.0
        self.calibration_ready_event = __import__('threading').Event()
        # Cache boolean untuk normalize lighting — dibaca dari background thread
        # WAJIB Python bool biasa, bukan akses GTK widget (thread-unsafe → SIGABRT)
        self._normalize_enabled = False
        self.key_queue = queue.Queue()
        
        self.connect("destroy", self.on_destroy)
        self.setup_ui()
        
    def setup_ui(self):
        hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        self.add(hbox)
        
        # ─── Sidebar Kiri (Controls) ───
        vbox_ctrl = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=5)
        vbox_ctrl.set_border_width(10)
        vbox_ctrl.set_size_request(320, -1)
        
        # System Status Frame
        f_sys = Gtk.Frame(label="System Status")
        vb_s = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=5)
        vb_s.set_border_width(5)
        self.lbl_status_calib = Gtk.Label(label="Mode: Live")
        self.lbl_status_ai = Gtk.Label(label="AI: Ready")
        vb_s.pack_start(self.lbl_status_calib, False, False, 0)
        vb_s.pack_start(self.lbl_status_ai, False, False, 0)
        
        # Smart Exposure Control
        # Menggabungkan Exposure (waktu), Gain (ISO), dan Brightness (EV) ke dalam satu slider 1.0 - 10.0
        hb_exp = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=3)
        hb_exp.pack_start(Gtk.Label(label="Smart Exp:"), False, False, 0)
        
        btn_exp_mm = Gtk.Button(label="--")
        btn_exp_mm.connect("clicked", self.on_adj_smart_exposure, -2.0)
        btn_exp_min = Gtk.Button(label=" - ")
        btn_exp_min.connect("clicked", self.on_adj_smart_exposure, -0.5)
        btn_exp_plus = Gtk.Button(label=" + ")
        btn_exp_plus.connect("clicked", self.on_adj_smart_exposure, +0.5)
        btn_exp_pp = Gtk.Button(label="++")
        btn_exp_pp.connect("clicked", self.on_adj_smart_exposure, +2.0)
        
        display_val = max(1.0, min(10.0, round(self.initial_exposure / 1000, 1)))
        self.entry_exposure = Gtk.Entry(text=str(display_val))
        self.entry_exposure.set_tooltip_text("1.0 (Gelap) - 10.0 (Sangat Terang). Otomatis mengatur Shutter, ISO & EV.")
        self.entry_exposure.set_width_chars(6)
        
        hb_exp.pack_start(btn_exp_mm, False, False, 0)
        hb_exp.pack_start(btn_exp_min, False, False, 0)
        hb_exp.pack_start(self.entry_exposure, True, True, 0)
        hb_exp.pack_start(btn_exp_plus, False, False, 0)
        hb_exp.pack_start(btn_exp_pp, False, False, 0)
        
        btn_apply_exp = Gtk.Button(label="Apply")
        btn_apply_exp.connect("clicked", self.on_apply_smart_exposure)
        hb_exp.pack_start(btn_apply_exp, False, False, 0)
        
        vb_s.pack_start(hb_exp, False, False, 5)
        
        f_sys.add(vb_s)
        vbox_ctrl.pack_start(f_sys, False, False, 5)
        
        # Moildev Anypoint Frame
        f_moil = Gtk.Frame(label="Moildev Anypoint Controls")
        vb_m = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=5)
        vb_m.set_border_width(5)
        
        # Alpha (Pitch)
        hb_a = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
        hb_a.pack_start(Gtk.Label(label="Alpha (Pitch):"), False, False, 0)
        btn_a_min = Gtk.Button(label=" - ")
        btn_a_min.connect("clicked", self.on_adj_alpha, -5.0)
        btn_a_plus = Gtk.Button(label=" + ")
        btn_a_plus.connect("clicked", self.on_adj_alpha, +5.0)
        self.entry_alpha = Gtk.Entry(text="0.0")
        hb_a.pack_start(btn_a_min, False, False, 0)
        hb_a.pack_start(self.entry_alpha, True, True, 0)
        hb_a.pack_start(btn_a_plus, False, False, 0)
        vb_m.pack_start(hb_a, False, False, 0)
        
        # Beta (Yaw)
        hb_b = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
        hb_b.pack_start(Gtk.Label(label="Beta (Yaw):  "), False, False, 0)
        btn_b_min = Gtk.Button(label=" - ")
        btn_b_min.connect("clicked", self.on_adj_beta, -5.0)
        btn_b_plus = Gtk.Button(label=" + ")
        btn_b_plus.connect("clicked", self.on_adj_beta, +5.0)
        self.entry_beta = Gtk.Entry(text="0.0")
        hb_b.pack_start(btn_b_min, False, False, 0)
        hb_b.pack_start(self.entry_beta, True, True, 0)
        hb_b.pack_start(btn_b_plus, False, False, 0)
        vb_m.pack_start(hb_b, False, False, 0)
        
        # Zoom
        hb_z = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
        hb_z.pack_start(Gtk.Label(label="Zoom:          "), False, False, 0)
        btn_z_min = Gtk.Button(label=" - ")
        btn_z_min.connect("clicked", self.on_adj_zoom, -0.1)
        btn_z_plus = Gtk.Button(label=" + ")
        btn_z_plus.connect("clicked", self.on_adj_zoom, +0.1)
        self.entry_zoom = Gtk.Entry(text="1.4")
        hb_z.pack_start(btn_z_min, False, False, 0)
        hb_z.pack_start(self.entry_zoom, True, True, 0)
        hb_z.pack_start(btn_z_plus, False, False, 0)
        vb_m.pack_start(hb_z, False, False, 0)
        
        # Apply & Reset Buttons
        hb_moil_btns = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
        btn_apply = Gtk.Button(label="Apply Anypoint")
        btn_apply.connect("clicked", self.on_apply_anypoint)
        btn_reset = Gtk.Button(label="Reset View")
        btn_reset.connect("clicked", self.on_reset_anypoint)
        hb_moil_btns.pack_start(btn_apply, True, True, 0)
        hb_moil_btns.pack_start(btn_reset, True, True, 0)
        vb_m.pack_start(hb_moil_btns, False, False, 5)
        
        f_moil.add(vb_m)
        vbox_ctrl.pack_start(f_moil, False, False, 5)
        
        # Actions Frame
        f_act = Gtk.Frame(label="Actions")
        vb_a = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=5)
        vb_a.set_border_width(5)
        
        # Tombol Start Calibration — hanya aktif di setup mode sebelum kalibrasi
        self.btn_start_calib = Gtk.Button(label="▶  Start Calibration")
        self.btn_start_calib.connect("clicked", self.on_start_calibration)
        self.btn_start_calib.get_style_context().add_class("suggested-action")
        vb_a.pack_start(self.btn_start_calib, False, False, 0)

        # Tombol Next Step (Pengganti Spasi)
        self.btn_next_step = Gtk.Button(label="⏭  Next Step (Space)")
        self.btn_next_step.connect("clicked", lambda w: self.queue_key(ord(' ')))
        self.btn_next_step.get_style_context().add_class("suggested-action")
        vb_a.pack_start(self.btn_next_step, False, False, 0)
        
        # Toggle Normalize Lighting
        self.chk_normalize = Gtk.CheckButton(label="Enable Normalize Lighting")
        self.chk_normalize.set_active(False)  # default mati
        # Update cache Python saat checkbox diubah (JANGAN baca widget dari background thread!)
        self.chk_normalize.connect("toggled", self._on_normalize_toggled)
        vb_a.pack_start(self.chk_normalize, False, False, 5)
        
        self.lbl_setup_hint = Gtk.Label(label="")
        self.lbl_setup_hint.set_line_wrap(True)
        vb_a.pack_start(self.lbl_setup_hint, False, False, 0)
        
        btn_record = Gtk.Button(label="Toggle Record (R)")
        btn_record.connect("clicked", lambda w: self.queue_key(ord('r')))
        
        btn_screenshot = Gtk.Button(label="Screenshot (S)")
        btn_screenshot.connect("clicked", lambda w: self.queue_key(ord('s')))
        
        btn_cap = Gtk.Button(label="Capture Calib Point (C)")
        btn_cap.connect("clicked", lambda w: self.queue_key(ord('c')))
        
        btn_quit = Gtk.Button(label="Quit (ESC)")
        btn_quit.connect("clicked", lambda w: self.queue_key(27))
        
        vb_a.pack_start(btn_record, False, False, 0)
        vb_a.pack_start(btn_screenshot, False, False, 0)
        vb_a.pack_start(btn_cap, False, False, 0)
        vb_a.pack_start(btn_quit, False, False, 0)
        
        # Sembunyikan tombol Start Calibration & Next Step secara default
        self.btn_start_calib.hide()
        self.btn_next_step.hide()
        self.lbl_setup_hint.hide()
        
        f_act.add(vb_a)
        vbox_ctrl.pack_start(f_act, False, False, 5)
        
        hbox.pack_start(vbox_ctrl, False, False, 0)
        
        # ─── Main Video Feed (Kanan) ───
        self.image = Gtk.Image()
        hbox.pack_start(self.image, True, True, 0)
        
        # Listen for key presses on the window itself
        self.connect("key-press-event", self.on_key_press)
        
        # Inisialisasi nilai awal
        if self.moil_undistorter:
            self.entry_alpha.set_text(str(self.moil_undistorter.pitch))
            self.entry_beta.set_text(str(self.moil_undistorter.yaw))
            self.entry_zoom.set_text(str(self.moil_undistorter.zoom))

    def update_image(self, frame_bgr):
        """Called by background thread. Throttled to max 20 FPS to avoid GLib queue flooding."""
        if self.headless or not self._alive:
            return

        import time
        now = time.monotonic()
        if now - self._last_ui_frame_t < 0.05:  # max 20 FPS
            return
        self._last_ui_frame_t = now

        # Lakukan resize dan konversi BGR→RGB di background thread (aman, tidak menyentuh GTK)
        h, w = frame_bgr.shape[:2]
        target_w = 960
        if w > target_w:
            scale = target_w / float(w)
            new_h = int(h * scale)
            # Menggunakan INTER_AREA sangat penting untuk downscaling resolusi tinggi agar tidak ada aliasing/garis gergaji di layar
            frame_bgr = cv2.resize(frame_bgr, (target_w, new_h), interpolation=cv2.INTER_AREA)

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        out_h, out_w = rgb.shape[:2]
        # Salin data ke GLib.Bytes — dikelola oleh GLib (aman lintas thread)
        glib_bytes = GLib.Bytes.new(rgb.tobytes())
        GLib.idle_add(self._update_image_from_glib_bytes, glib_bytes, out_w, out_h)

    def _update_image_from_glib_bytes(self, glib_bytes, w, h):
        """Dijalankan di GTK main thread — aman memanggil widget GTK.
        WAJIB pakai new_from_bytes (bukan new_from_data): new_from_data tidak menyalin
        data sehingga saat Python GC membebaskan glib_bytes, GTK akan akses dangling pointer.
        new_from_bytes menyalin data ke memory yang dikelola GLib — aman dari GC Python.
        """
        if self.headless or not self._alive:
            return False
        try:
            pb = GdkPixbuf.Pixbuf.new_from_bytes(
                glib_bytes,
                GdkPixbuf.Colorspace.RGB,
                False,   # has_alpha
                8,       # bits_per_sample
                w, h,
                w * 3,   # rowstride
            )
            self.image.set_from_pixbuf(pb)
        except Exception:
            pass
        return False

    def on_adj_alpha(self, widget, delta):
        try:
            val = float(self.entry_alpha.get_text()) + delta
            self.entry_alpha.set_text(str(round(val, 1)))
            self.on_apply_anypoint(None)
        except ValueError:
            pass

    def on_adj_beta(self, widget, delta):
        try:
            val = float(self.entry_beta.get_text()) + delta
            self.entry_beta.set_text(str(round(val, 1)))
            self.on_apply_anypoint(None)
        except ValueError:
            pass

    def on_adj_zoom(self, widget, delta):
        try:
            val = float(self.entry_zoom.get_text()) + delta
            self.entry_zoom.set_text(str(round(val, 2)))
            self.on_apply_anypoint(None)
        except ValueError:
            pass
            
    def on_adj_smart_exposure(self, widget, delta):
        try:
            val = round(float(self.entry_exposure.get_text()) + delta, 1)
            val = max(1.0, min(10.0, val))
            self.entry_exposure.set_text(str(val))
            self.on_apply_smart_exposure(None)
        except ValueError:
            pass

    def on_apply_smart_exposure(self, widget):
        if self.smart_exposure_callback:
            try:
                val = float(self.entry_exposure.get_text())
                val = max(1.0, min(10.0, val))
                
                # Pemetaan Smart Exposure (1.0 - 10.0):
                # Exposure time (Raw): 1000 -> 10000
                raw_exp = int(round(val * 1000))
                
                # Gain (0 - 255): Scale linearly from val 1.0 to 10.0
                raw_gain = int(round((val - 1.0) / 9.0 * 255.0))
                
                # Brightness EV (-64 to +64): Scale linearly
                raw_bri = int(round((val - 1.0) / 9.0 * 128.0 - 64.0))
                
                # Panggil satu callback dengan 3 nilai
                self.smart_exposure_callback(raw_exp, raw_gain, raw_bri)
            except ValueError:
                pass

    def on_apply_anypoint(self, widget):
        if not self.moil_undistorter:
            return
        try:
            a = float(self.entry_alpha.get_text())
            b = float(self.entry_beta.get_text())
            z = float(self.entry_zoom.get_text())
            self.moil_undistorter.pitch = a
            self.moil_undistorter.yaw = b
            self.moil_undistorter.zoom = z
        except ValueError:
            pass

    def on_reset_anypoint(self, widget):
        self.entry_alpha.set_text("0.0")
        self.entry_beta.set_text("0.0")
        self.entry_zoom.set_text("1.4")
        self.on_apply_anypoint(None)

    def on_start_calibration(self, widget):
        """Dipanggil saat user menekan tombol Start Calibration."""
        self.calibration_ready_event.set()
        GLib.idle_add(lambda: (self.btn_start_calib.set_sensitive(False),
                               self.btn_start_calib.hide(),
                               self.btn_next_step.show(),
                               self.lbl_setup_hint.set_text("Calibration in progress...\nClick 'Next Step' when prompted.")) or False)

    def enter_setup_mode(self, calib_name="Calibration"):
        """Dipanggil dari worker thread untuk menampilkan tombol Start Calibration."""
        hint = f"Setup mode: adjust exposure & anypoint,\nthen click \u25b6 Start {calib_name}"
        GLib.idle_add(lambda: (self.btn_start_calib.show(),
                               self.lbl_setup_hint.show(),
                               self.lbl_setup_hint.set_text(hint),
                               self.lbl_status_calib.set_text(f"Setup: {calib_name}")) or False)

    def _on_normalize_toggled(self, widget):
        """Dijalankan di GTK main thread — update cache Python yang aman dibaca dari thread lain."""
        self._normalize_enabled = widget.get_active()

    def is_normalize_enabled(self):
        """Dibaca dari background thread — hanya kembalikan Python bool (BUKAN akses GTK widget)."""
        return self._normalize_enabled

    def on_key_press(self, widget, event):
        # Convert GDK keyval to ascii
        keyname = Gdk.keyval_name(event.keyval)
        if keyname and len(keyname) == 1:
            self.queue_key(ord(keyname.lower()))
        elif keyname == "Escape":
            self.queue_key(27)
        return False

    def queue_key(self, key_code):
        self.key_queue.put(key_code)

    def get_key(self):
        """Pengganti cv2.waitKey(1) untuk worker thread."""
        try:
            return self.key_queue.get_nowait()
        except queue.Empty:
            return -1

    def on_destroy(self, widget):
        self._alive = False   # Hentikan semua idle callbacks sebelum Gtk.main_quit
        self.queue_key(27)    # Kirim ESC ke background thread agar loop berhenti
        Gtk.main_quit()
