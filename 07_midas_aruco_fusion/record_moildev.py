import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, Gdk, GdkPixbuf, GLib
import cv2
cv2.ocl.setUseOpenCL(False)  # Disable OpenCL for stability
import os
import sys
import time
from datetime import datetime
import threading
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_MOIL_DIR = os.path.join(_THIS_DIR, "moildev")
if _MOIL_DIR not in sys.path:
    sys.path.insert(0, _MOIL_DIR)

try:
    from Moildev import Moildev as MoildevLib
except ImportError as e:
    print(f"[ERROR] Gagal import Moildev: {e}")
    sys.exit(1)

SAVE_DIR = os.path.join(_THIS_DIR, "recorded_videos")
os.makedirs(SAVE_DIR, exist_ok=True)


class MoildevRecordingWindow(Gtk.Window):
    def __init__(self):
        super().__init__(title="Moildev Remap Recording")
        self.set_default_size(1100, 640)
        self.connect("destroy", self.on_destroy)

        self.cap = None
        self.running = False
        self.is_recording = False
        self.out_video = None
        self.fps = 20.0
        self.recorded_frames_count = 0
        self.prev_time = time.time()

        self.moil = None
        self.map_x = None
        self.map_y = None
        self.current_moil_params = None
        self.requested_cam_id = None
        self.lock = threading.Lock()

        # --- Smart Exposure State (diakses dari background thread) ---
        # Simpan sebagai Python state biasa (BUKAN akses GTK dari thread) untuk thread-safety
        self._desired_exp   = [None]  # raw exposure value, None = belum diset
        self._desired_gain  = [None]  # 0-255
        self._desired_bri   = [None]  # -64 to 64
        self._current_exp   = [None]
        self._current_gain  = [None]
        self._current_bri   = [None]
        self._hw_flush      = [0]     # frame flush counter pasca hw-change

        self.setup_ui()
        self.init_moildev()
        self.lbl_status.set_text("Ready. Camera Starting...")
        self.start_camera()

    # ─── Moildev ─────────────────────────────────────────────────────────────

    def init_moildev(self):
        cam_name = self.entry_moil_cam.get_text()
        json_path = os.path.join(_THIS_DIR, "camera_parameters.json")
        try:
            self.moil = MoildevLib(json_path, cam_name)
            self.update_maps()
        except Exception as e:
            print(f"[ERROR] Inisiasi Moildev gagal: {e}")

    def update_maps(self):
        if not self.moil:
            return
        alpha = self.scale_alpha_adj.get_value()
        beta  = self.scale_beta_adj.get_value()
        zoom  = self.scale_zoom_adj.get_value()
        params = (alpha, beta, zoom)
        if params != self.current_moil_params:
            map_x, map_y = self.moil.maps_anypoint_mode1(alpha, beta, zoom)
            with self.lock:
                self.map_x = cv2.UMat(map_x.astype(np.float32))
                self.map_y = cv2.UMat(map_y.astype(np.float32))
            self.current_moil_params = params

    # ─── UI ──────────────────────────────────────────────────────────────────

    def setup_ui(self):
        hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        self.add(hbox)

        vbox_ctrl = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=5)
        vbox_ctrl.set_border_width(10)
        vbox_ctrl.set_size_request(340, -1)

        # ── System Settings ──────────────────────────────────────────────────
        f_sys = Gtk.Frame(label="System Settings")
        vb_s = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=5)

        # Camera index
        hb_cam = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
        self.entry_cam = Gtk.Entry(text="0")
        btn_switch = Gtk.Button(label="Switch")
        btn_switch.connect("clicked", self.on_switch_camera)
        hb_cam.pack_start(Gtk.Label(label="Cam Index:"), False, False, 0)
        hb_cam.pack_start(self.entry_cam, True, True, 0)
        hb_cam.pack_start(btn_switch, False, False, 0)
        vb_s.pack_start(hb_cam, False, False, 2)

        # Smart Exposure slider  1.0 – 10.0
        hb_exp = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=3)
        hb_exp.pack_start(Gtk.Label(label="Smart Exp:"), False, False, 0)
        btn_emm = Gtk.Button(label="--")
        btn_emm.connect("clicked", self._on_adj_exp, -2.0)
        btn_em  = Gtk.Button(label=" - ")
        btn_em.connect("clicked", self._on_adj_exp, -0.5)
        btn_ep  = Gtk.Button(label=" + ")
        btn_ep.connect("clicked", self._on_adj_exp, +0.5)
        btn_epp = Gtk.Button(label="++")
        btn_epp.connect("clicked", self._on_adj_exp, +2.0)
        self.entry_exp = Gtk.Entry(text="5.0")
        self.entry_exp.set_tooltip_text("1.0 (Gelap) → 10.0 (Sangat Terang). Otomatis atur Shutter+ISO+EV.")
        self.entry_exp.set_width_chars(5)
        for w in (btn_emm, btn_em, self.entry_exp, btn_ep, btn_epp):
            hb_exp.pack_start(w, False, False, 0)
        btn_apply_exp = Gtk.Button(label="Apply")
        btn_apply_exp.connect("clicked", self._on_apply_exp)
        hb_exp.pack_start(btn_apply_exp, False, False, 0)
        vb_s.pack_start(hb_exp, False, False, 2)

        f_sys.add(vb_s)
        vbox_ctrl.pack_start(f_sys, False, False, 0)

        # ── Moildev Fisheye ──────────────────────────────────────────────────
        f_moil = Gtk.Frame(label="Moildev Fisheye (Mode 1)")
        vb_m = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=5)
        self.entry_moil_cam = Gtk.Entry(text="syue_7730v1_6")
        vb_m.pack_start(Gtk.Label(label="Camera Name:"), 0, 0, 0)
        vb_m.pack_start(self.entry_moil_cam, 0, 0, 0)

        btn_reload = Gtk.Button(label="Reload Camera Params")
        btn_reload.connect("clicked", self.on_reload_moil)
        vb_m.pack_start(btn_reload, False, False, 0)

        self.scale_alpha_adj = Gtk.Adjustment(value=0, lower=0, upper=110, step_increment=1, page_increment=10, page_size=0)
        self.scale_beta_adj  = Gtk.Adjustment(value=0, lower=0, upper=360, step_increment=1, page_increment=10, page_size=0)
        self.scale_zoom_adj  = Gtk.Adjustment(value=4, lower=1, upper=20,  step_increment=1, page_increment=2,  page_size=0)

        def on_moil_scale_changed(widget):
            self.update_maps()

        self.scale_alpha_adj.connect("value-changed", on_moil_scale_changed)
        self.scale_beta_adj.connect("value-changed", on_moil_scale_changed)
        self.scale_zoom_adj.connect("value-changed", on_moil_scale_changed)

        s_alpha = Gtk.Scale(orientation=Gtk.Orientation.HORIZONTAL, adjustment=self.scale_alpha_adj)
        s_alpha.set_digits(0); s_alpha.set_value_pos(Gtk.PositionType.RIGHT)
        s_beta  = Gtk.Scale(orientation=Gtk.Orientation.HORIZONTAL, adjustment=self.scale_beta_adj)
        s_beta.set_digits(0);  s_beta.set_value_pos(Gtk.PositionType.RIGHT)
        s_zoom  = Gtk.Scale(orientation=Gtk.Orientation.HORIZONTAL, adjustment=self.scale_zoom_adj)
        s_zoom.set_digits(1);  s_zoom.set_value_pos(Gtk.PositionType.RIGHT)

        vb_m.pack_start(Gtk.Label(label="Alpha:"), 0, 0, 0); vb_m.pack_start(s_alpha, 0, 0, 0)
        vb_m.pack_start(Gtk.Label(label="Beta:"),  0, 0, 0); vb_m.pack_start(s_beta,  0, 0, 0)
        vb_m.pack_start(Gtk.Label(label="Zoom:"),  0, 0, 0); vb_m.pack_start(s_zoom,  0, 0, 0)

        f_moil.add(vb_m)
        vbox_ctrl.pack_start(f_moil, False, False, 10)

        # ── Recording ────────────────────────────────────────────────────────
        f_rec = Gtk.Frame(label="Recording")
        vb_r = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=5)
        self.btn_record = Gtk.Button(label="⏺  Start Recording")
        self.btn_record.connect("clicked", self.on_toggle_record)
        vb_r.pack_start(self.btn_record, 0, 0, 0)
        f_rec.add(vb_r)
        vbox_ctrl.pack_start(f_rec, False, False, 5)

        # ── Extract Frames ───────────────────────────────────────────────────
        f_ext = Gtk.Frame(label="Extract Frames from Video")
        vb_e = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=5)

        # File chooser row
        hb_file = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
        self.entry_vid_path = Gtk.Entry(placeholder_text="Path video .avi / .mp4")
        btn_browse = Gtk.Button(label="Browse…")
        btn_browse.connect("clicked", self.on_browse_video)
        hb_file.pack_start(self.entry_vid_path, True, True, 0)
        hb_file.pack_start(btn_browse, False, False, 0)
        vb_e.pack_start(hb_file, False, False, 0)

        # Every-N-frame option
        hb_nth = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
        hb_nth.pack_start(Gtk.Label(label="Setiap N frame:"), False, False, 0)
        self.spin_nth = Gtk.SpinButton.new_with_range(1, 300, 1)
        self.spin_nth.set_value(1)
        self.spin_nth.set_tooltip_text("1 = ekstrak semua frame, 5 = ambil 1 setiap 5 frame, dst.")
        hb_nth.pack_start(self.spin_nth, False, False, 0)
        vb_e.pack_start(hb_nth, False, False, 0)

        btn_extract = Gtk.Button(label="⬇  Extract Frames Sekarang")
        btn_extract.connect("clicked", self.on_extract_frames)
        vb_e.pack_start(btn_extract, False, False, 0)

        self.lbl_extract_status = Gtk.Label(label="")
        self.lbl_extract_status.set_line_wrap(True)
        vb_e.pack_start(self.lbl_extract_status, False, False, 0)

        f_ext.add(vb_e)
        vbox_ctrl.pack_start(f_ext, False, False, 5)

        # ── Status ───────────────────────────────────────────────────────────
        self.lbl_status = Gtk.Label(label="Status...")
        vbox_ctrl.pack_start(self.lbl_status, False, False, 0)
        self.lbl_fps = Gtk.Label(label="FPS: 0.0")
        vbox_ctrl.pack_start(self.lbl_fps, False, False, 0)

        hbox.pack_start(vbox_ctrl, False, False, 0)
        self.image_widget = Gtk.Image()
        hbox.pack_start(self.image_widget, True, True, 0)

    # ─── Smart Exposure helpers ──────────────────────────────────────────────

    def _smart_exp_values(self, val: float):
        """Dari slider 1.0-10.0 hitung raw_exp, raw_gain, raw_bri."""
        val = max(1.0, min(10.0, val))
        raw_exp  = int(round(val * 1000))
        raw_gain = int(round((val - 1.0) / 9.0 * 255.0))
        raw_bri  = int(round((val - 1.0) / 9.0 * 128.0 - 64.0))
        return raw_exp, raw_gain, raw_bri

    def _on_adj_exp(self, widget, delta):
        try:
            val = round(float(self.entry_exp.get_text()) + delta, 1)
            val = max(1.0, min(10.0, val))
            self.entry_exp.set_text(str(val))
            self._on_apply_exp(None)
        except ValueError:
            pass

    def _on_apply_exp(self, widget):
        try:
            val = max(1.0, min(10.0, float(self.entry_exp.get_text())))
            raw_exp, raw_gain, raw_bri = self._smart_exp_values(val)
            # Simpan ke shared state — akan dibaca oleh background thread
            self._desired_exp[0]  = raw_exp
            self._desired_gain[0] = raw_gain
            self._desired_bri[0]  = raw_bri
            print(f"[EXP] Smart Exp {val:.1f} → Shutter:{raw_exp}, ISO:{raw_gain}, EV:{raw_bri}")
        except ValueError:
            pass

    # ─── Camera loop ─────────────────────────────────────────────────────────

    def start_camera(self):
        self.running = True
        threading.Thread(target=self.run_loop, daemon=True).start()

    def _apply_hw_if_needed(self):
        """Cek dan terapkan perubahan hardware exposure ke kamera. Thread-safe (hanya akses Python state)."""
        changed = False

        if self._desired_exp[0] is not None and self._desired_exp[0] != self._current_exp[0]:
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # Manual mode V4L2
            self.cap.set(cv2.CAP_PROP_EXPOSURE, self._desired_exp[0])
            self._current_exp[0] = self._desired_exp[0]
            changed = True

        if self._desired_gain[0] is not None and self._desired_gain[0] != self._current_gain[0]:
            self.cap.set(cv2.CAP_PROP_GAIN, self._desired_gain[0])
            self._current_gain[0] = self._desired_gain[0]
            changed = True

        if self._desired_bri[0] is not None and self._desired_bri[0] != self._current_bri[0]:
            self.cap.set(cv2.CAP_PROP_BRIGHTNESS, self._desired_bri[0])
            self._current_bri[0] = self._desired_bri[0]
            changed = True

        if changed:
            self._hw_flush[0] = 6  # Buang 6 frame stale pasca-perubahan hardware

    def run_loop(self):
        current_idx = None
        while self.running:
            try:
                with self.lock:
                    if self.requested_cam_id is not None:
                        new_idx = self.requested_cam_id
                        self.requested_cam_id = None
                        if new_idx != current_idx:
                            if self.cap is not None:
                                self.cap.release()
                                self.cap = None
                            current_idx = new_idx
                            GLib.idle_add(self.lbl_status.set_text, f"Switching to Camera {current_idx}...")

                if self.cap is None:
                    if current_idx is None:
                        cam_id = self.entry_cam.get_text()
                        current_idx = int(cam_id) if cam_id.isdigit() else cam_id

                    self.cap = cv2.VideoCapture(current_idx)
                    self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2592)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1944)
                    # Default: auto exposure (user bisa ubah via Smart Exp)
                    self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
                    self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)

                    # Reset state hw agar slider pertama kali langsung diterapkan
                    self._current_exp[0]  = None
                    self._current_gain[0] = None
                    self._current_bri[0]  = None
                    self._hw_flush[0] = 0

                    if not self.cap or not self.cap.isOpened():
                        GLib.idle_add(self.lbl_status.set_text, f"FAILED to open Camera {current_idx}. Retrying...")
                        if self.cap:
                            self.cap.release()
                        self.cap = None
                        time.sleep(2.0)
                        continue
                    else:
                        GLib.idle_add(self.lbl_status.set_text, f"Camera {current_idx} Active")

                # Terapkan perubahan hardware jika ada
                self._apply_hw_if_needed()

                ret, frame = self.cap.read()
                if not ret:
                    time.sleep(0.01)
                    continue

                # Buang frame stale pasca hardware change
                if self._hw_flush[0] > 0:
                    self._hw_flush[0] -= 1
                    ret2, frame2 = self.cap.read()
                    if ret2 and frame2 is not None:
                        frame = frame2

                with self.lock:
                    map_x_copy = self.map_x
                    map_y_copy = self.map_y

                if self.moil and map_x_copy is not None and map_y_copy is not None:
                    umat_frame   = cv2.UMat(frame)
                    remapped_umat = cv2.remap(umat_frame, map_x_copy, map_y_copy,
                                              cv2.INTER_LINEAR,
                                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                    remapped_frame = remapped_umat.get()
                else:
                    remapped_frame = frame.copy()

                disp_frame = remapped_frame.copy()

                is_rec_ui = False
                with self.lock:
                    if self.is_recording and self.out_video is not None:
                        # Ensure contiguous memory for FFmpeg backend safety
                        self.out_video.write(np.ascontiguousarray(remapped_frame))
                        self.recorded_frames_count += 1
                        is_rec_ui = True

                if is_rec_ui:
                    cv2.circle(disp_frame, (30, 30), 10, (0, 0, 255), -1)
                    cv2.putText(disp_frame, "REC", (50, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                curr_time = time.time()
                fps_val = 1.0 / (curr_time - self.prev_time) if (curr_time - self.prev_time) > 0 else 0
                self.prev_time = curr_time
                GLib.idle_add(self.lbl_fps.set_text, f"FPS: {fps_val:.1f}")

                h_disp, w_disp = disp_frame.shape[:2]
                cv2.putText(disp_frame, f"FPS: {fps_val:.1f}", (10, h_disp - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                GLib.idle_add(self.update_ui, disp_frame)

            except Exception as e:
                print(f"Error in run_loop: {e}")
                time.sleep(1.0)

        if self.cap:
            self.cap.release()
            self.cap = None
        if self.out_video:
            self.out_video.release()
            self.out_video = None

    # ─── UI update ───────────────────────────────────────────────────────────

    def update_ui(self, frame_bgr):
        if not self.running:
            return False
        h, w = frame_bgr.shape[:2]
        if w > 1280:
            scale = 1280 / float(w)
            frame_bgr = cv2.resize(frame_bgr, (1280, int(h * scale)))

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        glib_bytes = GLib.Bytes.new(rgb.tobytes())
        pb = GdkPixbuf.Pixbuf.new_from_bytes(glib_bytes, GdkPixbuf.Colorspace.RGB, False, 8, w, h, w * 3)
        self.image_widget.set_from_pixbuf(pb)
        return False

    # ─── Callbacks ───────────────────────────────────────────────────────────

    def on_switch_camera(self, widget):
        cam_id = self.entry_cam.get_text()
        idx = int(cam_id) if cam_id.isdigit() else cam_id
        with self.lock:
            self.requested_cam_id = idx

    def on_reload_moil(self, widget):
        self.init_moildev()

    def on_toggle_record(self, widget):
        with self.lock:
            if not self.is_recording:
                # Dapatkan resolusi frame
                w, h = 2592, 1944 # default fallback
                if self.map_x is not None:
                    h, w = self.map_x.get().shape[:2]
                elif self.cap and self.cap.isOpened():
                    w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = os.path.join(SAVE_DIR, f"moildev_remap_{ts}.avi")
                # Gunakan MJPG karena lebih stabil dan minim konflik thread dengan GTK
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                self.out_video = cv2.VideoWriter(output_path, fourcc, int(self.fps), (w, h))
                self._last_video_path = output_path
                self.recorded_frames_count = 0
                self.is_recording = True
                
                self.btn_record.set_label("⏹  Stop Recording")
                self.lbl_status.set_text(f"Recording: {output_path}")
                print("[INFO] Recording started.")
            else:
                self.is_recording = False
                self.btn_record.set_label("⏺  Start Recording")
                if self.out_video:
                    self.out_video.release()
                    self.out_video = None
                self.lbl_status.set_text("Recording Saved.")
                print(f"\n[REPORT] Selesai merekam. Total frame: {self.recorded_frames_count}\n")
                # Otomatis isi path ke kolom extract
                if hasattr(self, '_last_video_path') and self._last_video_path:
                    self.entry_vid_path.set_text(self._last_video_path)

    def on_browse_video(self, widget):
        dialog = Gtk.FileChooserDialog(
            title="Pilih file video",
            parent=self,
            action=Gtk.FileChooserAction.OPEN,
        )
        dialog.add_buttons(Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,
                           Gtk.STOCK_OPEN,   Gtk.ResponseType.OK)
        ff = Gtk.FileFilter()
        ff.set_name("Video (*.avi, *.mp4, *.mkv)")
        ff.add_pattern("*.avi"); ff.add_pattern("*.mp4"); ff.add_pattern("*.mkv")
        dialog.add_filter(ff)
        if dialog.run() == Gtk.ResponseType.OK:
            self.entry_vid_path.set_text(dialog.get_filename())
        dialog.destroy()

    def on_extract_frames(self, widget):
        vid_path = self.entry_vid_path.get_text().strip()
        if not vid_path or not os.path.isfile(vid_path):
            self.lbl_extract_status.set_text("❌ File video tidak ditemukan!")
            return
        nth = int(self.spin_nth.get_value())
        self.lbl_extract_status.set_text("⏳ Sedang mengekstrak...")
        # Jalankan di thread agar UI tidak freeze
        threading.Thread(target=self._extract_thread, args=(vid_path, nth), daemon=True).start()

    def _extract_thread(self, vid_path: str, nth: int):
        """Ekstrak video menjadi gambar PNG frame-by-frame di background thread."""
        base = os.path.splitext(os.path.basename(vid_path))[0]
        out_dir = os.path.join(SAVE_DIR, f"{base}_frames")
        os.makedirs(out_dir, exist_ok=True)

        cap = cv2.VideoCapture(vid_path)
        if not cap.isOpened():
            GLib.idle_add(self.lbl_extract_status.set_text, "❌ Gagal membuka video!")
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        saved = 0
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % nth == 0:
                fname = os.path.join(out_dir, f"frame_{frame_idx:06d}.png")
                cv2.imwrite(fname, frame)
                saved += 1
                # Update status setiap 10 frame tersimpan
                if saved % 10 == 0:
                    GLib.idle_add(
                        self.lbl_extract_status.set_text,
                        f"⏳ {saved} frame tersimpan... ({frame_idx}/{total_frames})"
                    )
            frame_idx += 1

        cap.release()
        msg = f"✅ {saved} frame disimpan ke:\n{out_dir}"
        print(f"\n[EXTRACT] {msg}\n")
        GLib.idle_add(self.lbl_extract_status.set_text, msg)

    def on_destroy(self, widget):
        self.running = False
        Gtk.main_quit()


if __name__ == "__main__":
    win = MoildevRecordingWindow()
    win.show_all()
    Gtk.main()
