import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, Gdk, GdkPixbuf, GLib
import cv2
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
        self.set_default_size(1000, 600)
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
        
        self.setup_ui()
        
        # Load Moildev
        self.init_moildev()
        
        self.lbl_status.set_text("Ready. Camera Starting...")
        self.start_camera()

    def init_moildev(self):
        cam_name = self.entry_moil_cam.get_text()
        json_path = os.path.join(_THIS_DIR, "camera_parameters.json")
        try:
            self.moil = MoildevLib(json_path, cam_name)
            self.update_maps()
        except Exception as e:
            print(f"[ERROR] Inisiasi Moildev gagal: {e}")

    def update_maps(self):
        if not self.moil: return
        alpha = self.scale_alpha_adj.get_value()
        beta = self.scale_beta_adj.get_value()
        zoom = self.scale_zoom_adj.get_value()
        params = (alpha, beta, zoom)
        
        if params != self.current_moil_params:
            map_x, map_y = self.moil.maps_anypoint_mode1(alpha, beta, zoom)
            # Konversi map ke format yang lebih optimal dan gunakan UMat untuk akselerasi OpenCL
            self.map_x = cv2.UMat(map_x.astype(np.float32))
            self.map_y = cv2.UMat(map_y.astype(np.float32))
            self.current_moil_params = params

    def setup_ui(self):
        hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        self.add(hbox)
        
        vbox_ctrl = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=5)
        vbox_ctrl.set_border_width(10)
        vbox_ctrl.set_size_request(320, -1)
        
        # System Frame
        f_sys = Gtk.Frame(label="System Settings")
        vb_s = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=5)
        hb_cam = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
        self.entry_cam = Gtk.Entry(text="0")
        btn_switch = Gtk.Button(label="Switch")
        btn_switch.connect("clicked", self.on_switch_camera)
        hb_cam.pack_start(self.entry_cam, True, True, 0)
        hb_cam.pack_start(btn_switch, False, False, 0)
        vb_s.pack_start(Gtk.Label(label="Cam Index:"), 0, 0, 0)
        vb_s.pack_start(hb_cam, 0, 0, 0)
        f_sys.add(vb_s)
        vbox_ctrl.pack_start(f_sys, False, False, 0)

        # Moil Frame
        f_moil = Gtk.Frame(label="Moildev Fisheye (Mode 1)")
        vb_m = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=5)
        self.entry_moil_cam = Gtk.Entry(text="syue_7730v1_6")
        vb_m.pack_start(Gtk.Label(label="Camera Name:"), 0, 0, 0)
        vb_m.pack_start(self.entry_moil_cam, 0, 0, 0)
        
        btn_reload = Gtk.Button(label="Reload Camera Params")
        btn_reload.connect("clicked", self.on_reload_moil)
        vb_m.pack_start(btn_reload, False, False, 0)

        # Sliders for Alpha, Beta, Zoom
        self.scale_alpha_adj = Gtk.Adjustment(value=0, lower=0, upper=110, step_increment=1, page_increment=10, page_size=0)
        self.scale_beta_adj = Gtk.Adjustment(value=0, lower=0, upper=360, step_increment=1, page_increment=10, page_size=0)
        self.scale_zoom_adj = Gtk.Adjustment(value=4, lower=1, upper=20, step_increment=1, page_increment=2, page_size=0)

        def on_moil_scale_changed(widget):
            self.update_maps()

        self.scale_alpha_adj.connect("value-changed", on_moil_scale_changed)
        self.scale_beta_adj.connect("value-changed", on_moil_scale_changed)
        self.scale_zoom_adj.connect("value-changed", on_moil_scale_changed)

        s_alpha = Gtk.Scale(orientation=Gtk.Orientation.HORIZONTAL, adjustment=self.scale_alpha_adj); s_alpha.set_digits(0); s_alpha.set_value_pos(Gtk.PositionType.RIGHT)
        s_beta = Gtk.Scale(orientation=Gtk.Orientation.HORIZONTAL, adjustment=self.scale_beta_adj); s_beta.set_digits(0); s_beta.set_value_pos(Gtk.PositionType.RIGHT)
        s_zoom = Gtk.Scale(orientation=Gtk.Orientation.HORIZONTAL, adjustment=self.scale_zoom_adj); s_zoom.set_digits(1); s_zoom.set_value_pos(Gtk.PositionType.RIGHT)

        vb_m.pack_start(Gtk.Label(label="Alpha:"), 0, 0, 0); vb_m.pack_start(s_alpha, 0, 0, 0)
        vb_m.pack_start(Gtk.Label(label="Beta:"), 0, 0, 0); vb_m.pack_start(s_beta, 0, 0, 0)
        vb_m.pack_start(Gtk.Label(label="Zoom:"), 0, 0, 0); vb_m.pack_start(s_zoom, 0, 0, 0)

        f_moil.add(vb_m)
        vbox_ctrl.pack_start(f_moil, False, False, 10)

        # Recording Frame
        f_rec = Gtk.Frame(label="Recording")
        vb_r = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=5)
        self.btn_record = Gtk.Button(label="Start Recording")
        self.btn_record.connect("clicked", self.on_toggle_record)
        vb_r.pack_start(self.btn_record, 0, 0, 0)
        f_rec.add(vb_r)
        vbox_ctrl.pack_start(f_rec, False, False, 10)

        self.lbl_status = Gtk.Label(label="Status...")
        vbox_ctrl.pack_start(self.lbl_status, False, False, 0)
        
        self.lbl_fps = Gtk.Label(label="FPS: 0.0")
        vbox_ctrl.pack_start(self.lbl_fps, False, False, 0)
        
        hbox.pack_start(vbox_ctrl, False, False, 0)
        self.image = Gtk.Image()
        hbox.pack_start(self.image, True, True, 0)

    def start_camera(self):
        self.running = True
        threading.Thread(target=self.run_loop, daemon=True).start()

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
                    self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
                    self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
                    
                    if not self.cap or not self.cap.isOpened():
                        GLib.idle_add(self.lbl_status.set_text, f"FAILED to open Camera {current_idx}. Retrying...")
                        if self.cap: self.cap.release()
                        self.cap = None
                        time.sleep(2.0)
                        continue
                    else:
                        GLib.idle_add(self.lbl_status.set_text, f"Camera {current_idx} Active")

                ret, frame = self.cap.read()
                if not ret: 
                    time.sleep(0.01)
                    continue

                if self.moil and self.map_x is not None and self.map_y is not None:
                    # UMat dan INTER_LINEAR untuk boosting FPS yang signifikan
                    umat_frame = cv2.UMat(frame)
                    remapped_umat = cv2.remap(umat_frame, self.map_x, self.map_y, cv2.INTER_LINEAR,
                                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                    remapped_frame = remapped_umat.get()
                else:
                    remapped_frame = frame.copy()

                disp_frame = remapped_frame.copy()

                if self.is_recording:
                    if self.out_video is None:
                        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                        output_path = os.path.join(SAVE_DIR, f"moildev_remap_{ts}.avi")
                        fourcc = cv2.VideoWriter_fourcc(*'XVID')
                        h, w = remapped_frame.shape[:2]
                        self.out_video = cv2.VideoWriter(output_path, fourcc, int(self.fps), (w, h))
                        GLib.idle_add(self.lbl_status.set_text, f"Recording: {output_path}")
                    
                    self.out_video.write(remapped_frame)
                    self.recorded_frames_count += 1
                    
                    # Add Red dot overlay for recording (only on display)
                    cv2.circle(disp_frame, (30, 30), 10, (0, 0, 255), -1)
                    cv2.putText(disp_frame, "REC", (50, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # Calculate FPS
                curr_time = time.time()
                fps_val = 1.0 / (curr_time - self.prev_time) if (curr_time - self.prev_time) > 0 else 0
                self.prev_time = curr_time
                GLib.idle_add(self.lbl_fps.set_text, f"FPS: {fps_val:.1f}")
                
                # Add FPS overlay (only on display)
                h_disp, w_disp = disp_frame.shape[:2]
                cv2.putText(disp_frame, f"FPS: {fps_val:.1f}", (10, h_disp - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

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

    def update_ui(self, frame_bgr):
        if not self.running: return
        h, w = frame_bgr.shape[:2]
        if w > 1280:
            scale = 1280 / float(w)
            new_h = int(h * scale)
            frame_bgr = cv2.resize(frame_bgr, (1280, new_h))
            
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, d = rgb.shape
        pb = GdkPixbuf.Pixbuf.new_from_data(rgb.tobytes(), GdkPixbuf.Colorspace.RGB, False, 8, w, h, w*3)
        self.image.set_from_pixbuf(pb)
        return False

    def on_switch_camera(self, widget):
        cam_id = self.entry_cam.get_text()
        idx = int(cam_id) if cam_id.isdigit() else cam_id
        with self.lock:
            self.requested_cam_id = idx

    def on_reload_moil(self, widget):
        self.init_moildev()

    def on_toggle_record(self, widget):
        if not self.is_recording:
            self.is_recording = True
            self.recorded_frames_count = 0
            self.btn_record.set_label("Stop Recording")
            print("[INFO] Recording started.")
        else:
            self.is_recording = False
            self.btn_record.set_label("Start Recording")
            if self.out_video:
                self.out_video.release()
                self.out_video = None
            self.lbl_status.set_text("Recording Saved.")
            print(f"\n[REPORT] Selesai merekam. Total frame yang terekam: {self.recorded_frames_count}\n")

    def on_destroy(self, widget):
        self.running = False
        Gtk.main_quit()

if __name__ == "__main__":
    win = MoildevRecordingWindow()
    win.show_all()
    Gtk.main()
