import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, Gdk, GdkPixbuf, GLib
import cv2
import yaml
import numpy as np
import os
import sys
import json
import time

# Ensure project root is in path so we can import midas_volumecup
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_dir not in sys.path:
    sys.path.append(root_dir)

fusion_dir = os.path.join(root_dir, '07_midas_aruco_fusion')
if fusion_dir not in sys.path:
    sys.path.append(fusion_dir)

from midas_volumecup.depth import MidasDepthEstimator
from midas_volumecup.detector import YoloDetector

class TestDataCollectionWindow(Gtk.Window):
    def __init__(self):
        super().__init__(title="MiDaS Test Dataset Collection")
        self.set_default_size(1000, 600)
        self.connect("destroy", self.on_destroy)
        
        self.cap = None
        self.depth_estimator = None
        self.detector = None
        self.running = False
        
        self.dataset_points = []
        self.latest_frame = None
        self.latest_depth = None
        self.latest_boxes = None
        self.requested_cam_id = None
        import threading
        self.lock = threading.Lock()
        
        self.moil_undistorter = None
        
        # Ensure data directories exist in the dataset folder
        self.points_file = os.path.join(os.path.dirname(__file__), 'test_points.json')
        self.snapshots_dir = os.path.join(os.path.dirname(__file__), 'test_snapshots')
        self.config_file = os.path.join(root_dir, 'midas_calibration.yaml')

        if not os.path.exists(self.snapshots_dir):
            os.makedirs(self.snapshots_dir)
        
        self.setup_ui()
        self.load_config()
        self.load_historical_points()
        
        import threading
        self.lbl_status.set_text("Loading Models (No Crashing Initializing)...")
        threading.Thread(target=self.init_ai, daemon=True).start()

    def init_ai(self):
        self.depth_estimator = MidasDepthEstimator()
        self.detector = YoloDetector()
        GLib.idle_add(self.lbl_status.set_text, "Ready. Camera Starting...")
        GLib.idle_add(self.start_camera)

    def start_camera(self):
        self.running = True
        import threading
        threading.Thread(target=self.run_loop, daemon=True).start()

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
        vb_s.pack_start(Gtk.Label(label="Cam Index:"),0,0,0)
        vb_s.pack_start(hb_cam, 0,0,0)
        self.entry_f = Gtk.Entry(text="846"); vb_s.pack_start(Gtk.Label(label="Focal Length (px):"),0,0,0); vb_s.pack_start(self.entry_f,0,0,0)
        f_sys.add(vb_s); vbox_ctrl.pack_start(f_sys, 0,0,0)

        # Data Collection Frame
        f_p = Gtk.Frame(label="Data Collection")
        vb_p = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=5)
        self.entry_roi = Gtk.Entry(text="10,400,100,470")
        
        self.scale_x1_adj = Gtk.Adjustment(value=10, lower=0, upper=2592, step_increment=10, page_increment=100, page_size=0)
        self.scale_y1_adj = Gtk.Adjustment(value=400, lower=0, upper=1944, step_increment=10, page_increment=100, page_size=0)
        self.scale_x2_adj = Gtk.Adjustment(value=100, lower=0, upper=2592, step_increment=10, page_increment=100, page_size=0)
        self.scale_y2_adj = Gtk.Adjustment(value=470, lower=0, upper=1944, step_increment=10, page_increment=100, page_size=0)

        def on_roi_scale_changed(widget):
            x1, y1 = int(self.scale_x1_adj.get_value()), int(self.scale_y1_adj.get_value())
            x2, y2 = int(self.scale_x2_adj.get_value()), int(self.scale_y2_adj.get_value())
            self.entry_roi.set_text(f"{x1},{y1},{x2},{y2}")

        self.scale_x1_adj.connect("value-changed", on_roi_scale_changed)
        self.scale_y1_adj.connect("value-changed", on_roi_scale_changed)
        self.scale_x2_adj.connect("value-changed", on_roi_scale_changed)
        self.scale_y2_adj.connect("value-changed", on_roi_scale_changed)

        sx1 = Gtk.Scale(orientation=Gtk.Orientation.HORIZONTAL, adjustment=self.scale_x1_adj); sx1.set_digits(0); sx1.set_value_pos(Gtk.PositionType.RIGHT)
        sy1 = Gtk.Scale(orientation=Gtk.Orientation.HORIZONTAL, adjustment=self.scale_y1_adj); sy1.set_digits(0); sy1.set_value_pos(Gtk.PositionType.RIGHT)
        sx2 = Gtk.Scale(orientation=Gtk.Orientation.HORIZONTAL, adjustment=self.scale_x2_adj); sx2.set_digits(0); sx2.set_value_pos(Gtk.PositionType.RIGHT)
        sy2 = Gtk.Scale(orientation=Gtk.Orientation.HORIZONTAL, adjustment=self.scale_y2_adj); sy2.set_digits(0); sy2.set_value_pos(Gtk.PositionType.RIGHT)

        vb_p.pack_start(Gtk.Label(label="Tray ROI (Sliders):"),0,0,0)
        bx_x = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=2); bx_x.pack_start(Gtk.Label(label="X1:"),0,0,0); bx_x.pack_start(sx1,1,1,0); bx_x.pack_start(Gtk.Label(label="X2:"),0,0,0); bx_x.pack_start(sx2,1,1,0)
        vb_p.pack_start(bx_x, 0,0,0)
        bx_y = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=2); bx_y.pack_start(Gtk.Label(label="Y1:"),0,0,0); bx_y.pack_start(sy1,1,1,0); bx_y.pack_start(Gtk.Label(label="Y2:"),0,0,0); bx_y.pack_start(sy2,1,1,0)
        vb_p.pack_start(bx_y, 0,0,0)

        self.entry_tray_z = Gtk.Entry(text="35.0"); vb_p.pack_start(Gtk.Label(label="True Z Tray (cm):"),0,0,0); vb_p.pack_start(self.entry_tray_z,0,0,0)
        self.entry_rim_z = Gtk.Entry(text="15.0"); vb_p.pack_start(Gtk.Label(label="True Z Rim (cm):"),0,0,0); vb_p.pack_start(self.entry_rim_z,0,0,0)
        btn_cap = Gtk.Button(label="Capture Dataset Point"); btn_cap.connect("clicked", self.on_capture); vb_p.pack_start(btn_cap,0,0,0)
        self.lbl_pts = Gtk.Label(label="Points: 0"); vb_p.pack_start(self.lbl_pts,0,0,0)
        f_p.add(vb_p); vbox_ctrl.pack_start(f_p, 0,0,10)

        # Moil Frame
        f_moil = Gtk.Frame(label="Moildev Fisheye (Mode 1)")
        vb_m = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=5)
        self.entry_moil_cam = Gtk.Entry(text="syue_7730v1_6"); vb_m.pack_start(Gtk.Label(label="Camera Name:"),0,0,0); vb_m.pack_start(self.entry_moil_cam,0,0,0)
        hb_moil_ab = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
        self.entry_moil_alpha = Gtk.Entry(text="0.0"); hb_moil_ab.pack_start(Gtk.Label(label="Alpha:"),0,0,0); hb_moil_ab.pack_start(self.entry_moil_alpha, True, True, 0)
        self.entry_moil_beta = Gtk.Entry(text="0.0"); hb_moil_ab.pack_start(Gtk.Label(label="Beta:"),0,0,0); hb_moil_ab.pack_start(self.entry_moil_beta, True, True, 0)
        vb_m.pack_start(hb_moil_ab, 0, 0, 0)
        hb_moil_z = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
        self.entry_moil_zoom = Gtk.Entry(text="1.4"); hb_moil_z.pack_start(Gtk.Label(label="Zoom:"),0,0,0); hb_moil_z.pack_start(self.entry_moil_zoom, True, True, 0)
        btn_apply_moil = Gtk.Button(label="Apply Moil"); btn_apply_moil.connect("clicked", self.on_apply_moil); hb_moil_z.pack_start(btn_apply_moil, False, False, 0)
        vb_m.pack_start(hb_moil_z, 0, 0, 0)
        f_moil.add(vb_m); vbox_ctrl.pack_start(f_moil, 0,0,10)

        self.lbl_status = Gtk.Label(label="Status..."); vbox_ctrl.pack_start(self.lbl_status, 0,0,0)
        hbox.pack_start(vbox_ctrl, 0,0,0)
        self.image = Gtk.Image(); hbox.pack_start(self.image, 1,1,0)

    def load_config(self):
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    config = yaml.safe_load(f)
                    self.entry_f.set_text(str(config.get('focal_length', 846.0)))
                    self.entry_cam.set_text(str(config.get('camera_index', 0)))
                    roi = config.get('tray_roi', [10,400,100,470])
                    self.entry_roi.set_text(f"{roi[0]},{roi[1]},{roi[2]},{roi[3]}")
                    try:
                        self.scale_x1_adj.set_value(roi[0])
                        self.scale_y1_adj.set_value(roi[1])
                        self.scale_x2_adj.set_value(roi[2])
                        self.scale_y2_adj.set_value(roi[3])
                    except: pass
        except: pass

    def load_historical_points(self):
        try:
            if os.path.exists(self.points_file):
                with open(self.points_file, 'r') as f:
                    self.dataset_points = json.load(f)
                self.lbl_pts.set_text(f"Points (Loaded): {len(self.dataset_points)}")
        except Exception as e:
            print(f"Error loading points: {e}")

    def run_loop(self):
        current_idx = None
        
        while self.running:
            try:
                # 1. Handle camera change request
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

                # 2. Ensure camera is initialized
                if self.cap is None:
                    # If this was the very first run, get it from entry
                    if current_idx is None:
                        cam_id = self.entry_cam.get_text()
                        current_idx = int(cam_id) if cam_id.isdigit() else cam_id
                    
                    self.cap = cv2.VideoCapture(current_idx)
                    
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

                # 3. Read Frame
                ret, frame = self.cap.read()
                if not ret: 
                    time.sleep(0.01)
                    continue
                
                # 4. Processing
                if self.moil_undistorter is not None:
                    frame = self.moil_undistorter.undistort(frame)

                self.latest_frame = frame.copy()
                self.latest_depth = self.depth_estimator.process(frame)
                self.latest_boxes = self.detector.detect(frame)
                
                depth_norm = cv2.normalize(self.latest_depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                if self.latest_boxes:
                    b = self.latest_boxes[0]['bbox']
                    cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
                
                try:
                    roi_text = self.entry_roi.get_text()
                    roi = tuple(map(int, roi_text.split(',')))
                    cv2.rectangle(frame, (roi[0], roi[1]), (roi[2], roi[3]), (255, 0, 0), 2)
                except: pass

                dv = cv2.applyColorMap(depth_norm, cv2.COLORMAP_INFERNO)
                comb = np.hstack((frame, dv))
                GLib.idle_add(self.update_ui, comb)

            except Exception as e:
                print(f"Error in run_loop: {e}")
                time.sleep(1.0)
            
        if self.cap: 
            self.cap.release()
            self.cap = None

    def update_ui(self, frame_bgr):
        if not self.running: return
        
        h, w = frame_bgr.shape[:2]
        if w > 1280:
            scale = 1280 / float(w)
            frame_bgr = cv2.resize(frame_bgr, (1280, int(h * scale)))
            
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, d = rgb.shape
        pb = GdkPixbuf.Pixbuf.new_from_data(rgb.tobytes(), GdkPixbuf.Colorspace.RGB, False, 8, w, h, w*3)
        self.image.set_from_pixbuf(pb)
        return False

    def on_capture(self, widget):
        if self.latest_depth is None or not self.latest_boxes:
            GLib.idle_add(self.lbl_status.set_text, "Capture Failed: No cup detected.")
            return
        try:
            tz_tray = float(self.entry_tray_z.get_text())
            tz_rim = float(self.entry_rim_z.get_text())
            raw_roi = tuple(map(int, self.entry_roi.get_text().split(',')))
            roi = (min(raw_roi[0], raw_roi[2]), min(raw_roi[1], raw_roi[3]), max(raw_roi[0], raw_roi[2]), max(raw_roi[1], raw_roi[3]))
        except Exception as e:
            GLib.idle_add(self.lbl_status.set_text, "Capture Failed: Invalid input values.")
            return
            
        dn = cv2.normalize(self.latest_depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        mr, mt = self.depth_estimator.get_rim_depth(dn, self.latest_boxes[0]['bbox']), self.depth_estimator.get_tray_depth(dn, roi)
        if mt > 0:
            point = {'M_rim': mr, 'M_tray': mt, 'Z_tray': tz_tray, 'Z_rim': tz_rim, 'timestamp': time.time()}
            self.dataset_points.append(point)
            
            # 1. Save point to JSON
            with open(self.points_file, 'w') as f:
                json.dump(self.dataset_points, f, indent=4)
                
            # 2. Save image snapshot
            img_path = os.path.join(self.snapshots_dir, f"test_tray{tz_tray}cm_rim{tz_rim}cm_{int(time.time())}.jpg")
            cv2.imwrite(img_path, self.latest_frame)
            
            self.lbl_pts.set_text(f"Points: {len(self.dataset_points)}")
            self.lbl_status.set_text(f"Captured & Saved: {img_path}")

    def on_switch_camera(self, widget):
        cam_id = self.entry_cam.get_text()
        idx = int(cam_id) if cam_id.isdigit() else cam_id
        with self.lock:
            self.requested_cam_id = idx
        self.lbl_status.set_text(f"Requesting camera switch to {idx}...")

    def on_apply_moil(self, widget):
        cam_name = self.entry_moil_cam.get_text()
        try:
            alpha = float(self.entry_moil_alpha.get_text())
            beta = float(self.entry_moil_beta.get_text())
            zoom = float(self.entry_moil_zoom.get_text())
        except ValueError:
            GLib.idle_add(self.lbl_status.set_text, "Error: Alpha, Beta, Zoom must be numbers")
            return
            
        json_path = os.path.join(root_dir, '07_midas_aruco_fusion', 'camera_parameters.json')
        try:
            from core.moil_undistorter import MoilUndistorter
            self.moil_undistorter = MoilUndistorter(
                json_path=json_path,
                camera_name=cam_name,
                pitch=alpha,
                yaw=beta,
                zoom=zoom,
                mode=1,
                use_opencl=True,
                frame_width=2592,
                frame_height=1944
            )
            GLib.idle_add(self.lbl_status.set_text, f"Moil applied: {cam_name} (Mode 1, Alpha:{alpha}, Beta:{beta}, Z:{zoom})")
        except Exception as e:
            GLib.idle_add(self.lbl_status.set_text, f"Moil Error: {e}")
            self.moil_undistorter = None

    def on_destroy(self, widget):
        self.running = False; Gtk.main_quit()

if __name__ == "__main__":
    win = TestDataCollectionWindow(); win.show_all(); Gtk.main()
