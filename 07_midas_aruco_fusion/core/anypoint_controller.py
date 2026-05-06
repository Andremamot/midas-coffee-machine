"""
core/anypoint_controller.py — Mouse-driven anypoint view controller
====================================================================
Melampirkan mouse callback ke jendela OpenCV agar user bisa menggeser
sudut pandang Moildev secara interaktif:

  • Drag kiri-kanan  → yaw   (geser horizontal)
  • Drag atas-bawah  → pitch (geser vertikal)
  • Scroll wheel     → zoom  (dekat / jauh)
  • Tekan R          → reset ke nilai awal
  • Tekan S          → print nilai saat ini ke terminal (untuk disimpan)

Kontrol ini aktif di semua mode (kalibrasi maupun live).
Dipanggil dari run_fusion.py, hanya jika --fisheye aktif.
"""

import cv2


class AnypointController:
    """
    Mengelola interaksi mouse untuk mengontrol parameter Moildev anypoint.

    Parameters
    ----------
    undistorter : MoilUndistorter
        Instance MoilUndistorter yang akan dikontrol.
    sensitivity_deg_per_px : float
        Seberapa banyak derajat berubah per piksel drag (default 0.15°/px).
    zoom_step : float
        Perubahan zoom per scroll tick (default 0.1).
    """

    def __init__(
        self,
        undistorter,
        sensitivity_deg_per_px: float = 0.15,
        zoom_step: float = 0.1,
    ):
        self._u             = undistorter
        self._sens          = sensitivity_deg_per_px
        self._zoom_step     = zoom_step

        # simpan nilai awal untuk reset
        self._init_pitch    = undistorter.pitch
        self._init_yaw      = undistorter.yaw
        self._init_roll     = undistorter.roll
        self._init_zoom     = undistorter.zoom

        # state mouse drag
        self._dragging      = False
        self._drag_start_x  = 0
        self._drag_start_y  = 0
        self._pitch_at_drag = 0.0
        self._yaw_at_drag   = 0.0

        # flag agar update_maps tidak dipanggil terlalu sering saat drag
        self._dirty         = False

    # ── Callback utama ──────────────────────────────────────────────────────

    def mouse_callback(self, event, x, y, flags, param):
        """Dipasang ke jendela cv2 via cv2.setMouseCallback."""

        if event == cv2.EVENT_LBUTTONDOWN:
            self._dragging      = True
            self._drag_start_x  = x
            self._drag_start_y  = y
            self._pitch_at_drag = self._u.pitch
            self._yaw_at_drag   = self._u.yaw

        elif event == cv2.EVENT_MOUSEMOVE and self._dragging:
            dx = x - self._drag_start_x     # geser horizontal → yaw
            dy = y - self._drag_start_y     # geser vertikal   → pitch
            new_pitch = self._pitch_at_drag - dy * self._sens
            new_yaw   = self._yaw_at_drag   + dx * self._sens
            self._u.update_maps(pitch=new_pitch, yaw=new_yaw)
            self._dirty = True

        elif event == cv2.EVENT_LBUTTONUP:
            self._dragging = False

        elif event == cv2.EVENT_MOUSEWHEEL:
            # flags > 0  → scroll up  → zoom in (nilai naik)
            # flags < 0  → scroll down→ zoom out
            delta = self._zoom_step if flags > 0 else -self._zoom_step
            self._u.update_maps(zoom=self._u.zoom + delta)
            self._dirty = True

    def reset(self):
        """Reset ke nilai awal saat MoilUndistorter diinisiasi."""
        self._u.update_maps(
            pitch=self._init_pitch,
            yaw=self._init_yaw,
            roll=self._init_roll,
            zoom=self._init_zoom,
        )
        self._dirty = True

    # ── Overlay HUD ─────────────────────────────────────────────────────────

    def draw_overlay(self, frame):
        """
        Gambar panel info Anypoint di pojok kanan bawah frame.
        Kembalikan frame yang sudah diberi anotasi (tidak mengubah original).
        """
        h, w = frame.shape[:2]
        
        # UI Scaling factor (2.5x for 2.5K resolution)
        S = 2.5
        x0, y0 = w - int(280 * S), h - int(120 * S)
        x1, y1 = w - int(15 * S),   h - int(15 * S)

        cv2.rectangle(frame, (x0, y0), (x1, y1), (10, 10, 30), -1)
        cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 180, 255), 2)

        def put(text, row, color=(200, 230, 255)):
            cv2.putText(frame, text, (x0 + int(10*S), y0 + int(25*S) + row * int(22 * S)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45 * S, color, 2, cv2.LINE_AA)

        if getattr(self._u, "mode", 2) == 1:
            put("[ ANYPOINT (MODE 1) ]", 0, (0, 200, 255))
            put(f"Alpha : {self._u.pitch:+.1f}deg  (drag up/dn)", 1)
            put(f"Beta  : {self._u.yaw:+.1f}deg  (drag L/R)", 2)
            put(f"Zoom  : {self._u.zoom:.2f}x  (scroll)", 3)
            put("R=reset  S=print params", 4, (100, 200, 100))
        else:
            put("[ ANYPOINT CTRL ]", 0, (0, 200, 255))
            put(f"Pitch : {self._u.pitch:+.1f}deg  (drag up/dn)", 1)
            put(f"Yaw   : {self._u.yaw:+.1f}deg  (drag L/R)", 2)
            put(f"Zoom  : {self._u.zoom:.2f}x  (scroll)", 3)
            put("R=reset  S=print params", 4, (100, 200, 100))

        return frame

    def attach(self, window_name: str):
        """Pasang callback ke jendela OpenCV yang sudah dibuat."""
        cv2.setMouseCallback(window_name, self.mouse_callback)
