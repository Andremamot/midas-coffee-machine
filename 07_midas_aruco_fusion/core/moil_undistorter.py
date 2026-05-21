"""
core/moil_undistorter.py — Wrapper Moildev fisheye undistortion dengan akselerasi OpenCL
==========================================================================================
Modul ini membungkus library Moildev (di ../moildev/) dan menyediakan
interface undistortion yang transparan untuk pipeline MiDaS+ArUco.

Filosofi:
  - Folder moildev/ TIDAK DISENTUH sama sekali (read-only dependency).
  - Jika OpenCL tersedia, map_x / map_y diupload ke cv2.UMat untuk
    akselerasi GPU terintegrasi (iGPU / ARM Mali / OpenCL device).
  - Jika OpenCL tidak tersedia, fallback ke CPU numpy biasa tanpa crash.
  - Matriks remap dihitung SEKALI saat inisiasi, lalu dipakai ulang
    di setiap frame → overhead per-frame minimum.

Dipanggil dari:
  run_fusion.py → hanya jika argumen --fisheye diberikan.
"""

import os
import sys
import json
import warnings
import threading

import cv2
import numpy as np

# ── Import Moildev dari folder sibling (../moildev/) ───────────────────────
_THIS_DIR    = os.path.dirname(os.path.abspath(__file__))
_MOIL_DIR    = os.path.abspath(os.path.join(_THIS_DIR, "..", "moildev"))

if _MOIL_DIR not in sys.path:
    sys.path.insert(0, _MOIL_DIR)

try:
    from Moildev import Moildev as _MoildevLib
except ImportError as _e:
    raise ImportError(
        f"[MoilUndistorter] Gagal import Moildev dari {_MOIL_DIR}.\n"
        f"Pastikan folder moildev/ sudah ada dan berisi Moildev.py.\n"
        f"Error asli: {_e}"
    )


class MoilUndistorter:
    """
    Wrapper Moildev untuk undistortion lensa fisheye secara real-time.

    Menggunakan strategi **Hybrid Zoom**:
    - Moildev menangani fisheye undistortion dengan zoom aman (≤ MAX_MOIL_ZOOM).
    - Sisa zoom ditangani oleh digital crop + resize (tanpa batas atas).
    - User cukup set zoom=2, 3, 5, dst. — pembagian otomatis.

    Parameters
    ----------
    json_path : str
        Path absolut atau relatif ke camera_parameters.json.
    camera_name : str
        Nama profil kamera di dalam JSON (contoh: 'lrcp_imx586_240_17').
    pitch : float
        Rotasi pitch dalam derajat (default 0 = lurus ke tengah lensa).
    yaw : float
        Rotasi yaw dalam derajat (default 0).
    roll : float
        Rotasi roll dalam derajat (default 0).
    zoom : float
        Faktor zoom total yang diinginkan (default 2). Secara otomatis
        dibagi menjadi moil_zoom (≤ MAX_MOIL_ZOOM) + digital_zoom.
    mode : int
        Mode Moildev (1 = Alpha/Beta, 2 = Pitch/Yaw/Roll). Default 2.
    use_opencl : bool
        Aktifkan akselerasi OpenCL jika tersedia (default True).
    frame_width : int
        Lebar resolusi stream aktual (default 640). Maps di-rescale ke ukuran ini.
    frame_height : int
        Tinggi resolusi stream aktual (default 480). Maps di-rescale ke ukuran ini.
    target_size : tuple | None
        Resize output ke (width, height) setelah remap. None = tidak diubah.

    Attributes
    ----------
    opencl_active : bool
        True jika OpenCL berhasil diaktifkan dan dipakai.
    adjusted_focal_length : float
        Perkiraan focal length baru berbasis parameter5 Moildev.
        Injeksikan ke aruco.camera_matrix setelah inisiasi.
    moil_zoom : float
        Komponen zoom yang dikirim ke Moildev (≤ MAX_MOIL_ZOOM).
    digital_zoom : float
        Komponen zoom tambahan via crop+resize (≥ 1.0).
    """

    # Batas aman zoom Moildev sebelum polinomial kalibrasi wrap-around.
    # Ditentukan secara empiris: di atas nilai ini, maps mulai "terbalik".
    MAX_MOIL_ZOOM = 1.5

    def __init__(
        self,
        json_path: str,
        camera_name: str = "lrcp_imx586_240_17",
        pitch: float = 0.0,
        yaw: float = 0.0,
        roll: float = 0.0,
        zoom: float = 2.0,
        mode: int = 2,
        use_opencl: bool = True,
        frame_width: int = 640,
        frame_height: int = 480,
        target_size: tuple = None,
    ):
        self.camera_name   = camera_name
        self.pitch         = pitch
        self.yaw           = yaw
        self.roll          = roll
        self.zoom          = max(1.0, zoom)
        self.mode          = mode
        self.frame_width   = frame_width
        self.frame_height  = frame_height
        self.target_size   = target_size
        self.opencl_active = False

        # Lock untuk thread-safety: update_maps() (GTK main thread)
        # vs undistort() (background camera thread) bisa terjadi bersamaan.
        # Tanpa lock, cv2.remap bisa mengakses map yang sedang di-overwrite → SIGABRT.
        self._maps_lock = threading.Lock()

        # ── Hybrid Zoom: pisahkan zoom menjadi moil + digital ──────────
        self.moil_zoom, self.digital_zoom = self._split_zoom(self.zoom)

        # ── Validasi json_path ──────────────────────────────────────────────
        if not os.path.isabs(json_path):
            json_path = os.path.join(_THIS_DIR, "..", json_path)
        json_path = os.path.abspath(json_path)

        if not os.path.exists(json_path):
            raise FileNotFoundError(
                f"[MoilUndistorter] camera_parameters.json tidak ditemukan: {json_path}"
            )

        # ── Validasi camera_name ada di JSON ───────────────────────────────
        with open(json_path, "r") as f:
            _params = json.load(f)
        if camera_name not in _params:
            available = [k for k in _params.keys()][:10]
            raise KeyError(
                f"[MoilUndistorter] camera_name '{camera_name}' tidak ditemukan di JSON.\n"
                f"Beberapa profil yang tersedia: {available} ..."
            )

        # ── Instansiasi Moildev Natively ─────────────────────────────────────────
        # Sesuai request user: menggunakan modul Moildev secara native seperti di README.
        # Kita lewati parameter manual dan biarkan Moildev membaca JSON langsung.
        self._moil = _MoildevLib(json_path, camera_name)
        
        # Ekstrak beberapa parameter penting untuk informasi/scaling jika perlu
        self._parameter5   = self._moil.param_5
        self._calibRatio   = self._moil._Moildev__calibration_ratio if hasattr(self._moil, "_Moildev__calibration_ratio") else 1.0
        
        # Simpan resolusi asli sensor dari JSON untuk rescaling
        self._sensor_width  = self._moil.image_width
        self._sensor_height = self._moil.image_height
        print(f"[MOIL] Native Init: {self._sensor_width}x{self._sensor_height} "
              f"→ stream: {frame_width}x{frame_height}")

        # Generate maps dengan moil_zoom (komponen aman, ≤ MAX_MOIL_ZOOM)
        # mode 0 atau 1 → maps_anypoint_mode1 (Alpha/Beta), sama seperti C++ --moil-mode 0
        if self.mode in (0, 1):
            # pitch dipetakan ke alpha, yaw dipetakan ke beta untuk Mode 1
            map_x_np, map_y_np = self._moil.maps_anypoint_mode1(pitch, yaw, self.moil_zoom)
        else:
            map_x_np, map_y_np = self._moil.maps_anypoint_mode2(pitch, yaw, roll, self.moil_zoom)

        # KRITIS: Rescale maps dari resolusi sensor JSON ke resolusi stream aktual.
        # Maps berisi koordinat piksel dalam ruang sensor (mis. 0-2592 x 0-1944).
        # Jika stream 640x480, semua koordinat itu out-of-bounds → frame hitam.
        self._map_x_cpu, self._map_y_cpu = self._rescale_maps(
            map_x_np, map_y_np, frame_width, frame_height
        )

        # ── Aktifkan OpenCL jika diminta & tersedia ────────────────────────
        self._map_x = self._map_x_cpu
        self._map_y = self._map_y_cpu

        if use_opencl:
            try:
                if cv2.ocl.haveOpenCL() and cv2.ocl.useOpenCL():
                    # Upload map ke device memory
                    self._map_x = cv2.UMat(self._map_x_cpu)
                    self._map_y = cv2.UMat(self._map_y_cpu)
                    self.opencl_active = True
                    print("[MOIL] OpenCL aktif — remap akan diakselerasi oleh GPU/iGPU.")
                else:
                    print("[MOIL] OpenCL dinonaktifkan atau tidak tersedia. Menggunakan CPU.")
            except Exception as _ocl_err:
                warnings.warn(
                    f"[MOIL] Gagal mengaktifkan OpenCL: {_ocl_err}. Fallback ke CPU."
                )

        if not self.opencl_active:
            print("[MOIL] OpenCL: OFF — menggunakan CPU numpy.")

        print(f"[MOIL] Maps siap. Adjusted focal length ≈ {self.adjusted_focal_length:.1f} px")
        print(f"[MOIL] Hybrid Zoom: total={self.zoom:.2f}x → moil={self.moil_zoom:.2f}x + digital={self.digital_zoom:.2f}x")

    # ── Properti ────────────────────────────────────────────────────────────

    @property
    def adjusted_focal_length(self) -> float:
        """
        Estimasi focal length ekivalen piksel setelah koreksi Moildev.
        Menggunakan parameter5 dibagi calibrationRatio sebagai pendekatan.
        Gunakan nilai ini untuk meng-override aruco.camera_matrix[0,0] & [1,1].
        """
        if self._calibRatio > 0:
            return self._parameter5 / self._calibRatio
        return self._parameter5

    # ── Helper: Map Rescaling ────────────────────────────────────────────────

    def _rescale_maps(
        self,
        map_x: np.ndarray,
        map_y: np.ndarray,
        target_w: int,
        target_h: int,
    ) -> tuple:
        """
        Rescale remap maps dari resolusi sensor JSON ke resolusi stream aktual.

        Moildev selalu menghasilkan maps di resolusi kamera yang tersimpan di
        JSON (mis. 2592x1944). Jika stream aktual lebih kecil (mis. 640x480),
        perlu dua operasi:
          1. Scale nilai koordinat piksel (nilai di map) proporsional ke resolusi target.
          2. Resize ukuran map matrix itu sendiri ke (target_w, target_h).

        Parameters
        ----------
        map_x, map_y : np.ndarray
            Maps raw dari Moildev (resolusi sensor JSON).
        target_w, target_h : int
            Resolusi tujuan (ukuran frame stream aktual).

        Returns
        -------
        tuple (map_x_rescaled, map_y_rescaled) dalam np.float32.
        """
        src_w = self._sensor_width
        src_h = self._sensor_height

        if src_w == target_w and src_h == target_h:
            # Resolusi sama — tidak perlu rescale apapun
            return map_x.astype(np.float32), map_y.astype(np.float32)

        # Faktor skala nilai koordinat
        scale_x = target_w  / src_w
        scale_y = target_h / src_h

        # Scale nilai koordinat DULU (float arithmetic, tidak ada data loss)
        scaled_x = (map_x * scale_x).astype(np.float32)
        scaled_y = (map_y * scale_y).astype(np.float32)

        # Lalu resize dimensi map agar sesuai target frame size
        # INTER_LINEAR cocok untuk floating-point maps
        resized_x = cv2.resize(scaled_x, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        resized_y = cv2.resize(scaled_y, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        return resized_x, resized_y

    # ── Helper: Hybrid Zoom ────────────────────────────────────────────────

    def _split_zoom(self, total_zoom: float) -> tuple:
        """
        Pisahkan zoom total menjadi (moil_zoom, digital_zoom).

        - moil_zoom  : dikirim ke Moildev (di-clamp ke MAX_MOIL_ZOOM)
        - digital_zoom: sisa zoom via crop+resize (≥ 1.0)

        Contoh:
            total=1.2 → moil=1.2, digital=1.0  (murni Moildev)
            total=2.0 → moil=1.5, digital=1.33  (hybrid)
            total=4.0 → moil=1.5, digital=2.67  (heavy digital)
        """
        total_zoom = max(1.0, total_zoom)
        moil_z  = min(total_zoom, self.MAX_MOIL_ZOOM)
        digi_z  = total_zoom / moil_z  # selalu ≥ 1.0
        return moil_z, digi_z

    def _digital_crop(self, frame: np.ndarray) -> np.ndarray:
        """
        Terapkan digital zoom via center-crop + resize ke ukuran asli.
        Hanya aktif jika digital_zoom > 1.0.

        Parameters
        ----------
        frame : np.ndarray
            Frame BGR hasil remap Moildev.

        Returns
        -------
        np.ndarray
            Frame BGR yang sudah di-crop dan di-resize.
        """
        if self.digital_zoom <= 1.001:  # toleransi float
            return frame

        h, w = frame.shape[:2]
        # Hitung ukuran crop region (center)
        crop_w = int(w / self.digital_zoom)
        crop_h = int(h / self.digital_zoom)

        # Pastikan minimal 1 piksel
        crop_w = max(1, crop_w)
        crop_h = max(1, crop_h)

        # Center crop
        x1 = (w - crop_w) // 2
        y1 = (h - crop_h) // 2
        x2 = x1 + crop_w
        y2 = y1 + crop_h

        cropped = frame[y1:y2, x1:x2]

        # Resize kembali ke ukuran asli dengan LANCZOS4 untuk anti-aliasing halus
        return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LANCZOS4)

    # ── Method Utama ────────────────────────────────────────────────────────

    def update_maps(
        self,
        pitch: float = None,
        yaw: float = None,
        roll: float = None,
        zoom: float = None,
    ) -> None:
        """
        Regenerasi remap maps secara real-time dengan parameter baru.
        Dipanggil oleh AnypointController saat user drag mouse.

        Parameters yang None akan menggunakan nilai saat ini.
        Zoom otomatis dibagi menjadi moil_zoom + digital_zoom.

        Thread-safe: dilindungi oleh _maps_lock agar tidak bentrok
        dengan undistort() yang berjalan di background thread.
        """
        if pitch is not None: self.pitch = pitch
        if yaw   is not None: self.yaw   = yaw
        if roll  is not None: self.roll  = roll
        if zoom  is not None: self.zoom  = max(1.0, zoom)

        # Hitung ulang pembagian hybrid zoom
        self.moil_zoom, self.digital_zoom = self._split_zoom(self.zoom)

        # Generate maps dengan moil_zoom saja (komponen aman)
        # mode 0 atau 1 → maps_anypoint_mode1, sama seperti C++ --moil-mode 0
        if self.mode in (0, 1):
            map_x_np, map_y_np = self._moil.maps_anypoint_mode1(
                self.pitch, self.yaw, self.moil_zoom
            )
        else:
            map_x_np, map_y_np = self._moil.maps_anypoint_mode2(
                self.pitch, self.yaw, self.roll, self.moil_zoom
            )

        # Rescale maps ke resolusi stream aktual
        new_x, new_y = self._rescale_maps(
            map_x_np, map_y_np, self.frame_width, self.frame_height
        )

        # Atomik swap di bawah lock — background thread (undistort) tidak akan
        # melihat map setengah jadi.
        with self._maps_lock:
            self._map_x_cpu = new_x
            self._map_y_cpu = new_y
            if self.opencl_active:
                self._map_x = cv2.UMat(self._map_x_cpu)
                self._map_y = cv2.UMat(self._map_y_cpu)
            else:
                self._map_x = self._map_x_cpu
                self._map_y = self._map_y_cpu

    def undistort(self, frame: np.ndarray) -> np.ndarray:
        """
        Terapkan koreksi Moildev anypoint + digital zoom pada satu frame.

        Pipeline:
          1. Remap fisheye → anypoint (moil_zoom, aman ≤ MAX_MOIL_ZOOM)
          2. Digital crop + resize (digital_zoom, sisa dari total zoom)
          3. Resize ke target_size jika dispesifikasi

        Parameters
        ----------
        frame : np.ndarray
            Frame BGR dari cv2.VideoCapture(), shape (H, W, 3).

        Returns
        -------
        np.ndarray
            Frame BGR yang sudah terkoreksi dan di-zoom, shape sama atau target_size.
        """
        if frame is None:
            return frame

        # INTER_LINEAR dipilih karena INTER_CUBIC dan INTER_LANCZOS4 menyebabkan
        # crash C++ (std::terminate / SIGSEGV) yang TIDAK BISA di-catch Python
        # saat OpenCL menerima frame transisi pasca-perubahan exposure.
        # Di resolusi 2592x1944 perbedaan visual LINEAR vs CUBIC tidak terlihat.
        if self.opencl_active:
            frame_in = cv2.UMat(frame)
        else:
            frame_in = frame

        # Stage 1: Moildev remap (undistortion + zoom aman)
        # Ambil referensi maps di bawah lock agar atomic terhadap update_maps().
        with self._maps_lock:
            mx = self._map_x
            my = self._map_y
            digi_z = self.digital_zoom  # snapshot nilai saat ini

        remapped = cv2.remap(
            frame_in,
            mx,
            my,
            interpolation=cv2.INTER_LANCZOS4,   # LANCZOS4: anti-aliasing terbaik
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        if self.opencl_active:
            remapped = remapped.get()

        # Stage 2: Digital zoom via center-crop + resize
        remapped = self._digital_crop(remapped)

        # Resize jika target_size dispesifikasi
        if self.target_size is not None:
            remapped = cv2.resize(remapped, self.target_size, interpolation=cv2.INTER_LANCZOS4)

        return remapped

    def build_aruco_camera_matrix(
        self,
        frame_width: int,
        frame_height: int,
    ) -> np.ndarray:
        """
        Buat camera matrix 3×3 untuk ArUco berdasarkan adjusted_focal_length
        dan pusat gambar yang disesuaikan dengan resolusi streaming.

        Parameters
        ----------
        frame_width, frame_height : int
            Resolusi aktual frame streaming (bukan resolusi sensor penuh).

        Returns
        -------
        np.ndarray shape (3,3)
            Camera matrix K yang siap diinjeksikan ke aruco.camera_matrix.
        """
        # Hitung skala rasio antara resolusi streaming dan resolusi sensor JSON
        scale_x = frame_width  / max(self._moil.image_width,  1)
        scale_y = frame_height / max(self._moil.image_height, 1)
        scale   = (scale_x + scale_y) / 2.0

        # Focal length dikalikan dengan TOTAL zoom (moil + digital).
        # Moildev zoom memperbesar objek lewat remap, digital zoom memperbesar
        # lewat crop — keduanya meningkatkan focal length efektif secara linear.
        fl   = self.adjusted_focal_length * scale * self.zoom
        
        cx   = frame_width  / 2.0
        cy   = frame_height / 2.0

        K = np.array([
            [fl,  0., cx],
            [0.,  fl, cy],
            [0.,  0.,  1.],
        ], dtype=np.float64)

        return K

    def __repr__(self) -> str:
        return (
            f"MoilUndistorter(camera='{self.camera_name}', mode={self.mode}, "
            f"pitch={self.pitch}, yaw={self.yaw}, roll={self.roll}, "
            f"zoom={self.zoom} [moil={self.moil_zoom:.2f}+digi={self.digital_zoom:.2f}], "
            f"opencl={self.opencl_active})"
        )
