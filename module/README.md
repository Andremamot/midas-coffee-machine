# MIDAS Coffee Machine — C++ Modules

Direktori ini berisi module-module C++ (berupa shared library) yang digunakan dalam pipeline MIDAS Coffee Machine. Setiap module mengikuti pola yang seragam dengan struktur `include/<name>/` untuk public headers dan `src/` untuk implementasi.

---

## Daftar Module

1. [mod_aruco](#1-mod_aruco--aruco-marker-detection) — Deteksi marker ArUco & estimasi pose
2. [mod_moil](#2-mod_moil--fisheye-undistortion) — Koreksi distorsi lensa fisheye (via libmoildevren.a)
3. [mod_volume](#3-mod_volume--cup-volume-estimator) — Estimasi tinggi & volume gelas

---

## 1. mod_aruco — ArUco Marker Detection

Module C++ shared library untuk deteksi marker ArUco dan estimasi pose (posisi & orientasi) menggunakan OpenCV. Digunakan untuk mengukur jarak kamera ke tray (Z_tray) pada pipeline MIDAS Coffee Machine.

### Dependensi
- **OpenCV 4**: Modul `aruco`, `calib3d`, `imgproc`, `core`
- **yaml-cpp**: Load file kalibrasi kamera (`.yml`)

### Integrasi CMake
```cmake
add_subdirectory(module/aruco)
target_link_libraries(your_target PRIVATE mod_aruco)
```

### API Publik (`#include <aruco/aruco_detector.h>`)

- `ArucoResult` — Struct hasil deteksi satu marker (ID, corners, jarak, euler angles).
- `BestDistanceResult` — Struct hasil agregat jarak median terbaik.
- `ArucoDetector` — Class utama untuk deteksi dan estimasi pose.

**Contoh Penggunaan**:
```cpp
ArucoDetector aruco(5.0f, "DICT_4X4_50", "calibration_params.yml");
auto results = aruco.detect(frame);
BestDistanceResult best = aruco.get_best_distance(results);
```

---

## 2. mod_moil — Fisheye Undistortion

Module C++ shared library untuk koreksi distorsi lensa fisheye menggunakan library Moildev (`libmoildevren.a`). Mendukung anypoint undistortion mode 1 (alpha/beta) dan mode 2 (pitch/yaw/roll), hybrid zoom, serta pembuatan camera matrix untuk ArUco pose estimation.

### Dependensi
- **OpenCV 4**: `core`, `imgproc` (remap, resize, GaussianBlur)
- **nlohmann_json**: Parse `camera_parameters.json`
- **libmoildevren.a**: **INTERFACE** — tidak di-embed, harus di-link manual

> **Catatan**: `libmoildevren.a` tidak dikompilasi dengan `-fPIC` sehingga **tidak dapat di-embed** ke dalam `.so`. Proyek pemanggil harus me-link library ini secara eksplisit.

### Integrasi CMake
```cmake
add_subdirectory(module/moil)
target_link_libraries(your_target PRIVATE mod_moil /path/to/libmoildevren.a)
```

### API Publik (`#include <moil/moil_undistorter.h>`)

- `MoilUndistorter` — Class utama untuk konfigurasi Moildev dan apply undistortion.

**Contoh Penggunaan**:
```cpp
MoilUndistorter moil("camera_parameters.json", "syue_7730v1_6",
                     -15.0f, 0.0f, 0.0f, 2.0f, 2);  // pitch, yaw, roll, zoom, mode
cv::Mat corrected = moil.undistort(raw_frame);
cv::Mat K = moil.build_aruco_camera_matrix(corrected.cols, corrected.rows);
```

---

## 3. mod_volume — Cup Volume Estimator

Module C++ shared library untuk estimasi volume gelas kopi dari data frame kamera. Menyediakan kalkulasi tinggi gelas (7 mode kalibrasi), deteksi lebar rim via Otsu threshold, estimasi diameter via model pinhole, dan kalkulasi volume via model silinder.

### Dependensi
- **OpenCV 4**: `core`, `imgproc` (threshold, resize, cvtColor)
- **pthread**: Thread safety untuk operasi concurrent

### Integrasi CMake
```cmake
add_subdirectory(module/volume)
target_link_libraries(your_target PRIVATE mod_volume)
```

### API Publik

#### `height_math.h`
Menyediakan struct `BBox` dan 7 fungsi kalibrasi tinggi gelas (termasuk `calc_height_geom`, `calc_height_zgrid`, dll).

#### `volume_math.h`
Menyediakan fungsi pipeline estimasi volume:
- `measure_rim_width_px` (Otsu threshold strip detection)
- `calc_diameter` (Model pinhole)
- `calc_volume` (Model silinder)

#### `volume_config.h`
Konstanta operasional, seperti `MARKER_SIZE_CM`, `MOIL_ZOOM`, dan `REMAP_INTERP`.

**Contoh Penggunaan Lengkap**:
```cpp
#include <volume/height_math.h>
#include <volume/volume_math.h>

using namespace fusion;

BBox bbox = {120, 80, 280, 340}; // Dari YOLO
double h_cup_cm = calc_height_geom(z_tray_cm, bbox, focal_px, poly_Kgeom);

double z_rim_cm  = std::max(0.0, z_tray_cm - h_cup_cm);
double rim_w_px  = measure_rim_width_px(undistorted_frame, bbox);
double diam_cm   = calc_diameter(rim_w_px, z_rim_cm, focal_px);
double volume_ml = calc_volume(h_cup_cm, diam_cm);
```
