# `mod_volume` — Cup Volume Estimator Module

> **Konteks untuk AI Agent:** Ini adalah modul C++ mandiri mengikuti pola `Model_kakip/module/` yang menghitung volume gelas kopi (mL) dari raw fisheye frame. Modul ini digunakan setelah YOLO/ArUco/MiDaS sudah menghasilkan bounding box, jarak tray, dan tinggi gelas.

---

## Posisi dalam Ekosistem Model_kakip

```
Model_kakip/module/
├── camera/         ← akuisisi frame kamera
├── detections/     ← YOLO + MiDaS + ArUco inference
└── volume/         ← [MODUL INI] estimasi volume gelas
```

Pipeline lengkap:
```
Kamera (fisheye) → [mod_camera] → raw_frame
raw_frame         → [mod_detect] → bbox, h_cup_cm, z_tray_cm
raw_frame + data  → [mod_volume] → volume_ml  ← hasil akhir
```

**mod_volume TIDAK melakukan** deteksi AI. Ia hanya menerima hasil dari modul lain dan menghitung volume.

---

## Struktur Modul

```
module/volume/
├── CMakeLists.txt                      ← definisi mod_volume (SHARED library)
├── README.md                           ← file ini
├── include/
│   └── volume/                         ← public headers (diakses sebagai <volume/...>)
│       ├── cup_volume_estimator.h      ← API UTAMA (satu-satunya yang di-include caller)
│       ├── volume_config.h             ← konstanta hardcoded
│       ├── volume_math.h               ← fungsi math pure
│       └── height_math.h              ← definisi struct BBox
└── src/                                ← implementasi (tidak terekspos)
    ├── cup_volume_estimator.cpp
    ├── volume_math.cpp
    ├── height_math.cpp
    └── fisheye/                        ← internal Moildev wrapper
        ├── moil_undistorter.h          ← internal, JANGAN di-include langsung
        ├── moil_undistorter.cpp
        ├── anypoint_controller.h
        └── anypoint_controller.cpp
```

> **Aturan include:** Caller **HANYA** boleh `#include <volume/cup_volume_estimator.h>`. Jangan include header dari `src/` — itu internal modul.

---

## Cara Integrasi ke Proyek Lain

### 1. Tambahkan modul via `add_subdirectory()`

```cmake
# Di CMakeLists.txt proyek Anda:
add_subdirectory(path/to/Model_kakip/module/volume)

target_link_libraries(your_app PRIVATE
    mod_volume
    /path/to/weights/moil/libmoildev.a   # ← WAJIB di-link terpisah
)
```

> **Kenapa `libmoildev.a` terpisah?**
> `libmoildev.a` tidak di-compile dengan `-fPIC` (Position Independent Code)
> sehingga tidak bisa di-embed ke dalam `.so`. Library ini harus di-link
> oleh binary akhir (executable Anda), bukan oleh `mod_volume.so`.

### 2. Include HANYA satu header

```cpp
#include <volume/cup_volume_estimator.h>
```

---

## API Reference

### Struct `BBox` (dari `height_math.h`)

```cpp
struct BBox {
    int x1, y1, x2, y2;  // Pixel coordinates dari YOLO output
};
```

### Struct `VolumeInput`

```cpp
struct VolumeInput {
    cv::Mat frame;       // Raw fisheye frame (BELUM di-undistort — modul yang handle)
    BBox    bbox;        // Bounding box gelas dari YOLO
    double  h_cup_cm;   // Tinggi gelas (cm) dari MiDaS/height pipeline
    double  z_tray_cm;  // Jarak kamera ke meja via ArUco (cm)
    double  focal_px;   // Focal length efektif dari adjusted_focal_px()
};
```

### Struct `VolumeResult`

```cpp
struct VolumeResult {
    double rim_w_px;    // Lebar rim dalam piksel (debug)
    double z_rim_cm;    // Jarak kamera ke bibir gelas = z_tray - h_cup
    double diameter_cm; // Diameter fisik gelas (cm)
    double volume_ml;   // Volume estimasi dalam mL
    bool   valid;       // true jika semua tahap berhasil — CEK INI DULU
};
```

### Class `CupVolumeEstimator`

```cpp
// Konstruktor (panggil SEKALI di startup)
CupVolumeEstimator(
    const std::string& camera_params_json,  // path ke camera_parameters.json
    const std::string& camera_name = "syue_7730v1_6",
    int frame_width  = 2592,
    int frame_height = 1944
);

VolumeResult estimate(const VolumeInput& input);  // per-frame
double adjusted_focal_px() const;                  // inject ke ArUco
static void apply_exposure(cv::VideoCapture& cap); // set V4L2=5000
bool is_ready() const;
```

---

## Contoh Penggunaan Lengkap

```cpp
#include <volume/cup_volume_estimator.h>
#include <opencv2/opencv.hpp>

int main() {
    // ── STARTUP (SEKALI) ─────────────────────────────────────────────────────
    fusion::CupVolumeEstimator vol(
        "../../weights/moil/camera_parameters.json",
        "syue_7730v1_6"
    );
    if (!vol.is_ready()) { return 1; }

    cv::VideoCapture cap(0);
    cap.set(cv::CAP_PROP_FRAME_WIDTH,  2592);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 1944);
    fusion::CupVolumeEstimator::apply_exposure(cap);  // V4L2 = 5000

    // Gunakan focal length dari modul untuk ArUco (bukan yang dari JSON mentah)
    double focal_px = vol.adjusted_focal_px();

    // ── PER-FRAME ────────────────────────────────────────────────────────────
    while (true) {
        cv::Mat raw;
        cap >> raw;

        // === Hasil dari pipeline Anda (YOLO + ArUco + MiDaS) ===
        fusion::BBox bbox = {x1, y1, x2, y2};  // dari YOLO
        double z_tray_cm  = /* dari ArUco */;
        double h_cup_cm   = /* dari MiDaS/height pipeline */;

        // === Panggil mod_volume ===
        fusion::VolumeInput in { raw, bbox, h_cup_cm, z_tray_cm, focal_px };
        auto res = vol.estimate(in);

        if (res.valid) {
            printf("Volume: %.0f mL | Diameter: %.1f cm\n",
                   res.volume_ml, res.diameter_cm);
        }
    }
}
```

---

## Parameter Hardcoded (dari `volume_config.h`)

| Parameter | Nilai | Penjelasan |
|-----------|-------|------------|
| `MOIL_ZOOM` | `2.0x` | Hybrid: Moildev 1.5x + digital crop 1.33x |
| `MOIL_MODE` | `2` | AnyPointM2 = Pitch/Yaw/Roll |
| `MOIL_PITCH_DEG` | `-15.0°` | Kamera miring 15° ke bawah |
| `V4L2_EXPOSURE` | `5000` | Ekivalen smart_exposure=5.0 di Python |
| Interpolasi | `INTER_LINEAR` | **JANGAN ganti** — CUBIC/LANCZOS crash di OpenCL |

---

## Hubungan ke Pipeline Python

| Python (`07_midas_aruco_fusion`) | C++ (`mod_volume`) |
|---|---|
| `core/volume_math.py::measure_rim_width_px()` | `fusion::measure_rim_width_px()` |
| `core/volume_math.py::calc_diameter()` | `fusion::calc_diameter()` |
| `core/volume_math.py::calc_volume()` | `fusion::calc_volume()` |
| `core/moil_undistorter.py::undistort()` | `fusion::MoilUndistorter::undistort()` |
| `run_fusion.py` → blok volume di live loop | `fusion::CupVolumeEstimator::estimate()` |

---

## File Runtime yang Dibutuhkan

| File | Dibutuhkan oleh | Lokasi |
|------|----------------|--------|
| `libmod_volume.so` | Binary proyek Anda | Sama dengan executable / `LD_LIBRARY_PATH` |
| `camera_parameters.json` | `CupVolumeEstimator` constructor | Path yang diberikan ke constructor |
| `libmoildev.a` | Linker saat build | `weights/moil/libmoildev.a` |

---

## Troubleshooting

| Masalah | Penyebab | Solusi |
|---------|----------|--------|
| `is_ready()` = false | JSON tidak ditemukan / nama profil salah | Pastikan path benar dan nama ada di JSON |
| `valid` = false | `h_cup_cm` atau `bbox` tidak valid | Cek output YOLO dan height pipeline |
| Link error saat build | `libmoildev.a` tidak di-link | Tambah `libmoildev.a` ke `target_link_libraries` |
| `volume_ml` tidak wajar | `focal_px` salah | Gunakan `adjusted_focal_px()`, bukan focal dari JSON |
| Crash runtime `.so` tidak ditemukan | Library path salah | Set `LD_LIBRARY_PATH` atau install ke `/usr/lib` |

---

## Catatan Penting untuk AI Agent

1. **Hanya include satu header:** `#include <volume/cup_volume_estimator.h>`
2. **Frame harus RAW fisheye** — jangan berikan frame yang sudah di-undistort
3. **`focal_px` dari `adjusted_focal_px()`** — sudah memperhitungkan Moildev zoom 2.0x
4. **Inisialisasi SEKALI** — konstruktor mahal (load Moildev maps), jangan per-frame
5. **`libmoildev.a` link terpisah** — bukan dari `mod_volume`, tapi dari binary Anda langsung
6. **Thread safety** — buat instance terpisah per thread jika perlu paralel
