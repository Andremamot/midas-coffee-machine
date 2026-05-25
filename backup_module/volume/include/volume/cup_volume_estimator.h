/**
 * @file cup_volume_estimator.h
 * @brief API publik utama mod_volume — estimasi volume gelas dari raw fisheye frame.
 *
 * Include HANYA file ini di proyek Anda:
 *   #include <volume/cup_volume_estimator.h>
 *
 * Pipeline internal (otomatis):
 *   raw_frame → [MoilUndistorter] → [measure_rim_width_px] → [calc_diameter] → [calc_volume]
 *
 * Semua parameter Moildev sudah hardcoded via volume_config.h:
 *   pitch = -15°, zoom = 2.0, mode = 2, interpolasi = INTER_LINEAR
 */

#pragma once

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <string>
#include <memory>

#include "height_math.h"    // BBox
#include "volume_config.h"  // hardcoded constants

namespace fusion {

// Forward declaration — implementasi internal di src/fisheye/
class MoilUndistorter;

// ─────────────────────────────────────────────────────────────────────────────

/**
 * @brief Input per-frame untuk estimasi volume.
 *
 * Semua nilai kecuali frame harus sudah tersedia dari pipeline YOLO/ArUco/MiDaS.
 */
struct VolumeInput {
    cv::Mat frame;       ///< Raw fisheye frame (BELUM di-undistort)
    BBox    bbox;        ///< Bounding box gelas dari YOLO {x1, y1, x2, y2}
    double  h_cup_cm;   ///< Tinggi gelas dalam cm (dari MiDaS + height pipeline)
    double  z_tray_cm;  ///< Jarak kamera ke tray via ArUco (cm)
    double  focal_px;   ///< Focal length efektif (dari adjusted_focal_px())
};

/**
 * @brief Hasil estimasi volume untuk satu frame.
 *
 * Cek `valid` sebelum menggunakan nilai lain.
 */
struct VolumeResult {
    double rim_w_px    = 0.0; ///< Lebar rim gelas dalam piksel (debug)
    double z_rim_cm    = 0.0; ///< Jarak kamera ke bibir gelas = z_tray - h_cup
    double diameter_cm = 0.0; ///< Diameter fisik gelas (cm)
    double volume_ml   = 0.0; ///< Volume estimasi dalam mL
    bool   valid       = false;///< true jika semua tahap berhasil
};

// ─────────────────────────────────────────────────────────────────────────────

/**
 * @class CupVolumeEstimator
 * @brief Modul mandiri estimasi volume gelas dari raw fisheye frame.
 *
 * Contoh penggunaan:
 * @code
 *   // Di startup (SEKALI):
 *   fusion::CupVolumeEstimator vol("camera_parameters.json", "syue_7730v1_6");
 *   double focal_px = vol.adjusted_focal_px();
 *
 *   // Per-frame:
 *   fusion::VolumeInput in { raw_frame, bbox, h_cup_cm, z_tray_cm, focal_px };
 *   auto res = vol.estimate(in);
 *   if (res.valid) printf("Volume: %.0f mL\n", res.volume_ml);
 * @endcode
 */
class CupVolumeEstimator {
public:
    /**
     * @brief Konstruktor — inisialisasi MoilUndistorter dengan params hardcoded.
     *
     * @param camera_params_json  Path ke camera_parameters.json Moildev
     * @param camera_name         Nama profil kamera di JSON (mis. "syue_7730v1_6")
     * @param frame_width         Lebar resolusi stream (default 2592)
     * @param frame_height        Tinggi resolusi stream (default 1944)
     * @throws std::runtime_error jika JSON tidak ditemukan
     */
    explicit CupVolumeEstimator(
        const std::string& camera_params_json,
        const std::string& camera_name = "syue_7730v1_6",
        int frame_width  = 2592,
        int frame_height = 1944
    );

    ~CupVolumeEstimator();

    CupVolumeEstimator(const CupVolumeEstimator&)            = delete;
    CupVolumeEstimator& operator=(const CupVolumeEstimator&) = delete;
    CupVolumeEstimator(CupVolumeEstimator&&)                  = default;
    CupVolumeEstimator& operator=(CupVolumeEstimator&&)       = default;

    // ── Core API ─────────────────────────────────────────────────────────────

    /**
     * @brief Estimasi volume dari satu raw fisheye frame.
     *
     * Otomatis undistort frame sebelum deteksi rim.
     * Thread safety: tidak thread-safe — gunakan instance terpisah per thread.
     */
    VolumeResult estimate(const VolumeInput& input);

    // ── Utilities ─────────────────────────────────────────────────────────────

    /**
     * @brief Focal length (px) yang sudah disesuaikan dengan Moildev zoom.
     *
     * Inject nilai ini ke ArUco detector agar konsisten:
     *   aruco.camera_matrix[0][0] = vol.adjusted_focal_px();
     */
    double adjusted_focal_px() const;

    /**
     * @brief Hitung volume langsung dari tinggi + diameter (tanpa frame).
     * Helper statis jika diameter sudah diketahui.
     */
    static double from_height_diameter(double h_cup_cm, double diameter_cm);

    /**
     * @brief Set V4L2 manual exposure hardcoded (5000) ke VideoCapture.
     * Panggil SEKALI setelah cap.open().
     */
    static void apply_exposure(cv::VideoCapture& cap);

    bool is_ready() const { return ready_; }

private:
    bool ready_ = false;
    std::unique_ptr<MoilUndistorter> moil_;
    int frame_w_;
    int frame_h_;
};

}  // namespace fusion
