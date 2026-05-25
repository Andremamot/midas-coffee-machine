/**
 * @file moil_undistorter.h
 * @brief Public API untuk mod_moil — Fisheye undistortion via Moildev library.
 *
 * Include HANYA file ini dari proyek yang menggunakan mod_moil:
 *   #include <moil/moil_undistorter.h>
 *
 * Port dari: 07_midas_aruco_fusion/core/moil_undistorter.hpp
 * Dipaketkan sebagai shared library (mod_moil) dengan pola backup_module.
 *
 * CATATAN libmoildevren.a:
 *   Library tidak di-embed ke .so karena tidak dikompilasi dengan -fPIC.
 *   Proyek pemanggil harus link libmoildevren.a secara eksplisit:
 *     target_link_libraries(your_app PRIVATE mod_moil /path/to/libmoildevren.a)
 */

#pragma once

#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>
#include <mutex>

// Forward-declare the legacy Moildev class dari libmoildevren.a
// (tidak include moildev.hpp karena itu versi baru yang tidak match library)
class Moildev;

/**
 * @class MoilUndistorter
 * @brief Fisheye undistortion wrapper menggunakan Moildev (via libmoildevren.a).
 *
 * C++ port dari core/moil_undistorter.py.
 *
 * Contoh penggunaan:
 * @code
 *   MoilUndistorter moil("camera_parameters.json", "syue_7730v1_6",
 *                        pitch, yaw, roll, zoom, mode);
 *   cv::Mat corrected = moil.undistort(raw_frame);
 *   cv::Mat K = moil.build_aruco_camera_matrix(width, height);
 * @endcode
 */
class MoilUndistorter {
public:
    /**
     * @brief Konstruktor — baca JSON, konfigurasi Moildev, pre-compute remap maps.
     *
     * @param json_path      Path ke camera_parameters.json
     * @param camera_name    Nama profil kamera di JSON (mis. "syue_7730v1_6")
     * @param pitch          Anypoint pitch dalam derajat (default 0 = lurus)
     * @param yaw            Anypoint yaw dalam derajat (default 0)
     * @param roll           Anypoint roll dalam derajat (default 0)
     * @param zoom           Zoom factor (default 1.4)
     * @param mode           1 = AnyPointM (alpha/beta), 2 = AnyPointM2 (pitch/yaw/roll)
     * @param frame_w        Lebar frame kamera aktual (0 = ambil dari JSON)
     * @param frame_h        Tinggi frame kamera aktual (0 = ambil dari JSON)
     * @param output_w       Lebar output remap (0 = sama dengan frame)
     * @param output_h       Tinggi output remap (0 = sama dengan frame)
     * @throws std::runtime_error jika JSON tidak ditemukan atau kamera tidak ada
     */
    MoilUndistorter(const std::string& json_path,
                    const std::string& camera_name = "syue_7730v1_6",
                    float  pitch    = 0.0f,
                    float  yaw      = 0.0f,
                    float  roll     = 0.0f,
                    float  zoom     = 1.4f,
                    int    mode     = 2,
                    int    frame_w  = 0,   ///< 0 = gunakan resolusi sensor dari JSON
                    int    frame_h  = 0,
                    int    output_w = 0,   ///< 0 = sama dengan frame
                    int    output_h = 0);

    ~MoilUndistorter();

    // Non-copyable (karena Moildev* tidak bisa di-copy)
    MoilUndistorter(const MoilUndistorter&)            = delete;
    MoilUndistorter& operator=(const MoilUndistorter&) = delete;

    // ── Core API ───────────────────────────────────────────────────────────

    /**
     * @brief Terapkan anypoint undistortion ke satu frame BGR.
     *
     * Pipeline internal:
     *   1. Moildev remap (undistortion + zoom ≤ MAX_MOIL_ZOOM)
     *   2. Digital zoom via center-crop + resize (sisa zoom)
     *   3. Unsharp mask sharpening (jika aktif)
     *
     * @param frame  Frame BGR input (belum di-undistort)
     * @return       Frame BGR setelah undistortion
     */
    cv::Mat undistort(const cv::Mat& frame);

    /**
     * @brief Update remap maps dengan parameter anypoint baru.
     *
     * Panggil ketika pitch/yaw/roll/zoom berubah (mis. dari GUI drag).
     * Thread-safe (lock internal).
     */
    void update_maps(float pitch, float yaw, float roll, float zoom);

    /**
     * @brief Bangun camera matrix K (3×3) untuk ArUco detection.
     *
     * Harus dipanggil setiap frame karena zoom mempengaruhi focal length.
     *
     * @param frame_width   Lebar frame streaming aktual
     * @param frame_height  Tinggi frame streaming aktual
     * @return              cv::Mat 3×3, CV_64F
     */
    cv::Mat build_aruco_camera_matrix(int frame_width, int frame_height) const;

    // ── Accessors ──────────────────────────────────────────────────────────

    float pitch_deg()     const { return pitch_;        }
    float yaw_deg()       const { return yaw_;          }
    float roll_deg()      const { return roll_;         }
    float zoom_factor()   const { return zoom_;         }
    float moil_zoom()     const { return moil_zoom_;    }   ///< Komponen zoom ke Moildev (≤ MAX_MOIL_ZOOM)
    float digital_zoom()  const { return digital_zoom_; }   ///< Komponen zoom digital (≥ 1.0)
    int   mode()          const { return mode_;         }

    float image_width()   const { return img_w_; }
    float image_height()  const { return img_h_; }

    /**
     * @brief Focal length ekuivalen (piksel) = parameter5 / calibrationRatio.
     */
    float adjusted_focal_length() const;

    /**
     * @brief Set kekuatan sharpening setelah remap.
     * @param amount  0.0 = tidak ada, 1.0 = moderat, 2.0 = kuat
     */
    void  set_sharpen(float amount) { sharpen_amount_ = std::max(0.0f, amount); }
    float sharpen_amount() const    { return sharpen_amount_; }

    /**
     * @brief Set zoom reference untuk perhitungan focal length ArUco.
     *
     * Formula empiris: fl = param5_ * zoom / zoom_ref²
     * Default: 1.6 (dikalibrasi untuk libmoildevren.a / syue_7730v1 cameras).
     */
    void  set_aruco_zoom_ref(float zoom_ref) { zoom_ref_ = std::max(0.1f, zoom_ref); }
    float aruco_zoom_ref() const { return zoom_ref_; }

private:
    // Moildev instance (old API dari libmoildevren.a)
    Moildev* moil_;

    // Parameter kamera untuk focal length & ArUco matrix
    double param5_;
    double calib_ratio_;
    float  img_w_, img_h_;       ///< Resolusi sensor JSON
    float  frame_w_, frame_h_;   ///< Resolusi frame kamera aktual
    float  output_w_, output_h_; ///< Target resolusi remap output

    // Kekuatan sharpening (0 = off)
    float sharpen_amount_ = 0.0f;

    // Parameter anypoint saat ini
    float pitch_, yaw_, roll_, zoom_;
    int   mode_;

    // Zoom reference untuk perhitungan focal length ArUco
    float zoom_ref_;

    // ── Hybrid Zoom ────────────────────────────────────────────────────────
    // Batas zoom aman Moildev sebelum polynomial wrap-around.
    // Sama persis dengan Python: MAX_MOIL_ZOOM = 1.5
    static constexpr float MAX_MOIL_ZOOM = 1.5f;

    float moil_zoom_;     ///< Komponen zoom yang dikirim ke Moildev (≤ MAX_MOIL_ZOOM)
    float digital_zoom_;  ///< Komponen zoom tambahan via center-crop (≥ 1.0)

    // Split zoom_ menjadi (moil_zoom_, digital_zoom_)
    void split_zoom_();

    // Terapkan center-crop + resize (digital zoom)
    cv::Mat digital_crop_(const cv::Mat& frame) const;

    // Remap maps (float32)
    cv::Mat map_x_, map_y_;
    cv::Mat map1_16s_, map2_16s_;    ///< Pre-converted Fixed-Point maps for faster remap
    mutable std::mutex maps_mutex_;  ///< Protects maps dari concurrent undistort()

    // Internal: re-generate maps dari params saat ini
    void rebuild_maps_();
};
