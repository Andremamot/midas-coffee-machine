/*******************************************************************************
 * core/moildev_applicator.hpp
 *
 * Fisheye undistortion engine — diport dari MoildevApplicator (unicorn-solution).
 *
 * Keunggulan vs implementasi lama:
 *   - LUT Alpha-Rho via Horner's Method (jauh lebih cepat dari pow())
 *   - getAlphaBeta(x,y) — konversi koordinat klik ke sudut fisheye
 *   - Formula focal length yang benar: param5 / calibRatio (bukan empiris)
 *   - Thread-safe dengan std::mutex
 *   - INTER_LINEAR + fixed-point maps (CV_16SC2) untuk performa optimal di ARM
 *   - update_maps() thread-safe untuk kontrol real-time
 *
 * Digunakan oleh:
 *   - run_fusion.cpp (C++ pipeline)
 *   - core/gui_fusion.cpp (GUI overlay)
 ******************************************************************************/
#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>

// Forward-declare engine CPU agar tidak ditarik ke setiap TU yang include header ini
namespace moildev::cpu {
class Moildev;
}

/**
 * @class MoildevApplicator
 * @brief High-level fisheye undistortion engine berbasis MoildevApplicator dari unicorn-solution.
 *
 * Menyediakan:
 *   - undistort(frame)       — remap fisheye ke anypoint
 *   - update_maps(...)       — update parameter secara real-time (thread-safe)
 *   - getAlphaBeta(x, y)     — konversi koordinat piksel ke sudut (Alpha, Beta)
 *   - adjusted_focal_length()— focal length koreksi yang benar (param5/calibRatio)
 *   - build_aruco_camera_matrix() — camera matrix K untuk ArUco
 *
 * Tidak mendukung panorama (kamera selalu menghadap ke bawah, alpha/beta ≈ 0).
 */
class MoildevApplicator {
public:
    /**
     * @brief Inisialisasi engine fisheye.
     *
     * Membaca profil kamera dari JSON, inisialisasi moildev::cpu::Moildev,
     * pre-compute LUT Alpha-Rho (Horner's Method), dan build remap maps.
     *
     * @param json_path      Path ke camera_parameters.json
     * @param camera_name    Nama profil di dalam JSON (contoh: "syue_7730v1_6")
     * @param pitch          Sudut pitch dalam derajat (default 0.0)
     * @param yaw            Sudut yaw dalam derajat   (default 0.0)
     * @param roll           Sudut roll dalam derajat  (default 0.0)
     * @param zoom           Faktor zoom                (default 1.4)
     * @param mode           1=AnyPointM (alpha/beta), 2=AnyPointM2 (pitch/yaw) (default 2)
     * @param frame_w        Lebar frame input (0 = gunakan ukuran JSON)
     * @param frame_h        Tinggi frame input (0 = gunakan ukuran JSON)
     * @param output_w       Lebar output map (0 = sama dengan frame)
     * @param output_h       Tinggi output map (0 = sama dengan frame)
     */
    MoildevApplicator(const std::string &json_path,
               const std::string &camera_name = "syue_7730v1_6",
               float pitch  = 0.0f, float yaw  = 0.0f, float roll = 0.0f,
               float zoom   = 1.4f, int   mode  = 2,
               int   frame_w = 0,  int   frame_h = 0,
               int   output_w = 0, int   output_h = 0);

    ~MoildevApplicator();

    // Non-copyable
    MoildevApplicator(const MoildevApplicator &) = delete;
    MoildevApplicator &operator=(const MoildevApplicator &) = delete;

    // ── Core API ─────────────────────────────────────────────────────────────

    /**
     * @brief Terapkan undistortion fisheye ke satu frame BGR.
     * @param frame  Frame BGR input (fisheye mentah)
     * @return       Frame BGR hasil undistortion
     *               (INTER_LINEAR + fixed-point maps untuk performa ARM optimal)
     */
    cv::Mat undistort(const cv::Mat &frame);

    /**
     * @brief Update parameter anypoint dan regenerasi remap maps (thread-safe).
     *
     * Dipanggil oleh AnypointController saat user drag mouse / scroll.
     * Aman dipanggil dari thread lain sementara undistort() berjalan.
     */
    void update_maps(float pitch, float yaw, float roll, float zoom);

    /**
     * @brief Konversi koordinat piksel pada frame fisheye ke sudut (Alpha, Beta).
     *
     * Menggunakan LUT Rho-to-Alpha yang sudah di-pre-compute saat konstruktor.
     * Setara dengan MoildevApplicator::getAlphaBeta() dari unicorn-solution.
     *
     * @param x  Koordinat piksel horizontal
     * @param y  Koordinat piksel vertikal
     * @return   {Alpha (elevasi derajat), Beta (azimuth derajat)}
     */
    std::pair<float, float> get_alpha_beta(int x, int y) const;

    /**
     * @brief Buat camera matrix 3×3 untuk deteksi ArUco.
     *
     * Formula yang benar: fl = adjusted_focal_length()
     * (BUKAN formula empiris zoom/zoom_ref² yang menyebabkan Z_tray meleset).
     *
     * @param frame_width   Lebar frame aktual
     * @param frame_height  Tinggi frame aktual
     * @return cv::Mat (3×3, CV_64F)
     */
    cv::Mat build_aruco_camera_matrix(int frame_width, int frame_height) const;

    // ── Accessor parameter ─────────────────────────────────────────────────

    float pitch_deg()   const { return pitch_; }
    float yaw_deg()     const { return yaw_;   }
    float roll_deg()    const { return roll_;  }
    float zoom_factor() const { return zoom_;  }
    int   mode()        const { return mode_;  }

    // Alias agar kompatibel dengan AnypointController lama
    float pitch         = 0.0f; ///< Dibaca oleh AnypointController::draw_overlay()
    float yaw           = 0.0f;
    float roll          = 0.0f;
    float zoom          = 1.4f;

    float image_width()  const { return img_w_; }
    float image_height() const { return img_h_; }

    /**
     * @brief Focal length ekivalen piksel yang benar.
     *
     * Formula: param5_ / calib_ratio_
     * Ini adalah formula yang sama dengan unicorn-solution (MoildevApplicator).
     */
    float adjusted_focal_length() const;

    /**
     * @brief Nama backend yang aktif ("CPU").
     * Diperluas ke "OpenCL" / "CUDA" di masa depan.
     */
    std::string getBackendName() const { return "CPU"; }

    /**
     * @brief Set sharpening setelah remap (opsional).
     * @param amount  0.0 = tidak ada, 1.0 = sedang, 2.0 = kuat
     */
    void set_sharpen(float amount);
    float sharpen_amount() const { return sharpen_amount_; }

    /**
     * @brief Toggle debug output (default: OFF).
     *
     * Saat verbose=true, rebuild_maps_() mencetak info diagnostik:
     *   "[MOIL-DBG] maps rebuilt ..."
     * Hanya aktifkan saat debugging; JANGAN aktifkan di production
     * karena melakukan cv::minMaxLoc pada map besar setiap frame.
     *
     * @param v  true = aktifkan debug output, false = silent (default)
     */
    void set_verbose(bool v) { verbose_ = v; }
    bool is_verbose()  const { return verbose_; }

private:
    // ── Engine Moildev CPU (libmoildev_cpu.so) ────────────────────────────
    std::unique_ptr<moildev::cpu::Moildev> moil_;

    // ── Parameter kamera (dari JSON) ──────────────────────────────────────
    double param5_;       ///< parameter5 dari kalibrasi
    double calib_ratio_;  ///< calibrationRatio dari kalibrasi
    double icx_, icy_;    ///< pusat optik pada resolusi output

    float img_w_, img_h_;        ///< resolusi sensor dari JSON
    float frame_w_, frame_h_;    ///< resolusi frame input aktual
    float output_w_, output_h_;  ///< resolusi output remap

    // ── Parameter anypoint saat ini ───────────────────────────────────────
    float pitch_, yaw_, roll_, zoom_;
    int   mode_;

    // ── Sharpening pasca-remap ─────────────────────────────────────────────
    float sharpen_amount_ = 0.0f;

    // ── Verbose/debug flag ────────────────────────────────────────────────
    // Default false — tidak ada output di production.
    // Set ke true via set_verbose(true) hanya untuk debugging lokal.
    bool verbose_ = false;

    // ── Remap maps ────────────────────────────────────────────────────────
    cv::Mat map_x_, map_y_;           ///< Float32 maps (source of truth)
    cv::Mat map_x_fixed_, map_y_fixed_; ///< Fixed-point maps (CV_16SC2 + CV_16UC1)
                                      ///< Pre-konversi via cv::convertMaps untuk
                                      ///< performa ARM/RZ-V2H optimal (~2-4x speedup)
    mutable std::mutex maps_mutex_;

    // ── LUT Alpha-Rho (diport dari MoildevApplicator::initializeAlphaRhoTables) ──
    std::vector<double> alpha_to_rho_table_;  ///< alpha (derajat*10) → rho (piksel)
    std::vector<int>    rho_to_alpha_table_;  ///< rho (piksel)       → alpha (derajat*10)

    // ── Private helpers ───────────────────────────────────────────────────

    /** Regenerasi maps dari parameter saat ini. Harus dipanggil di bawah maps_mutex_. */
    void rebuild_maps_();

    /**
     * Pre-compute LUT Alpha-Rho menggunakan Horner's Method.
     * Diport dari MoildevApplicator::initializeAlphaRhoTables() unicorn-solution.
     * Horner's Method: O(n) vs O(n log n) dari pow() biasa.
     */
    void init_alpha_rho_tables_(double p0, double p1, double p2,
                                double p3, double p4, double p5,
                                double calib);
};
