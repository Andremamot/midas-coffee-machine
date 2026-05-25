/**
 * @file aruco_detector.h
 * @brief Public API untuk mod_aruco — ArUco marker detection & pose estimation.
 *
 * Include HANYA file ini dari proyek yang menggunakan mod_aruco:
 *   #include <aruco/aruco_detector.h>
 *
 * Port dari: 06_aruco_marker/aruco_detector.hpp
 * Dipaketkan sebagai shared library (mod_aruco) dengan pola backup_module.
 */

#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <string>
#include <vector>
#include <map>

// ─────────────────────────────────────────────────────────────────────────────
// Structs
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @brief Hasil deteksi satu ArUco marker.
 */
struct ArucoResult {
    int                        id;               ///< Marker ID dari dictionary
    std::vector<cv::Point2f>   corners;          ///< 4 sudut marker dalam piksel
    float                      distance_cm;      ///< Jarak kamera ke marker (cm), dari tvec[2]
    cv::Vec3d                  rvec;             ///< Rotation vector (Rodrigues)
    cv::Vec3d                  tvec;             ///< Translation vector (cm)
    float                      euler_roll;       ///< Roll dalam derajat
    float                      euler_pitch;      ///< Pitch dalam derajat
    float                      euler_yaw;        ///< Yaw dalam derajat
    cv::Point2f                center;           ///< Titik tengah marker
    float                      reprojection_error; ///< Error reproyeksi rata-rata (piksel)
};

/**
 * @brief Hasil agregat dari get_best_distance().
 *
 * Mengembalikan jarak terbaik (median dari marker valid)
 * beserta statistik filtering.
 */
struct BestDistanceResult {
    float              distance_cm;    ///< Jarak median terbaik (cm)
    int                used_count;     ///< Jumlah marker yang digunakan
    int                rejected_count; ///< Jumlah marker yang dibuang (reproj error terlalu besar)
    std::vector<float> all_distances;  ///< Semua jarak marker yang valid
};

// ─────────────────────────────────────────────────────────────────────────────
// ArucoDetector
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @class ArucoDetector
 * @brief Wrapper ArUco marker detector dengan pose estimation.
 *
 * Contoh penggunaan:
 * @code
 *   ArucoDetector aruco(5.0f, "DICT_4X4_50", "calibration_params.yml");
 *   auto results = aruco.detect(frame);
 *   BestDistanceResult best = aruco.get_best_distance(results);
 *   if (best.distance_cm > 0) {
 *       std::cout << "Z_tray = " << best.distance_cm << " cm\n";
 *   }
 * @endcode
 */
class ArucoDetector {
public:
    /**
     * @brief Konstruktor — load dictionary dan kalibrasi kamera.
     *
     * @param marker_size_cm    Ukuran fisik sisi marker (cm)
     * @param dictionary_name   Nama dictionary, contoh "DICT_4X4_50"
     * @param params_path       Path ke file kalibrasi YAML (camera_matrix + dist_coeffs)
     */
    ArucoDetector(float              marker_size_cm   = 5.0f,
                  const std::string& dictionary_name  = "DICT_4X4_50",
                  const std::string& params_path      = "../calibration_params.yml");

    ~ArucoDetector() = default;

    // ── Core API ───────────────────────────────────────────────────────────

    /**
     * @brief Deteksi marker ArUco pada frame dan estimasi pose-nya.
     *
     * @param frame  Frame BGR atau grayscale
     * @return       Vektor hasil deteksi (kosong jika tidak ada marker)
     */
    std::vector<ArucoResult> detect(const cv::Mat& frame);

    /**
     * @brief Gambar anotasi marker (corners, axes, label) pada frame.
     *
     * @param frame    Frame BGR input
     * @param results  Hasil detect()
     * @return         Frame baru dengan anotasi
     */
    cv::Mat annotate_frame(const cv::Mat& frame,
                           const std::vector<ArucoResult>& results);

    /**
     * @brief Pilih jarak terbaik (median) dari semua marker yang valid.
     *
     * Marker dengan reprojection_error > max_reproj_error dibuang.
     * Jika semua dibuang, gunakan marker dengan error terkecil.
     *
     * @param results           Hasil detect()
     * @param max_reproj_error  Threshold error reproyeksi (piksel), default 0.5
     * @return                  BestDistanceResult
     */
    BestDistanceResult get_best_distance(const std::vector<ArucoResult>& results,
                                         float max_reproj_error = 0.5f);

    // ── Public members (diakses langsung oleh pipeline) ────────────────────

    std::string  dictionary_name;   ///< Nama dictionary yang dipakai
    float        marker_size_cm;    ///< Ukuran fisik marker (cm)
    cv::Mat      camera_matrix;     ///< 3×3 camera matrix K
    cv::Mat      dist_coeffs;       ///< 1×5 distortion coefficients
                                    ///< (bisa di-zero setelah Moildev undistortion)

private:
    cv::Ptr<cv::aruco::Dictionary>         aruco_dict;
    cv::Ptr<cv::aruco::DetectorParameters> aruco_params;
    float f_pixel_;  ///< Focal length rata-rata (px), dari (fx+fy)/2

    void  load_camera_calibration(const std::string& params_path);
    void  rvec_to_euler(const cv::Vec3d& rvec,
                        float& roll, float& pitch, float& yaw);
    float compute_reprojection_error(const std::vector<cv::Point2f>& corners_2d,
                                     const cv::Vec3d& rvec,
                                     const cv::Vec3d& tvec);

    std::map<std::string, cv::aruco::PREDEFINED_DICTIONARY_NAME> dict_map_ = {
        {"DICT_4X4_50",  cv::aruco::DICT_4X4_50},
        {"DICT_4X4_100", cv::aruco::DICT_4X4_100},
        {"DICT_4X4_250", cv::aruco::DICT_4X4_250},
        {"DICT_5X5_50",  cv::aruco::DICT_5X5_50},
        {"DICT_5X5_100", cv::aruco::DICT_5X5_100},
        {"DICT_5X5_250", cv::aruco::DICT_5X5_250},
        {"DICT_6X6_50",  cv::aruco::DICT_6X6_50},
        {"DICT_6X6_100", cv::aruco::DICT_6X6_100},
        {"DICT_6X6_250", cv::aruco::DICT_6X6_250},
        {"DICT_7X7_50",  cv::aruco::DICT_7X7_50},
        {"DICT_7X7_100", cv::aruco::DICT_7X7_100},
        {"DICT_7X7_250", cv::aruco::DICT_7X7_250},
    };
};
