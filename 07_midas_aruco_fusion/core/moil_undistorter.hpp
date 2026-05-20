/*******************************************************************************
 * core/moil_undistorter.hpp  (v2 — menggunakan API Moildev lama via libmoildevren.a)
 *
 * API yang ada di libmoildevren.a:
 *   class Moildev {
 *       Moildev();  // default ctor
 *       Config(string camera_name,
 *              double sensorW, double sensorH,
 *              double icx, double icy, double ratio,
 *              double imgW, double imgH,
 *              double calibRatio,
 *              double para0..para5, double)  // 14 doubles + 1 string
 *       AnyPointM (float* mapX, float* mapY, double alpha, double beta, double zoom)
 *       AnyPointM2(float* mapX, float* mapY, double pitch, double yaw,  double zoom)
 *       getImageWidth()  / getImageHeight()
 *   }
 ******************************************************************************/
#pragma once

#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

// Forward-declare the legacy Moildev class from libmoildevren.a
// (tidak include moildev.hpp karena itu adalah versi baru yang tidak match library)
class Moildev;

/**
 * @brief Fisheye undistortion wrapper using Moildev (via libmoildevren.a).
 *
 * C++ port of core/moil_undistorter.py.
 *
 * Usage:
 *   MoilUndistorter moil("camera_parameters.json", "syue_7730v1_6",
 *                        pitch, yaw, roll, zoom, mode);
 *   cv::Mat corrected = moil.undistort(raw_frame);
 *   cv::Mat K = moil.build_aruco_camera_matrix(width, height);
 */
class MoilUndistorter {
public:
    /**
     * @brief Construct and initialize the undistorter.
     *
     * Reads camera profile from JSON, configures Moildev instance,
     * and pre-computes the remap maps at init time (one-time cost).
     *
     * @param json_path      Absolute path to camera_parameters.json
     * @param camera_name    Profile key inside the JSON (e.g. "syue_7730v1_6")
     * @param pitch          Anypoint pitch in degrees (default 0 = straight)
     * @param yaw            Anypoint yaw in degrees  (default 0)
     * @param roll           Anypoint roll in degrees  (default 0)
     * @param zoom           Anypoint zoom factor      (default 1.4)
     * @param mode           1 = AnyPointM (alpha/beta), 2 = AnyPointM2 (pitch/yaw/roll)
     */
    MoilUndistorter(const std::string& json_path,
                    const std::string& camera_name = "syue_7730v1_6",
                    float pitch   = 0.0f,
                    float yaw     = 0.0f,
                    float roll    = 0.0f,
                    float zoom    = 1.4f,
                    int   mode    = 2,
                    int   frame_w = 0,   // 0 = use JSON sensor size
                    int   frame_h = 0,
                    int   output_w = 0,  // target output map size
                    int   output_h = 0);

    ~MoilUndistorter();

    // Non-copyable
    MoilUndistorter(const MoilUndistorter&)            = delete;
    MoilUndistorter& operator=(const MoilUndistorter&) = delete;

    // ── Core API ───────────────────────────────────────────────────────────

    /**
     * @brief Apply anypoint undistortion to a single BGR frame.
     * @param frame  Input BGR frame
     * @return       Undistorted BGR frame
     */
    cv::Mat undistort(const cv::Mat& frame);

    /**
     * @brief Regenerate remap maps with updated parameters.
     * Call this when pitch/yaw/roll/zoom changes (e.g. mouse drag).
     */
    void update_maps(float pitch, float yaw, float roll, float zoom);

    /**
     * @brief Build a 3×3 camera matrix K for ArUco detection.
     *
     * Should be called every frame (zoom affects focal length).
     *
     * @param frame_width   Actual width of streaming frame
     * @param frame_height  Actual height of streaming frame
     * @return cv::Mat (3×3, CV_64F)
     */
    cv::Mat build_aruco_camera_matrix(int frame_width, int frame_height) const;

    // ── Parameter accessors ────────────────────────────────────────────────
<<<<<<< Updated upstream
    float pitch_deg()   const { return pitch_;  }
    float yaw_deg()     const { return yaw_;    }
    float roll_deg()    const { return roll_;   }
    float zoom_factor() const { return zoom_;   }
    int   mode()        const { return mode_;   }
=======
    float pitch_deg()    const { return pitch_;        }
    float yaw_deg()      const { return yaw_;          }
    float roll_deg()     const { return roll_;         }
    float zoom_factor()  const { return zoom_;         }
    float moil_zoom()    const { return moil_zoom_;    }  ///< komponen zoom yg dikirim ke Moildev (≤ MAX_MOIL_ZOOM)
    float digital_zoom() const { return digital_zoom_; }  ///< komponen zoom digital (≥ 1.0)
    int   mode()         const { return mode_;         }
>>>>>>> Stashed changes

    float image_width()  const { return img_w_; }
    float image_height() const { return img_h_; }

    /**
     * @brief Estimated equivalent focal length (pixels) = parameter5 / calibrationRatio.
     */
    float adjusted_focal_length() const;

    /**
     * @brief Set sharpening strength applied after remap.
     * @param amount  0.0 = no sharpen, 1.0 = moderate, 2.0 = strong
     */
    void  set_sharpen(float amount) { sharpen_amount_ = std::max(0.0f, amount); }
    float sharpen_amount() const    { return sharpen_amount_; }

    /**
     * @brief Set zoom reference for ArUco focal length calculation.
     * Empirical formula: fl = param5_ * zoom / zoom_ref²
     * Default: 1.6 (calibrated for libmoildevren.a / syue_7730v1 cameras).
     */
    void  set_aruco_zoom_ref(float zoom_ref) { zoom_ref_ = std::max(0.1f, zoom_ref); }
    float aruco_zoom_ref() const { return zoom_ref_; }

private:
    // Moildev instance (old API from libmoildevren.a)
    Moildev* moil_;

    // Camera params needed for focal length & ArUco matrix
    double param5_;
    double calib_ratio_;
    float  img_w_, img_h_;           // sensor JSON resolution
    float  frame_w_, frame_h_;       // actual camera frame resolution
    float  output_w_, output_h_;     // target remap output resolution

    // Sharpening strength (0 = off)
    float  sharpen_amount_ = 0.0f;

    // Current anypoint params
    float pitch_, yaw_, roll_, zoom_;
    int   mode_;

    // Zoom referensi untuk perhitungan focal length ArUco.
    // Secara empiris: fl = param5_ * zoom / zoom_ref_^2
<<<<<<< Updated upstream
    // zoom_ref_ adalah zoom di mana formula fl=param5_/zoom kebetulan benar.
    // Untuk library libmoildevren.a dengan kamera syue_7730v1_6: zoom_ref ≈ 1.6
    // Dapat diubah via set_aruco_zoom_ref().
    float zoom_ref_;

    // Remap maps (float32, native sensor resolution)
    cv::Mat map_x_, map_y_;

    // Internal: re-generate maps from current params
=======
    float zoom_ref_;

    // ── Hybrid Zoom ───────────────────────────────────────────────────────────
    // Batas zoom aman Moildev sebelum polynomial wrap-around.
    // Sama persis dengan Python: MAX_MOIL_ZOOM = 1.5
    static constexpr float MAX_MOIL_ZOOM = 1.5f;

    float moil_zoom_;     ///< komponen zoom yang dikirim ke Moildev (≤ MAX_MOIL_ZOOM)
    float digital_zoom_;  ///< komponen zoom tambahan via center-crop (≥ 1.0)

    // Split zoom_ menjadi (moil_zoom_, digital_zoom_)
    void split_zoom_();

    // Terapkan center-crop + resize (digital zoom)
    cv::Mat digital_crop_(const cv::Mat& frame) const;

    // Remap maps (float32)
    cv::Mat map_x_, map_y_;

    // Internal: re-generate maps dari params saat ini
>>>>>>> Stashed changes
    void rebuild_maps_();
};
