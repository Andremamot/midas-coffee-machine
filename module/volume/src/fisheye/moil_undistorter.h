/**
 * @file moil_undistorter.h
 * @brief Fisheye undistortion via Moildev maps.
 * Port of Python core/moil_undistorter.py.
 */

#pragma once

#include <mutex>
#include <opencv2/core.hpp>
#include <string>

namespace fusion {

class moil_facade {
public:
  static constexpr double MAX_MOIL_ZOOM = 1.5;

  explicit moil_facade(const std::string &camera_params_json,
                       const std::string &camera_name = "", int mode = 1);

  /// Undistort a frame using current maps
  cv::Mat undistort(const cv::Mat &frame);

  /// Thread-safe map update
  void update_maps(double pitch, double yaw, double roll, double zoom);

  /// Build ArUco camera matrix from current undistortion params
  cv::Matx33d build_aruco_camera_matrix(int w, int h) const;

  // Getters
  double pitch() const { return pitch_; }
  double yaw() const { return yaw_; }
  double roll() const { return roll_; }
  double zoom() const { return zoom_; }
  double adjusted_focal_length() const { return adjusted_focal_; }
  bool is_ready() const { return maps_ready_; }

private:
  double pitch_ = 0, yaw_ = 0, roll_ = 0, zoom_ = 1.0;
  double adjusted_focal_ = 0;
  int mode_ = 1;
  bool maps_ready_ = false;

  // Moildev camera params
  double focal_length_ = 0;
  int image_width_ = 0, image_height_ = 0;

  // Remap LUTs
  cv::Mat map_x_, map_y_;
  cv::Mat scaled_map_x_, scaled_map_y_;
  cv::Size stream_size_;
  mutable std::mutex map_mutex_;

  /// Split total zoom → moil_zoom + digital_zoom
  static std::pair<double, double> split_zoom(double total_zoom);

  /// Rescale maps from native to stream resolution
  void rescale_maps(const cv::Size &stream_size);

  /// Apply digital crop (center crop + resize)
  cv::Mat digital_crop(const cv::Mat &frame, double digital_zoom) const;

  /// Load camera params from JSON
  void load_params(const std::string &json_path);
  void load_params_with_name(const std::string &json_path,
                             const std::string &camera_name);

  /// Generate undistortion maps via Moildev algorithm
  void generate_maps(double pitch, double yaw, double roll, double moil_zoom);
};

} // namespace fusion
