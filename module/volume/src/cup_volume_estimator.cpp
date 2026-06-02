/**
 * @file cup_volume_estimator.cpp
 * @brief Implementasi CupVolumeEstimator — pipeline volume dari raw fisheye
 * frame.
 *
 * Pipeline:
 *   raw_fisheye → [MoildevApplicator] → [measure_rim_width_px] → [calc_diameter] →
 * [calc_volume]
 */

#include <volume/cup_volume_estimator.h>
#include <volume/volume_math.h>

// Internal header (dari src/fisheye/ — tidak terekspos ke luar modul)
#include "fisheye/moil_undistorter.h"

#include <iostream>
#include <stdexcept>

namespace fusion {

// ─────────────────────────────────────────────────────────────────────────────
// Constructor / Destructor
// ─────────────────────────────────────────────────────────────────────────────

CupVolumeEstimator::CupVolumeEstimator(const std::string &camera_params_json,
                                       const std::string &camera_name,
                                       int frame_width, int frame_height)
    : frame_w_(frame_width), frame_h_(frame_height) {
  std::cout << "[mod_volume] Init MoildevApplicator\n"
            << "            JSON   : " << camera_params_json << "\n"
            << "            Profil : " << camera_name << "\n"
            << "            Mode   : " << volume_config::MOIL_MODE << "\n"
            << "            Pitch  : " << volume_config::MOIL_PITCH_DEG << "°\n"
            << "            Zoom   : " << volume_config::MOIL_ZOOM << "x\n"
            << "            Stream : " << frame_width << "×" << frame_height
            << "\n";
  try {
    moil_ = std::make_unique<MoildevApplicator>(camera_params_json, camera_name,
                                          volume_config::MOIL_MODE);
    moil_->update_maps(volume_config::MOIL_PITCH_DEG,
                       volume_config::MOIL_YAW_DEG,
                       volume_config::MOIL_ROLL_DEG, volume_config::MOIL_ZOOM);
    if (!moil_->is_ready()) {
      throw std::runtime_error(
          "[mod_volume] MoildevApplicator gagal init — cek JSON & nama profil.");
    }
    ready_ = true;
    std::cout << "[mod_volume] Siap. adjusted_focal = "
              << moil_->adjusted_focal_length() << " px\n";
  } catch (const std::exception &e) {
    std::cerr << "[mod_volume] ERROR: " << e.what() << "\n";
    ready_ = false;
  }
}

CupVolumeEstimator::~CupVolumeEstimator() = default;

// ─────────────────────────────────────────────────────────────────────────────
// Core estimate()
// ─────────────────────────────────────────────────────────────────────────────

VolumeResult CupVolumeEstimator::estimate(const VolumeInput &input) {
  VolumeResult result;

  if (!ready_ || input.frame.empty())
    return result;
  if (input.h_cup_cm <= 0.0 || input.z_tray_cm <= 0.0)
    return result;
  if (input.focal_px <= 0.0)
    return result;
  if (input.bbox.x2 <= input.bbox.x1 || input.bbox.y2 <= input.bbox.y1)
    return result;

  // Stage 1: Undistort — INTER_LINEAR (hardcoded, jangan ganti ke CUBIC)
  cv::Mat undistorted = moil_->undistort(input.frame);
  if (undistorted.empty()) {
    std::cerr << "[mod_volume] Undistort kosong!\n";
    return result;
  }

  // Stage 2: Rim detection (Otsu strip)
  double rim_w_px = measure_rim_width_px(undistorted, input.bbox);
  result.rim_w_px = rim_w_px;

  // Stage 3: Diameter (pinhole model)
  double z_rim_cm = std::max(0.0, input.z_tray_cm - input.h_cup_cm);
  result.z_rim_cm = z_rim_cm;
  double diameter_cm = calc_diameter(rim_w_px, z_rim_cm, input.focal_px);
  result.diameter_cm = diameter_cm;

  // Stage 4: Volume (silinder)
  double volume_ml = calc_volume(input.h_cup_cm, diameter_cm);
  result.volume_ml = volume_ml;

  result.valid = (rim_w_px > 0.0 && diameter_cm > 0.0 && volume_ml > 0.0);
  return result;
}

// ─────────────────────────────────────────────────────────────────────────────
// Utilities
// ─────────────────────────────────────────────────────────────────────────────

double CupVolumeEstimator::adjusted_focal_px() const {
  if (!moil_ || !ready_)
    return 0.0;
  auto K = moil_->build_aruco_camera_matrix(frame_w_, frame_h_);
  return K(0, 0); // fx
}

double CupVolumeEstimator::from_height_diameter(double h_cup_cm,
                                                double diameter_cm) {
  return calc_volume(h_cup_cm, diameter_cm);
}

void CupVolumeEstimator::apply_exposure(cv::VideoCapture &cap) {
  if (!cap.isOpened()) {
    std::cerr << "[mod_volume] apply_exposure: cap tidak terbuka!\n";
    return;
  }
  cap.set(cv::CAP_PROP_AUTO_EXPOSURE, 0.25); // V4L2: manual mode
  cap.set(cv::CAP_PROP_EXPOSURE,
          static_cast<double>(volume_config::V4L2_EXPOSURE));
  std::cout << "[mod_volume] Exposure: V4L2=" << volume_config::V4L2_EXPOSURE
            << " (smart_exposure=5.0, auto=OFF)\n";
}

} // namespace fusion
