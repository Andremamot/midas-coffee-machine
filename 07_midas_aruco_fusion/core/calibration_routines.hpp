/*******************************************************************************
 * calibration_routines.hpp
 * Port of 07_midas_aruco_fusion/core/calibration_routines.py
 * Calibration modes 1-7 matching the Python implementation exactly.
 ******************************************************************************/
#pragma once

#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

/* Forward declare detector types — resolved at compile time */
#include <detections/ai.h>
#include <detections/box.h>
#include "calibration_storage.hpp"

/* ArUco detector lives in 06_aruco_marker */
#include "aruco_detector.hpp"

/**
 * @brief Calibration routines namespace.
 *
 * Each function corresponds to one calibration mode in the Python version.
 * All functions share the same signature convention:
 *   - cap / ai  : references to live camera and AI singleton
 *   - aruco     : ArUco detector (for distance measurement)
 *   - headless  : suppress cv::imshow calls (for embedded targets)
 *   - storage   : CalibrationStorage instance to save results
 *   - Returns   : filled nlohmann::json with calib_data on success, empty json on failure/abort
 *
 * The midas and cup detectors are accessed via AI::get_instance().
 */
namespace CalibRoutines {

/**
 * Helper: get ArUco distance + ROI from a detection result.
 * Returns true if a valid marker was found.
 */
bool get_aruco_roi(const std::vector<ArucoResult>& results,
                   ArucoDetector& aruco,
                   double& z_cm,
                   cv::Rect& roi_out);

/**
 * Helper: extract cup bounding box (first detected cup_rim detection).
 * Returns true if at least one detection was found.
 */
bool get_cup_bbox(const std::vector<Detection>& detections,
                  cv::Rect& bbox_out);

/* ──────────────────────────────────────────────────────────────────────── */
/* Mode 1 & 2: 1-Point and 2-Point K-Factor / Linear calibration           */
/* ──────────────────────────────────────────────────────────────────────── */
nlohmann::json run_calib_1p_2p(cv::VideoCapture& cap,
                                ArucoDetector& aruco,
                                CalibrationStorage& storage,
                                bool headless,
                                double true_height,
                                double true_height_2,
                                int calibrate_mode);

/* Mode 3: Z-Grid polynomial */
nlohmann::json run_calib_zgrid(cv::VideoCapture& cap,
                                ArucoDetector& aruco,
                                CalibrationStorage& storage,
                                bool headless,
                                double true_height,
                                int n_positions);

/* Mode 4: BBox-area compensated */
nlohmann::json run_calib_bbox(cv::VideoCapture& cap,
                               ArucoDetector& aruco,
                               CalibrationStorage& storage,
                               bool headless,
                               double true_height);

/* Mode 5: Geometric Z-Grid */
nlohmann::json run_calib_geom(cv::VideoCapture& cap,
                               ArucoDetector& aruco,
                               CalibrationStorage& storage,
                               bool headless,
                               double true_height,
                               int n_positions);

/* Mode 6: Bilateral Z-Grid */
nlohmann::json run_calib_bilateral(cv::VideoCapture& cap,
                                    ArucoDetector& aruco,
                                    CalibrationStorage& storage,
                                    bool headless,
                                    double true_height,
                                    double true_height_2,
                                    int n_positions);

/* Mode 7: Universal Analytic Geometry */
nlohmann::json run_calib_analytic(cv::VideoCapture& cap,
                                   ArucoDetector& aruco,
                                   CalibrationStorage& storage,
                                   bool headless,
                                   double true_height,
                                   double true_height_2);

} // namespace CalibRoutines
