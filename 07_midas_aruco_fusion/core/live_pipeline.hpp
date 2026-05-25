/*******************************************************************************
 * live_pipeline.hpp
 * Port of 07_midas_aruco_fusion/core/live_pipeline.py
 ******************************************************************************/
#pragma once

#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <atomic>



#include "aruco_detector.hpp"
#include "moil_undistorter.hpp"
#include "gui_fusion.hpp"
#include <camera/camera.h>

/**
 * @brief Run the live cup-height estimation pipeline.
 *
 * Equivalent to run_live_pipeline() in the Python version.
 * Uses AI::get_instance() for all AI inference (cup + midas).
 *
 * @param cam             Camera module instance pointer
 * @param aruco           Initialized ArucoDetector
 * @param headless        If true, suppress all cv::imshow calls
 * @param calib_data      Calibration JSON loaded from calibration.json
 * @param marker_size_cm  Physical ArUco marker size (cm)
 * @param active_poly_Kgeom  Active poly_Kgeom coefficients (type 5 only)
 * @param active_cup_str  Human-readable active cup label (type 5 only)
 * @param screenshot_dir  Directory to save screenshots
 * @param video_dir       Directory to save recorded videos
 * @param moil            MoilUndistorter pointer (nullptr = fisheye disabled)
 * @param gui             GuiFusion pointer
 * @param no_anypoint     If true, keep fisheye mode but skip anypoint remap
 */
void run_live_pipeline(Camera*             cam,
                       ArucoDetector&      aruco,
                       bool                headless,
                       const nlohmann::json& calib_data,
                       double              marker_size_cm,
                       const std::vector<double>& active_poly_Kgeom,
                       const std::string&  active_cup_str,
                       const std::string&  screenshot_dir,
                       const std::string&  video_dir,
                       MoilUndistorter*    moil          = nullptr,
                       GuiFusion*          gui           = nullptr,
                       bool                no_anypoint   = false,
                       int                 output_w      = 0,
                       int                 output_h      = 0);

