/*******************************************************************************
 * session_reporter.hpp
 * Port of 07_midas_aruco_fusion/core/session_reporter.py
 * Writes a markdown + JSON session report after the live pipeline ends.
 ******************************************************************************/
#pragma once

#include <nlohmann/json.hpp>
#include <string>
#include <vector>
#include <map>

/**
 * @brief Generates a plain-text / markdown session report.
 *
 * Note: The Python version used matplotlib for charts.
 * The C++ port writes the same markdown + JSON data files but skips the
 * plot image (no matplotlib on embedded target). Screenshots are still copied.
 */
class SessionReporter {
public:
    /**
     * @param report_dir  Base directory for all report output (e.g. "results/report")
     */
    explicit SessionReporter(const std::string& report_dir = "results/report");

    /**
     * Generate and save the session report.
     *
     * @param calib_data        Calibration JSON used during the session
     * @param marker_size_cm    Physical ArUco marker side length (cm)
     * @param focal_len_px      Camera focal length in pixels
     * @param total_frames      Total camera frames captured
     * @param midas_runs        Number of MiDaS inference calls
     * @param history_z_tray    Z-tray values per inference step
     * @param history_cup_h     Cup height per cup index per inference step
     * @param history_frames    Frame indices matching history_z_tray
     * @param screenshots       Paths to captured screenshots to include
     */
    void generate(const nlohmann::json& calib_data,
                  double marker_size_cm,
                  double focal_len_px,
                  int    total_frames,
                  int    midas_runs,
                  const std::vector<double>&             history_z_tray,
                  const std::map<int, std::vector<double>>& history_cup_h,
                  const std::vector<int>&                history_frames,
                  const std::vector<std::string>&        screenshots);

private:
    std::string report_dir_;

    static std::string timestamp_folder();   ///< "YYYY-MM-DD_HH-MM-SS"
    static std::string timestamp_display();  ///< "YYYY-MM-DD HH:MM:SS"

    /** Compute basic stats; returns {mean, std, min, max, median, p5, p95} */
    struct Stats {
        double mean, std_dev, min_v, max_v, median, p5, p95;
    };
    static Stats compute_stats(const std::vector<double>& vals);
};
