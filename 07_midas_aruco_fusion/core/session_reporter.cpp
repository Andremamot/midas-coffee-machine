/*******************************************************************************
 * session_reporter.cpp
 * Port of 07_midas_aruco_fusion/core/session_reporter.py
 ******************************************************************************/
#include "session_reporter.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>

namespace fs = std::filesystem;

/*---------------------------------------------------------------------------*/
/* Construction                                                               */
/*---------------------------------------------------------------------------*/

SessionReporter::SessionReporter(const std::string& report_dir)
    : report_dir_(report_dir)
{}

/*---------------------------------------------------------------------------*/
/* Timestamp helpers                                                          */
/*---------------------------------------------------------------------------*/

std::string SessionReporter::timestamp_folder()
{
    auto now  = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::ostringstream oss;
    oss << std::put_time(std::localtime(&time), "%Y-%m-%d_%H-%M-%S");
    return oss.str();
}

std::string SessionReporter::timestamp_display()
{
    auto now  = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::ostringstream oss;
    oss << std::put_time(std::localtime(&time), "%Y-%m-%d %H:%M:%S");
    return oss.str();
}

/*---------------------------------------------------------------------------*/
/* Stats computation                                                          */
/*---------------------------------------------------------------------------*/

SessionReporter::Stats SessionReporter::compute_stats(
    const std::vector<double>& vals)
{
    Stats s{};
    if (vals.empty()) return s;

    /* mean */
    s.mean = std::accumulate(vals.begin(), vals.end(), 0.0) /
             static_cast<double>(vals.size());

    /* std_dev */
    double sq_sum = 0.0;
    for (double v : vals) sq_sum += (v - s.mean) * (v - s.mean);
    s.std_dev = std::sqrt(sq_sum / static_cast<double>(vals.size()));

    /* sort copy for percentiles */
    std::vector<double> sorted(vals);
    std::sort(sorted.begin(), sorted.end());

    s.min_v  = sorted.front();
    s.max_v  = sorted.back();

    auto at_pct = [&](double pct) -> double {
        double idx = pct * (sorted.size() - 1);
        size_t lo  = static_cast<size_t>(idx);
        size_t hi  = std::min(lo + 1, sorted.size() - 1);
        double frac = idx - lo;
        return sorted[lo] * (1.0 - frac) + sorted[hi] * frac;
    };
    s.median = at_pct(0.50);
    s.p5     = at_pct(0.05);
    s.p95    = at_pct(0.95);

    return s;
}

/*---------------------------------------------------------------------------*/
/* Main generate()                                                            */
/*---------------------------------------------------------------------------*/

void SessionReporter::generate(
    const nlohmann::json& calib_data,
    double marker_size_cm,
    double focal_len_px,
    int    total_frames,
    int    midas_runs,
    const std::vector<double>&             history_z_tray,
    const std::map<int, std::vector<double>>& history_cup_h,
    const std::vector<int>&                history_frames,
    const std::vector<std::string>&        screenshots)
{
    /* Aggregate height history across all cups */
    std::vector<double> all_cups;
    for (auto& [idx, h_list] : history_cup_h) {
        for (double h : h_list) {
            if (h > 0.0) all_cups.push_back(h);
        }
    }
    std::vector<double> valid_trays;
    for (double z : history_z_tray) {
        if (z > 0.0) valid_trays.push_back(z);
    }

    Stats cup_s  = compute_stats(all_cups);
    Stats tray_s = compute_stats(valid_trays);

    /* Print summary to console */
    std::cout << std::string(50, '=') << "\n";
    std::cout << "  ARUCO+MIDAS FUSION SESSION REPORT\n";
    std::cout << std::string(50, '=') << "\n";
    std::cout << "Total Camera Frames  : " << total_frames  << "\n";
    std::cout << "Total MiDaS Inferences: " << midas_runs  << "\n";
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Average Tray Z-dist  : " << tray_s.mean  << " cm\n";
    std::cout << "Average Cup Height   : " << cup_s.mean   << " cm\n";
    std::cout << "Cup Height Variance  : +/- " << cup_s.std_dev << " cm\n";
    std::cout << std::string(50, '=') << "\n\n";

    if (history_frames.empty()) return;

    /* Create timestamped report folder */
    std::string ts        = timestamp_folder();
    std::string ts_disp   = timestamp_display();
    fs::path    rep_dir   = fs::path(report_dir_) / ts;
    fs::create_directories(rep_dir);

    /* ------------------------------------------------------------------ */
    /* Copy screenshots                                                     */
    /* ------------------------------------------------------------------ */
    std::vector<std::string> ss_rel;
    if (!screenshots.empty()) {
        fs::path ss_sub = rep_dir / "screenshots";
        fs::create_directories(ss_sub);
        for (auto& src : screenshots) {
            std::string basename = fs::path(src).filename().string();
            fs::path    dst      = ss_sub / basename;
            try {
                fs::copy_file(src, dst, fs::copy_options::overwrite_existing);
                ss_rel.push_back("screenshots/" + basename);
            } catch (...) {}
        }
    }

    /* ------------------------------------------------------------------ */
    /* Markdown report                                                      */
    /* ------------------------------------------------------------------ */
    std::ofstream md(rep_dir / "report.md");
    if (md.is_open()) {
        md << "# ArUco + MiDaS Fusion Session Report\n\n";
        md << "**Date/Time:** " << ts_disp << "\n\n";

        /* Parameters */
        md << "## 1. Parameters\n\n";
        md << "| Parameter | Value |\n";
        md << "| :--- | :--- |\n";
        md << "| **Physical Marker Size** | " << marker_size_cm << " cm |\n";

        int ctype = calib_data.value("type", 1);
        std::string calib_str = "1-Point K-Factor";
        if (ctype == 1)
            calib_str = "1-Point K-Factor (K=" +
                        std::to_string(calib_data.value("K", 0.0)) + ")";
        else if (ctype == 2)
            calib_str = "2-Point Linear";
        else if (ctype == 3)
            calib_str = "3-Point Z-Grid";
        else if (ctype == 4)
            calib_str = "4-BBox Area";
        else if (ctype == 5)
            calib_str = "5-Geometric Projection";
        else if (ctype == 6)
            calib_str = "6-Bilateral MiDaS";
        else if (ctype == 7)
            calib_str = "7-Analytic Geometry";

        md << "| **Calibration Model** | " << calib_str << " |\n";
        md << "| **Camera Focal Length** | " << std::fixed
           << std::setprecision(1) << focal_len_px << " px |\n\n";

        /* Statistics */
        md << "## 2. Global Stability Summary\n\n";
        md << "| Metric | Value | Description |\n";
        md << "| :--- | :--- | :--- |\n";
        md << "| **Average Cup Height** | **" << std::fixed
           << std::setprecision(2) << cup_s.mean
           << " cm** | Mean of all valid predictions. |\n";
        if (!all_cups.empty()) {
            md << "| **Median Height (P50)** | **" << cup_s.median
               << " cm** | Most representative single value. |\n";
            md << "| **Precision Error (P95-P5)** | **"
               << (cup_s.p95 - cup_s.p5)
               << " cm** | 90% of readings fall within this range. |\n";
        }
        md << "| **Standard Deviation** | " << cup_s.std_dev
           << " cm | Consistency / jitter of the AI model. |\n";
        md << "| **Tray Anchor Depth (Z)** | " << tray_s.mean
           << " cm | Average physical depth of the tray. |\n";
        md << "| **Min / Max Height** | " << cup_s.min_v << " / "
           << cup_s.max_v << " cm | Extremes recorded. |\n";
        md << "| **Total Frames / Inferences** | " << total_frames
           << " / " << midas_runs << " | Pipeline tracking efficiency. |\n\n";

        /* Screenshots */
        if (!ss_rel.empty()) {
            md << "## 3. Screenshots\n\n";
            for (auto& ss : ss_rel) {
                md << "- ![](" << ss << ")\n";
            }
        }

        md.close();
    }

    /* ------------------------------------------------------------------ */
    /* JSON data dump                                                        */
    /* ------------------------------------------------------------------ */
    nlohmann::json json_data = {
        {"session_timestamp", ts},
        {"parameters", {
            {"marker_size_cm",    marker_size_cm},
            {"calibration_model", calib_data},
            {"focal_length_px",   focal_len_px}
        }},
        {"summary", {
            {"total_frames",           total_frames},
            {"midas_inferences",       midas_runs},
            {"avg_cup_height_cm",      cup_s.mean},
            {"min_cup_height_cm",      cup_s.min_v},
            {"max_cup_height_cm",      cup_s.max_v},
            {"std_dev_cup_height_cm",  cup_s.std_dev},
            {"avg_z_tray_cm",          tray_s.mean}
        }},
        {"frame_metrics_history", {
            {"frame_indices",      history_frames},
            {"z_tray_history",     history_z_tray}
        }},
        {"screenshots", ss_rel}
    };

    /* Serialize history_cup_h (map<int,vector<double>>) */
    nlohmann::json cup_history_json = nlohmann::json::object();
    for (auto& [idx, h_list] : history_cup_h) {
        cup_history_json[std::to_string(idx)] = h_list;
    }
    json_data["frame_metrics_history"]["cup_height_history"] = cup_history_json;

    std::ofstream jf(rep_dir / "session_data.json");
    if (jf.is_open()) {
        jf << json_data.dump(2);
        jf.close();
        auto size_bytes = fs::file_size(rep_dir / "session_data.json");
        std::cout << "Report saved to: " << rep_dir.string()
                  << " (session_data.json "
                  << std::fixed << std::setprecision(1)
                  << size_bytes / 1024.0 << " KB)\n";
    }
}
