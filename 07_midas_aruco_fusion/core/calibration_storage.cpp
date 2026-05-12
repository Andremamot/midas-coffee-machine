/*******************************************************************************
 * calibration_storage.cpp
 * Port of 07_midas_aruco_fusion/core/calibration_storage.py
 ******************************************************************************/
#include "calibration_storage.hpp"

#include <chrono>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

/*---------------------------------------------------------------------------*/
/* Construction / path management                                             */
/*---------------------------------------------------------------------------*/

CalibrationStorage::CalibrationStorage(const std::string& calib_path)
    : calib_path_(calib_path)
{}

void CalibrationStorage::set_path(const std::string& path)
{
    calib_path_ = path;
}

/*---------------------------------------------------------------------------*/
/* Helpers                                                                    */
/*---------------------------------------------------------------------------*/

std::string CalibrationStorage::timestamp()
{
    auto now  = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::ostringstream oss;
    oss << std::put_time(std::localtime(&time), "%Y-%m-%dT%H:%M:%S");
    return oss.str();
}

void CalibrationStorage::write(const nlohmann::json& data) const
{
    std::ofstream f(calib_path_);
    if (!f.is_open()) {
        std::cerr << "[CALIB] ERROR: Cannot open " << calib_path_
                  << " for writing\n";
        return;
    }
    f << data.dump(2);
    f.close();
}

/*---------------------------------------------------------------------------*/
/* Load                                                                       */
/*---------------------------------------------------------------------------*/

nlohmann::json CalibrationStorage::load() const
{
    std::ifstream f(calib_path_);
    if (!f.is_open()) {
        std::cerr << "[CALIB] File not found: " << calib_path_ << "\n";
        return nlohmann::json{};
    }
    try {
        nlohmann::json data;
        f >> data;

        int ctype = data.value("type", 1);
        const char* labels[] = {
            "1-Point K-Factor",
            "2-Point Linear",
            "3-Point Z-Grid",
            "4-BBox Area",
            "5-Geometric Projection",
            "6-Bilateral MiDaS",
            "7-Analytic Geometry"
        };
        if (ctype >= 1 && ctype <= 7) {
            std::cout << "[CALIB] Loaded calibration model: "
                      << labels[ctype - 1]
                      << " (from " << calib_path_ << ")\n";
        }
        return data;
    } catch (const std::exception& e) {
        std::cerr << "[CALIB] Failed to parse calibration: " << e.what() << "\n";
        return nlohmann::json{};
    }
}

/*---------------------------------------------------------------------------*/
/* Save helpers                                                               */
/*---------------------------------------------------------------------------*/

void CalibrationStorage::save_1p(double K, double z_tray_ref,
                                 double ratio_ref, double true_height)
{
    nlohmann::json data = {
        {"type",           1},
        {"K",              K},
        {"z_tray_ref_cm",  z_tray_ref},
        {"ratio_ref",      ratio_ref},
        {"true_height_cm", true_height},
        {"calibrated_at",  timestamp()}
    };
    write(data);
    std::cout << "[CALIB] Saved 1-Point -> " << calib_path_ << "\n";
}

void CalibrationStorage::save_2p(double m, double c,
                                 double R1, double Z1, double H1,
                                 double R2, double Z2, double H2)
{
    nlohmann::json data = {
        {"type",  2},
        {"m",     m},
        {"c",     c},
        {"data1", {{"R", R1}, {"Z", Z1}, {"H", H1}}},
        {"data2", {{"R", R2}, {"Z", Z2}, {"H", H2}}},
        {"calibrated_at", timestamp()}
    };
    write(data);
    std::cout << "[CALIB] Saved 2-Point -> " << calib_path_ << "\n";
}

void CalibrationStorage::save_3p(const std::vector<double>& poly_K,
                                 const std::vector<double>& z_grid,
                                 double true_height)
{
    nlohmann::json data = {
        {"type",           3},
        {"poly_K",         poly_K},
        {"z_grid_points",  z_grid},
        {"true_height_cm", true_height},
        {"calibrated_at",  timestamp()}
    };
    write(data);
    std::cout << "[CALIB] Saved Z-Grid (type 3) -> " << calib_path_ << "\n";
}

void CalibrationStorage::save_4p(double m_ref, double c_ref, double ref_area,
                                 double z_low, double z_high, double true_height)
{
    nlohmann::json data = {
        {"type",               4},
        {"m_ref",              m_ref},
        {"c_ref",              c_ref},
        {"ref_bbox_area_px",   ref_area},
        {"z_ref",              z_low},
        {"z_high",             z_high},
        {"true_height_cm",     true_height},
        {"calibrated_at",      timestamp()}
    };
    write(data);
    std::cout << "[CALIB] Saved BBox Area (type 4) -> " << calib_path_ << "\n";
}

void CalibrationStorage::save_5p(const std::vector<double>& poly_Kgeom,
                                 const std::vector<double>& z_grid,
                                 double true_height)
{
    /* Type 5 accumulates multiple cup-height profiles */
    nlohmann::json data = {{"type", 5}, {"profiles", nlohmann::json::object()}};

    /* Try to merge with existing file */
    std::ifstream f(calib_path_);
    if (f.is_open()) {
        try {
            nlohmann::json old;
            f >> old;
            if (old.value("type", 0) == 5) {
                if (old.contains("profiles")) {
                    data["profiles"] = old["profiles"];
                } else if (old.contains("poly_Kgeom")) {
                    /* Upgrade old flat format to profiles map */
                    std::string old_h =
                        std::to_string(old.value("true_height_cm", 7.6));
                    data["profiles"][old_h] = {
                        {"poly_Kgeom",    old["poly_Kgeom"]},
                        {"z_grid_points", old.value("z_grid_points",
                                          std::vector<double>{})},
                        {"calibrated_at", old.value("calibrated_at", "")}
                    };
                }
            }
        } catch (...) {}
    }

    /* Insert/overwrite this cup's profile */
    std::ostringstream key_ss;
    key_ss << true_height;
    data["profiles"][key_ss.str()] = {
        {"poly_Kgeom",    poly_Kgeom},
        {"z_grid_points", z_grid},
        {"calibrated_at", timestamp()}
    };

    write(data);
    std::cout << "[CALIB] Saved Geometric Z-Grid (type 5) for Menu ["
              << true_height << "cm] -> " << calib_path_ << "\n";
}

void CalibrationStorage::save_6p(const std::vector<double>& poly_m,
                                 const std::vector<double>& poly_c,
                                 const std::vector<double>& z_grid,
                                 double h1, double h2)
{
    nlohmann::json data = {
        {"type",              6},
        {"poly_m",            poly_m},
        {"poly_c",            poly_c},
        {"z_grid_points",     z_grid},
        {"true_height_cm_1",  h1},
        {"true_height_cm_2",  h2},
        {"calibrated_at",     timestamp()}
    };
    write(data);
    std::cout << "[CALIB] Saved Bilateral Z-Grid (type 6) -> "
              << calib_path_ << "\n";
}

void CalibrationStorage::save_7(double A, double B, double h1, double h2)
{
    nlohmann::json data = {
        {"type",              7},
        {"A",                 A},
        {"B",                 B},
        {"true_height_cm_1",  h1},
        {"true_height_cm_2",  h2},
        {"calibrated_at",     timestamp()}
    };
    write(data);
    std::cout << "[CALIB] Saved Universal Analytic Geometry (type 7) -> "
              << calib_path_ << "\n";
}
