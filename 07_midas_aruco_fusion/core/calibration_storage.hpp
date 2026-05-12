/*******************************************************************************
 * calibration_storage.hpp
 * Port of 07_midas_aruco_fusion/core/calibration_storage.py
 * Load / save calibration.json using nlohmann::json
 ******************************************************************************/
#pragma once

#include <nlohmann/json.hpp>
#include <string>
#include <vector>

/**
 * @brief Thin wrapper around a calibration JSON file.
 *
 * Mirrors the Python calibration_storage module.
 * The JSON schema is identical so files are cross-compatible between
 * the Python and C++ versions of the application.
 */
class CalibrationStorage {
public:
    explicit CalibrationStorage(const std::string& calib_path = "calibration.json");

    /** Set the calibration file path at runtime. */
    void set_path(const std::string& path);
    const std::string& get_path() const { return calib_path_; }

    /* ------------------------------------------------------------------ */
    /* Load                                                                 */
    /* ------------------------------------------------------------------ */

    /**
     * Load calibration data from disk.
     * @return Parsed JSON object, or empty json{} on failure.
     */
    nlohmann::json load() const;

    /* ------------------------------------------------------------------ */
    /* Save — one method per calibration type                               */
    /* ------------------------------------------------------------------ */

    void save_1p(double K, double z_tray_ref,
                 double ratio_ref, double true_height);

    void save_2p(double m, double c,
                 double R1, double Z1, double H1,
                 double R2, double Z2, double H2);

    void save_3p(const std::vector<double>& poly_K,
                 const std::vector<double>& z_grid,
                 double true_height);

    void save_4p(double m_ref, double c_ref, double ref_area,
                 double z_low, double z_high, double true_height);

    /** Type 5 accumulates profiles keyed by cup height string. */
    void save_5p(const std::vector<double>& poly_Kgeom,
                 const std::vector<double>& z_grid,
                 double true_height);

    void save_6p(const std::vector<double>& poly_m,
                 const std::vector<double>& poly_c,
                 const std::vector<double>& z_grid,
                 double h1, double h2);

    void save_7(double A, double B, double h1, double h2);

private:
    std::string calib_path_;

    /** Write json object to disk with indent=2. */
    void write(const nlohmann::json& data) const;

    /** ISO-8601 timestamp string. */
    static std::string timestamp();
};
