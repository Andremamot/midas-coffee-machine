/*******************************************************************************
 * calibration_routines.cpp
 * Port of 07_midas_aruco_fusion/core/calibration_routines.py
 * All 7 calibration modes — exact logic match to Python.
 ******************************************************************************/
#include "calibration_routines.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>
#include <thread>

/* ── Local time helpers ─────────────────────────────────────────────────── */
static double now_sec()
{
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

/* ── UI overlay constants ─────────────────────────────────────────────────*/
static constexpr char WIN[] = "ArUco + MiDaS | Cup Height Estimator";

/* ── Median helper ────────────────────────────────────────────────────────*/
static double vec_median(std::vector<double> v)
{
    if (v.empty()) return 0.0;
    std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
    return v[v.size() / 2];
}

static double vec_mean(const std::vector<double>& v)
{
    if (v.empty()) return 0.0;
    return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
}

/*---------------------------------------------------------------------------*/
/* Shared helpers                                                             */
/*---------------------------------------------------------------------------*/

bool CalibRoutines::get_aruco_roi(const std::vector<ArucoResult>& results,
                                  ArucoDetector& aruco,
                                  double& z_cm,
                                  cv::Rect& roi_out)
{
    if (results.empty()) return false;
    BestDistanceResult best = aruco.get_best_distance(results);
    if (best.distance_cm <= 0.0f) return false;

    z_cm = best.distance_cm;

    /* Build ROI from first result corners */
    const auto& corners = results[0].corners;
    if (corners.empty()) return false;

    float x_min = corners[0].x, y_min = corners[0].y;
    float x_max = corners[0].x, y_max = corners[0].y;
    for (auto& pt : corners) {
        x_min = std::min(x_min, pt.x);
        y_min = std::min(y_min, pt.y);
        x_max = std::max(x_max, pt.x);
        y_max = std::max(y_max, pt.y);
    }

    roi_out = cv::Rect(
        static_cast<int>(x_min) + 2,
        static_cast<int>(y_min) + 2,
        static_cast<int>(x_max - x_min) - 4,
        static_cast<int>(y_max - y_min) - 4
    );
    return roi_out.width > 0 && roi_out.height > 0;
}

bool CalibRoutines::get_cup_bbox(const std::vector<Detection>& detections,
                                 cv::Rect& bbox_out)
{
    /* Find the first cup_rim detection (class_id == 0) */
    for (const auto& d : detections) {
        if (d.score > 0.0f) {
            bbox_out = cv::Rect(
                static_cast<int>(d.box.x),
                static_cast<int>(d.box.y),
                static_cast<int>(d.box.w),
                static_cast<int>(d.box.h)
            );
            return true;
        }
    }
    return false;
}

/*---------------------------------------------------------------------------*/
/* Draw helpers                                                               */
/*---------------------------------------------------------------------------*/

static void draw_status_box(cv::Mat& frame,
                            const std::string& line1,
                            const std::string& line2,
                            const std::string& status_line,
                            cv::Scalar bg_color = {20, 20, 40})
{
    cv::rectangle(frame, cv::Point(8, 8), cv::Point(535, 100), bg_color, -1);
    cv::rectangle(frame, cv::Point(8, 8), cv::Point(535, 100),
                  cv::Scalar(0, 200, 255), 1);
    cv::putText(frame, line1, cv::Point(18, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 200, 255), 1);
    cv::putText(frame, line2, cv::Point(18, 58),
                cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(150, 200, 255), 1);
    cv::putText(frame, status_line, cv::Point(18, 92),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(200, 200, 200), 1);
}

/*===========================================================================*/
/* MODE 1 & 2: 1-Point / 2-Point calibration                                 */
/*===========================================================================*/

nlohmann::json CalibRoutines::run_calib_1p_2p(cv::VideoCapture& cap,
                                               ArucoDetector& aruco,
                                               CalibrationStorage& storage,
                                               bool headless,
                                               double true_height,
                                               double true_height_2,
                                               int calibrate_mode)
{
    constexpr double WARMUP_SEC  = 5.0;
    constexpr double SAMPLE_TIMEOUT = 30.0;

    AI* ai = AI::get_instance();

    std::string phase = "warmup_1";
    double phase_start = now_sec();

    std::vector<double> ratios_1, z_trays_1;
    std::vector<double> ratios_2, z_trays_2;

    double last_midas_t  = 0.0;
    std::vector<Detection> last_detections;
    std::vector<ArucoResult> last_aruco;

    std::cout << std::string(55, '-') << "\n";
    std::cout << "  CALIBRATION " << calibrate_mode << "-POINT\n";
    std::cout << std::string(55, '-') << "\n";

    if (!headless) cv::namedWindow(WIN, cv::WINDOW_NORMAL);

    while (phase != "done") {
        cv::Mat frame;
        if (!cap.read(frame) || frame.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            continue;
        }

        double t       = now_sec();
        double elapsed = t - phase_start;

        /* ArUco detection */
        last_aruco = aruco.detect(frame);
        double z_calib = 0.0;
        cv::Rect aruco_roi;
        bool aruco_ok = get_aruco_roi(last_aruco, aruco, z_calib, aruco_roi);

        /* Cup + MiDaS at ~5fps when conditions are met */
        bool in_sample = (phase == "sampling_1" || phase == "sampling_2");
        if ((t - last_midas_t) > 0.2 && z_calib > 0 && aruco_ok && in_sample)
        {
            auto [dets] = ai->cup_detector->detect(frame);
            last_detections = dets;
            cv::Rect cup_bbox;
            if (get_cup_bbox(dets, cup_bbox)) {
                cv::Mat depth_map = ai->midas_estimator->inference(frame);
                last_midas_t = t;

                float m_rim  = ai->midas_estimator->get_rim_depth(depth_map, cup_bbox);
                float m_tray = ai->midas_estimator->get_tray_depth(depth_map, aruco_roi);

                if (m_rim > 0 && m_tray > 0) {
                    double ratio = m_rim / m_tray;
                    if (phase == "sampling_1") {
                        ratios_1.push_back(ratio);
                        z_trays_1.push_back(z_calib);
                    } else {
                        ratios_2.push_back(ratio);
                        z_trays_2.push_back(z_calib);
                    }
                }
            }
            last_midas_t = t;
        }

        /* Phase transitions */
        if (phase == "warmup_1" && elapsed >= WARMUP_SEC) {
            phase = "sampling_1"; phase_start = now_sec();
        } else if (phase == "sampling_1") {
            if ((int)ratios_1.size() >= 5) {
                phase = (calibrate_mode == 1) ? "done" : "swap_wait";
            } else if (elapsed > SAMPLE_TIMEOUT) {
                std::cout << "[CALIB] Timeout: cup 1 detection failed.\n";
                phase = "done";
            }
        } else if (phase == "warmup_2" && elapsed >= WARMUP_SEC) {
            phase = "sampling_2"; phase_start = now_sec();
        } else if (phase == "sampling_2") {
            if ((int)ratios_2.size() >= 5)
                phase = "done";
            else if (elapsed > SAMPLE_TIMEOUT) {
                std::cout << "[CALIB] Timeout: cup 2 detection failed.\n";
                phase = "done";
            }
        }

        /* UI overlay */
        cv::Mat disp = frame.clone();
        if (!last_aruco.empty()) disp = aruco.annotate_frame(disp, last_aruco);
        for (auto& d : last_detections) {
            if (d.score > 0.0f) {
                cv::rectangle(disp,
                    cv::Rect((int)d.box.x, (int)d.box.y,
                             (int)d.box.w, (int)d.box.h),
                    cv::Scalar(0, 255, 80), 2);
            }
        }

        std::string s1 = std::string("ArUco: ") + (z_calib > 0 ? "OK" : "NOT FOUND") +
                         "  YOLO: " + (last_detections.empty() ? "NOT FOUND" : "OK");

        if (phase == "warmup_1" || phase == "warmup_2") {
            int idx = (phase == "warmup_1") ? 1 : 2;
            double H_t = (idx == 1) ? true_height : true_height_2;
            int pct = std::min(100, (int)((elapsed / WARMUP_SEC) * 100));
            draw_status_box(disp,
                "WARMING UP CUP " + std::to_string(idx) + " (H=" +
                std::to_string(H_t).substr(0, 4) + "cm)",
                "Keep cup still. Prog: " + std::to_string(pct) + "%",
                s1);
        } else if (phase == "sampling_1" || phase == "sampling_2") {
            int idx  = (phase == "sampling_1") ? 1 : 2;
            int n    = (idx == 1) ? (int)ratios_1.size()
                                  : (int)ratios_2.size();
            draw_status_box(disp,
                "SAMPLING DATA CUP " + std::to_string(idx) +
                " (Count: " + std::to_string(n) + "/5)",
                "Ensure camera and cup are visible...", s1,
                cv::Scalar(20, 20, 40));
        } else if (phase == "swap_wait") {
            draw_status_box(disp,
                "SWAP THE CUP NOW",
                "Place a cup with height " +
                    std::to_string(true_height_2).substr(0, 4) +
                    " cm on the tray.",
                "Then press SPACE to continue.",
                cv::Scalar(50, 20, 200));
        }

        if (!headless) {
            cv::imshow(WIN, disp);
            int key = cv::waitKey(1) & 0xFF;
            if (key == 27) return nlohmann::json{};  // ESC — abort
            if (phase == "swap_wait" && key == ' ') {
                phase = "warmup_2"; phase_start = now_sec();
            }
        }
    }

    /* Compute calibration */
    if ((int)ratios_1.size() < 3) {
        std::cerr << "[CALIB] Error: insufficient data for cup 1.\n";
        return nlohmann::json{};
    }

    double R1 = vec_mean(ratios_1);
    double Z1 = vec_mean(z_trays_1);

    if (calibrate_mode == 1) {
        double K = R1 * (1.0 - true_height / Z1);
        storage.save_1p(K, Z1, R1, true_height);
        std::cout << "[CALIB] 1-Point done. K=" << K << "\n";
        return nlohmann::json{{"type", 1}, {"K", K}};
    }

    /* 2-Point */
    if ((int)ratios_2.size() < 3) {
        std::cerr << "[CALIB] Error: insufficient data for cup 2.\n";
        return nlohmann::json{};
    }
    double R2 = vec_mean(ratios_2);
    double Z2 = vec_mean(z_trays_2);

    double Y1 = true_height   / Z1;
    double Y2 = true_height_2 / Z2;

    if (std::fabs(R2 - R1) < 0.05) {
        std::cerr << "[CALIB] Error: both cups have nearly identical MiDaS ratio.\n";
        return nlohmann::json{};
    }
    double m = (Y2 - Y1) / (R2 - R1);
    double c = Y1 - m * R1;
    storage.save_2p(m, c, R1, Z1, true_height, R2, Z2, true_height_2);
    std::cout << "[CALIB] 2-Point done. m=" << m << " c=" << c << "\n";
    return nlohmann::json{{"type", 2}, {"m", m}, {"c", c}};
}

/*===========================================================================*/
/* MODE 3: Z-Grid polynomial                                                  */
/*===========================================================================*/

nlohmann::json CalibRoutines::run_calib_zgrid(cv::VideoCapture& cap,
                                               ArucoDetector& aruco,
                                               CalibrationStorage& storage,
                                               bool headless,
                                               double true_height,
                                               int n_positions)
{
    AI* ai = AI::get_instance();

    std::cout << "  Z-GRID CALIBRATION (" << n_positions << " positions)\n";
    std::cout << "  Cup reference height: " << true_height << " cm\n";

    constexpr double WARMUP_SEC = 4.0;
    constexpr double TIMEOUT    = 30.0;

    struct GridPt { double R, Z; };
    std::vector<GridPt> grid_data;

    int    pos_idx     = 0;
    std::string phase  = "warmup";
    double phase_start = now_sec();
    double last_midas  = 0.0;

    std::vector<double> pos_ratios, pos_z;
    std::vector<Detection>   last_dets;
    std::vector<ArucoResult> last_aruco;

    while (pos_idx < n_positions || phase == "warmup") {
        cv::Mat frame;
        if (!cap.read(frame) || frame.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            continue;
        }

        double t       = now_sec();
        double elapsed = t - phase_start;

        last_aruco = aruco.detect(frame);
        double z_calib = 0.0; cv::Rect aruco_roi;
        bool aruco_ok = get_aruco_roi(last_aruco, aruco, z_calib, aruco_roi);

        if ((t - last_midas) > 0.2 && z_calib > 0 && aruco_ok
            && phase == "sampling")
        {
            auto [dets] = ai->cup_detector->detect(frame);
            last_dets = dets;
            cv::Rect cup_bbox;
            if (get_cup_bbox(dets, cup_bbox)) {
                cv::Mat dm = ai->midas_estimator->inference(frame);
                last_midas = t;
                float m_rim  = ai->midas_estimator->get_rim_depth(dm, cup_bbox);
                float m_tray = ai->midas_estimator->get_tray_depth(dm, aruco_roi);
                if (m_rim > 0 && m_tray > 0) {
                    pos_ratios.push_back(m_rim / m_tray);
                    pos_z.push_back(z_calib);
                }
            }
            last_midas = t;
        }

        if (phase == "warmup" && elapsed >= WARMUP_SEC) {
            phase = "sampling"; phase_start = now_sec();
        } else if (phase == "sampling") {
            if ((int)pos_ratios.size() >= 5 || elapsed > TIMEOUT) {
                if ((int)pos_ratios.size() >= 3) {
                    GridPt pt{vec_mean(pos_ratios), vec_mean(pos_z)};
                    grid_data.push_back(pt);
                    std::cout << "[CALIB] Position " << (pos_idx + 1) << "/"
                              << n_positions << " committed: Z=" << pt.Z
                              << "cm, R=" << pt.R << "\n";
                }
                pos_ratios.clear(); pos_z.clear();
                ++pos_idx;
                if (pos_idx >= n_positions) break;
                phase = "swap_wait";
            }
        }

        /* UI overlay */
        cv::Mat disp = frame.clone();
        if (!last_aruco.empty()) disp = aruco.annotate_frame(disp, last_aruco);
        for (auto& d : last_dets)
            if (d.score > 0)
                cv::rectangle(disp,
                    cv::Rect((int)d.box.x, (int)d.box.y,
                             (int)d.box.w, (int)d.box.h),
                    cv::Scalar(0, 255, 80), 2);

        std::string s1 = "ArUco: " + std::string(z_calib > 0 ? "OK" : "NONE") +
                         "  YOLO: " + (last_dets.empty() ? "NONE" : "OK");

        if (phase == "warmup") {
            int pct = std::min(100, (int)((elapsed / WARMUP_SEC) * 100));
            draw_status_box(disp,
                "Z-GRID: Warming up pos " + std::to_string(pos_idx + 1) +
                "/" + std::to_string(n_positions),
                "Keep cup still. Prog: " + std::to_string(pct) + "%", s1);
        } else if (phase == "sampling") {
            draw_status_box(disp,
                "Z-GRID: Sampling pos " + std::to_string(pos_idx + 1) +
                "/" + std::to_string(n_positions) +
                " (" + std::to_string((int)pos_ratios.size()) + "/5)",
                "Z_tray = " + std::to_string(z_calib).substr(0, 5) + " cm",
                s1, cv::Scalar(20, 20, 40));
        } else if (phase == "swap_wait") {
            draw_status_box(disp,
                "MOVE NOZZLE TO NEXT POSITION",
                "(" + std::to_string(pos_idx) + "/" +
                std::to_string(n_positions) +
                " done) Keep same cup visible.",
                "Press SPACE when ready.",
                cv::Scalar(30, 50, 160));
        }

        if (!headless) {
            cv::imshow(WIN, disp);
            int key = cv::waitKey(1) & 0xFF;
            if (key == 27) return nlohmann::json{};
            if (phase == "swap_wait" && key == ' ') {
                phase = "warmup"; phase_start = now_sec();
            }
        }
    }

    if ((int)grid_data.size() < 2) {
        std::cerr << "[CALIB] Not enough Z-Grid data.\n";
        return nlohmann::json{};
    }

    /* Polynomial fit: K_i = R_i * (1 - H / Z_i).
     * We fit K(Z) with degree min(n-1, 2).
     * Simple implementation: degree 1 if 2 pts, degree 2 if 3+ pts.        */
    int n_pts = (int)grid_data.size();
    int deg   = std::min(n_pts - 1, 2);

    /* Vandermonde system for poly fit */
    std::vector<double> Z_pts(n_pts), K_pts(n_pts);
    for (int i = 0; i < n_pts; ++i) {
        Z_pts[i] = grid_data[i].Z;
        K_pts[i] = grid_data[i].R * (1.0 - true_height / grid_data[i].Z);
    }

    /* For degrees 1 or 2 we solve the small normal equations directly */
    std::vector<double> poly_K;
    if (deg == 1) {
        /* Linear fit: K = a*Z + b */
        double sum_z  = 0, sum_z2 = 0, sum_k = 0, sum_zk = 0;
        int    n      = n_pts;
        for (int i = 0; i < n; ++i) {
            sum_z  += Z_pts[i];
            sum_z2 += Z_pts[i] * Z_pts[i];
            sum_k  += K_pts[i];
            sum_zk += Z_pts[i] * K_pts[i];
        }
        double denom = n * sum_z2 - sum_z * sum_z;
        double a     = (denom != 0) ? (n * sum_zk - sum_z * sum_k) / denom : 0.0;
        double b     = (sum_k - a * sum_z) / n;
        poly_K = {a, b};  /* [a, b] for a*Z + b = polyval convention */
    } else {
        /* Quadratic fit: K = a*Z^2 + b*Z + c (degree 2) */
        /* Build 3x3 normal equations */
        double S[5] = {};  /* S0=n, S1=sum Z, S2=sum Z^2, S3=sum Z^3, S4=sum Z^4 */
        double T[3] = {};  /* T0=sum K, T1=sum K*Z, T2=sum K*Z^2 */
        S[0] = n_pts;
        for (int i = 0; i < n_pts; ++i) {
            double z = Z_pts[i], k = K_pts[i];
            S[1] += z; S[2] += z*z; S[3] += z*z*z; S[4] += z*z*z*z;
            T[0] += k; T[1] += k*z; T[2] += k*z*z;
        }
        /* Gauss elimination on 3x3 */
        double A[3][4] = {
            {S[0], S[1], S[2], T[0]},
            {S[1], S[2], S[3], T[1]},
            {S[2], S[3], S[4], T[2]}
        };
        for (int col = 0; col < 3; ++col) {
            /* Pivot */
            int pivot = col;
            for (int r = col+1; r < 3; ++r)
                if (std::fabs(A[r][col]) > std::fabs(A[pivot][col]))
                    pivot = r;
            std::swap(A[col], A[pivot]);
            if (std::fabs(A[col][col]) < 1e-12) continue;
            for (int r = 0; r < 3; ++r) {
                if (r == col) continue;
                double f = A[r][col] / A[col][col];
                for (int c2 = col; c2 < 4; ++c2) A[r][c2] -= f * A[col][c2];
            }
        }
        double c_coef = A[2][3] / A[2][2];
        double b_coef = (A[1][3] - A[1][2]*c_coef) / A[1][1];
        double a_coef = (A[0][3] - A[0][2]*c_coef - A[0][1]*b_coef) / A[0][0];
        poly_K = {a_coef, b_coef, c_coef};  /* [a, b, c] for a*Z^2 + b*Z + c */
    }

    storage.save_3p(poly_K, Z_pts, true_height);
    std::cout << "[CALIB] Z-Grid calibration done!\n";
    return nlohmann::json{{"type", 3}, {"poly_K", poly_K}};
}

/*===========================================================================*/
/* MODE 4: BBox-area compensated                                              */
/*===========================================================================*/

nlohmann::json CalibRoutines::run_calib_bbox(cv::VideoCapture& cap,
                                              ArucoDetector& aruco,
                                              CalibrationStorage& storage,
                                              bool headless,
                                              double true_height)
{
    AI* ai = AI::get_instance();

    std::cout << "  BBOX AREA COMPENSATION CALIBRATION (Type 4)\n";
    std::cout << "  Cup reference height: " << true_height << " cm\n";

    constexpr double WARMUP_SEC = 4.0;
    constexpr double TIMEOUT    = 30.0;
    constexpr int    N_POS      = 2;

    struct BBoxPt { double R, Z, area; };
    std::vector<BBoxPt> positions;

    int    pos_idx     = 0;
    std::string phase  = "warmup";
    double phase_start = now_sec();
    double last_midas  = 0.0;

    std::vector<double> pr4, pz4, pa4;
    std::vector<Detection>   last_dets;
    std::vector<ArucoResult> last_aruco;

    while (pos_idx < N_POS) {
        cv::Mat frame;
        if (!cap.read(frame) || frame.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            continue;
        }
        double t = now_sec(), elapsed = t - phase_start;

        last_aruco = aruco.detect(frame);
        double z_calib = 0.0; cv::Rect aruco_roi;
        bool aruco_ok = get_aruco_roi(last_aruco, aruco, z_calib, aruco_roi);

        if ((t - last_midas) > 0.2 && z_calib > 0 && aruco_ok
            && phase == "sampling")
        {
            auto [dets] = ai->cup_detector->detect(frame);
            last_dets = dets;
            cv::Rect cup_bbox;
            if (get_cup_bbox(dets, cup_bbox)) {
                cv::Mat dm = ai->midas_estimator->inference(frame);
                last_midas = t;
                float m_rim  = ai->midas_estimator->get_rim_depth(dm, cup_bbox);
                float m_tray = ai->midas_estimator->get_tray_depth(dm, aruco_roi);
                if (m_rim > 0 && m_tray > 0) {
                    double area = (double)(cup_bbox.width * cup_bbox.height);
                    pr4.push_back(m_rim / m_tray);
                    pz4.push_back(z_calib);
                    pa4.push_back(area);
                }
            }
            last_midas = t;
        }

        if (phase == "warmup" && elapsed >= WARMUP_SEC) {
            phase = "sampling"; phase_start = now_sec();
        } else if (phase == "sampling") {
            if ((int)pr4.size() >= 5 || elapsed > TIMEOUT) {
                if ((int)pr4.size() >= 3) {
                    positions.push_back({vec_mean(pr4), vec_mean(pz4), vec_mean(pa4)});
                    std::cout << "[CALIB] BBox pos " << (pos_idx+1) << "/2 committed\n";
                }
                pr4.clear(); pz4.clear(); pa4.clear();
                ++pos_idx;
                if (pos_idx < N_POS) phase = "swap_wait";
            }
        }

        cv::Mat disp = frame.clone();
        if (!last_aruco.empty()) disp = aruco.annotate_frame(disp, last_aruco);
        std::string s1 = "ArUco: " + std::string(z_calib>0?"OK":"NONE") +
                         "  YOLO: " + (last_dets.empty()?"NONE":"OK");
        if (phase == "warmup") {
            int pct = std::min(100, (int)((elapsed/WARMUP_SEC)*100));
            draw_status_box(disp, "BBOX-AREA: Warming up pos " +
                std::to_string(pos_idx+1)+"/2",
                "Keep cup still. Prog: "+std::to_string(pct)+"%", s1);
        } else if (phase == "sampling") {
            draw_status_box(disp, "BBOX-AREA: Sampling pos "+
                std::to_string(pos_idx+1)+"/2 ("+
                std::to_string((int)pr4.size())+"/5)",
                "Z_tray="+std::to_string(z_calib).substr(0,5)+"cm", s1,
                cv::Scalar(20,20,40));
        } else if (phase == "swap_wait") {
            draw_status_box(disp, "MOVE NOZZLE TO DIFFERENT HEIGHT",
                "(1/2 done) Keep same cup visible.",
                "Press SPACE when ready.", cv::Scalar(30,50,160));
        }

        if (!headless) {
            cv::imshow(WIN, disp);
            int key = cv::waitKey(1)&0xFF;
            if (key == 27) return nlohmann::json{};
            if (phase == "swap_wait" && key == ' ') {
                phase = "warmup"; phase_start = now_sec();
            }
        }
    }

    if ((int)positions.size() < 2) {
        std::cerr << "[CALIB] Not enough BBox data.\n";
        return nlohmann::json{};
    }

    auto& p1 = positions[0]; auto& p2 = positions[1];
    double Y1_4 = true_height / p1.Z;
    double Y2_4 = true_height / p2.Z;
    double dR   = p2.R - p1.R;
    double m_ref = (std::fabs(dR) > 0.02) ? (Y2_4 - Y1_4) / dR : 0.15;
    double c_ref = Y1_4 - m_ref * p1.R;

    /* Reference is the position with largest bbox (closest to camera) */
    double ref_area = (positions[0].area >= positions[1].area)
                      ? positions[0].area : positions[1].area;

    storage.save_4p(m_ref, c_ref, ref_area,
                    std::min(p1.Z, p2.Z), std::max(p1.Z, p2.Z), true_height);
    std::cout << "[CALIB] BBox Area calibration done!\n";
    return nlohmann::json{
        {"type", 4}, {"m_ref", m_ref}, {"c_ref", c_ref},
        {"ref_bbox_area_px", ref_area}
    };
}

/*===========================================================================*/
/* MODE 5: Geometric Z-Grid                                                   */
/*===========================================================================*/

nlohmann::json CalibRoutines::run_calib_geom(cv::VideoCapture& cap,
                                              ArucoDetector& aruco,
                                              CalibrationStorage& storage,
                                              bool headless,
                                              double true_height,
                                              int n_positions)
{
    AI*    ai       = AI::get_instance();
    double focal_px = aruco.camera_matrix.at<double>(0, 0);

    std::cout << "  GEOMETRIC Z-GRID CALIBRATION (" << n_positions << " positions)\n";
    std::cout << "  Cup height: " << true_height << " cm  focal: " << focal_px << " px\n";

    constexpr double WARMUP_SEC  = 5.0;
    constexpr double SAMPLE_COUNT = 30.0;

    struct GridPt { double Z, H_px; };
    std::vector<GridPt> grid_data;

    int    pos_idx    = 0;
    std::string phase = "warmup";
    double phase_start = now_sec();
    double last_det   = 0.0;

    std::vector<double> cur_z, cur_h;
    std::vector<Detection>   last_dets;
    std::vector<ArucoResult> last_aruco;

    while (phase != "done") {
        cv::Mat frame;
        if (!cap.read(frame) || frame.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            continue;
        }
        double t = now_sec(), elapsed = t - phase_start;

        last_aruco = aruco.detect(frame);
        double z_calib = 0.0; cv::Rect aruco_roi;
        bool aruco_ok = get_aruco_roi(last_aruco, aruco, z_calib, aruco_roi);

        if ((t - last_det) > 0.15 && z_calib > 0 && phase == "sampling") {
            auto [dets] = ai->cup_detector->detect(frame);
            last_dets = dets;
            last_det  = t;
            cv::Rect cup_bbox;
            if (get_cup_bbox(dets, cup_bbox)) {
                double bbox_h = (double)cup_bbox.height;
                if (bbox_h > 5) {
                    cur_z.push_back(z_calib);
                    cur_h.push_back(bbox_h);
                }
            }
        }

        if (phase == "warmup" && elapsed >= WARMUP_SEC) {
            phase = "sampling"; phase_start = now_sec();
        } else if (phase == "sampling" && (int)cur_z.size() >= 30) {
            double avg_z  = vec_median(cur_z);
            double avg_h  = vec_median(cur_h);
            grid_data.push_back({avg_z, avg_h});
            std::cout << "[CALIB] Geom pos " << (pos_idx+1) << "/"
                      << n_positions << ": Z=" << avg_z << " h_px=" << avg_h << "\n";
            ++pos_idx;
            cur_z.clear(); cur_h.clear();
            if (pos_idx >= n_positions) phase = "done";
            else phase = "swap_wait";
        }

        cv::Mat disp = frame.clone();
        if (!last_aruco.empty()) disp = aruco.annotate_frame(disp, last_aruco);
        std::string s1 = "ArUco: " + std::string(aruco_ok?"OK":"NONE") +
                         "  YOLO: " + (last_dets.empty()?"NONE":"OK");
        if (phase == "warmup") {
            int pct = std::min(100,(int)((elapsed/WARMUP_SEC)*100));
            draw_status_box(disp,"GEO-GRID ("+std::to_string(pos_idx+1)+"/"+
                std::to_string(n_positions)+"): Warming up... "+std::to_string(pct)+"%",
                "Keep cup "+std::to_string(true_height).substr(0,4)+"cm still.", s1);
        } else if (phase == "sampling") {
            draw_status_box(disp,"GEO-GRID ("+std::to_string(pos_idx+1)+"/"+
                std::to_string(n_positions)+"): Sampling ("+
                std::to_string((int)cur_z.size())+"/30)",
                "Z_tray="+std::to_string(z_calib).substr(0,5)+"cm", s1,
                cv::Scalar(20,20,40));
        } else if (phase == "swap_wait") {
            draw_status_box(disp,"MOVE NOZZLE TO NEW HEIGHT",
                "Wait for focus, then press SPACE.", s1,
                cv::Scalar(40,40,150));
        }

        if (!headless) {
            cv::imshow(WIN, disp);
            int key = cv::waitKey(1)&0xFF;
            if (key == 27) return nlohmann::json{};
            if (phase == "swap_wait" && key == ' ') {
                phase = "warmup"; phase_start = now_sec();
                cur_z.clear(); cur_h.clear();
            }
        }
    }

    if ((int)grid_data.size() < 2) {
        std::cerr << "[CALIB] Not enough geometric data.\n";
        return nlohmann::json{};
    }

    /* K_geom_i = H_true / (Z_i * H_px_i / focal) */
    int n_pts = (int)grid_data.size();
    std::vector<double> Z_pts(n_pts), K_pts(n_pts);
    for (int i = 0; i < n_pts; ++i) {
        Z_pts[i] = grid_data[i].Z;
        K_pts[i] = true_height / (grid_data[i].Z * grid_data[i].H_px / focal_px);
    }

    /* Reuse degree logic from mode 3 */
    int deg = std::min(n_pts - 1, 2);
    std::vector<double> poly_Kgeom;
    if (deg == 1) {
        double sz=0,sz2=0,sk=0,szk=0;
        for (int i=0;i<n_pts;++i){sz+=Z_pts[i];sz2+=Z_pts[i]*Z_pts[i];
            sk+=K_pts[i];szk+=Z_pts[i]*K_pts[i];}
        double d=n_pts*sz2-sz*sz;
        double a=(d!=0)?(n_pts*szk-sz*sk)/d:0;
        double b=(sk-a*sz)/n_pts;
        poly_Kgeom={a,b};
    } else {
        double S[5]={},T[3]={};
        S[0]=n_pts;
        for(int i=0;i<n_pts;++i){double z=Z_pts[i],k=K_pts[i];
            S[1]+=z;S[2]+=z*z;S[3]+=z*z*z;S[4]+=z*z*z*z;
            T[0]+=k;T[1]+=k*z;T[2]+=k*z*z;}
        double A[3][4]={{S[0],S[1],S[2],T[0]},{S[1],S[2],S[3],T[1]},{S[2],S[3],S[4],T[2]}};
        for(int col=0;col<3;++col){
            int piv=col;
            for(int r=col+1;r<3;++r) if(std::fabs(A[r][col])>std::fabs(A[piv][col])) piv=r;
            std::swap(A[col],A[piv]);
            if(std::fabs(A[col][col])<1e-12) continue;
            for(int r=0;r<3;++r){if(r==col)continue;double f=A[r][col]/A[col][col];
                for(int c2=col;c2<4;++c2) A[r][c2]-=f*A[col][c2];}}
        double cc=A[2][3]/A[2][2];
        double bc=(A[1][3]-A[1][2]*cc)/A[1][1];
        double ac=(A[0][3]-A[0][2]*cc-A[0][1]*bc)/A[0][0];
        poly_Kgeom={ac,bc,cc};
    }

    storage.save_5p(poly_Kgeom, Z_pts, true_height);
    std::cout << "[CALIB] Geometric Z-Grid done!\n";
    return nlohmann::json{{"type", 5}, {"poly_Kgeom", poly_Kgeom}};
}

/*===========================================================================*/
/* MODE 6: Bilateral Z-Grid                                                   */
/*===========================================================================*/

nlohmann::json CalibRoutines::run_calib_bilateral(cv::VideoCapture& cap,
                                                   ArucoDetector& aruco,
                                                   CalibrationStorage& storage,
                                                   bool headless,
                                                   double true_height,
                                                   double true_height_2,
                                                   int n_positions)
{
    AI* ai = AI::get_instance();

    std::cout << "  BILATERAL Z-GRID (" << n_positions
              << " positions x 2 cups)\n";

    constexpr double WARMUP_SEC = 4.0;
    constexpr int    SAMPLE_N   = 8;

    std::vector<double> m_pts, c_pts, Z_pts;
    std::vector<double> r1_s, r2_s, z_s;

    int    pos_idx    = 0;
    std::string phase = "warmup_c1";
    double phase_start = now_sec();
    double last_midas  = 0.0;

    std::vector<Detection>   last_dets;
    std::vector<ArucoResult> last_aruco;

    while (phase != "done") {
        cv::Mat frame;
        if (!cap.read(frame) || frame.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            continue;
        }
        double t = now_sec(), elapsed = t - phase_start;

        last_aruco = aruco.detect(frame);
        double z_calib = 0.0; cv::Rect aruco_roi;
        bool aruco_ok = get_aruco_roi(last_aruco, aruco, z_calib, aruco_roi);

        bool in_sample = (phase == "sample_c1" || phase == "sample_c2");
        if ((t - last_midas) > 0.2 && z_calib > 0 && aruco_ok && in_sample) {
            auto [dets] = ai->cup_detector->detect(frame);
            last_dets = dets;
            cv::Rect cup_bbox;
            if (get_cup_bbox(dets, cup_bbox)) {
                cv::Mat dm = ai->midas_estimator->inference(frame);
                last_midas = t;
                float m_rim  = ai->midas_estimator->get_rim_depth(dm, cup_bbox);
                float m_tray = ai->midas_estimator->get_tray_depth(dm, aruco_roi);
                if (m_rim > 0 && m_tray > 0) {
                    double ratio = m_rim / m_tray;
                    if (phase == "sample_c1") {
                        r1_s.push_back(ratio);
                        z_s.push_back(z_calib);
                    } else {
                        r2_s.push_back(ratio);
                    }
                }
            }
            last_midas = t;
        }

        /* Transitions */
        if      (phase == "warmup_c1" && elapsed >= WARMUP_SEC) { phase = "sample_c1"; phase_start = now_sec(); }
        else if (phase == "sample_c1" && (int)r1_s.size() >= SAMPLE_N) { phase = "swap_c2"; phase_start = now_sec(); }
        else if (phase == "warmup_c2" && elapsed >= WARMUP_SEC) { phase = "sample_c2"; phase_start = now_sec(); }
        else if (phase == "sample_c2" && (int)r2_s.size() >= SAMPLE_N) {
            double R1a = vec_median(r1_s);
            double R2a = vec_median(r2_s);
            double Za  = vec_median(z_s);
            double Y1  = true_height   / Za;
            double Y2  = true_height_2 / Za;
            double dR  = R2a - R1a;
            if (std::fabs(dR) < 0.005) {
                std::cerr << "[CALIB] Identical ratios for both cups at pos "
                          << (pos_idx+1) << ".\n";
                return nlohmann::json{};
            }
            double mi = (Y2 - Y1) / dR;
            double ci = Y1 - mi * R1a;
            m_pts.push_back(mi); c_pts.push_back(ci); Z_pts.push_back(Za);
            std::cout << "[CALIB] Z-pos " << (pos_idx+1) << "/" << n_positions
                      << ": Z=" << Za << " m=" << mi << " c=" << ci << "\n";
            ++pos_idx;
            r1_s.clear(); r2_s.clear(); z_s.clear();
            if (pos_idx >= n_positions) phase = "done";
            else phase = "swap_z";
        }

        cv::Mat disp = frame.clone();
        if (!last_aruco.empty()) disp = aruco.annotate_frame(disp, last_aruco);
        std::string s1 = "ArUco: " + std::string(z_calib>0?"OK":"NO") +
                         "  pos: " + std::to_string(pos_idx+1) +
                         "/" + std::to_string(n_positions);
        if (phase.substr(0,7) == "warmup_") {
            int pct = std::min(100,(int)((elapsed/WARMUP_SEC)*100));
            int idx = (phase == "warmup_c1") ? 1 : 2;
            draw_status_box(disp,"BILATERAL: Warming Cup "+std::to_string(idx)+
                "... "+std::to_string(pct)+"%",
                "Place cup still.", s1);
        } else if (phase.substr(0,7) == "sample_") {
            int idx = (phase == "sample_c1") ? 1 : 2;
            int cnt = (idx==1)?(int)r1_s.size():(int)r2_s.size();
            draw_status_box(disp,"BILATERAL: Sampling Cup "+std::to_string(idx)+
                " ("+std::to_string(cnt)+"/"+std::to_string(SAMPLE_N)+")",
                "Z_tray="+std::to_string(z_calib).substr(0,5)+"cm", s1,
                cv::Scalar(20,20,40));
        } else if (phase == "swap_c2") {
            draw_status_box(disp,"SWAP TO TALL CUP ("+
                std::to_string(true_height_2).substr(0,4)+"cm)",
                "Press SPACE to scan cup 2.", s1,
                cv::Scalar(150,50,80));
        } else if (phase == "swap_z") {
            draw_status_box(disp,"MOVE NOZZLE TO DIFFERENT HEIGHT",
                "Wait for focus. Place "+std::to_string(true_height).substr(0,4)+
                "cm cup. Press SPACE.", s1,
                cv::Scalar(40,40,150));
        }

        if (!headless) {
            cv::imshow(WIN, disp);
            int key = cv::waitKey(1)&0xFF;
            if (key == 27) return nlohmann::json{};
            if (key == ' ') {
                if (phase == "swap_c2") { phase = "warmup_c2"; phase_start = now_sec(); }
                else if (phase == "swap_z") { phase = "warmup_c1"; phase_start = now_sec(); }
            }
        }
    }

    /* Fit poly_m(Z) and poly_c(Z) */
    int n_pts = (int)Z_pts.size();
    if (n_pts < 1) { std::cerr << "[CALIB] No bilateral data.\n"; return nlohmann::json{}; }

    auto fit_poly = [&](const std::vector<double>& Y) -> std::vector<double> {
        int deg2 = std::min(n_pts-1, 2);
        if (deg2 == 0) return {Y[0]};
        if (deg2 == 1) {
            double sz=0,sz2=0,sy=0,szy=0;
            for(int i=0;i<n_pts;++i){sz+=Z_pts[i];sz2+=Z_pts[i]*Z_pts[i];
                sy+=Y[i];szy+=Z_pts[i]*Y[i];}
            double d=n_pts*sz2-sz*sz;
            double a=(d!=0)?(n_pts*szy-sz*sy)/d:0;
            double b=(sy-a*sz)/n_pts;
            return {a,b};
        }
        /* quad */
        double S[5]={},T[3]={};
        S[0]=n_pts;
        for(int i=0;i<n_pts;++i){double z=Z_pts[i],y=Y[i];
            S[1]+=z;S[2]+=z*z;S[3]+=z*z*z;S[4]+=z*z*z*z;
            T[0]+=y;T[1]+=y*z;T[2]+=y*z*z;}
        double A[3][4]={{S[0],S[1],S[2],T[0]},{S[1],S[2],S[3],T[1]},{S[2],S[3],S[4],T[2]}};
        for(int col=0;col<3;++col){
            int piv=col;
            for(int r=col+1;r<3;++r) if(std::fabs(A[r][col])>std::fabs(A[piv][col])) piv=r;
            std::swap(A[col],A[piv]);
            if(std::fabs(A[col][col])<1e-12) continue;
            for(int r=0;r<3;++r){if(r==col)continue;double f=A[r][col]/A[col][col];
                for(int c2=col;c2<4;++c2)A[r][c2]-=f*A[col][c2];}}
        double cc=A[2][3]/A[2][2];
        double bc=(A[1][3]-A[1][2]*cc)/A[1][1];
        double ac=(A[0][3]-A[0][2]*cc-A[0][1]*bc)/A[0][0];
        return {ac,bc,cc};
    };

    auto poly_m = fit_poly(m_pts);
    auto poly_c = fit_poly(c_pts);

    storage.save_6p(poly_m, poly_c, Z_pts, true_height, true_height_2);
    std::cout << "[CALIB] Bilateral Z-Grid done!\n";
    return nlohmann::json{{"type",6},{"poly_m",poly_m},{"poly_c",poly_c}};
}

/*===========================================================================*/
/* MODE 7: Universal Analytic Geometry                                        */
/*===========================================================================*/

nlohmann::json CalibRoutines::run_calib_analytic(cv::VideoCapture& cap,
                                                  ArucoDetector& aruco,
                                                  CalibrationStorage& storage,
                                                  bool headless,
                                                  double true_height,
                                                  double true_height_2)
{
    AI* ai = AI::get_instance();

    std::cout << "  UNIVERSAL ANALYTIC GEOMETRY (Type 7)\n";
    std::cout << "  Cup 1: " << true_height << " cm, Cup 2: "
              << true_height_2 << " cm\n";

    constexpr int    SAMPLE_N  = 30;
    std::string phase = "warmup_1";
    double phase_start = now_sec();
    double last_det   = 0.0;

    std::vector<double> y1_samples, y2_samples;
    std::vector<Detection>   last_dets;
    std::vector<ArucoResult> last_aruco;

    while (phase != "done") {
        cv::Mat frame;
        if (!cap.read(frame) || frame.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            continue;
        }
        double t = now_sec(), elapsed = t - phase_start;

        last_aruco = aruco.detect(frame);
        double z_calib = 0.0; cv::Rect aruco_roi;
        get_aruco_roi(last_aruco, aruco, z_calib, aruco_roi);

        bool in_sample = (phase == "sample_1" || phase == "sample_2");
        if ((t - last_det) > 0.1 && z_calib > 0 && in_sample) {
            auto [dets] = ai->cup_detector->detect(frame);
            last_dets = dets;
            last_det  = t;
            cv::Rect cup_bbox;
            if (get_cup_bbox(dets, cup_bbox)) {
                double bbox_h = (double)cup_bbox.height;
                if (bbox_h > 5) {
                    if (phase == "sample_1")
                        y1_samples.push_back(bbox_h * (z_calib - true_height));
                    else
                        y2_samples.push_back(bbox_h * (z_calib - true_height_2));
                }
            }
        }

        if      (phase == "warmup_1" && elapsed >= 3.0) { phase = "sample_1"; phase_start = now_sec(); }
        else if (phase == "sample_1" && (int)y1_samples.size() >= SAMPLE_N) { phase = "swap"; phase_start = now_sec(); }
        else if (phase == "warmup_2" && elapsed >= 3.0) { phase = "sample_2"; phase_start = now_sec(); }
        else if (phase == "sample_2" && (int)y2_samples.size() >= SAMPLE_N) {
            double Y1 = vec_median(y1_samples);
            double Y2 = vec_median(y2_samples);
            double dH = true_height_2 - true_height;
            if (std::fabs(dH) < 0.1) {
                std::cerr << "[CALIB] Cups must have different heights!\n";
                return nlohmann::json{};
            }
            double B = (Y2 - Y1) / dH;
            double A = Y1 - B * true_height;
            storage.save_7(A, B, true_height, true_height_2);
            std::cout << "[CALIB] A=" << A << " B=" << B << "\n";
            std::cout << "[CALIB] Analytic Geometry done!\n";
            return nlohmann::json{{"type",7},{"A",A},{"B",B}};
        }

        cv::Mat disp = frame.clone();
        if (!last_aruco.empty()) disp = aruco.annotate_frame(disp, last_aruco);
        for (auto& d : last_dets)
            if (d.score > 0)
                cv::rectangle(disp,
                    cv::Rect((int)d.box.x,(int)d.box.y,(int)d.box.w,(int)d.box.h),
                    cv::Scalar(0,255,80),2);
        cv::rectangle(disp, cv::Point(8,8), cv::Point(535,100),
                      cv::Scalar(20,20,60),-1);

        if (phase == "warmup_1" || phase == "sample_1") {
            cv::putText(disp,"YOLO ANALYTIC: Cup 1 ("+
                std::to_string(true_height).substr(0,4)+"cm)",
                cv::Point(18,30),cv::FONT_HERSHEY_SIMPLEX,0.55,
                cv::Scalar(0,220,255),2);
            if (phase == "sample_1")
                cv::putText(disp,"Sampling ["+
                    std::to_string((int)y1_samples.size())+"/30]",
                    cv::Point(18,60),cv::FONT_HERSHEY_SIMPLEX,0.45,
                    cv::Scalar(0,255,150),1);
        } else if (phase == "swap") {
            cv::rectangle(disp,cv::Point(8,8),cv::Point(535,100),
                          cv::Scalar(80,50,150),-1);
            cv::putText(disp,"SWAP TO CUP 2 ("+
                std::to_string(true_height_2).substr(0,4)+"cm)",
                cv::Point(18,30),cv::FONT_HERSHEY_SIMPLEX,0.55,
                cv::Scalar(255,255,255),2);
            cv::putText(disp,"Press SPACE when ready.",
                cv::Point(18,60),cv::FONT_HERSHEY_SIMPLEX,0.45,
                cv::Scalar(200,220,255),1);
        } else if (phase == "warmup_2" || phase == "sample_2") {
            cv::putText(disp,"YOLO ANALYTIC: Cup 2 ("+
                std::to_string(true_height_2).substr(0,4)+"cm)",
                cv::Point(18,30),cv::FONT_HERSHEY_SIMPLEX,0.55,
                cv::Scalar(0,220,255),2);
            if (phase == "sample_2")
                cv::putText(disp,"Sampling ["+
                    std::to_string((int)y2_samples.size())+"/30]",
                    cv::Point(18,60),cv::FONT_HERSHEY_SIMPLEX,0.45,
                    cv::Scalar(0,255,150),1);
        }

        if (!headless) {
            cv::imshow(WIN, disp);
            int key = cv::waitKey(1)&0xFF;
            if (key == 27) return nlohmann::json{};
            if (phase == "swap" && key == ' ') {
                phase = "warmup_2"; phase_start = now_sec();
            }
        }
    }

    return nlohmann::json{};  // unreachable but satisfies compiler
}
