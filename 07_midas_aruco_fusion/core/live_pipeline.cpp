/*******************************************************************************
 * live_pipeline.cpp
 * Port of 07_midas_aruco_fusion/core/live_pipeline.py
 * Uses AI::get_instance() for all AI inference.
 * Optimized with Multi-threading: Camera (Thread 1), Inference (Thread 2), Display (Main)
 ******************************************************************************/
#include "live_pipeline.hpp"

#include "calibration_routines.hpp"
#include "height_math.hpp"
#include "session_reporter.hpp"
#include "moildev_applicator.hpp"
#include "gui_fusion.hpp"
#include "volume_math.hpp"

#include <detections/ai.h>

#include <chrono>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>

namespace fs = std::filesystem;

/*---------------------------------------------------------------------------*/
/* Timestamp string for filenames                                             */
/*---------------------------------------------------------------------------*/
static std::string ts_filename()
{
    auto now  = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::ostringstream oss;
    oss << std::put_time(std::localtime(&time), "%Y-%m-%d_%H%M%S");
    return oss.str();
}

static double now_sec()
{
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

/*---------------------------------------------------------------------------*/
/* Shared structures for multi-threading                                     */
/*---------------------------------------------------------------------------*/

struct InferenceResult {
    double                z_tray_live = 0.0;
    cv::Rect              aruco_roi;
    bool                  aruco_roi_valid = false;
    std::vector<cv::Rect> cup_bboxes;
    std::array<double, 2> cup_heights_ema = {0.0, 0.0};
    std::array<bool, 2>   cup_heights_valid = {false, false};
    /* cup_diameters: diameter fisik (cm), dihitung di inference thread */
    std::array<double, 2> cup_diameters = {0.0, 0.0};
    std::array<double, 2> cup_volumes   = {0.0, 0.0};
    /* cup_zrims: z_tray - h_cup (cm), untuk display Z_rim label */
    std::array<double, 2> cup_zrims     = {0.0, 0.0};
    std::array<bool, 2>   cup_vol_valid = {false, false};
    cv::Mat               last_depth_norm;
    std::vector<ArucoResult> aruco_results;
    
    /* Stats */
    int stats_total_frames = 0;
    int stats_midas_runs   = 0;
    
    /* History for reporter */
    std::vector<double> history_z_tray;
    std::map<int, std::vector<double>> history_cup_h = {{0, {}}, {1, {}}};
    std::vector<int> history_frames;
};

static std::mutex              g_result_mutex;
static std::condition_variable g_result_cv;
static InferenceResult         g_shared_result;
static std::atomic<bool>       g_pipeline_running{true};
static std::atomic<bool>       g_midas_enabled{false};

/*---------------------------------------------------------------------------*/
/* Inference Worker Thread                                                    */
/*---------------------------------------------------------------------------*/

void inference_worker(Camera* cam, ArucoDetector* aruco_ptr, const nlohmann::json calib_data,
                      double marker_size_cm, std::vector<double> active_poly_Kgeom, double focal_px,
                      double true_height_cm,
                      MoildevApplicator* moil, bool no_anypoint, int output_w, int output_h,
                      GuiFusion* gui)
{
    AI* ai = AI::get_instance();
    ArucoDetector& aruco = *aruco_ptr;

    constexpr double MIDAS_FPS_LIMIT = 30.0;
    constexpr double MIDAS_INTERVAL  = 1.0 / MIDAS_FPS_LIMIT;
    constexpr double EMA_ALPHA       = 0.35;

    double last_midas_t = 0.0;
    int ctype = calib_data.value("type", 1);
    
    /* Local state for persistence within thread */
    double z_tray_live = 0.0;
    cv::Rect aruco_roi;
    bool aruco_roi_valid = false;
    std::array<double, 2>  cup_heights_ema = {0.0, 0.0};
    std::array<bool, 2>    cup_heights_valid = {false, false};
    std::array<double, 2>  cup_diameters = {0.0, 0.0};
    std::array<double, 2>  cup_volumes   = {0.0, 0.0};
    std::array<double, 2>  cup_zrims     = {0.0, 0.0};
    std::array<bool, 2>    cup_vol_valid = {false, false};

    /* Self-calibrating K factor:
     * K = true_height * focal / (z_tray * bbox_h_px)
     * Dihitung SEKALI dari observasi pertama saat true_height tersedia.
     * Konstan untuk setup kamera + model YOLO yang sama.
     * Memungkinkan estimasi tinggi gelas APAPUN secara proporsional:
     *   h_est = z_tray * (bbox_h / focal) * K_self_calib */
    double K_self_calib    = 0.0;
    bool   K_self_ready    = false;

    int stats_total_frames = 0;
    int stats_midas_runs   = 0;
    
    /* History */
    std::vector<double> history_z_tray;
    std::map<int, std::vector<double>> history_cup_h = {{0, {}}, {1, {}}};
    std::vector<int> history_frames;

    while (g_pipeline_running) {
        cv::Mat frame = cam->get_frame();
        if (frame.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            continue;
        }

        /* ── Normalize Lighting (jika GUI aktif dan diaktifkan user) ───── */
        if (gui != nullptr && gui->is_normalize_enabled()) {
            /* CLAHE-based normalization: equalize luminance di YCrCb */
            cv::Mat ycrcb;
            cv::cvtColor(frame, ycrcb, cv::COLOR_BGR2YCrCb);
            std::vector<cv::Mat> ch;
            cv::split(ycrcb, ch);
            cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
            clahe->apply(ch[0], ch[0]);
            cv::merge(ch, ycrcb);
            cv::cvtColor(ycrcb, frame, cv::COLOR_YCrCb2BGR);
        }

        /* ── Black & White Mode (jika GUI aktif dan diaktifkan user) ───── */
        if (gui != nullptr && gui->is_bw_enabled()) {
            cv::Mat gray;
            cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
            cv::cvtColor(gray, frame, cv::COLOR_GRAY2BGR);
        }
        /* ── Simpan raw fisheye sebelum undistorsi ────────────────────── */
        /* raw_frame digunakan untuk referensi; saat ini tidak dikirim ke volume
         * module (moil_facade dihapus). Frame undistorted dipakai untuk semua
         * kalkulasi (ArUco, YOLO, rim measurement). */
        cv::Mat raw_frame;
        if (moil != nullptr && !no_anypoint) {
            raw_frame = frame.clone();  // simpan sebelum remap
        }

        /* ── Apply fisheye undistortion (if enabled) ──────────────────── */
        if (moil != nullptr && !no_anypoint) {
            frame = moil->undistort(frame);

            /* Setelah Moildev undistortion, gambar sudah rektifikasi (pinhole).
             * 1. dist_coeffs di-zero: tidak ada distorsi residual.
             * 2. camera_matrix di-update ke effective focal length undistorted frame.
             *    Effective fl = FOCAL_LENGTH_FOR_ZOOM(250) * zoom_internal
             *                 = 250 * (param5/250) = param5 ≈ 504px.
             *    Principal point = center output frame. */
            aruco.dist_coeffs = cv::Mat::zeros(1, 5, CV_64F);
            aruco.camera_matrix = moil->build_aruco_camera_matrix(
                frame.cols, frame.rows);
        }

        double now = now_sec();
        stats_total_frames++;

        /* ── ArUco detection ── */
        cv::Mat aruco_frame;
        if (frame.channels() == 3) {
            cv::cvtColor(frame, aruco_frame, cv::COLOR_BGR2GRAY);
        } else if (frame.channels() == 4) {
            cv::cvtColor(frame, aruco_frame, cv::COLOR_BGRA2GRAY);
        } else {
            aruco_frame = frame.clone();
        }
        /* JANGAN lakukan global threshold di sini!
         * Global threshold 128 membunuh ArUco detection pada gambar cerah:
         * gambar menjadi hampir semua putih sehingga pola marker hilang.
         * ArUco detector sudah menggunakan adaptive threshold internal — 
         * cukup berikan gambar grayscale langsung. */
        auto aruco_results = aruco.detect(aruco_frame);
        if (!aruco_results.empty()) {
            BestDistanceResult best = aruco.get_best_distance(aruco_results);
            if (best.distance_cm > 0) {
                z_tray_live = best.distance_cm;
                const auto& corners = aruco_results[0].corners;
                if (!corners.empty()) {
                    float x_min = corners[0].x, y_min = corners[0].y;
                    float x_max = corners[0].x, y_max = corners[0].y;
                    for (auto& p : corners) {
                        x_min = std::min(x_min, p.x);
                        y_min = std::min(y_min, p.y);
                        x_max = std::max(x_max, p.x);
                        y_max = std::max(y_max, p.y);
                    }
                    int pad_x = std::max(2, (int)((x_max-x_min)/10));
                    int pad_y = std::max(2, (int)((y_max-y_min)/10));
                    aruco_roi = cv::Rect(
                        (int)x_min + pad_x, (int)y_min + pad_y,
                        (int)(x_max - x_min) - 2*pad_x,
                        (int)(y_max - y_min) - 2*pad_y
                    );
                    aruco_roi_valid = (aruco_roi.width > 0 && aruco_roi.height > 0);
                }
            }
        }

        /* ── Cup detection + MiDaS ── */
        std::vector<cv::Rect> current_cup_bboxes;
        cv::Mat current_depth_norm;
        bool ran_midas_this_loop = false;

        // Jalankan YOLO cup detection jika interval terpenuhi (tidak perlu tunggu ArUco)
        if ((now - last_midas_t) >= MIDAS_INTERVAL) {
            auto [dets] = ai->cup_detector->detect(frame);

            /* Debug: log semua detections setiap 60 frame */
            if (stats_total_frames % 60 == 1) {
                std::cout << "[YOLO] frame=" << stats_total_frames
                          << " total_dets=" << dets.size() << "\n";
                for (size_t i = 0; i < std::min(dets.size(), size_t(5)); ++i) {
                    std::cout << "  det[" << i << "] class=" << dets[i].class_id
                              << " score=" << dets[i].score
                              << " box=(" << dets[i].box.x << "," << dets[i].box.y
                              << " " << dets[i].box.w << "x" << dets[i].box.h << ")\n";
                }
            }

            if (!dets.empty()) {
                struct CupDet { cv::Rect bbox; float score; };
                std::vector<CupDet> valid_dets;
                for (auto& d : dets) {
                    // class_id=0: cup_rim, class_id=1: cup_body
                    // Terima keduanya, utamakan cup_rim untuk height estimation
                    if (d.score >= 0.35f && (d.class_id == 0 || d.class_id == 1))
                        valid_dets.push_back({
                            cv::Rect((int)d.box.x, (int)d.box.y, (int)d.box.w, (int)d.box.h),
                            d.score
                        });
                }
                std::sort(valid_dets.begin(), valid_dets.end(),
                          [](const CupDet& a, const CupDet& b){ return a.bbox.x < b.bbox.x; });
                if (valid_dets.size() > 2) valid_dets.resize(2);

                for (auto& cd : valid_dets) current_cup_bboxes.push_back(cd.bbox);

                /* ── Geometric Height (ctype 5/7) ── */
                if (z_tray_live > 0) {
                    for (int i = 0; i < 2; ++i) {
                        double height_raw = 0.0;
                        if (i < (int)current_cup_bboxes.size()) {
                            const cv::Rect& bbox = current_cup_bboxes[i];
                            if (ctype == 5) {
                                height_raw = HeightMath::calc_height_geom(
                                    z_tray_live, bbox.x, bbox.y, bbox.x + bbox.width, bbox.y + bbox.height,
                                    focal_px,
                                    active_poly_Kgeom);
                                /* DEBUG — print K_live inline untuk verifikasi poly_Kgeom */
                                if (stats_total_frames % 30 == 1) {
                                    double k_dbg = 0.0;
                                    for (const double c : active_poly_Kgeom)
                                        k_dbg = k_dbg * z_tray_live + c;
                                    std::cout << "[HEIGHT-DBG] cup=" << i
                                              << " z=" << z_tray_live
                                              << " bbox_h=" << bbox.height
                                              << " focal=" << focal_px
                                              << " K_live=" << k_dbg
                                              << " poly_sz=" << active_poly_Kgeom.size()
                                              << " h_raw=" << height_raw << "\n";
                                    std::cout.flush();
                                }
                                /* FALLBACK: jika poly extrapolasi ke luar range kalibrasi
                                 * (K_live < 0 → height_raw=0), gunakan Self-calibrating K.
                                 *
                                 * K_self_calib di-bootstrap SEKALI dari true_height_cm:
                                 *   K = true_height * focal / (z_tray * bbox_h)
                                 * lalu dibekukan. Untuk gelas berbeda ukuran:
                                 *   h_est = z * (bbox_h / focal) * K_self_calib
                                 * → proporsional terhadap bbox, bukan hardcoded. */
                                if (height_raw <= 0.0) {
                                    const double h_geo = z_tray_live *
                                        (static_cast<double>(bbox.height) / focal_px);
                                    /* Bootstrap K dari true_height sekali */
                                    if (!K_self_ready && true_height_cm > 0.0 && h_geo > 0.0) {
                                        double K_candidate = true_height_cm / h_geo;
                                        if (K_candidate > 0.05 && K_candidate < 5.0) {
                                            K_self_calib = K_candidate;
                                            K_self_ready = true;
                                            std::cout << "[K-CALIB] Bootstrap K_self="
                                                      << K_self_calib
                                                      << " dari true_height=" << true_height_cm
                                                      << " z=" << z_tray_live
                                                      << " bbox_h=" << bbox.height << "\n";
                                            std::cout.flush();
                                        }
                                    }
                                    /* Estimasi tinggi proporsional (adaptif per bbox_h) */
                                    if (K_self_ready && h_geo > 0.0) {
                                        height_raw = h_geo * K_self_calib;
                                        if (stats_total_frames % 30 == 1) {
                                            std::cout << "[HEIGHT-EST] cup=" << i
                                                      << " h_geo=" << h_geo
                                                      << " K=" << K_self_calib
                                                      << " h_est=" << height_raw << " cm\n";
                                            std::cout.flush();
                                        }
                                    } else if (true_height_cm > 0.0) {
                                        /* Belum ter-kalibrasi, fallback hardcoded sementara */
                                        height_raw = true_height_cm;
                                    }
                                }
                            } else if (ctype == 7) {
                                height_raw = HeightMath::calc_height_analytic(
                                    z_tray_live, bbox.x, bbox.y, bbox.x + bbox.width, bbox.y + bbox.height,
                                    calib_data.value("A", 0.0), calib_data.value("B", 0.0));
                            }
                            // Catatan: ctype 1-4, 6 sebelumnya bergantung pada MiDaS. Karena MiDaS dicopot, tinggi akan 0.0.
                            if (height_raw > 0.0) {
                                if (!cup_heights_valid[i]) {
                                    cup_heights_ema[i] = height_raw;
                                    cup_heights_valid[i] = true;
                                } else {
                                    cup_heights_ema[i] = EMA_ALPHA * height_raw + (1.0 - EMA_ALPHA) * cup_heights_ema[i];
                                }
                            }
                            if (cup_heights_valid[i]) {
                                history_cup_h[i].push_back(cup_heights_ema[i]);
                                double z_rim_val = std::max(0.0, z_tray_live - cup_heights_ema[i]);
                                cup_zrims[i] = z_rim_val;

                                /* ── Hitung diameter & volume di inference thread ──
                                 * Gunakan undistorted frame (sudah diremap) +
                                 * focal efektif dari moildev camera matrix.
                                 * Ini lebih akurat daripada display frame (scaled) +
                                 * focal dari calibration_params.yml. */
                                if (!frame.empty() && z_rim_val > 0.0) {
                                    double focal_eff = aruco.camera_matrix.empty()
                                                       ? focal_px
                                                       : aruco.camera_matrix.at<double>(0, 0);
                                    float rim_w_px = VolumeMath::measureRimWidthPx(frame, bbox);
                                    cup_diameters[i] = VolumeMath::calcDiameter(
                                        rim_w_px, z_rim_val, focal_eff);
                                    cup_volumes[i]   = VolumeMath::calcVolume(
                                        cup_heights_ema[i], cup_diameters[i]);
                                    cup_vol_valid[i] = true;
                                } else {
                                    cup_vol_valid[i] = false;
                                }
                            } else {
                                cup_vol_valid[i] = false;
                                history_cup_h[i].push_back(0.0);
                            }
                        } else {
                            cup_heights_valid[i] = false;
                            cup_vol_valid[i]     = false;
                            history_cup_h[i].push_back(0.0);
                        }
                    }
                    history_z_tray.push_back(z_tray_live);
                    history_frames.push_back(stats_total_frames);
                }
            } else {
                /* No cups detected */
                for (int i = 0; i < 2; ++i) {
                    cup_heights_valid[i] = false;
                    cup_vol_valid[i] = false;
                    history_cup_h[i].push_back(0.0);
                }
                history_z_tray.push_back(z_tray_live);
                history_frames.push_back(stats_total_frames);
            }
            last_midas_t = now;
        }

        /* ── Update Shared Result ── */
        {
            std::lock_guard<std::mutex> lock(g_result_mutex);
            g_shared_result.z_tray_live = z_tray_live;
            g_shared_result.aruco_roi = aruco_roi;
            g_shared_result.aruco_roi_valid = aruco_roi_valid;
            if (!current_cup_bboxes.empty() || ran_midas_this_loop) {
                 g_shared_result.cup_bboxes = current_cup_bboxes;
            }
            g_shared_result.cup_heights_ema = cup_heights_ema;
            g_shared_result.cup_heights_valid = cup_heights_valid;
            g_shared_result.cup_diameters = cup_diameters;
            g_shared_result.cup_volumes = cup_volumes;
            g_shared_result.cup_zrims = cup_zrims;
            g_shared_result.cup_vol_valid = cup_vol_valid;
            if (!current_depth_norm.empty()) {
                g_shared_result.last_depth_norm = current_depth_norm.clone();
            }
            g_shared_result.aruco_results = aruco_results;
            g_shared_result.stats_total_frames = stats_total_frames;
            g_shared_result.stats_midas_runs = stats_midas_runs;
            
            /* Sync history for final report */
            g_shared_result.history_z_tray = history_z_tray;
            g_shared_result.history_cup_h = history_cup_h;
            g_shared_result.history_frames = history_frames;
        }
        g_result_cv.notify_one();
    }
}

/*---------------------------------------------------------------------------*/
/* Main live pipeline                                                         */
/*---------------------------------------------------------------------------*/

void run_live_pipeline(Camera*                 cam,
                       ArucoDetector&          aruco,
                       bool                    headless,
                       const nlohmann::json&   calib_data,
                       double                  marker_size_cm,
                       const std::vector<double>& active_poly_Kgeom,
                       const std::string&      active_cup_str,
                       const std::string&      screenshot_dir,
                       const std::string&      video_dir,
                       double                  true_height_cm,
                       MoildevApplicator*      moil,
                       GuiFusion*              gui,
                       bool                    no_anypoint,
                       int                     output_w,
                       int                     output_h)
{
    /* Read focal length from aruco camera matrix */
    std::cout << "[DEBUG-PL] run_live_pipeline entered\n"; std::cout.flush();
    std::cout << "[DEBUG-PL] aruco.camera_matrix empty=" << aruco.camera_matrix.empty() << "\n"; std::cout.flush();
    double focal_px = aruco.camera_matrix.empty() ? 800.0
                      : aruco.camera_matrix.at<double>(0, 0);
    std::cout << "[DEBUG-PL] focal_px=" << focal_px << "\n"; std::cout.flush();

    int ctype = calib_data.value("type", 1);
    std::cout << "[DEBUG-PL] ctype=" << ctype << "\n"; std::cout.flush();

    if (gui && !headless) {
        std::cout << "[GUI] Connected to GTK interface.\n";
    }

    /* Reset shared state */
    std::cout << "[DEBUG-PL] Resetting shared state...\n"; std::cout.flush();
    {
        std::lock_guard<std::mutex> lock(g_result_mutex);
        g_shared_result = InferenceResult();
        g_pipeline_running = true;
    }
    std::cout << "[DEBUG-PL] Shared state reset OK\n"; std::cout.flush();

    /* Start Inference Thread */
    std::cout << "[DEBUG-PL] Starting inference thread...\n"; std::cout.flush();
    std::thread inf_thread(inference_worker, cam, &aruco, calib_data,
                           marker_size_cm, active_poly_Kgeom, focal_px,
                           true_height_cm,
                           moil, no_anypoint, output_w, output_h, gui);
    std::cout << "[DEBUG-PL] Inference thread started\n"; std::cout.flush();

    /* Recording state */
    bool              is_recording = false;
    cv::VideoWriter   video_writer;
    std::vector<std::string> screenshot_paths;

    /* FPS counter state */
    double fps_display      = 0.0;
    int    fps_frame_count  = 0;
    double fps_last_time    = now_sec();

    try {
        while (g_pipeline_running) {
            cv::Mat frame = cam->get_frame();
            if (frame.empty()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }

            /* ── Normalize Lighting untuk display frame ────────────────── */
            if (gui != nullptr && gui->is_normalize_enabled()) {
                cv::Mat ycrcb;
                cv::cvtColor(frame, ycrcb, cv::COLOR_BGR2YCrCb);
                std::vector<cv::Mat> ch;
                cv::split(ycrcb, ch);
                cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
                clahe->apply(ch[0], ch[0]);
                cv::merge(ch, ycrcb);
                cv::cvtColor(ycrcb, frame, cv::COLOR_YCrCb2BGR);
            }

            /* ── Black & White Mode untuk display frame ────────────────── */
            if (gui != nullptr && gui->is_bw_enabled()) {
                cv::Mat gray;
                cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
                cv::cvtColor(gray, frame, cv::COLOR_GRAY2BGR);
            }
            /* ── Apply fisheye undistortion to display frame ── */
            if (moil != nullptr && !no_anypoint) {
                frame = moil->undistort(frame);
            }

            InferenceResult res;
            {
                /* Wait briefly for newest result if needed, but don't block display loop too long */
                std::unique_lock<std::mutex> lock(g_result_mutex);
                /* We don't necessarily want to wait if we want smooth display, 
                   but we need some data to draw. If no data yet, wait. */
                if (g_shared_result.stats_total_frames == 0) {
                    g_result_cv.wait_for(lock, std::chrono::milliseconds(100));
                }
                res = g_shared_result;
            }

            int h_frame = frame.rows, w_frame = frame.cols;
            cv::Mat disp = frame.clone();

            /* ── FPS Calculation ── */
            fps_frame_count++;
            double now_disp = now_sec();
            double elapsed  = now_disp - fps_last_time;
            if (elapsed >= 0.5) {  /* update setiap 0.5 detik */
                fps_display    = fps_frame_count / elapsed;
                fps_frame_count = 0;
                fps_last_time  = now_disp;
            }

            /* ── Build display frame ── */
            /* NOTE: aruco object is owned by inference thread — do NOT call aruco.detect()
             * or aruco.annotate_frame() here. Use the snapshot from g_shared_result instead.
             * Annotate manually using the corners stored in aruco_results. */
            for (const auto& ar : res.aruco_results) {
                if (ar.corners.size() == 4) {
                    std::vector<cv::Point> pts;
                    for (const auto& c : ar.corners)
                        pts.push_back(cv::Point((int)c.x, (int)c.y));
                    cv::polylines(disp, pts, true, cv::Scalar(0, 255, 0), 2);
                    cv::putText(disp, "ID:" + std::to_string(ar.id),
                                pts[0], cv::FONT_HERSHEY_SIMPLEX, 0.5,
                                cv::Scalar(0, 220, 0), 1);
                }
            }
            if (res.aruco_roi_valid)
                cv::rectangle(disp, res.aruco_roi, cv::Scalar(255, 140, 0), 1);
            for (auto& bbox : res.cup_bboxes)
                cv::rectangle(disp, bbox, cv::Scalar(0, 255, 80), 2);

            /* UI Scaling factor for Kakip */
            double S = std::max(0.5, (double)w_frame / 1000.0);

            /* Info panel */
            int panel_w = (int)(560 * S), panel_h = (int)(155 * S);
            cv::rectangle(disp, cv::Point(20, 20), cv::Point(20 + panel_w, 20 + panel_h), cv::Scalar(25, 25, 25), -1);
            cv::rectangle(disp, cv::Point(20, 20), cv::Point(20 + panel_w, 20 + panel_h), cv::Scalar(90, 90, 90), 2);

            /* MiDaS depth PiP */
            if (!res.last_depth_norm.empty()) {
                cv::Mat depth_color;
                cv::applyColorMap(res.last_depth_norm, depth_color, cv::COLORMAP_JET);
                int pip_h = (int)(h_frame / 3.2), pip_w = (int)(w_frame / 3.2);
                cv::Mat pip;
                cv::resize(depth_color, pip, cv::Size(pip_w, pip_h));
                int pip_margin = (int)(40 * S);
                int y1 = h_frame - pip_h - pip_margin, x1 = w_frame - pip_w - (int)(10 * S);
                if (x1 >= 0 && y1 >= 0 && x1+pip_w <= disp.cols && y1+pip_h <= disp.rows) {
                    pip.copyTo(disp(cv::Rect(x1, y1, pip_w, pip_h)));
                    cv::rectangle(disp, cv::Point(x1, y1), cv::Point(x1+pip_w, y1+pip_h), cv::Scalar(200,200,200), 2);
                    cv::putText(disp, "MiDaS Depth", cv::Point(x1+(int)(12*S), y1+(int)(28*S)), cv::FONT_HERSHEY_SIMPLEX, 0.5 * S, cv::Scalar(255,255,255), 2);
                }
            }

            /* Cup height text */
            cv::putText(disp, "CUP HEIGHTS", cv::Point((int)(40*S), (int)(45*S)), cv::FONT_HERSHEY_SIMPLEX, 0.55 * S, cv::Scalar(170,170,170), 2);
            {
                std::ostringstream oss;
                oss << std::fixed << std::setprecision(1) << "Z_tray: " << res.z_tray_live << " cm";
                cv::putText(disp, oss.str(), cv::Point((int)(230*S), (int)(40*S)), cv::FONT_HERSHEY_SIMPLEX, 0.5 * S, cv::Scalar(255,160,60), 2);
            }
            /* MiDaS status badge */
            {
                bool midas_on = g_midas_enabled.load();
                std::string midas_lbl = midas_on ? "MiDaS:ON" : "MiDaS:OFF";
                cv::Scalar  midas_clr = midas_on ? cv::Scalar(80,220,80) : cv::Scalar(60,60,200);
                cv::putText(disp, midas_lbl, cv::Point((int)(395*S), (int)(40*S)), cv::FONT_HERSHEY_SIMPLEX, 0.45 * S, midas_clr, 2);
            }

            int base_y = (int)(95 * S);
            for (int i = 0; i < 2; ++i) {
                int y_pos = base_y + i * (int)(50 * S);
                std::string lbl = "CUP " + std::to_string(i+1) + ":";
                if (res.cup_heights_valid[i] && res.cup_heights_ema[i] > 0) {
                    std::ostringstream oss;
                    oss << std::fixed << std::setprecision(1) << res.cup_heights_ema[i] << " cm";
                    cv::putText(disp, lbl, cv::Point((int)(40*S), y_pos-(int)(15*S)), cv::FONT_HERSHEY_SIMPLEX, 0.6 * S, cv::Scalar(200,200,200), 2);
                    cv::putText(disp, oss.str(), cv::Point((int)(115*S), y_pos+(int)(5*S)), cv::FONT_HERSHEY_DUPLEX, 1.3 * S, cv::Scalar(0,255,100), 3);
                    /* Z_rim dari inference thread (sudah dihitung dengan h_cup EMA) */
                    double z_rim_val = (i < 2) ? res.cup_zrims[i]
                                               : std::max(0.0, res.z_tray_live - res.cup_heights_ema[i]);
                    std::ostringstream oss2;
                    oss2 << "Z_rim: " << std::fixed << std::setprecision(1) << z_rim_val << " cm";
                    cv::putText(disp, oss2.str(), cv::Point((int)(280*S), y_pos-(int)(4*S)), cv::FONT_HERSHEY_SIMPLEX, 0.5 * S, cv::Scalar(100,255,100), 2);

                    if (res.cup_vol_valid[i]) {
                        /* Diameter & volume sudah dihitung di inference thread
                         * dengan frame undistorted + focal yang benar.
                         * Display thread cukup baca nilai yang sudah ada. */
                        double diameter  = res.cup_diameters[i];
                        double volume_ml = res.cup_volumes[i];

                        std::ostringstream ossD;
                        ossD << "D:" << std::fixed << std::setprecision(1) << diameter << "cm";
                        cv::putText(disp, ossD.str(), cv::Point((int)(395*S), y_pos-(int)(4*S)), cv::FONT_HERSHEY_SIMPLEX, 0.45 * S, cv::Scalar(200,200,255), 2);

                        std::ostringstream ossV;
                        if (volume_ml > 0) {
                            ossV << "V:" << std::fixed << std::setprecision(0) << volume_ml << "mL";
                            cv::putText(disp, ossV.str(), cv::Point((int)(480*S), y_pos-(int)(4*S)), cv::FONT_HERSHEY_SIMPLEX, 0.45 * S, cv::Scalar(255,220,80), 2);
                        } else {
                            cv::putText(disp, "V:--mL", cv::Point((int)(480*S), y_pos-(int)(4*S)), cv::FONT_HERSHEY_SIMPLEX, 0.45 * S, cv::Scalar(100,100,100), 2);
                        }
                    } else {
                        /* cup_vol_valid false: tampilkan D:-- V:-- */
                        cv::putText(disp, "D:-- cm", cv::Point((int)(395*S), y_pos-(int)(4*S)), cv::FONT_HERSHEY_SIMPLEX, 0.45 * S, cv::Scalar(100,100,100), 2);
                        cv::putText(disp, "V:--mL",  cv::Point((int)(480*S), y_pos-(int)(4*S)), cv::FONT_HERSHEY_SIMPLEX, 0.45 * S, cv::Scalar(100,100,100), 2);
                    }
                } else {
                    cv::putText(disp, lbl, cv::Point((int)(40*S), y_pos-(int)(15*S)), cv::FONT_HERSHEY_SIMPLEX, 0.6 * S, cv::Scalar(100,100,100), 2);
                    cv::putText(disp, "-- cm", cv::Point((int)(115*S), y_pos+(int)(5*S)), cv::FONT_HERSHEY_DUPLEX, 1.3 * S, cv::Scalar(70,70,70), 3);
                    cv::putText(disp, "Z_rim: -- cm", cv::Point((int)(280*S), y_pos-(int)(4*S)), cv::FONT_HERSHEY_SIMPLEX, 0.5 * S, cv::Scalar(100,100,100), 2);
                    cv::putText(disp, "D:-- cm", cv::Point((int)(395*S), y_pos-(int)(4*S)), cv::FONT_HERSHEY_SIMPLEX, 0.45 * S, cv::Scalar(100,100,100), 2);
                    cv::putText(disp, "V:--mL",  cv::Point((int)(480*S), y_pos-(int)(4*S)), cv::FONT_HERSHEY_SIMPLEX, 0.45 * S, cv::Scalar(100,100,100), 2);
                }
            }

            /* Recording indicator */
            if (is_recording) {
                auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
                if ((ms / 500) % 2 == 0) {
                    cv::circle(disp, cv::Point(w_frame-65, 25), 6, cv::Scalar(0,0,255), -1);
                    cv::putText(disp, "REC", cv::Point(w_frame-50, 30), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0,0,255), 2);
                }
                if (video_writer.isOpened()) video_writer.write(disp);
            }

            /* Status bar */
            int bar_h = (int)(45 * S);
            cv::rectangle(disp, cv::Point(0, h_frame - bar_h), cv::Point(w_frame, h_frame), cv::Scalar(15,15,15), -1);
            std::string bar_txt =
                "ArUco: " + std::string(res.z_tray_live > 0 ? "OK" : "X") +
                " | YOLO: " + std::string(res.cup_bboxes.empty() ? "X" : "OK");
            bar_txt += " | [V]=Rec [S]=Shot [M]=MiDaS [Q]=Quit";
            /* FPS di status bar */
            {
                std::ostringstream oss_fps_bar;
                oss_fps_bar << std::fixed << std::setprecision(1) << "  FPS:" << fps_display;
                bar_txt += oss_fps_bar.str();
            }
            cv::putText(disp, bar_txt, cv::Point((int)(20*S), h_frame - (int)(15*S)), cv::FONT_HERSHEY_SIMPLEX, 0.5 * S, cv::Scalar(130,200,130), 2);

            if (ctype == 5) {
                /* Tampilkan TARGET MENU di BAWAH panel, bukan di dalam row Cup 2 */
                int target_y = 20 + (int)(155 * S) + (int)(30 * S);
                cv::putText(disp, "TARGET MENU: " + active_cup_str + " cm", cv::Point((int)(40*S), target_y), cv::FONT_HERSHEY_SIMPLEX, 0.65 * S, cv::Scalar(0,255,255), 2);
            }

            if (!headless) {
                if (gui) {
                    gui->update_image(disp);
                }
                int key = -1;
                if (gui) key = gui->get_key();
                
                if (key == 27 || key == 'q') {
                    g_pipeline_running = false;
                    break;
                }
                else if (key == 's') {
                    /* Screenshot — 's' is reserved for screenshot only */
                    std::string ts  = ts_filename();
                    std::string ss_path = screenshot_dir + "/fusion_" + ts + ".jpg";
                    cv::imwrite(ss_path, disp);
                    screenshot_paths.push_back(ss_path);
                    std::cout << "[SHOT] Screenshot saved: " << ss_path << "\n";
                } else if (key == 'm' || key == 'M') {
                    /* Toggle MiDaS on/off */
                    bool prev = g_midas_enabled.load();
                    g_midas_enabled.store(!prev);
                    std::cout << "[MIDAS] " << (prev ? "Disabled" : "Enabled") << "\n";
                } else if (key == 'v') {
                    /* Video recording toggle — changed from 'r' to 'v' to free 'r' for reset */
                    if (!is_recording) {
                        std::string ts  = ts_filename();
                        std::string vid_path = video_dir + "/fusion_" + ts + ".mp4";
                        double fps = 30.0;
                        int fourcc = cv::VideoWriter::fourcc('m','p','4','v');
                        video_writer.open(vid_path, fourcc, fps, cv::Size(w_frame, h_frame));
                        is_recording = true;
                        std::cout << "[REC] Recording started: " << vid_path << "\n";
                    } else {
                        is_recording = false;
                        video_writer.release();
                        std::cout << "[REC] Recording stopped.\n";
                    }
                }
            } else {
                /* Headless: print data periodically */
                static double last_print = 0;
                double now = now_sec();
                if (now - last_print > 0.5) {
                    std::cout << std::fixed << std::setprecision(2) << "[DATA] z_tray=" << res.z_tray_live << "cm";
                    for (int i = 0; i < 2; ++i) {
                        std::cout << " | C" << (i+1) << ":";
                        if (res.cup_heights_valid[i]) std::cout << res.cup_heights_ema[i] << "cm";
                        else std::cout << "--";
                    }
                    std::cout << " | Inf: " << res.stats_total_frames << " frames\n";
                    last_print = now;
                }
                /* Need a way to quit headless? usually Ctrl+C or timeout. 
                   But let's assume it runs until external termination or 
                   we could add a simple timer. */
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in live pipeline: " << e.what() << "\n";
        g_pipeline_running = false;
    }

    if (inf_thread.joinable()) inf_thread.join();

    /* Final results for report */
    InferenceResult final_res;
    {
        std::lock_guard<std::mutex> lock(g_result_mutex);
        final_res = g_shared_result;
    }

    if (video_writer.isOpened()) video_writer.release();

    std::cout << "\n[DONE] Pipeline closed. Generating session report...\n";

    SessionReporter reporter("results/report");
    reporter.generate(
        calib_data,
        marker_size_cm,
        focal_px,
        final_res.stats_total_frames,
        final_res.stats_midas_runs,
        final_res.history_z_tray,
        final_res.history_cup_h,
        final_res.history_frames,
        screenshot_paths
    );
}
