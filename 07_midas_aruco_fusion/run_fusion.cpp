/*******************************************************************************
 * run_fusion.cpp
 * ArUco + MiDaS + YOLOv8 Cup Height Estimator — C++ version for RZ/V2H
 * Port of 07_midas_aruco_fusion/run_fusion.py
 *
 * Usage:
 *   ./run_fusion_cpp [options]
 *
 * Options:
 *   --camera          <int>    Camera index (default: 0)
 *   --cap-width       <int>    Capture width  (default: 2592)
 *   --cap-height      <int>    Capture height (default: 1944)
 *   --headless                 No display (terminal mode)
 *   --marker-size     <f>      Physical ArUco marker side in cm (default: 5.0)
 *   --calibrate       <int>    0=Live, 1-7=Calibration mode (default: 0)
 *   --true-height     <f>      Reference cup height in cm
 *   --true-height-2   <f>      Second height in cm
 *   --target-cup      <f>      LIVE: target menu cup height for type-5
 *   --n-positions     <int>    Number of Z positions for grid calib (default: 3)
 *   --cup-profile     <str>    Profile name for calibration file
 *   --fisheye                  Enable fisheye undistortion via Moildev
 *   --moil-camera-name <str>   Camera profile in camera_parameters.json
 *   --moil-pitch      <f>      Anypoint pitch in degrees (default: 0.0)
 *   --moil-yaw        <f>      Anypoint yaw in degrees   (default: 0.0)
 *   --moil-roll       <f>      Anypoint roll in degrees  (default: 0.0)
 *   --moil-zoom       <f>      Zoom factor               (default: 1.4)
 *   --moil-mode       <int>    1=AnyPointM 2=AnyPointM2 (default: 2)
 *   --no-anypoint              Fisheye ON but skip anypoint remap
 *   --manual-exposure <int>    Manual exposure value (0=auto)
 *   --calib-params    <str>    Camera calibration yml (default: calibration_params.yml)
 ******************************************************************************/

#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>
#include <filesystem>
#include <linux/drpai.h>
#include <opencv2/opencv.hpp>

#include <detections/ai.h>

#include <camera/camera.h>

#include "core/calibration_routines.hpp"
#include "core/calibration_storage.hpp"
#include "core/live_pipeline.hpp"
#include "core/moil_undistorter.hpp"
#include "core/gui_fusion.hpp"
#include "aruco_detector.hpp"

#include <gtk/gtk.h>
#include <thread>


namespace fs = std::filesystem;

/* ── Simple argument parsing ─────────────────────────────────────────────── */
struct Args {
    int         camera            = 0;
    int         cap_width         = 2592;
    int         cap_height        = 1944;
    int         output_width      = 0;   // 0 = no upscaling
    int         output_height     = 0;
    bool        headless          = false;
    float       marker_size       = 5.0f;
    int         calibrate         = 0;
    double      true_height       = 0.0;
    double      true_height_2     = 0.0;
    double      target_cup        = -1.0;
    int         n_positions       = 3;
    std::string cup_profile       = "default";
    bool        fisheye           = false;
    std::string moil_camera_name  = "syue_7730v1_6";
    float       moil_pitch        = 0.0f;
    float       moil_yaw          = 0.0f;
    float       moil_roll         = 0.0f;
    float       moil_zoom         = 1.0f;
    int         moil_mode         = 2;
    bool        no_anypoint       = false;
    int         manual_exposure   = 0;
    std::string calib_params      = "../calibration_params.yml";
    std::string calib_path        = "calibration.json";
    std::string cam_params_json   = "camera_parameters.json";
};

static void print_usage(const char* prog)
{
    std::cout <<
        "Usage: " << prog << " [options]\n"
        "  --camera <int>              Camera index (default: 0)\n"
        "  --cap-width <int>           Capture resolution width  (default: 2592)\n"
        "  --cap-height <int>          Capture resolution height (default: 1944)\n"
        "  --headless                  No display\n"
        "  --marker-size <float>       ArUco marker side length in cm (default: 5.0)\n"
        "  --calibrate <int>           0=Live 1-7=Calibration mode\n"
        "  --true-height <float>       Cup reference height (cm)\n"
        "  --true-height-2 <f>         Second cup height (cm)\n"
        "  --target-cup <float>        LIVE type-5: target cup height (cm)\n"
        "  --n-positions <int>         Z positions for grid calib (default: 3)\n"
        "  --cup-profile <str>         Profile name for calib file\n"
        "  --fisheye                   Enable fisheye undistortion via Moildev\n"
        "  --moil-camera-name <str>    Camera profile in camera_parameters.json\n"
        "                              (default: syue_7730v1_6)\n"
        "  --moil-pitch <float>        Anypoint pitch in degrees  (default: 0.0)\n"
        "  --moil-yaw   <float>        Anypoint yaw in degrees    (default: 0.0)\n"
        "  --moil-roll  <float>        Anypoint roll in degrees   (default: 0.0)\n"
        "  --moil-zoom  <float>        Anypoint zoom factor       (default: 1.0)\n"
        "  --moil-mode  <int>          1=AnyPointM 2=AnyPointM2  (default: 2)\n"
        "  --no-anypoint               Fisheye ON but skip anypoint remap\n"
        "  --output-width  <int>       Upscale output to this width  (0=off)\n"
        "  --output-height <int>       Upscale output to this height (0=off)\n"
        "  --manual-exposure <int>     Manual exposure (0=auto)\n"
        "  --calib-params <path>       Camera calibration yml file\n";
}

static Args parse_args(int argc, char* argv[])
{
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg == "--help" || arg == "-h") { print_usage(argv[0]); std::exit(0); }
        else if (arg == "--camera"            && i+1 < argc) a.camera           = std::atoi(argv[++i]);
        else if (arg == "--cap-width"          && i+1 < argc) a.cap_width        = std::atoi(argv[++i]);
        else if (arg == "--cap-height"         && i+1 < argc) a.cap_height       = std::atoi(argv[++i]);
        else if (arg == "--headless")                         a.headless         = true;
        else if (arg == "--marker-size"        && i+1 < argc) a.marker_size      = std::atof(argv[++i]);
        else if (arg == "--calibrate"          && i+1 < argc) a.calibrate        = std::atoi(argv[++i]);
        else if (arg == "--true-height"        && i+1 < argc) a.true_height      = std::atof(argv[++i]);
        else if (arg == "--true-height-2"      && i+1 < argc) a.true_height_2    = std::atof(argv[++i]);
        else if (arg == "--target-cup"         && i+1 < argc) a.target_cup       = std::atof(argv[++i]);
        else if (arg == "--n-positions"        && i+1 < argc) a.n_positions      = std::atoi(argv[++i]);
        else if (arg == "--cup-profile"        && i+1 < argc) a.cup_profile      = argv[++i];
        else if (arg == "--fisheye")                          a.fisheye          = true;
        else if (arg == "--moil-camera-name"   && i+1 < argc) a.moil_camera_name = argv[++i];
        else if (arg == "--moil-pitch"         && i+1 < argc) a.moil_pitch       = std::atof(argv[++i]);
        else if (arg == "--moil-yaw"           && i+1 < argc) a.moil_yaw         = std::atof(argv[++i]);
        else if (arg == "--moil-roll"          && i+1 < argc) a.moil_roll        = std::atof(argv[++i]);
        else if (arg == "--moil-zoom"          && i+1 < argc) a.moil_zoom        = std::atof(argv[++i]);
        else if (arg == "--moil-mode"          && i+1 < argc) a.moil_mode        = std::atoi(argv[++i]);
        else if (arg == "--no-anypoint")                      a.no_anypoint      = true;
        else if (arg == "--output-width"       && i+1 < argc) a.output_width     = std::atoi(argv[++i]);
        else if (arg == "--output-height"      && i+1 < argc) a.output_height    = std::atoi(argv[++i]);
        else if (arg == "--manual-exposure"    && i+1 < argc) a.manual_exposure  = std::atoi(argv[++i]);
        else if (arg == "--calib-params"       && i+1 < argc) a.calib_params     = argv[++i];
        else {
            std::cerr << "[WARN] Unknown argument: " << arg << "\n";
        }
    }

    /* Build calibration JSON path based on profile / fisheye */
    if (a.fisheye) {
        a.calib_path = "calibration_fisheye_" + a.cup_profile + ".json";
    } else if (a.cup_profile != "default") {
        a.calib_path = "calibration_" + a.cup_profile + ".json";
    }

    return a;
}

/* ── Validate required args for each calibration mode ───────────────────── */
static bool validate_args(const Args& a)
{
    if (a.calibrate > 0) {
        if ((a.calibrate == 1 || a.calibrate == 3 ||
             a.calibrate == 4 || a.calibrate == 5)
            && a.true_height <= 0.0) {
            std::cerr << "[ERROR] Calibration mode " << a.calibrate
                      << " requires --true-height\n";
            return false;
        }
        if ((a.calibrate == 2 || a.calibrate == 6 || a.calibrate == 7)
            && (a.true_height <= 0.0 || a.true_height_2 <= 0.0)) {
            std::cerr << "[ERROR] Calibration mode " << a.calibrate
                      << " requires --true-height AND --true-height-2\n";
            return false;
        }
        if (a.calibrate < 1 || a.calibrate > 7) {
            std::cerr << "[ERROR] Unknown calibration mode: " << a.calibrate << "\n";
            return false;
        }
    }
    return true;
}

/* ══════════════════════════════════════════════════════════════════════════ */

int main(int argc, char* argv[])
{
    unsigned long OCA_list[16];
    for (int i=0; i < 16; i++) {
        OCA_list[i] = 0;
    }
    OCA_Activate( &OCA_list[0] );

    std::cout << "=======================================================\n";
    std::cout << "  ArUco + MiDaS + YOLOv8 | Cup Height Estimator (V2H)\n";
    std::cout << "=======================================================\n\n";

    /* Parse arguments */
    Args args = parse_args(argc, argv);
    if (!validate_args(args)) return 1;

    /* Ensure output directories exist */
    fs::create_directories("results/report");
    fs::create_directories("results/video");
    fs::create_directories("results/live_cam");

    /* ── Initialize AI singleton ───────────────────────────────────────── */
    std::cout << "[INIT] Initializing AI (DRP-AI cup detector + MiDaS)...\n";
    AI* ai = AI::get_instance();
    (void)ai;  /* triggers singleton constructor */
    std::cout << "[INIT] AI initialized.\n\n";

    /* ── Initialize ArUco ─────────────────────────────────────────────── */
    std::cout << "[INIT] Loading ArucoDetector (calibration: "
              << args.calib_params << ")...\n";
    ArucoDetector aruco(args.marker_size, "DICT_4X4_50", args.calib_params);
    std::cout << "[INIT] ArucoDetector ready.\n\n";

    /* ── Open camera ──────────────────────────────────────────────────── */
    std::cout << "[INIT] Opening camera index " << args.camera << "...\n";
    Camera cam(args.camera, /*autostart=*/true);
    std::cout << "[INIT] Camera ready (threaded capture).\n\n";

    /* ── Initialize Moildev fisheye undistorter (only if --fisheye) ──── */
    std::unique_ptr<MoilUndistorter>  moil_undistorter;

    if (args.fisheye) {
        std::cout << "[MOIL] Initializing fisheye undistorter...\n";
        std::cout << "[MOIL] Camera profile : " << args.moil_camera_name << "\n";
        std::cout << "[MOIL] Mode=" << args.moil_mode
                  << "  pitch=" << args.moil_pitch
                  << "  yaw="   << args.moil_yaw
                  << "  roll="  << args.moil_roll
                  << "  zoom="  << args.moil_zoom << "\n";
        try {
            moil_undistorter = std::make_unique<MoilUndistorter>(
                args.cam_params_json,
                args.moil_camera_name,
                args.moil_pitch,
                args.moil_yaw,
                args.moil_roll,
                args.moil_zoom,
                args.moil_mode,
                args.cap_width,   // actual camera frame width
                args.cap_height,  // actual camera frame height
                args.output_width,
                args.output_height
            );

            /* Override ArUco camera matrix with Moildev focal length */
            /* Use cap_width/cap_height as the expected stream resolution */
            cv::Mat new_K = moil_undistorter->build_aruco_camera_matrix(
                args.cap_width, args.cap_height);
            aruco.camera_matrix = new_K;
            std::cout << "[MOIL] ArUco camera matrix overridden:"
                      << " fx=" << new_K.at<double>(0,0)
                      << " fy=" << new_K.at<double>(1,1)
                      << " cx=" << new_K.at<double>(0,2)
                      << " cy=" << new_K.at<double>(1,2) << "\n";

            std::cout << "[MOIL] Fisheye undistorter ready.\n\n";
        } catch (const std::exception& e) {
            std::cerr << "[MOIL ERROR] Failed to initialize MoilUndistorter: "
                      << e.what() << "\n";
            std::cerr << "[MOIL] Continuing WITHOUT fisheye undistortion.\n\n";
            moil_undistorter.reset();
        }
    }

    /* ── Initialize GUI ───────────────────────────────────────────────── */
    std::shared_ptr<GuiFusion> gui;
    if (!args.headless) {
        gtk_init(&argc, &argv);
        gui = std::make_shared<GuiFusion>(moil_undistorter.get(), args.headless, args.manual_exposure);
        std::cout << "[GUI] GTK3 Interface Active\n\n";
    }

    /* ── Worker Thread ───────────────────────────────────────────────── */
    auto worker_thread = [&]() {
        /* ── Calibration storage ─────────────────────────────────────────── */
        CalibrationStorage storage(args.calib_path);

        /* ── Calibration path ─────────────────────────────────────────────── */
        nlohmann::json calib_data;

        if (args.calibrate > 0) {
            if (gui) {
                // Determine calib name
                std::string cname = "Unknown";
                switch(args.calibrate) {
                    case 1: cname = "1-Point"; break;
                    case 2: cname = "2-Point"; break;
                    case 3: cname = "Z-Grid"; break;
                    case 4: cname = "BBox"; break;
                    case 5: cname = "Geometric"; break;
                    case 6: cname = "Bilateral"; break;
                    case 7: cname = "Analytic"; break;
                }
                gui->enter_setup_mode(cname);
                std::cout << "[SETUP] Waiting for user to configure camera and start " << cname << " calibration...\n";
                gui->wait_for_calibration_ready();
            }

            cv::VideoCapture cap_calib(args.camera);
            cap_calib.set(cv::CAP_PROP_FRAME_WIDTH, 640);
            cap_calib.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

            /* Run the selected calibration mode */
            switch (args.calibrate) {
                case 1:
                case 2:
                    calib_data = CalibRoutines::run_calib_1p_2p(
                        cap_calib, aruco, storage, args.headless,
                        args.true_height, args.true_height_2, args.calibrate);
                    break;
                case 3:
                    calib_data = CalibRoutines::run_calib_zgrid(
                        cap_calib, aruco, storage, args.headless,
                        args.true_height, args.n_positions);
                    break;
                case 4:
                    calib_data = CalibRoutines::run_calib_bbox(
                        cap_calib, aruco, storage, args.headless, args.true_height);
                    break;
                case 5:
                    calib_data = CalibRoutines::run_calib_geom(
                        cap_calib, aruco, storage, args.headless,
                        args.true_height, args.n_positions);
                    break;
                case 6:
                    calib_data = CalibRoutines::run_calib_bilateral(
                        cap_calib, aruco, storage, args.headless,
                        args.true_height, args.true_height_2, args.n_positions);
                    break;
                case 7:
                    calib_data = CalibRoutines::run_calib_analytic(
                        cap_calib, aruco, storage, args.headless,
                        args.true_height, args.true_height_2);
                    break;
                default:
                    std::cerr << "[ERROR] Unknown calibration mode.\n";
                    if (gui) gui->queue_key(27);
                    return;
            }
            cap_calib.release();

            if (calib_data.empty() || calib_data.is_null()) {
                std::cerr << "[CALIB] Calibration failed or aborted.\n";
                if (gui) gui->queue_key(27);
                return;
            }
            std::cout << "[CALIB] Calibration complete. Entering LIVE mode...\n\n";

        } else {
            /* Load existing calibration */
            calib_data = storage.load();
            if (calib_data.empty()) {
                std::cerr << "[ERROR] " << args.calib_path
                          << " not found! Calibrate first, e.g.:\n"
                          << "  ./run_fusion_cpp --calibrate 5 --true-height 7.6\n";
                if (gui) gui->queue_key(27);
                return;
            }
        }

        /* ── Resolve active poly_Kgeom for type-5 ────────────────────────── */
        std::vector<double> active_poly_Kgeom = {1.0};
        std::string         active_cup_str    = "LEGACY (1 Profile)";

        if (calib_data.value("type", 0) == 5 && calib_data.contains("profiles")) {
            auto profiles = calib_data["profiles"];
            if (args.target_cup > 0.0) {
                std::ostringstream key_ss;
                key_ss << args.target_cup;
                std::string key = key_ss.str();
                if (profiles.contains(key)) {
                    active_poly_Kgeom =
                        profiles[key]["poly_Kgeom"].get<std::vector<double>>();
                    active_cup_str = key;
                } else if (!profiles.empty()) {
                    /* Fall back to first profile */
                    auto it = profiles.begin();
                    active_cup_str    = it.key();
                    active_poly_Kgeom =
                        it.value()["poly_Kgeom"].get<std::vector<double>>();
                }
            } else if (!profiles.empty()) {
                auto it = profiles.begin();
                active_cup_str    = it.key();
                active_poly_Kgeom =
                    it.value()["poly_Kgeom"].get<std::vector<double>>();
            }
        } else if (calib_data.contains("poly_Kgeom")) {
            active_poly_Kgeom =
                calib_data["poly_Kgeom"].get<std::vector<double>>();
        }

        if (gui) {
            gui->set_status_calib("Mode: Live (" + active_cup_str + ")");
        }

        /* ── Run live pipeline ────────────────────────────────────────────── */
        run_live_pipeline(
            &cam, aruco,
            args.headless,
            calib_data,
            args.marker_size,
            active_poly_Kgeom,
            active_cup_str,
            "results/live_cam",
            "results/video",
            moil_undistorter.get(),
            gui.get(),
            args.no_anypoint,
            args.output_width,
            args.output_height
        );
        
        if (gui) {
            gui->queue_key(27);
        }
    };

    std::thread bg_thread(worker_thread);

    if (!args.headless && gui) {
        gui->show_all();
        gtk_main();
    }
    
    bg_thread.join();

    return 0;
}
