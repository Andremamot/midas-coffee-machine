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
#include <signal.h>
#include <execinfo.h>
#include <unistd.h>

#include <detections/ai.h>

#include <camera/camera.h>

#include "core/calibration_routines.hpp"
#include "core/calibration_storage.hpp"
#include "core/live_pipeline.hpp"
#include "core/moildev_applicator.hpp"
#include "core/gui_fusion.hpp"
#include "aruco_detector.hpp"

#include <gtk/gtk.h>
#include <thread>
#include <atomic>

/* ── GUI sync: set true by gtk_main's first idle, so worker thread
 * can know the event loop is running before posting g_idle_add()  ── */
std::atomic<bool> g_gui_ready{false};

namespace fs = std::filesystem;

/* ── SIGSEGV handler: print backtrace ───────────────────────────────────── */
static void sigsegv_handler(int sig)
{
    void* array[32];
    int size = backtrace(array, 32);
    char msg[] = "\n[FATAL] Caught signal SIGSEGV — backtrace:\n";
    write(STDERR_FILENO, msg, sizeof(msg) - 1);
    backtrace_symbols_fd(array, size, STDERR_FILENO);
    char done[] = "[FATAL] End of backtrace. Exiting.\n";
    write(STDERR_FILENO, done, sizeof(done) - 1);
    _exit(1);
}

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

    /* Install SIGSEGV handler for debugging */
    struct sigaction sa;
    sa.sa_handler = sigsegv_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = SA_RESTART;
    sigaction(SIGSEGV, &sa, nullptr);

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
    std::cout << "[INIT] AI initialized.\n";

    /* ── Warm up DRP-AI in main thread ──────────────────────────────────── */
    /* DRPQueue is lazy-initialized on first detect() call.
     * If detect() is first called from a secondary thread, the DRPQueue worker
     * thread crashes because DRP-AI hardware context is not inherited.
     * Solution: trigger the first detect() here in main() so the DRPQueue
     * is fully initialized in the correct thread context before inference thread. */
    std::cout << "[INIT] Warming up DRP-AI in main thread...\n";
    {
        cv::Mat dummy(64, 64, CV_8UC3, cv::Scalar(0, 0, 0));
        try {
            ai->cup_detector->detect(dummy);
            ai->midas_estimator->inference(dummy);
        } catch (...) { /* ignore dummy-frame errors */ }
    }
    std::cout << "[INIT] DRP-AI warm-up complete.\n\n";

    /* ── Initialize ArUco ─────────────────────────────────────────────── */
    std::cout << "[INIT] Loading ArucoDetector (calibration: "
              << args.calib_params << ")...\n";
    ArucoDetector aruco(args.marker_size, "DICT_4X4_50", args.calib_params);
    std::cout << "[INIT] ArucoDetector ready.\n\n";

    /* ── Open camera ──────────────────────────────────────────────────── */
    std::cout << "[INIT] Opening camera index " << args.camera << "...\n";
    Camera cam(args.camera, /*autostart=*/true, /*manual_exposure=*/args.manual_exposure);
    std::cout << "[INIT] Camera ready (threaded capture).\n\n";

    if (args.manual_exposure > 0) {
        // Konversi dari nilai raw (1000-10000) ke skala Smart Exposure (1.0-10.0)
        float smart_val = std::max(1.0f, std::min(10.0f, static_cast<float>(args.manual_exposure) / 1000.0f));
        std::cout << "[CAM] Menggunakan Smart Exposure: " << smart_val << " (raw=" << args.manual_exposure << ")\n";
        cam.set_smart_exposure(smart_val);
    }

    /* ── Initialize Moildev fisheye undistorter (only if --fisheye) ──── */
    std::unique_ptr<MoildevApplicator>  moil_undistorter;

    if (args.fisheye) {
        std::cout << "[MOIL] Initializing fisheye undistorter...\n";
        std::cout << "[MOIL] Camera profile : " << args.moil_camera_name << "\n";
        std::cout << "[MOIL] Mode=" << args.moil_mode
                  << "  pitch=" << args.moil_pitch
                  << "  yaw="   << args.moil_yaw
                  << "  roll="  << args.moil_roll
                  << "  zoom="  << args.moil_zoom << "\n";
        try {
            moil_undistorter = std::make_unique<MoildevApplicator>(
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

            /* NOTE: ArUco camera matrix TIDAK di-override dari Moildev.
             * Tetap gunakan matrix dari calibration_params.yml yang sudah dikalibrasi
             * dan memberikan Z_tray yang akurat. Override Moildev menyebabkan Z_tray 2x salah. */
            std::cout << "[MOIL] Fisheye undistorter ready. ArUco menggunakan matrix dari calibration_params.yml.\n\n";
        } catch (const std::exception& e) {
            std::cerr << "[MOIL ERROR] Failed to initialize MoildevApplicator: "
                      << e.what() << "\n";
            std::cerr << "[MOIL] Continuing WITHOUT fisheye undistortion.\n\n";
            moil_undistorter.reset();
        }
    }

    /* ── Initialize GUI ───────────────────────────────────────────────── */
    std::shared_ptr<GuiFusion> gui;
    if (!args.headless) {
        /* Log display environment — sangat berguna untuk debugging di Renesas Yocto */
        const char* disp_env    = getenv("DISPLAY");
        const char* wayland_env = getenv("WAYLAND_DISPLAY");
        const char* xdg_env     = getenv("XDG_RUNTIME_DIR");
        std::cout << "[ENV] DISPLAY         = " << (disp_env    ? disp_env    : "(not set)") << "\n";
        std::cout << "[ENV] WAYLAND_DISPLAY = " << (wayland_env ? wayland_env : "(not set)") << "\n";
        std::cout << "[ENV] XDG_RUNTIME_DIR = " << (xdg_env     ? xdg_env     : "(not set)") << "\n";

        if (!gtk_init_check(&argc, &argv)) {
            std::cerr << "[WARN] GTK init failed — no display available (DISPLAY/WAYLAND_DISPLAY not set?).\n";
            std::cerr << "[WARN] Falling back to headless mode automatically.\n";
            std::cerr << "[HINT] On Renesas Yocto+Wayland: export WAYLAND_DISPLAY=wayland-0 && export XDG_RUNTIME_DIR=/run/user/0\n";
            std::cerr << "[HINT] Then re-run the application.\n";
            args.headless = true;
        } else {
            gui = std::make_shared<GuiFusion>(moil_undistorter.get(), args.headless, args.manual_exposure, &cam);
            std::cout << "[GUI] GTK3 Interface created (Wayland/X11 backend active).\n\n";
        }
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

                /* Stream kamera ke GUI selama menunggu user tekan "Start Calibration".
                 * Tanpa ini, GUI kosong dan user tidak bisa lihat kamera untuk
                 * adjust exposure/anypoint sebelum kalibrasi. */
                while (!gui->is_calibration_ready() && gui->is_alive()) {
                    cv::Mat raw = cam.get_frame();
                    if (!raw.empty()) {
                        cv::Mat preview = raw;
                        if (moil_undistorter) {
                            preview = moil_undistorter->undistort(raw);
                        }
                        /* Overlay teks setup mode */
                        cv::putText(preview,
                            "SETUP: " + cname + " Calibration",
                            cv::Point(20, 50),
                            cv::FONT_HERSHEY_SIMPLEX, 1.2,
                            cv::Scalar(0, 220, 255), 3);
                        cv::putText(preview,
                            "Adjust exposure & anypoint, then click [Start Calibration]",
                            cv::Point(20, 95),
                            cv::FONT_HERSHEY_SIMPLEX, 0.65,
                            cv::Scalar(200, 200, 200), 2);
                        gui->update_image(preview);
                    }
                    std::this_thread::sleep_for(std::chrono::milliseconds(33)); // ~30fps
                }
                if (!gui->is_alive()) return;  /* user tutup window */

            }

            /* Stop camera stream utama agar tidak conflict dengan cap_calib.
             * Camera class menggunakan VideoCapture internal — membuka camera
             * index yang sama dari dua VideoCapture sekaligus menyebabkan
             * frame corrupt atau cap_calib.read() selalu gagal. */
            cam.stop_camera();

            cv::VideoCapture cap_calib(args.camera);
            /* KRITIS: Gunakan resolusi penuh yang sama dengan live pipeline.
             * Python get_frame() juga menggunakan resolusi 2592x1944 + Moildev undistortion.
             * Kalibrasi dengan 640x480 raw akan menghasilkan bbox_h dan focal yang
             * berbeda dari saat inferensi → K_geom salah. */
            cap_calib.set(cv::CAP_PROP_FRAME_WIDTH,  2592);
            cap_calib.set(cv::CAP_PROP_FRAME_HEIGHT, 1944);

            /* Update aruco.camera_matrix ke focal moildev sebelum kalibrasi,
             * sama seperti Python run_fusion.py baris 234-239. */
            if (moil_undistorter) {
                /* Ambil frame dummy untuk mengetahui resolusi output */
                cv::Mat dummy_frame;
                cap_calib.read(dummy_frame);
                if (!dummy_frame.empty()) {
                    cv::Mat dummy_undist = moil_undistorter->undistort(dummy_frame);
                    aruco.camera_matrix = moil_undistorter->build_aruco_camera_matrix(
                        dummy_undist.cols, dummy_undist.rows);
                    aruco.dist_coeffs = cv::Mat::zeros(1, 5, CV_64F);
                    std::cout << "[CALIB] aruco.camera_matrix updated to moildev focal: "
                              << "fx=" << aruco.camera_matrix.at<double>(0,0) << " px\n";
                }
            }

            /* Run the selected calibration mode */
            switch (args.calibrate) {
                case 1:
                case 2:
                    calib_data = CalibRoutines::run_calib_1p_2p(
                        cap_calib, aruco, storage, args.headless,
                        args.true_height, args.true_height_2, args.calibrate,
                        moil_undistorter.get(), gui.get());
                    break;
                case 3:
                    calib_data = CalibRoutines::run_calib_zgrid(
                        cap_calib, aruco, storage, args.headless,
                        args.true_height, args.n_positions,
                        moil_undistorter.get(), gui.get());
                    break;
                case 4:
                    calib_data = CalibRoutines::run_calib_bbox(
                        cap_calib, aruco, storage, args.headless, args.true_height,
                        moil_undistorter.get(), gui.get());
                    break;
                case 5:
                    calib_data = CalibRoutines::run_calib_geom(
                        cap_calib, aruco, storage, args.headless,
                        args.true_height, args.n_positions,
                        moil_undistorter.get(), gui.get());
                    break;
                case 6:
                    calib_data = CalibRoutines::run_calib_bilateral(
                        cap_calib, aruco, storage, args.headless,
                        args.true_height, args.true_height_2, args.n_positions,
                        moil_undistorter.get(), gui.get());
                    break;
                case 7:
                    calib_data = CalibRoutines::run_calib_analytic(
                        cap_calib, aruco, storage, args.headless,
                        args.true_height, args.true_height_2,
                        moil_undistorter.get(), gui.get());
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
        std::cout << "[DEBUG] Resolving poly_Kgeom...\n";
        std::vector<double> active_poly_Kgeom = {1.0};
        std::string         active_cup_str    = "LEGACY (1 Profile)";

        if (calib_data.value("type", 0) == 5 && calib_data.contains("profiles")) {
            auto profiles = calib_data["profiles"];
            std::cout << "[DEBUG] type=5, profiles size=" << profiles.size() << "\n";
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

        std::cout << "[DEBUG] poly_Kgeom resolved: [" << active_cup_str
                  << "] size=" << active_poly_Kgeom.size() << "\n";

        if (gui) {
            std::cout << "[DEBUG] Setting GUI status...\n";
            gui->set_status_calib("Mode: Live (" + active_cup_str + ")");
        }

        std::cout << "[DEBUG] Entering run_live_pipeline...\n";
        std::cout.flush();

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
            args.true_height,       /* fallback jika poly_Kgeom extrapolasi negatif */
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
        std::cout << "[GUI] Window shown — starting gtk_main() event loop...\n";
        /* Signal to worker thread that the GTK event loop is about to run.
         * g_idle_add() called from bg_thread before gtk_main() is risky on
         * Wayland — the idle source may never fire. We signal readiness here. */
        g_idle_add([](gpointer) -> gboolean {
            g_gui_ready.store(true);
            return G_SOURCE_REMOVE;
        }, nullptr);
        gtk_main();
    }
    
    bg_thread.join();

    return 0;
}
