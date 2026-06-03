/*******************************************************************************
 * core/moildev_applicator.cpp
 *
 * Implementasi MoildevApplicator — diport dari MoildevApplicator (unicorn-solution).
 *
 * Perubahan utama dari implementasi lama:
 *   1. LUT Alpha-Rho via Horner's Method (dari initializeAlphaRhoTables)
 *   2. getAlphaBeta(x,y) — konversi koordinat klik ke sudut (rho→alpha lookup)
 *   3. Formula focal length yang benar: param5_/calib_ratio_ (bukan empiris)
 *   4. INTER_LANCZOS4 untuk kualitas undistortion lebih baik
 *   5. Sync member pitch/yaw/roll/zoom agar AnypointController bekerja
 ******************************************************************************/

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif

#include "moildev_applicator.hpp"

// Full moildev CPU engine — hanya di-include di .cpp ini
#include "../../module/moil/lib/include/moildev_cpu.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include <opencv2/imgproc.hpp>  // cv::remap, INTER_LANCZOS4, GaussianBlur


#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ── Constructor ───────────────────────────────────────────────────────────────

MoildevApplicator::MoildevApplicator(const std::string &json_path,
                       const std::string &camera_name,
                       float pitch_in, float yaw_in, float roll_in,
                       float zoom_in,  int   mode_in,
                       int   frame_w,  int   frame_h,
                       int   output_w, int   output_h)
    : moil_(nullptr),
      pitch_(pitch_in), yaw_(yaw_in), roll_(roll_in), zoom_(zoom_in),
      mode_(mode_in)
{
    // Sync member publik (dibaca AnypointController)
    pitch = pitch_in;
    yaw   = yaw_in;
    roll  = roll_in;
    zoom  = zoom_in;

    // ── 1. Baca dan validasi JSON ─────────────────────────────────────────
    std::ifstream f(json_path);
    if (!f.is_open()) {
        throw std::runtime_error(
            "[MoildevApplicator] Tidak bisa buka camera_parameters.json: " + json_path);
    }

    nlohmann::json root;
    try {
        f >> root;
    } catch (const std::exception &e) {
        throw std::runtime_error(
            "[MoildevApplicator] JSON parse error: " + std::string(e.what()));
    }

    if (!root.contains(camera_name)) {
        std::string avail;
        int cnt = 0;
        for (auto it = root.begin(); it != root.end() && cnt < 8; ++it, ++cnt)
            avail += it.key() + ", ";
        throw std::runtime_error(
            "[MoildevApplicator] Profil kamera '" + camera_name +
            "' tidak ditemukan di JSON.\nTersedia (8 pertama): " + avail);
    }

    const auto &p = root[camera_name];

    // ── 2. Ekstrak parameter kalibrasi ────────────────────────────────────
    auto getd = [&](const std::string &key, double def = 0.0) -> double {
        if (p.contains(key)) return p[key].get<double>();
        return def;
    };

    double sensorW  = getd("cameraSensorWidth", 1.0);
    double sensorH  = getd("cameraSensorHeight", 1.0);
    double icx      = getd("iCx");
    double icy      = getd("iCy");
    double ratio    = getd("ratio", 1.0);
    double imgW     = getd("imageWidth");
    double imgH     = getd("imageHeight");
    double calibR   = getd("calibrationRatio", 1.0);
    double para0    = getd("parameter0");
    double para1    = getd("parameter1");
    double para2    = getd("parameter2");
    double para3    = getd("parameter3");
    double para4    = getd("parameter4");
    double para5    = getd("parameter5");

    img_w_      = static_cast<float>(imgW);
    img_h_      = static_cast<float>(imgH);
    param5_     = para5;
    calib_ratio_= (calibR > 0.0) ? calibR : 1.0;

    // ── 3. Tentukan resolusi target ───────────────────────────────────────
    int target_w = (output_w > 0) ? output_w
                 : ((frame_w > 0) ? frame_w : static_cast<int>(imgW));
    int target_h = (output_h > 0) ? output_h
                 : ((frame_h > 0) ? frame_h : static_cast<int>(imgH));

    // Scale iCx/iCy dan calibRatio ke resolusi output
    double configIcx = icx;
    double configIcy = icy;
    double configCal = calib_ratio_;

    if (target_w != static_cast<int>(imgW) || target_h != static_cast<int>(imgH)) {
        double sx = static_cast<double>(target_w) / imgW;
        double sy = static_cast<double>(target_h) / imgH;
        configIcx   = icx * sx;
        configIcy   = icy * sy;
        configCal   = calib_ratio_ * sx;
        std::cout << "[MOIL] Scale config ke output " << target_w << "x"
                  << target_h << " (sx=" << sx << ")\n";
    }

    icx_ = configIcx;
    icy_ = configIcy;

    frame_w_  = (frame_w  > 0) ? static_cast<float>(frame_w)  : img_w_;
    frame_h_  = (frame_h  > 0) ? static_cast<float>(frame_h)  : img_h_;
    output_w_ = static_cast<float>(target_w);
    output_h_ = static_cast<float>(target_h);

    std::cout << "[MOIL] Profil kamera   : " << camera_name << "\n"
              << "[MOIL] Sensor          : " << imgW << "x" << imgH << "\n"
              << "[MOIL] Frame input     : " << frame_w_ << "x" << frame_h_ << "\n"
              << "[MOIL] Output remap    : " << output_w_ << "x" << output_h_ << "\n"
              << "[MOIL] param5=" << para5 << "  calibRatio=" << calibR << "\n"
              << "[MOIL] Focal length adj= " << adjusted_focal_length() << " px\n";

    // ── 4. Instansiasi moildev::cpu::Moildev ─────────────────────────────
    std::cout << "[MOIL] Membuat instance moildev::cpu::Moildev...\n";
    moil_ = std::make_unique<moildev::cpu::Moildev>(
        static_cast<float>(sensorW), static_cast<float>(sensorH),
        static_cast<float>(configIcx), static_cast<float>(configIcy),
        static_cast<float>(ratio),
        static_cast<float>(target_w), static_cast<float>(target_h),
        static_cast<float>(configCal),
        static_cast<float>(para0), static_cast<float>(para1),
        static_cast<float>(para2), static_cast<float>(para3),
        static_cast<float>(para4), static_cast<float>(para5));
    std::cout << "[MOIL] moildev::cpu::Moildev OK.\n";

    float lib_w = moil_->getImageWidth();
    float lib_h = moil_->getImageHeight();
    std::cout << "[MOIL] Engine W=" << lib_w << " H=" << lib_h << "\n";
    if (lib_w <= 0 || lib_h <= 0) {
        throw std::runtime_error(
            "[MoildevApplicator] FATAL: engine melaporkan dimensi 0 atau negatif.");
    }

    // ── 5. Pre-compute LUT Alpha-Rho (Horner's Method) ───────────────────
    // Diport dari MoildevApplicator::initializeAlphaRhoTables() unicorn-solution
    std::cout << "[MOIL] Membangun LUT Alpha-Rho (Horner's Method)...\n";
    init_alpha_rho_tables_(para0, para1, para2, para3, para4, para5, calib_ratio_);
    std::cout << "[MOIL] LUT Alpha-Rho siap. Size rho→alpha: "
              << rho_to_alpha_table_.size() << "\n";

    // ── 6. Pre-compute remap maps ─────────────────────────────────────────
    std::cout << "[MOIL] Membangun remap maps ("
              << static_cast<int>(output_w_) << "x"
              << static_cast<int>(output_h_) << ")...\n";
    rebuild_maps_();

    std::cout << "[MOIL] Init OK. mode=" << mode_
              << "  pitch=" << pitch_  << "  yaw=" << yaw_
              << "  zoom=" << zoom_ << "\n";
}

// ── Destructor ────────────────────────────────────────────────────────────────

MoildevApplicator::~MoildevApplicator() = default;

// ── init_alpha_rho_tables_ ───────────────────────────────────────────────────
// Diport 1:1 dari MoildevApplicator::initializeAlphaRhoTables() unicorn-solution.
// Menggunakan Horner's Method untuk evaluasi polinomial orde-6:
//   rho = (((((p0*a + p1)*a + p2)*a + p3)*a + p4)*a + p5)*a * calibRatio

void MoildevApplicator::init_alpha_rho_tables_(double p0, double p1, double p2,
                                         double p3, double p4, double p5,
                                         double calib)
{
    alpha_to_rho_table_.clear();
    rho_to_alpha_table_.clear();
    alpha_to_rho_table_.reserve(1800);
    rho_to_alpha_table_.reserve(3600);

    const double DEG_TO_RAD = M_PI / 180.0;

    // Build Alpha→Rho table (1800 entry: 0.0° hingga 179.9° dalam 0.1° step)
    for (int i = 0; i < 1800; ++i) {
        double alpha = (static_cast<double>(i) / 10.0) * DEG_TO_RAD;
        // Horner's Method: O(n) evaluasi polinomial orde-6
        double rho = (((((p0 * alpha + p1) * alpha + p2) * alpha
                              + p3) * alpha + p4) * alpha + p5) * alpha;
        alpha_to_rho_table_.push_back(rho * calib);
    }

    // Build Rho→Alpha table (inverse lookup)
    int i = 0, index = 0;
    while (i < 1800) {
        while (index < static_cast<int>(alpha_to_rho_table_[i])) {
            rho_to_alpha_table_.push_back(i);
            index++;
        }
        i++;
    }
    // Isi sisa sampai 3600
    while (index < 3600) {
        rho_to_alpha_table_.push_back(i);
        index++;
    }
}

// ── get_alpha_beta ────────────────────────────────────────────────────────────
// Diport dari MoildevApplicator::getAlphaBeta() unicorn-solution.

std::pair<float, float> MoildevApplicator::get_alpha_beta(int x, int y) const
{
    if (rho_to_alpha_table_.empty()) {
        std::cerr << "[MoildevApplicator] get_alpha_beta: LUT belum diinisialisasi!\n";
        return {0.0f, 0.0f};
    }

    double deltaX = static_cast<double>(x) - icx_;
    double deltaY = -(static_cast<double>(y) - icy_);

    double r_px  = std::sqrt(deltaX * deltaX + deltaY * deltaY);
    int    r_int = static_cast<int>(std::round(r_px));

    if (r_int < 0 || static_cast<size_t>(r_int) >= rho_to_alpha_table_.size()) {
        return {0.0f, 0.0f};
    }

    float  alpha     = static_cast<float>(rho_to_alpha_table_[r_int]) / 10.0f;
    double angle_rad = std::atan2(deltaY, deltaX);
    double angle_deg = angle_rad * 180.0 / M_PI;
    float  beta      = 90.0f - static_cast<float>(angle_deg);

    // Normalisasi beta ke [-180, 180]
    while (beta <= -180.0f) beta += 360.0f;
    while (beta >   180.0f) beta -= 360.0f;

    return {alpha, beta};
}

// ── adjusted_focal_length ─────────────────────────────────────────────────────
// Formula yang benar: param5 / calibrationRatio
// Bukan formula empiris lama: param5 * zoom / zoom_ref²

float MoildevApplicator::adjusted_focal_length() const
{
    return static_cast<float>(
        (calib_ratio_ > 0.0) ? (param5_ / calib_ratio_) : param5_);
}

// ── build_aruco_camera_matrix ─────────────────────────────────────────────────
// Focal untuk ArUco = param5/calibRatio × zoom_
//
// Alasan: Moildev zoom meregangkan pusat fisheye ke seluruh frame output.
// Dengan zoom=2, object tampak 2× lebih besar dalam piksel.
// ArUco solvePnP membutuhkan focal yang sesuai dengan besaran piksel itu.
// Formula: fl = (param5/calibRatio) × zoom × scale_frame
//
// Tanpa zoom: ArUco melihat marker 2× besar → estimasi jarak 2× lebih kecil.
// Contoh: zoom=2, z_real=15cm → z_aruco=7.5cm (SALAH)
// Dengan zoom: z_aruco=15cm (BENAR)

cv::Mat MoildevApplicator::build_aruco_camera_matrix(int frame_width,
                                               int frame_height) const
{
    // Skala focal length ke resolusi frame aktual
    double scale_x = static_cast<double>(frame_width)  / static_cast<double>(output_w_);
    double scale_y = static_cast<double>(frame_height) / static_cast<double>(output_h_);
    double scale   = (scale_x + scale_y) / 2.0;

    /* Sertakan zoom_ factor: saat zoom=2, object 2× lebih besar di image,
     * sehingga focal efektif untuk ArUco distance estimation juga 2× lebih besar. */
    double fl = static_cast<double>(adjusted_focal_length()) * scale * static_cast<double>(zoom_);
    double cx = static_cast<double>(frame_width)  / 2.0;
    double cy = static_cast<double>(frame_height) / 2.0;

    std::cout << "[MOIL] build_aruco_camera_matrix:"
              << " param5/calibRatio=" << adjusted_focal_length()
              << " zoom=" << zoom_
              << " scale=" << scale
              << " -> fl=" << fl << " px\n";

    cv::Mat K = (cv::Mat_<double>(3, 3)
        << fl, 0., cx,
           0., fl, cy,
           0., 0.,  1.);
    return K;
}

// ── rebuild_maps_ ─────────────────────────────────────────────────────────────
//
// API moildev:
//   AnyPointM (alpha, beta, zoom)  → rotasi spherical (sudut nadir, azimuth)
//   AnyPointM2(alpha, beta, zoom)  → rotasi Euler/gimbal (pitch=thetaX, yaw=thetaY)
//
// User mengontrol lewat pitch_ / yaw_ → HARUS pakai AnyPointM2.
// AnyPointM dipakai jika input sudah dalam koordinat spherical (alpha dari nadir).
//
// pitch=0, yaw=0, zoom=1 dengan AnyPointM2 → view lurus ke bawah (nadir, tengah frame)

void MoildevApplicator::rebuild_maps_()
{
    // Native size dari engine (harus = imageWidth/imageHeight di konstruktor)
    int nativeW = static_cast<int>(moil_->getImageWidth());
    int nativeH = static_cast<int>(moil_->getImageHeight());

    if (nativeW <= 0) nativeW = static_cast<int>(output_w_);
    if (nativeH <= 0) nativeH = static_cast<int>(output_h_);

    // Alokasi canvas native size
    cv::Mat native_x(nativeH, nativeW, CV_32F);
    cv::Mat native_y(nativeH, nativeW, CV_32F);

    float *mx_ptr = reinterpret_cast<float *>(native_x.data);
    float *my_ptr = reinterpret_cast<float *>(native_y.data);

    // AnyPointM2: pitch_ = thetaX (rotasi sumbu X, maju/mundur)
    //             yaw_   = thetaY (rotasi sumbu Y, kiri/kanan)
    //             zoom_  = faktor zoom (>= 1)
    // pitch=0, yaw=0 → kamera menghadap tepat ke bawah (center frame)
    //
    // KALIBRASI ZOOM:
    // Moildev menggunakan FOCAL_LENGTH_FOR_ZOOM = 250.0f sebagai base.
    // Kamera nyata memiliki focal length = param5/calibRatio (≈504px untuk wxsj_7730_6).
    // Agar zoom=1 (user) menghasilkan tampilan seperti "kamera normal" (bukan fisheye lebar):
    //   zoom_internal = zoom_user × (param5 / calibRatio / 250)
    // Sehingga: zoom=1 → natural focal length → FOV seperti kamera biasa
    //           zoom=2 → 2× zoom in dari natural
    //           zoom=0.5 → 0.5× zoom out dari natural
    // Moildev base focal: FOCAL_LENGTH_FOR_ZOOM = 250.0f (macro di moildev_common.hpp)
    // param5/calibRatio = focal length kamera nyata
    // zoom_internal disekalakan agar zoom=1 user = tampilan normal kamera
    float natural_zoom = static_cast<float>(param5_ / calib_ratio_) / 250.0f;
    float zoom_internal = zoom_ * natural_zoom;


    // ── Debug: zoom internals (hanya saat verbose=true) ──────────────────
    if (verbose_) {
        std::cout << "[MOIL-DBG] zoom_user=" << zoom_
                  << " natural_zoom=" << natural_zoom
                  << " zoom_internal=" << zoom_internal
                  << " (param5=" << param5_ << "/250)\n";
        std::cout.flush();
    }

    moil_->AnyPointM2(mx_ptr, my_ptr, pitch_, yaw_, zoom_internal);

    // ── Debug: verifikasi nilai map (hanya saat verbose=true) ────────────
    // cv::minMaxLoc pada map besar (1280×720) = O(W×H) → jangan di production!
    if (verbose_) {
        float cx_map = native_x.at<float>(nativeH/2, nativeW/2);
        float cy_map = native_y.at<float>(nativeH/2, nativeW/2);
        float tl_x   = native_x.at<float>(0, 0);
        float tl_y   = native_y.at<float>(0, 0);
        double min_x, max_x, min_y, max_y;
        cv::minMaxLoc(native_x, &min_x, &max_x);
        cv::minMaxLoc(native_y, &min_y, &max_y);
        std::cout << "[MOIL-DBG] AnyPointM2(pitch=" << pitch_
                  << " yaw=" << yaw_ << " zoom=" << zoom_ << ")\n"
                  << "[MOIL-DBG]   map_x: min=" << min_x << " max=" << max_x
                  << " | center=(" << cx_map << "," << cy_map << ")"
                  << " | topleft=(" << tl_x << "," << tl_y << ")\n"
                  << "[MOIL-DBG]   map_y: min=" << min_y << " max=" << max_y << "\n"
                  << "[MOIL-DBG]   valid_range_x=[0," << nativeW << "]"
                  << " valid_range_y=[0," << nativeH << "]\n";
        std::cout.flush();
    }
    // ── END DEBUG ─────────────────────────────────────────────────────────


    // Jika resolusi output berbeda dari native, resize map koordinat.
    // PENTING: cv::resize pada remap maps menginterpolasi nilai koordinat,
    // sehingga map baru menunjuk ke pixel yang proporsional di resolusi lain.
    int outW = static_cast<int>(output_w_);
    int outH = static_cast<int>(output_h_);

    if (nativeW != outW || nativeH != outH) {
        // Scale koordinat agar sesuai resolusi output
        double sx = static_cast<double>(outW) / nativeW;
        double sy = static_cast<double>(outH) / nativeH;
        cv::resize(native_x, map_x_, cv::Size(outW, outH));
        cv::resize(native_y, map_y_, cv::Size(outW, outH));
        map_x_ *= static_cast<float>(sx);
        map_y_ *= static_cast<float>(sy);
    } else {
        map_x_ = native_x;
        map_y_ = native_y;
    }

    // Pre-konversi ke fixed-point integer untuk remap lebih cepat di ARM
    cv::convertMaps(map_x_, map_y_, map_x_fixed_, map_y_fixed_, CV_16SC2);
}


// ── update_maps ───────────────────────────────────────────────────────────────

void MoildevApplicator::update_maps(float pitch_in, float yaw_in, float roll_in, float zoom_in)
{
    pitch_ = pitch_in;
    yaw_   = yaw_in;
    roll_  = roll_in;
    zoom_  = std::max(1.0f, std::min(20.0f, zoom_in));

    // Sync member publik (dibaca AnypointController)
    pitch = pitch_;
    yaw   = yaw_;
    roll  = roll_;
    zoom  = zoom_;

    std::lock_guard<std::mutex> lock(maps_mutex_);
    rebuild_maps_();
}

// ── undistort ─────────────────────────────────────────────────────────────────

cv::Mat MoildevApplicator::undistort(const cv::Mat &frame)
{
    if (frame.empty()) return frame;

    std::lock_guard<std::mutex> lock(maps_mutex_);
    if (map_x_.empty() || map_y_.empty()) return frame;

    cv::Mat result;

    // Gunakan fixed-point maps (CV_16SC2) jika tersedia.
    // INTER_LINEAR + fixed-point adalah kombinasi terbaik untuk ARM/RZ/V2H:
    //   - INTER_LANCZOS4 kernel 8x8 → sangat lambat di embedded (0.2 FPS)
    //   - INTER_LINEAR   kernel 2x2 → ~10x lebih cepat, kualitas cukup untuk undistortion
    //   - Fixed-point maps           → ~2-4x lebih cepat vs float maps
    if (!map_x_fixed_.empty()) {
        cv::remap(frame, result, map_x_fixed_, map_y_fixed_,
                  cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    } else {
        cv::remap(frame, result, map_x_, map_y_,
                  cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    }

    // Unsharp mask opsional (disabled by default — GaussianBlur berat di ARM)
    if (sharpen_amount_ > 0.0f) {
        cv::Mat blurred;
        cv::GaussianBlur(result, blurred, cv::Size(0, 0), 3.0);
        cv::addWeighted(result, 1.0 + sharpen_amount_,
                        blurred, -sharpen_amount_, 0, result);
    }

    return result;
}

// ── set_sharpen ───────────────────────────────────────────────────────────────

void MoildevApplicator::set_sharpen(float amount)
{
    sharpen_amount_ = std::max(0.0f, amount);
}
