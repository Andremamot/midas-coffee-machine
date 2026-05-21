/*******************************************************************************
 * core/moil_undistorter.cpp  (v2 — menggunakan API Moildev lama via libmoildevren.a)
 *
 * API Moildev yang tersedia di libmoildevren.a (decompiled dari nm output):
 *
 *   class Moildev {
 *   public:
 *       Moildev();
 *       Config(std::string camera_name,
 *              double sensorW, double sensorH,
 *              double icx, double icy, double ratio,
 *              double imgW, double imgH,
 *              double calibRatio,
 *              double para0, double para1, double para2,
 *              double para3, double para4, double para5,
 *              double unused);           // 14 doubles
 *       AnyPointM (float* mapX, float* mapY, double alpha, double beta,  double zoom)
 *       AnyPointM2(float* mapX, float* mapY, double pitch, double yaw,   double zoom)
 *       double getImageWidth()
 *       double getImageHeight()
 *       double getiCx()
 *       double getiCy()
 *   };
 *
 * Referensi: demangled dari `nm -g libmoildevren.a`
 ******************************************************************************/

#include "moil_undistorter.hpp"

// Include header yang sesuai dengan API lama (dari moildev_common.hpp)
// Kita deklarasikan class Moildev minimal di sini sesuai ABI yang ada di library
#include "../lib/include/moildev_common.hpp"

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cmath>

// ──────────────────────────────────────────────────────────────────────────────
// Minimal C++ declaration agar linker resolve ke simbol di libmoildevren.a
// Harus persis sesuai dengan ABI: Moildev::Config(string, double×14)
// ──────────────────────────────────────────────────────────────────────────────
class Moildev {
private:
    // PENTING: Karena kita tidak punya definisi asli class Moildev dari library,
    // compiler C++ di file ini menganggap sizeof(Moildev) = 1 byte.
    // Saat kita memanggil `new Moildev()`, ia hanya mengalokasi 1 byte di heap.
    // Library libmoildevren.a akan menulis member variable ke `this` out-of-bounds
    // dan merusak struktur heap (heap corruption) yang menyebabkan Segfault.
    // Solusi: kita tambahkan padding besar (64 KB) agar alokasinya lebih dari cukup 
    // untuk menampung semua member variable internal library.
    alignas(16) char opaque_padding[65536];

public:
    Moildev();
    ~Moildev();

    /**
     * Configure the Moildev instance.
     * Simbol: _ZN7Moildev6ConfigENSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEdddddddddddddd
     * = Moildev::Config(std::string [by value], double×14)
     * PENTING: string by VALUE bukan const& agar mangle name cocok!
     */
    void Config(std::string camera_name,
                double cameraSensorWidth, double cameraSensorHeight,
                double icx,  double icy, double ratio,
                double imageWidth, double imageHeight,
                double calibrationRatio,
                double para0, double para1, double para2,
                double para3, double para4, double para5);

    /**
     * Internal init: pre-compute sin/cos lookup table.
     * Simbol: _ZN7Moildev7initSinEv
     * Harus dipanggil manual setelah Config() pada versi REN.
     */
    void initSin();

    /**
     * Internal init: pre-compute alpha↔rho lookup table.
     * Simbol: _ZN7Moildev18initAlphaRho_TableEv
     * Harus dipanggil manual setelah Config() pada versi REN.
     */
    void initAlphaRho_Table();

    /**
     * AnyPoint Mode 1: alpha/beta.
     * Simbol: _ZN7Moildev9AnyPointMEPfS0_ddd
     */
    int AnyPointM(float* mapX, float* mapY, double alpha, double beta, double zoom);

    /**
     * AnyPoint Mode 2: pitch/yaw.
     * Simbol: _ZN7Moildev10AnyPointM2EPfS0_ddd
     */
    int AnyPointM2(float* mapX, float* mapY, double pitch, double yaw, double zoom);

    /**
     * PanoramaCar: dipakai oleh Python binding sebagai 'AnypointCar' untuk mode 2.
     * Simbol: _ZN7Moildev11PanoramaCarEPfS0_dddbb
     * Parameters: mapX, mapY, alpha_max, iC_alpha_degree, iC_beta_degree, flip_h, flip_v
     */
    int PanoramaCar(float* mapX, float* mapY, double alpha_max, double iC_alpha_degree, double iC_beta_degree, bool flip_h, bool flip_v);

    double getImageWidth();
    double getImageHeight();
    double getiCx();
    double getiCy();
};

// ── Constructor ──────────────────────────────────────────────────────────────

MoilUndistorter::MoilUndistorter(const std::string& json_path,
                                 const std::string& camera_name,
                                 float pitch, float yaw, float roll,
                                 float zoom, int mode,
                                 int frame_w, int frame_h,
                                 int output_w, int output_h)
    : moil_(nullptr)
    , pitch_(pitch), yaw_(yaw), roll_(roll)
    , zoom_(std::max(1.0f, zoom))
    , mode_(mode)
    , zoom_ref_(1.6f)
    , moil_zoom_(0.0f), digital_zoom_(1.0f)
{
    // ── 1. Read and validate JSON ────────────────────────────────────────────
    std::ifstream f(json_path);
    if (!f.is_open()) {
        throw std::runtime_error(
            "[MoilUndistorter] Cannot open camera_parameters.json: " + json_path);
    }

    nlohmann::json root;
    try {
        f >> root;
    } catch (const std::exception& e) {
        throw std::runtime_error(
            "[MoilUndistorter] JSON parse error: " + std::string(e.what()));
    }

    if (!root.contains(camera_name)) {
        std::string avail;
        int cnt = 0;
        for (auto it = root.begin(); it != root.end() && cnt < 8; ++it, ++cnt)
            avail += it.key() + ", ";
        throw std::runtime_error(
            "[MoilUndistorter] Camera profile '" + camera_name +
            "' not found in JSON.\nAvailable (first 8): " + avail);
    }

    const auto& p = root[camera_name];

    // ── 2. Extract parameters (JSON uses float, API needs double) ────────────
    auto getd = [&](const std::string& key, double def = 0.0) -> double {
        if (p.contains(key)) return p[key].get<double>();
        return def;
    };

    double sensorW  = getd("cameraSensorWidth",  1.0);
    double sensorH  = getd("cameraSensorHeight", 1.0);
    double icx      = getd("iCx");
    double icy      = getd("iCy");
    double ratio    = getd("ratio",  1.0);
    double imgW     = getd("imageWidth");
    double imgH     = getd("imageHeight");
    double calibR   = getd("calibrationRatio", 1.0);
    double para0    = getd("parameter0");
    double para1    = getd("parameter1");
    double para2    = getd("parameter2");
    double para3    = getd("parameter3");
    double para4    = getd("parameter4");
    double para5    = getd("parameter5");

    img_w_       = static_cast<float>(imgW);
    img_h_       = static_cast<float>(imgH);
    param5_      = para5;
    calib_ratio_ = (calibR > 0.0) ? calibR : 1.0;

    // ── Scale parameters ke resolusi OUTPUT aktual ───────────────────────────
    int target_w = (output_w > 0) ? output_w : ((frame_w > 0) ? frame_w : static_cast<int>(imgW));
    int target_h = (output_h > 0) ? output_h : ((frame_h > 0) ? frame_h : static_cast<int>(imgH));

    double configW   = static_cast<double>(target_w);
    double configH   = static_cast<double>(target_h);
    double configIcx = icx;
    double configIcy = icy;
    double configCal = calib_ratio_;

    if (target_w != (int)imgW || target_h != (int)imgH) {
        double sx = static_cast<double>(target_w) / imgW;
        double sy = static_cast<double>(target_h) / imgH;
        configIcx = icx * sx;
        configIcy = icy * sy;
        configCal = calib_ratio_ * sx;
        std::cout << "[MOIL] Scaling Config to output " << target_w << "x" << target_h
                  << " (sx=" << sx << ")\n";
    }

    frame_w_  = (frame_w > 0) ? static_cast<float>(frame_w) : img_w_;
    frame_h_  = (frame_h > 0) ? static_cast<float>(frame_h) : img_h_;
    output_w_ = static_cast<float>(target_w);
    output_h_ = static_cast<float>(target_h);

    std::cout << "[MOIL] Camera profile   : " << camera_name << "\n"
              << "[MOIL] Sensor resolution: " << imgW << " x " << imgH << "\n"
              << "[MOIL] Frame resolution : " << frame_w_ << " x " << frame_h_ << "\n"
              << "[MOIL] Output resolution: " << output_w_ << " x " << output_h_ << "\n"
              << "[MOIL] param5=" << para5 << "  calibRatio=" << calibR << "\n"
              << "[MOIL] Adjusted focal length ≈ " << adjusted_focal_length() << " px\n";

    // ── 3. Instansiasi Moildev (API lama, sesuai libmoildevren.a) ─────────────
    std::cout << "[MOIL] Creating Moildev instance...\n";
    moil_ = new Moildev();
    std::cout << "[MOIL] Calling Config()...\n";

    // Gunakan configW/H/Icx/Icy/Cal yang sudah discale ke resolusi frame aktual
    moil_->Config(camera_name,
                  sensorW, sensorH,
                  configIcx, configIcy, ratio,
                  configW, configH,
                  configCal,
                  para0, para1, para2,
                  para3, para4, para5);
    std::cout << "[MOIL] Config() OK.\n";

    // ── PENTING: Pada versi REN, init tables harus dipanggil manual ──────────
    // initSin()         : pre-compute sin/cos lookup table (dipakai AnyPointM/M2)
    // initAlphaRho_Table: pre-compute alpha<->rho table   (dipakai Panorama* functions)
    // Tanpa ini, AnyPointM/M2 akan akses tabel yang belum diinisialisasi → SEGFAULT.
    std::cout << "[MOIL] Calling initSin()...\n";
    moil_->initSin();
    std::cout << "[MOIL] Calling initAlphaRho_Table()...\n";
    moil_->initAlphaRho_Table();
    std::cout << "[MOIL] Init tables OK.\n";
    
    // VERIFIKASI CONFIG:
    double lib_w = moil_->getImageWidth();
    double lib_h = moil_->getImageHeight();
    double lib_icx = moil_->getiCx();
    double lib_icy = moil_->getiCy();
    std::cout << "[MOIL] Lib Verification:\n"
              << "       Lib W: " << lib_w << " Lib H: " << lib_h << "\n"
              << "       Lib iCx: " << lib_icx << " Lib iCy: " << lib_icy << "\n";

    if (lib_w <= 0 || lib_h <= 0) {
        throw std::runtime_error("[MOIL] FATAL: libmoildevren.a reported 0 or negative image dimensions after Config(). The Config parameters might have mismatched ABI.");
    }

    // ── 4. Pre-compute remap maps ────────────────────────────────────────────
    std::cout << "[MOIL] Building remap maps (" << (int)output_w_ << "x" << (int)output_h_ << ")...\n";
    rebuild_maps_();

    std::cout << "[MOIL] Init OK. Mode=" << mode_
              << "  pitch=" << pitch_ << "  yaw=" << yaw_
              << "  zoom=" << zoom_
              << " [moil=" << moil_zoom_ << " + digital=" << digital_zoom_ << "x]\n"
              << "[MOIL] Remap maps ready (" << (int)output_w_ << "\xc3\x97" << (int)output_h_ << ")\n";
}

// ── Destructor ───────────────────────────────────────────────────────────────

MoilUndistorter::~MoilUndistorter()
{
    delete moil_;
}

// ── rebuild_maps_ ─────────────────────────────────────────────────────────────

void MoilUndistorter::rebuild_maps_()
{
    int W = static_cast<int>(moil_->getImageWidth());
    int H = static_cast<int>(moil_->getImageHeight());

    if (W <= 0) W = static_cast<int>(img_w_);
    if (H <= 0) H = static_cast<int>(img_h_);

    std::cout << "[MOIL-DEBUG] rebuild_maps W=" << W << " H=" << H << " allocating...\n";

    // Langsung alokasi ke cv::Mat
    map_x_.create(H, W, CV_32F);
    map_y_.create(H, W, CV_32F);

    float* mx_ptr = reinterpret_cast<float*>(map_x_.data);
    float* my_ptr = reinterpret_cast<float*>(map_y_.data);

    std::cout << "[MOIL-DEBUG] rebuild_maps calling ";

    // ── Hybrid Zoom: kirim hanya moil_zoom ke Moildev ────────────────────────
    split_zoom_();

    if (mode_ == 1 || mode_ == 0) {
        double alpha = pitch_;
        double beta = yaw_;
        if (beta < 0) {
            beta = beta + 360;
        }
        if (alpha < -110 || alpha > 110 || beta < 0 || beta > 360) {
            alpha = 0;
            beta = 0;
        } else {
            alpha = (alpha < -110) ? -110 : ((alpha > 110) ? 110 : alpha);
            beta = (beta < 0) ? 0 : ((beta > 360) ? 360 : beta);
        }
        std::cout << "AnyPointM(alpha=" << alpha << " beta=" << beta << " zoom=" << moil_zoom_ << ")\n";
        moil_->AnyPointM(mx_ptr, my_ptr,
                         alpha,
                         beta,
                         static_cast<double>(moil_zoom_));
    } else {
        double p_val = pitch_;
        double y_val = yaw_;
        if (p_val < -110 || p_val > 110 || y_val < -110 || y_val > 110) {
            p_val = 0;
            y_val = 0;
        } else {
            p_val = (p_val < -110) ? -110 : ((p_val > 110) ? 110 : p_val);
            y_val = (y_val < -110) ? -110 : ((y_val > 110) ? 110 : y_val);
        }
        std::cout << "AnyPointM2(pitch=" << p_val << " yaw=" << y_val << " zoom=" << moil_zoom_ << ")\n";
        moil_->AnyPointM2(mx_ptr, my_ptr,
                          p_val,
                          y_val,
                          static_cast<double>(moil_zoom_));
    }

    // ── Scale maps ke resolusi frame input ───────────────────────────────────
    // Peta map_x_ dan map_y_ sekarang berisi nilai koordinat untuk mengambil dari 
    // sumber gambar berukuran output_w_ x output_h_.
    // Namun, sumber gambar asli (frame) kita adalah frame_w_ x frame_h_.
    // Oleh karena itu, kita sesuaikan koordinat agar mengambil tepat pada frame yang kecil.
    if (frame_w_ != output_w_ || frame_h_ != output_h_) {
        float scale_x = frame_w_ / output_w_;
        float scale_y = frame_h_ / output_h_;
        map_x_ *= scale_x;
        map_y_ *= scale_y;
    }

    // Debug: print beberapa nilai map untuk validasi
    std::cout << "[MOIL-DEBUG] map_x sample: "
              << "[0,0]="   << map_x_.at<float>(0, 0)
              << " [ctr]="  << map_x_.at<float>(H/2, W/2)
              << " [end]="  << map_x_.at<float>(H-1, W-1) << "\n";
    std::cout << "[MOIL-DEBUG] map_y sample: "
              << "[0,0]="   << map_y_.at<float>(0, 0)
              << " [ctr]="  << map_y_.at<float>(H/2, W/2)
              << " [end]="  << map_y_.at<float>(H-1, W-1) << "\n";

    std::cout << "[MOIL-DEBUG] rebuild_maps finished.\n";
}

// ── split_zoom_ ───────────────────────────────────────────────────────────────
// Sama persis dengan Python: _split_zoom()
//   moil_zoom  = min(total_zoom, MAX_MOIL_ZOOM)  → dikirim ke Moildev
//   digital_zoom = total_zoom / moil_zoom          → center-crop sisa

void MoilUndistorter::split_zoom_()
{
    float total = std::max(1.0f, zoom_);
    moil_zoom_    = std::min(total, MAX_MOIL_ZOOM);
    digital_zoom_ = total / moil_zoom_;  // selalu >= 1.0
    std::cout << "[MOIL] Hybrid Zoom: total=" << total
              << "x → moil=" << moil_zoom_
              << "x + digital=" << digital_zoom_ << "x\n";
}

// ── digital_crop_ ─────────────────────────────────────────────────────────────
// Port dari Python: _digital_crop()
//   Crop region tengah sebesar (W/digital_zoom) x (H/digital_zoom),
//   lalu resize kembali ke ukuran asli.

cv::Mat MoilUndistorter::digital_crop_(const cv::Mat& frame) const
{
    if (digital_zoom_ <= 1.001f) return frame;  // toleransi float, tidak ada crop

    int w = frame.cols;
    int h = frame.rows;

    int crop_w = std::max(1, (int)(w / digital_zoom_));
    int crop_h = std::max(1, (int)(h / digital_zoom_));

    int x1 = (w - crop_w) / 2;
    int y1 = (h - crop_h) / 2;

    cv::Mat cropped = frame(cv::Rect(x1, y1, crop_w, crop_h));

    cv::Mat result;
    // LANCZOS4: anti-aliasing terbaik, mengurangi jagged/pixelated edges
    cv::resize(cropped, result, cv::Size(w, h), 0, 0, cv::INTER_LANCZOS4);
    return result;
}
// ── update_maps ───────────────────────────────────────────────────────────────

void MoilUndistorter::update_maps(float pitch, float yaw, float roll, float zoom)
{
    pitch_ = pitch;
    yaw_   = yaw;
    roll_  = roll;
    zoom_  = std::max(1.0f, zoom);  // clamp minimum 1.0
    // split_zoom_() dipanggil di dalam rebuild_maps_()
    std::lock_guard<std::mutex> lock(maps_mutex_);
    rebuild_maps_();
}

// ── undistort ─────────────────────────────────────────────────────────────────

cv::Mat MoilUndistorter::undistort(const cv::Mat& frame)
{
    if (frame.empty()) return frame;

    // Ambil snapshot maps di bawah lock agar thread-safe.
    // Clone ringan: hanya increment ref count, tidak copy data.
    cv::Mat mx, my;
    {
        std::lock_guard<std::mutex> lock(maps_mutex_);
        if (map_x_.empty() || map_y_.empty()) return frame;
        mx = map_x_;  // shared refcount, bukan deep copy
        my = map_y_;
    }

    // Stage 1: Moildev remap (undistortion + zoom aman ≤ MAX_MOIL_ZOOM)
    cv::Mat result;
    cv::remap(frame, result, mx, my,
              cv::INTER_LANCZOS4,        // Anti-aliasing terbaik untuk remap fisheye
              cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

    // Stage 2: Digital zoom via center-crop + resize (sisa zoom di atas moil_zoom)
    result = digital_crop_(result);

    // Stage 3: Unsharp mask sharpening (if enabled)
    if (sharpen_amount_ > 0.0f) {
        cv::Mat blurred;
        cv::GaussianBlur(result, blurred, cv::Size(0, 0), 3.0);
        cv::addWeighted(result,  1.0 + sharpen_amount_,
                        blurred, -sharpen_amount_,
                        0, result);
    }

    return result;
}

// ── adjusted_focal_length ────────────────────────────────────────────────────

float MoilUndistorter::adjusted_focal_length() const
{
    return static_cast<float>(
        (calib_ratio_ > 0.0) ? (param5_ / calib_ratio_) : param5_
    );
}

// ── build_aruco_camera_matrix ────────────────────────────────────────────────

cv::Mat MoilUndistorter::build_aruco_camera_matrix(int frame_width, int frame_height) const
{
    // Hitung skala rasio antara resolusi streaming dan resolusi sensor JSON
    float scale_x = static_cast<float>(frame_width)  / std::max(1.0f, static_cast<float>(moil_->getImageWidth()));
    float scale_y = static_cast<float>(frame_height) / std::max(1.0f, static_cast<float>(moil_->getImageHeight()));
    float scale   = (scale_x + scale_y) / 2.0f;

    // Focal length dikalikan dengan TOTAL zoom (moil + digital).
    float fl = adjusted_focal_length() * scale * zoom_;
    
    float cx = static_cast<float>(frame_width)  / 2.0f;
    float cy = static_cast<float>(frame_height) / 2.0f;

    cv::Mat K = (cv::Mat_<double>(3, 3) <<
        static_cast<double>(fl), 0., static_cast<double>(cx),
        0., static_cast<double>(fl), static_cast<double>(cy),
        0., 0.,  1.);

    return K;
}
