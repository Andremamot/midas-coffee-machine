/*******************************************************************************
 * volume_math.cpp
 ******************************************************************************/
#include <volume/volume_math.h>

#include <cmath>
#include <algorithm>
#include <iostream>

namespace VolumeMath {

float measureRimWidthPx(const cv::Mat& frame, const cv::Rect& bbox)
{
    float bbox_w = static_cast<float>(bbox.width);
    float bbox_h = static_cast<float>(bbox.height);

    if (bbox_w < 2.0f || bbox_h < 2.0f)
        return std::max(1.0f, bbox_w * 0.795f);

    /* ── Kalibrasi Geometris (Primary) ──────────────────────────────────────
     *
     * MASALAH LAMA (Otsu strip tengah):
     *   Otsu pada strip TENGAH cup selalu menghasilkan ~40% bbox (≈283px dari
     *   709px). Di tengah cup, interior cup vs eksterior memberikan kontras
     *   yang tidak merepresentasikan lebar RIM.
     *
     *   Bug tambahan: fallback BINARY Otsu menghasilkan nilai yang lolos guard
     *   condition (50%–92%), sehingga BBOX_FILL_RATIO tidak pernah aktif.
     *
     * SOLUSI BARU:
     *   1. Lakukan Sobel edge-scan pada strip ATAS bbox (top 15% bbox_h).
     *      Di posisi ini rim cup terlihat paling jelas dari sudut kamera atas.
     *   2. Jika edge-scan berhasil (hasilnya 65%–95% bbox_w), pakai itu.
     *   3. Jika gagal, gunakan BBOX_FILL_RATIO = 0.795 (kalibrasi geometris).
     *
     * Kalibrasi geometris:
     *   D_fisik = 7.6cm, z_rim = 13.6cm, focal = 1009.4px
     *   rim_w_expected = D × focal / z_rim = 7.6 × 1009.4 / 13.6 = 564px
     *   YOLO bbox_w tipikal ≈ 709px
     *   BBOX_FILL_RATIO = 564 / 709 = 0.795 */

    const float BBOX_FILL_RATIO = 0.795f;
    const float fill_w          = bbox_w * BBOX_FILL_RATIO;

    int h_frame = frame.rows;
    int w_frame = frame.cols;

    /* ── Edge-scan pada strip ATAS bbox (top 15% bbox_h) ── */
    float edge_w = -1.0f;

    {
        int top_y1 = std::max(0, bbox.y);
        int top_y2 = std::min(h_frame, bbox.y + std::max(4, static_cast<int>(bbox_h * 0.15f)));
        int rx1    = std::max(0, bbox.x);
        int rx2    = std::min(w_frame, bbox.x + bbox.width);

        if (top_y2 > top_y1 && rx2 > rx1) {
            cv::Mat strip = frame(cv::Rect(rx1, top_y1, rx2 - rx1, top_y2 - top_y1));
            if (!strip.empty()) {
                cv::Mat gray;
                if (strip.channels() == 3)
                    cv::cvtColor(strip, gray, cv::COLOR_BGR2GRAY);
                else if (strip.channels() == 4)
                    cv::cvtColor(strip, gray, cv::COLOR_BGRA2GRAY);
                else
                    gray = strip.clone();

                /* Sobel horizontal → deteksi tepi vertikal rim */
                cv::Mat sobel_x;
                cv::Sobel(gray, sobel_x, CV_16S, 1, 0, 3);
                cv::Mat sobel_abs;
                cv::convertScaleAbs(sobel_x, sobel_abs);

                /* Kolaps vertikal → profil edge horizontal */
                cv::Mat col_sum;
                cv::reduce(sobel_abs, col_sum, 0, cv::REDUCE_SUM, CV_32S);

                /* Threshold adaptif = 30% dari nilai maks */
                double maxVal = 0;
                cv::minMaxLoc(col_sum, nullptr, &maxVal);
                const double edge_thresh = maxVal * 0.30;

                /* Cari tepi paling kiri dan kanan yang melewati threshold */
                const int* cp = col_sum.ptr<int>(0);
                int left_col = -1, right_col = -1;
                for (int c = 0; c < col_sum.cols; ++c) {
                    if (cp[c] >= static_cast<int>(edge_thresh)) {
                        if (left_col < 0) left_col = c;
                        right_col = c;
                    }
                }
                if (left_col >= 0 && right_col > left_col)
                    edge_w = static_cast<float>(right_col - left_col);
            }
        }
    }

    /* ── Pilih hasil final ── */
    float result_w;
    if (edge_w >= bbox_w * 0.65f && edge_w <= bbox_w * 0.95f) {
        /* Edge-scan dari strip atas memberikan hasil yang masuk akal */
        result_w = edge_w;
    } else {
        /* Edge-scan gagal atau di luar rentang → pakai kalibrasi geometris */
        result_w = fill_w;
    }

    return std::max(1.0f, result_w);
}

double calcDiameter(double rim_w_px, double z_rim_cm, double focal_px)
{
    if (z_rim_cm <= 0.0 || focal_px <= 0.0 || rim_w_px <= 0.0)
        return 0.0;
    return (rim_w_px * z_rim_cm) / focal_px;
}

double calcVolume(double h_cup_cm, double diameter_cm)
{
    if (h_cup_cm <= 0.0 || diameter_cm <= 0.0)
        return 0.0;
    double radius = diameter_cm / 2.0;

    /* TAPERED_CUP_FACTOR: koreksi bentuk kerucut/trapesium.
     *
     * V_silinder(D=7.6cm, h=7.6cm) = π × 3.8² × 7.6 = 344ml
     * V_nyata cup kopi = 150ml
     * TAPERED_CUP_FACTOR = 150 / 344 = 0.436 ≈ 0.44
     *
     * Catatan: nilai ini dikalibrasi untuk cup kopi trapesium yang digunakan.
     * Ubah ke 1.0 untuk cup silindris sempurna. */
    const double TAPERED_CUP_FACTOR = 0.44;
    return M_PI * (radius * radius) * h_cup_cm * TAPERED_CUP_FACTOR;
}

} // namespace VolumeMath
