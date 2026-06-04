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
        return std::max(1.0f, bbox_w * 0.78f);

    int h_frame = frame.rows;
    int w_frame = frame.cols;

    /* ── Pendekatan Gabungan: Otsu + BBOX_FILL_RATIO ───────────────────────
     *
     * MASALAH SEBELUMNYA: Otsu dari strip atas/tengah selalu memberi ~40% bbox
     * terlepas dari posisi strip (atas vs tengah). Ini terjadi karena perbedaan
     * pencahayaan di dalam dan luar cup yang membuat Otsu mendeteksi wilayah
     * tertentu saja, bukan seluruh lebar cup.
     *
     * SOLUSI: Gunakan BBOX_FILL_RATIO sebagai basis kalibrasi.
     * Dari pengukuran fisik dan z_rim:
     *   D_fisik = 7.6cm, z_rim = 13.6cm, focal = 1009.4px
     *   rim_w_expected = D × focal / z_rim = 7.6 × 1009.4 / 13.6 = 564px
     *   YOLO bbox_w ≈ 709px
     *   BBOX_FILL_RATIO = 564 / 709 = 0.795 ≈ 0.80
     *
     * Strategi: coba Otsu dulu (dari strip tengah). Jika hasilnya tidak masuk
     * akal (< 50% bbox atau > 85% bbox), gunakan BBOX_FILL_RATIO fallback. */

    /* ── Strip TENGAH bbox untuk Otsu ── */
    int mid_y     = bbox.y + bbox.height / 2;
    int strip_h   = std::max(4, static_cast<int>(bbox_h * 0.06f));  // ±6% bbox_h
    int ry1 = std::max(0, mid_y - strip_h);
    int ry2 = std::min(h_frame, mid_y + strip_h);
    int rx1 = std::max(0, bbox.x);
    int rx2 = std::min(w_frame, bbox.x + bbox.width);

    float otsu_w = -1.0f;

    if (ry2 > ry1 && rx2 > rx1) {
        cv::Mat strip = frame(cv::Rect(rx1, ry1, rx2 - rx1, ry2 - ry1));
        if (!strip.empty()) {
            cv::Mat gray;
            if (strip.channels() == 3)
                cv::cvtColor(strip, gray, cv::COLOR_BGR2GRAY);
            else if (strip.channels() == 4)
                cv::cvtColor(strip, gray, cv::COLOR_BGRA2GRAY);
            else
                gray = strip.clone();

            cv::Mat blurred;
            cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 0);

            /* Coba THRESH_BINARY_INV */
            cv::Mat mask;
            cv::threshold(blurred, mask, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);
            cv::Mat col_max;
            cv::reduce(mask, col_max, 0, cv::REDUCE_MAX, CV_8U);

            int left_col = -1, right_col = -1;
            const uchar* rp = col_max.ptr<uchar>(0);
            for (int c = 0; c < col_max.cols; ++c) {
                if (rp[c] > 0) {
                    if (left_col < 0) left_col = c;
                    right_col = c;
                }
            }
            if (left_col >= 0 && right_col > left_col)
                otsu_w = static_cast<float>(right_col - left_col);

            /* Jika BINARY_INV tidak memuaskan, coba BINARY */
            if (otsu_w < bbox_w * 0.50f) {
                cv::Mat mask2;
                cv::threshold(blurred, mask2, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
                cv::Mat col2;
                cv::reduce(mask2, col2, 0, cv::REDUCE_MAX, CV_8U);
                int l2 = -1, r2 = -1;
                const uchar* rp2 = col2.ptr<uchar>(0);
                for (int c = 0; c < col2.cols; ++c) {
                    if (rp2[c] > 0) {
                        if (l2 < 0) l2 = c;
                        r2 = c;
                    }
                }
                if (l2 >= 0 && r2 > l2) {
                    float w2 = static_cast<float>(r2 - l2);
                    /* Ambil yang lebih besar (lebih mendekati lebar cup sebenarnya) */
                    if (w2 > otsu_w)
                        otsu_w = w2;
                }
            }
        }
    }

    /* ── BBOX_FILL_RATIO fallback ──────────────────────────────────────────
     * Kalibrasi: D_fisik=7.6cm, z_rim=13.6cm, focal=1009.4 → rim_w=564px
     * bbox_w YOLO tipikal ≈ 709px → RATIO = 564/709 = 0.795
     *
     * Gunakan ini jika Otsu memberikan hasil < 50% atau tidak berhasil.
     * Nilai ini akurat untuk gelas yang sudah dikalibrasi. */
    const float BBOX_FILL_RATIO = 0.795f;
    const float fill_w = bbox_w * BBOX_FILL_RATIO;

    float result_w;
    if (otsu_w >= bbox_w * 0.50f && otsu_w <= bbox_w * 0.92f) {
        /* Otsu memberi hasil yang masuk akal → gunakan */
        result_w = otsu_w;
    } else {
        /* Otsu gagal atau terlalu kecil → gunakan kalibrasi geometris */
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
     * V_silinder(D=7.6cm, h=7.6cm) = π × 3.8² × 7.6 = 344ml
     * V_nyata = 150ml
     * factor = 150/344 = 0.436 ≈ 0.44
     * Ubah ke 1.0 untuk cup silindris sempurna. */
    const double TAPERED_CUP_FACTOR = 0.44;
    return M_PI * (radius * radius) * h_cup_cm * TAPERED_CUP_FACTOR;
}

} // namespace VolumeMath
