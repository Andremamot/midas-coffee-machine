/**
 * @file volume_config.h
 * @brief Hardcoded operational parameters for mod_volume.
 *
 * Semua konstanta konfigurasi kamera dan Moildev yang sudah dikalibrasi
 * dikumpulkan di satu file agar mudah dicek dan diaudit.
 *
 * JANGAN ganti REMAP_INTERP ke INTER_CUBIC/INTER_LANCZOS4 —
 * menyebabkan crash SIGABRT karena race condition OpenCL ↔ frame update.
 */

#pragma once

#include <opencv2/imgproc.hpp>  // cv::INTER_LINEAR

namespace fusion {
namespace volume_config {

// ── ArUco Marker ─────────────────────────────────────────────────────────────
constexpr double MARKER_SIZE_CM = 2.5;

// ── Moildev Anypoint ─────────────────────────────────────────────────────────
constexpr double MOIL_ZOOM      = 2.0;    ///< Hybrid: moil 1.5x + digital 1.33x
constexpr int    MOIL_MODE      = 2;      ///< AnyPointM2 = Pitch/Yaw/Roll
constexpr double MOIL_PITCH_DEG = -15.0; ///< Kamera miring 15° ke bawah
constexpr double MOIL_YAW_DEG   = 0.0;
constexpr double MOIL_ROLL_DEG  = 0.0;
constexpr double MAX_MOIL_ZOOM  = 1.5;   ///< Batas zoom sebelum digital crop

// ── Exposure ─────────────────────────────────────────────────────────────────
constexpr int    V4L2_EXPOSURE  = 5000;  ///< = smart_exposure 5.0 di Python

// ── Remap Interpolation ───────────────────────────────────────────────────────
/// SELALU INTER_LINEAR — thread-safe, tidak trigger OpenCL race condition.
constexpr int    REMAP_INTERP   = cv::INTER_LINEAR;

// ── EMA ──────────────────────────────────────────────────────────────────────
constexpr double EMA_ALPHA      = 0.35;

}  // namespace volume_config
}  // namespace fusion
