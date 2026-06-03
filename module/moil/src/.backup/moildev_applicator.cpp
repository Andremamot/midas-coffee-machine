/**
 * @file moildev_applicator.cpp
 * @brief Implements the MoildevApplicator singleton class.
 *
 * @details This file contains the implementation for the high-level Moildev manager, 
 * including initialization, map generation, coordinate conversion, and map modification logic.
 */

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif

#include "utils/moildev_applicator.hpp"
#include "constants/constants.hpp"
#include "cores/mediator.hpp"
#include "utils/frame_processor.hpp" // For clearing CUDA cache
#include "helpers/logger.hpp" // Include the logger
#include "models/frame_model.hpp"
#include "models/function_model.hpp"
#include <cmath>
#include <fstream>
#include <iostream> // Keep if needed by other included headers
#include <opencv2/imgproc.hpp>
#include <algorithm> // Untuk std::replace

// FIX 2: Fallback jika M_PI tetap tidak terdefinisi (Safety net)
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace
{
  static constexpr const char *TAG = "MOILDEV APPLICATOR"; // File-local tag
}

bool MoildevApplicator::moilCreated = false;

const double MAX_HORIZONTAL_CROP_RATIO = 0.50;
const double MAX_VERTICAL_CROP_RATIO = 0.50;

/**
 * @brief Gets the singleton instance of the MoildevApplicator.
 * @return MoildevApplicator& A reference to the singleton instance.
 */
MoildevApplicator &MoildevApplicator::getInstance()
{
  static MoildevApplicator instance;
  return instance;
}

/**
 * @brief Private constructor to enforce the singleton pattern.
 * @details The constructor follows the initialization sequence:
 * 1. Loads camera parameters from the database.
 * 2. Initializes the core `Moildev` engine with these parameters.
 * 3. Pre-computes the alpha/rho lookup tables for fast coordinate conversions.
 */
MoildevApplicator::MoildevApplicator()
{
  Logger::info(TAG, "MoildevApplicator initializing...");
  loadCameraParametersFromDatabase();
  initializeMoil();
  initializeAlphaRhoTables();
}

/**
 * @brief Initializes the internal Moildev object with the loaded camera
 * parameters.
 * @details Resets any previous configuration and then configures the `moil`
 * member object with the camera parameters stored in the `Constants` namespace.
 */
void MoildevApplicator::initializeMoil()
{
  moil.reset();

  // Persiapkan variabel casting (agar kode ifdef di bawah lebih rapi)
  float sW = static_cast<float>(Constants::CAMERA_SENSOR_WIDTH);
  float sH = static_cast<float>(Constants::CAMERA_SENSOR_HEIGHT);
  float iCx = static_cast<float>(Constants::I_CX);
  float iCy = static_cast<float>(Constants::I_CY);
  float ratio = static_cast<float>(Constants::I_RATIO);
  float iW = static_cast<float>(Constants::WIDTH);
  float iH = static_cast<float>(Constants::HEIGHT);
  float calib = static_cast<float>(Constants::CALIBRATION_RATIO);
  float p0 = static_cast<float>(Constants::PARAMETER_0);
  float p1 = static_cast<float>(Constants::PARAMETER_1);
  float p2 = static_cast<float>(Constants::PARAMETER_2);
  float p3 = static_cast<float>(Constants::PARAMETER_3);
  float p4 = static_cast<float>(Constants::PARAMETER_4);
  float p5 = static_cast<float>(Constants::PARAMETER_5);

#ifdef _WIN32
  // ==========================================
  // KHUSUS WINDOWS (WRAPPER LOGIC)
  // ==========================================

  // Tentukan backend berdasarkan opsi CMake (USE_OPENCL_GPU)
  moildev::Backend targetBackend = moildev::Backend::CPU;

#ifdef USE_CUDA
  targetBackend = moildev::Backend::CUDA;
  Logger::info(TAG, "Windows Init: Menggunakan Backend CUDA (GPU)");
#elif defined(USE_OPENCL_GPU)
  targetBackend = moildev::Backend::OCL;
  Logger::info(TAG, "Windows Init: Menggunakan Backend OPENCL (GPU)");
#else
  Logger::info(TAG, "Windows Init: Menggunakan Backend CPU");
#endif

  // Panggil Constructor Wrapper (Ada parameter tambahan di akhir)
  moil = std::make_unique<MoilEngine>(sW, sH, iCx, iCy, ratio, iW, iH, calib,
                                      p0, p1, p2, p3, p4, p5,
                                      targetBackend // <--- BEDA DISINI
  );

#else
  // ==========================================
  // KHUSUS LINUX (KODE LAMA ANDA)
  // ==========================================
  // Tidak ada parameter backend, karena sudah terikat di .so

  moil = std::make_unique<MoilEngine>(sW, sH, iCx, iCy, ratio, iW, iH, calib,
                                      p0, p1, p2, p3, p4, p5);

#ifdef USE_CUDA
  Logger::info(TAG, "Moil object configured (Linux CUDA Mode) for camera '%s'.",
                Constants::CAMERA_NAME.c_str());
#elif defined(USE_OPENCL_GPU)
  Logger::info(TAG, "Moil object configured (Linux GPU Mode) for camera '%s'.",
                Constants::CAMERA_NAME.c_str());
#else
  Logger::info(TAG, "Moil object configured (Linux CPU Mode) for camera '%s'.",
                Constants::CAMERA_NAME.c_str());
#endif
#endif
}

/**
 * @brief Loads camera parameters from the database via the `Constants` utility.
 */
void MoildevApplicator::loadCameraParametersFromDatabase()
{
  // Logging is handled inside Constants::loadCameraParametersFromDatabase
  Constants::loadCameraParametersFromDatabase();
}

/**
 * @brief Reloads all camera parameters, re-initializes the Moildev engine, and
 * regenerates all maps.
 */
void MoildevApplicator::reconfigureAndRemap()
{
  std::lock_guard<std::recursive_mutex> lock(mtx);
  Logger::info(TAG, "Starting full reconfigure and map regeneration sequence.");
  loadCameraParametersFromDatabase();
  initializeMoil();
  initializeAlphaRhoTables();
  createMaps();

  // Clear the GPU remap cache in FrameProcessor to force re-upload of new maps
  FrameProcessor::clearGpuMapCache();

  Logger::info(TAG, "Reconfiguration complete.");
}

/**
 * @brief Pre-computes the lookup tables for fast alpha-to-rho and rho-to-alpha
 * conversions.
 * @details Optimizations applied:
 * 1. Horner's Method for polynomial evaluation (replaces expensive pow()
 * calls).
 * 2. Vector memory reservation to avoid dynamic reallocations.
 */
void MoildevApplicator::initializeAlphaRhoTables()
{
  // 1. Clear and Reserve Memory
  alpha_to_rho_table.clear();
  rho_to_alpha_table.clear();

  // Reserve known sizes to prevent expensive re-allocations during push_back
  alpha_to_rho_table.reserve(1800);
  rho_to_alpha_table.reserve(3600);

  // 2. Cache Constants locally to avoid repeated namespace lookups
  const double DEG_TO_RAD = static_cast<double>(M_PI) / 180.0;
  const double CALIB_RATIO = Constants::CALIBRATION_RATIO;
  const double P0 = Constants::PARAMETER_0;
  const double P1 = Constants::PARAMETER_1;
  const double P2 = Constants::PARAMETER_2;
  const double P3 = Constants::PARAMETER_3;
  const double P4 = Constants::PARAMETER_4;
  const double P5 = Constants::PARAMETER_5;

  // 3. Build Alpha-to-Rho Table using Horner's Method
  for (int i = 0; i < 1800; ++i)
  {
    // alpha in radians (i is tenths of a degree)
    double alpha = (static_cast<double>(i) / 10.0) * DEG_TO_RAD;

    // Horner's Method Optimization:
    // Original: P0*a^6 + P1*a^5 + ...
    // Optimized: Recursive multiplication/addition (O(n) vs O(n log n))
    double rho =
        (((((P0 * alpha + P1) * alpha + P2) * alpha + P3) * alpha + P4) *
             alpha +
         P5) *
        alpha;

    alpha_to_rho_table.push_back(rho * CALIB_RATIO);
  }

  // 4. Build Rho-to-Alpha Table (Reverse Lookup)
  int i = 0;
  int index = 0; // Represents the pixel radius (rho)

  // Map pixel radius 'index' to the corresponding angle 'i'
  while (i < 1800)
  {
    // While the current pixel radius is within the calculated rho for angle 'i'
    while (index < static_cast<int>(alpha_to_rho_table[i]))
    {
      rho_to_alpha_table.push_back(i);
      index++;
    }
    i++;
  }

  // Fill any remaining entries up to 3600 with the max angle
  while (index < 3600)
  {
    rho_to_alpha_table.push_back(i);
    index++;
  }

  Logger::info(TAG, "Alpha-Rho lookup tables initialized (Optimized). Size: %d",
               (int)rho_to_alpha_table.size());
}

/**
 * @brief Converts pixel coordinates (x, y) from the original fisheye image into
 * angular coordinates (alpha, beta).
 * @return std::pair<float, float> A pair containing the alpha (elevation) and
 * beta (azimuth) angles in degrees.
 */
std::pair<float, float> MoildevApplicator::getAlphaBeta(int x, int y)
{
  // Safety check: ensure table is initialized
  if (rho_to_alpha_table.empty())
  {
    Logger::error(TAG, "getAlphaBeta called but rho_to_alpha_table is empty! "
                       "Maps not initialized?");
    return std::make_pair(0.0f, 0.0f);
  }

  double iCx = Constants::I_CX;
  double iCy = Constants::I_CY;

  // LOG 1: Check inputs and Center Constants (If Center is 0,0 here, DB load
  // failed/delayed)
  Logger::debug(TAG, "getAlphaBeta Input: (%d, %d) | Center used: (%.2f, %.2f)",
                x, y, iCx, iCy);

  double deltaX = static_cast<double>(x) - iCx;
  double deltaY = -(static_cast<double>(y) - iCy);

  double r_px = sqrt(deltaX * deltaX + deltaY * deltaY);
  int r_int = static_cast<int>(round(r_px));

  // LOG 2: Check intermediate math (Delta and Radius)
  Logger::debug(TAG, " -> DeltaX: %.2f, DeltaY: %.2f, Radius: %.2f (int: %d)",
                deltaX, deltaY, r_px, r_int);

  if (r_int < 0 || r_int >= rho_to_alpha_table.size())
  {
    Logger::warn(TAG,
                 "Click at (%d, %d) (radius %.2f) is outside valid circle "
                 "(Table Size: %zu).",
                 x, y, r_px, rho_to_alpha_table.size());
    return std::make_pair(0.0f, 0.0f);
  }

  float alpha = static_cast<float>(rho_to_alpha_table[r_int]) / 10.0f;

  double angle_rad = atan2(deltaY, deltaX);
  double angle_deg = angle_rad * 180.0 / static_cast<double>(M_PI);

  float beta = 90.0f - static_cast<float>(angle_deg);

  // LOG 3: Check Raw Beta before normalization
  Logger::debug(TAG, " -> Raw Alpha: %.2f | Angle Deg: %.2f | Raw Beta: %.2f",
                alpha, angle_deg, beta);

  while (beta <= -180.0f)
    beta += 360.0f;
  while (beta > 180.0f)
    beta -= 360.0f;

  // LOG 4: Final result
  Logger::debug(TAG, " -> Final Result: Alpha=%.2f, Beta=%.2f", alpha, beta);

  return std::make_pair(alpha, beta);
}

std::string MoildevApplicator::getBackendName() const
{
#ifdef _WIN32
  if (moil)
    return moil->getBackendName();
  return "Not Initialized";
#else
#if defined(USE_CUDA)
  return "CUDA";
#elif defined(USE_OPENCL_GPU)
  return "OpenCL";
#else
  return "CPU";
#endif
#endif
}

/**
 * @brief Generates and saves the base map for a single, specific function
 * preset.
 * @param functionRecord The `FunctionRecord` to generate a map for.
 */
void MoildevApplicator::createMap(FunctionRecord &functionRecord)
{
  // Panggil helper generation
  cv::Mat mapX, mapY;
  generateBaseMapForFunction(functionRecord, mapX, mapY);

  if (mapX.empty())
    return;

  std::string sanitizedFuncName = functionRecord.name;
  std::replace(sanitizedFuncName.begin(), sanitizedFuncName.end(), ' ', '_');
  Mediator::getInstance().functionModel->saveMap(mapX, mapY, functionRecord,
                                                 sanitizedFuncName);
  Logger::info(TAG, "Base maps created (OCL) for: %s.",
               functionRecord.name.c_str());
}
/**
 * @brief Regenerates all function maps at a new target resolution.
 * @param targetResolution The new `cv::Size` to be used for all generated maps.
 */
void MoildevApplicator::regenerateAllMapsWithNewResolution(
    const cv::Size &targetResolution)
{
  auto *functionModel = Mediator::getInstance().functionModel.get();
  if (!functionModel)
  {
    // Replaced std::cerr with Logger::error
    Logger::error(TAG, "FunctionModel is null! Cannot regenerate maps.");
    return;
  }

  // Replaced std::cout with Logger::info
  Logger::info(TAG, "Regenerating all maps to new resolution: %dx%d.",
               targetResolution.width, targetResolution.height);

  for (auto &func : functionModel->getMutableRecords())
  {
    if (func.name.find("Original") != std::string::npos)
      continue;

    // OPTIMIZATION: Generate directly at target size using your existing helper
    std::pair<cv::Mat, cv::Mat> maps = generateMapWithDimension(
        (func.name.find("Panorama") != std::string::npos) ? "panorama"
                                                          : "anypoint",
        func.alpha, func.beta, func.zoom, targetResolution.width,
        targetResolution.height);

    // Save directly (No cv::resize needed)
    std::string sanitizedFuncName = func.name;
    std::replace(sanitizedFuncName.begin(), sanitizedFuncName.end(), ' ', '_');
    functionModel->saveMap(maps.first, maps.second, func, sanitizedFuncName);
  }

  // Replaced std::cout with Logger::info
  Logger::info(TAG, "All non-Original maps have been regenerated and resized.");
}

/**
 * @brief Gets the native (full) resolution of the maps.
 * @return cv::Size The native width and height.
 */
cv::Size MoildevApplicator::getNativeMapSize()
{
  if (moil)
    return cv::Size((int)moil->getImageWidth(), (int)moil->getImageHeight());
  return cv::Size(0, 0);
}

/**
 * @brief A helper function to generate a full-resolution base map for a given
 * function.
 * @param[out] mapX The output `cv::Mat` for the X-map.
 * @param[out] mapY The output `cv::Mat` for the Y-map.
 */
void MoildevApplicator::generateBaseMapForFunction(const FunctionRecord &func,
                                                   cv::Mat &mapX,
                                                   cv::Mat &mapY)
{
  if (!moil)
  {
    Logger::error(TAG, "Moil OCL instance not initialized!");
    return;
  }

  int w = static_cast<int>(moil->getImageWidth());
  int h = static_cast<int>(moil->getImageHeight());

  // Alokasi memori
  mapX.create(h, w, CV_32F);
  mapY.create(h, w, CV_32F);

  // Pastikan pointer float valid
  float *pMapX = reinterpret_cast<float *>(mapX.data);
  float *pMapY = reinterpret_cast<float *>(mapY.data);

  // Panggil fungsi OCL melalui pointer
  if (func.name.find("Anypoint") != std::string::npos)
  {
    moil->AnyPointM(pMapX, pMapY, func.alpha, func.beta, func.zoom);
  }
  else if (func.name.find("Panorama") != std::string::npos)
  {
    // Perhatikan parameter PanoramaCar OCL mungkin sedikit berbeda urutannya,
    // sesuaikan dengan header baru Header baru: PanoramaCar(mapX, mapY,
    // alpha_max, alpha_degree, beta_degree, flip_h, flip_v)
    moil->PanoramaCar(pMapX, pMapY,
                      func.alpha, // alpha_max? Cek mapping parameter logic Anda
                      func.beta,  // alpha_degree (center)
                      func.zoom,  // beta_degree
                      false, false);
  }
  else if (func.name.find("Original") != std::string::npos)
  {
    for (int i = 0; i < h; i++)
    {
      // Get pointer to the beginning of the row
      float *rowX = mapX.ptr<float>(i);
      float *rowY = mapY.ptr<float>(i);

      for (int j = 0; j < w; j++)
      {
        // Direct memory access
        rowX[j] = static_cast<float>(j);
        rowY[j] = static_cast<float>(i);
      }
    }
  }
  else
  {
    Logger::error(TAG,
                  "Base map generation not supported for function type: %s.",
                  func.name.c_str());
  }
}

/**
 * @brief Applies cropping (margins) to a base map and saves the result.
 * @param functionRecord The `FunctionRecord` containing the margin values to
 * apply.
 */
void MoildevApplicator::applyMarginToMap(FunctionRecord &functionRecord)
{
  // Replaced std::cout with Logger::debug
  Logger::debug(TAG, "Applying margins to map for function '%s'...",
                functionRecord.name.c_str());

  cv::Mat baseMapX, baseMapY;
  generateBaseMapForFunction(functionRecord, baseMapX, baseMapY);

  if (baseMapX.empty() || baseMapY.empty())
  {
    // Replaced std::cerr with Logger::error
    Logger::error(
        TAG,
        " -> Failed to generate base map for margin application. Aborting.");
    return;
  }

  double norm_left = functionRecord.leftMargin / Constants::MARGIN_MAX;
  double norm_right = functionRecord.rightMargin / Constants::MARGIN_MAX;
  double norm_top = functionRecord.topMargin / Constants::MARGIN_MAX;
  double norm_bottom = functionRecord.bottomMargin / Constants::MARGIN_MAX;

  double crop_ratio_left = norm_left * MAX_HORIZONTAL_CROP_RATIO;
  double crop_ratio_right = norm_right * MAX_HORIZONTAL_CROP_RATIO;
  double crop_ratio_top = norm_top * MAX_VERTICAL_CROP_RATIO;
  double crop_ratio_bottom = norm_bottom * MAX_VERTICAL_CROP_RATIO;

  int map_full_width = baseMapX.cols;
  int map_full_height = baseMapY.rows;

  int roi_x = static_cast<int>(map_full_width * crop_ratio_left);
  int roi_y = static_cast<int>(map_full_height * crop_ratio_top);

  int roi_width = static_cast<int>(map_full_width *
                                   (1.0 - crop_ratio_left - crop_ratio_right));
  int roi_height = static_cast<int>(map_full_height *
                                    (1.0 - crop_ratio_top - crop_ratio_bottom));

  if (roi_width <= 0 || roi_height <= 0)
  {
    // Replaced std::cerr with Logger::error
    Logger::error(
        TAG,
        " -> Invalid crop dimensions calculated (%dx%d). Aborting margin save.",
        roi_width, roi_height);
    return;
  }
  cv::Rect map_roi(roi_x, roi_y, roi_width, roi_height);

  cv::Mat finalMapX = baseMapX(map_roi);
  cv::Mat finalMapY = baseMapY(map_roi);

  std::string sanitizedFuncName = functionRecord.name;
  std::replace(sanitizedFuncName.begin(), sanitizedFuncName.end(), ' ', '_');

  Mediator::getInstance().functionModel->saveMap(
      finalMapX, finalMapY, functionRecord, sanitizedFuncName);

  // Replaced std::cout with Logger::debug
  Logger::debug(TAG, " -> Cropped map for '%s' saved. New size: %dx%d.",
                functionRecord.name.c_str(), finalMapX.cols, finalMapX.rows);
}

/**
 * @brief Generates and saves the base (full-resolution) maps for all function
 * presets.
 */
void MoildevApplicator::createMaps()
{
  auto *functionModel = Mediator::getInstance().functionModel.get();
  if (!functionModel)
    return;

  Logger::info(TAG, "Starting OCL base map generation.");

  for (auto &func : functionModel->getMutableRecords())
  {
    if (func.name.find("Original") != std::string::npos)
      continue;

    cv::Mat baseMapX, baseMapY;
    generateBaseMapForFunction(func, baseMapX,
                               baseMapY); // Sudah menggunakan OCL di dalamnya

    std::string sanitizedFuncName = func.name;
    std::replace(sanitizedFuncName.begin(), sanitizedFuncName.end(), ' ', '_');
    functionModel->saveMap(baseMapX, baseMapY, func, sanitizedFuncName);
  }
  Logger::info(TAG, "OCL Maps generation finished.");
}

/**
 * @brief Implementation of generateMapWithDimension
 * @details Generates map matrices on-the-fly without saving to the database.
 */
std::pair<cv::Mat, cv::Mat>
MoildevApplicator::generateMapWithDimension(const std::string &type,
                                            float alpha, float beta, float zoom,
                                            int width, int height)
{
  std::lock_guard<std::recursive_mutex> lock(mtx);
  if (!moil)
    return {cv::Mat(), cv::Mat()};

  // 1. Ambil ukuran asli yang diharapkan engine
  int nativeW = static_cast<int>(moil->getImageWidth());
  int nativeH = static_cast<int>(moil->getImageHeight());

  // 2. Selalu buat canvas sesuai ukuran native engine agar tidak overflow
  cv::Mat nativeMapX(nativeH, nativeW, CV_32F);
  cv::Mat nativeMapY(nativeH, nativeW, CV_32F);

  float *pMapX = reinterpret_cast<float *>(nativeMapX.data);
  float *pMapY = reinterpret_cast<float *>(nativeMapY.data);

  if (type == "panorama")
  {
    moil->PanoramaCar(pMapX, pMapY, alpha, beta, zoom, false, false);
  }
  else
  {
    moil->AnyPointM(pMapX, pMapY, alpha, beta, zoom);
  }

  // 3. Jika user meminta resolusi berbeda, resize SETELAH operasi engine selesai
  if (nativeW != width || nativeH != height)
  {
    cv::Mat resizedX, resizedY;
    cv::resize(nativeMapX, resizedX, cv::Size(width, height));
    cv::resize(nativeMapY, resizedY, cv::Size(width, height));
    return {resizedX, resizedY};
  }

  return {nativeMapX, nativeMapY};
}