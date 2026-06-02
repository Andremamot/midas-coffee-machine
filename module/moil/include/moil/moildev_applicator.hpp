/**
 * @file moildev_applicator.hpp
 * @brief Defines the MoildevApplicator singleton class for high-level Moildev operations.
 * @details This header file declares the `MoildevApplicator` class, which serves as a high-level
 * manager for the core `Moildev` library. It handles loading camera parameters, initializing the
 * `Moildev` engine, and provides an interface to generate and manage remapping matrices (maps).
 */
#ifndef MOILDEV_APPLICATOR_HPP
#define MOILDEV_APPLICATOR_HPP

#include <string>
#include <memory> // Untuk std::unique_ptr
#include <opencv2/core.hpp>

// Include header OpenCL yang baru
// #include "lib/moildev_ocl.hpp"
#include "constants/constants.hpp"
#include "models/frame_model.hpp"
#include <utility>
#include "models/function_model.hpp"
#include <mutex>

#ifdef _WIN32

#include "lib/moildev.hpp"

using MoilEngine = moildev::Moildev;

#else
#ifdef USE_CUDA
#include "lib/moildev_ocl.hpp"
using MoilEngine = moildev::ocl::Moildev;
#elif defined(USE_OPENCL_GPU)
#include "lib/moildev_ocl.hpp"
using MoilEngine = moildev::ocl::Moildev;
#else
#include "lib/moildev_cpu.hpp"
using MoilEngine = moildev::cpu::Moildev;
#endif
#endif

/**
 * @class MoildevApplicator
 * @brief High-Level Abstraction for the MOIL (Fisheye) SDK.
 * @details This Singleton acts as the "Optical Engine" of the application. It wraps the low-level `Moildev` library
 * and provides business-logic-aware methods to generate, cache, and manage the remapping matrices (maps)
 * required to dewarp fisheye images.
 *
 * **Key Responsibilities**:
 * - **Initialization**: Loads camera calibration (`.json`/`.xml`) into the Moil Engine.
 * - **Map Generation**: Creates discrete `mapX`/`mapY` files for specific views (Anypoint, Panorama).
 * - **Optimization**: Handles the "Regenerate Maps" feature to trade off between performance (low res) and quality (high res).
 */
class MoildevApplicator
{
public:
    /**
     * @brief Gets the native (full) resolution of the maps as determined by the initialized Moildev camera parameters.
     * @return cv::Size The native width and height.
     */
    cv::Size getNativeMapSize();

    // Deleted copy and move semantics to enforce the singleton pattern.
    MoildevApplicator(const MoildevApplicator &) = delete;
    MoildevApplicator &operator=(const MoildevApplicator &) = delete;
    MoildevApplicator(MoildevApplicator &&) = delete;
    MoildevApplicator &operator=(MoildevApplicator &&) = delete;

    /**
     * @brief Gets the singleton instance of the MoildevApplicator.
     * @return MoildevApplicator& A reference to the singleton instance.
     */
    static MoildevApplicator &getInstance();

    /**
     * @brief Generates/Caches the Remap Table for a Single Preset.
     * @details Calculates the `mapX` and `mapY` matrices based on the `FunctionRecord`'s alpha/beta/zoom
     * parameters and saves them to disk.
     * @param functionRecord The preset to process. Modified in-place (map paths updated).
     */
    void createMap(FunctionRecord &functionRecord);

    /**
     * @brief Batch Generator: Creates Maps for ALL Presets.
     * @details Iterates through the entire `FunctionModel` database and generates missing maps.
     * Called on startup to ensure the cache is warm.
     */
    void createMaps();

    /**
     * @brief Global Resolution Rescaling (Performance Tuning).
     * @details Re-computes ALL maps at a specific target resolution.
     *
     * **Use Case**:
     * - **Low Spec PC**: User sets "Performance Mode" -> Maps generated at 480p.
     * - **High Spec PC**: User sets "Quality Mode" -> Maps generated at 1080p.
     *
     * @param targetResolution The new dimensions for all cached maps.
     */
    void regenerateAllMapsWithNewResolution(const cv::Size &targetResolution);

    /**
     * @brief Applies cropping (margins) to a base map and saves the result.
     * @details It regenerates a clean, full-resolution base map, calculates a Region of Interest (ROI)
     * based on the margin parameters in the `functionRecord`, crops the base map to this ROI,
     * and then saves the final cropped map, overwriting the previous file.
     * @param functionRecord The `FunctionRecord` containing the margin values to apply.
     */
    void applyMarginToMap(FunctionRecord &functionRecord);

    /**
     * @brief A helper function to generate a full-resolution base map for a given function.
     * @details This function does not save the map; it only computes the `mapX` and `mapY` matrices in memory.
     * @param func The `FunctionRecord` containing the parameters (alpha, beta, zoom) for the map.
     * @param[out] mapX The output `cv::Mat` for the X-map.
     * @param[out] mapY The output `cv::Mat` for the Y-map.
     */
    void generateBaseMapForFunction(const FunctionRecord &func, cv::Mat &mapX, cv::Mat &mapY);

    /**
     * @brief Hot-Reloads Camera Calibration.
     * @details Called when the user changes the camera parameters (Center X/Y, Radius) in Settings.
     * It reinits the Moil Engine and forces a regeneration of all maps to match the new lens optics.
     */
    void reconfigureAndRemap();

    /**
     * @brief Generates a remapping map with specific dimensions and parameters.
     * @details This function creates a map of a given type (e.g., "Anypoint") with the specified
     * angular parameters and target dimensions, without relying on a stored `FunctionRecord`.
     * @param type The type of map to generate (e.g., "Anypoint", "Panorama").
     * @param alpha The vertical viewing angle (elevation) in degrees.
     * @param beta The horizontal viewing angle (azimuth) in degrees.
     * @param zoom The magnification factor.
     * @param width The target width of the output map.
     * @param height The target height of the output map.
     * @return std::pair<cv::Mat, cv::Mat> A pair containing the generated `mapX` and `mapY`.
     */
    std::pair<cv::Mat, cv::Mat> generateMapWithDimension(
        const std::string &type,
        float alpha, float beta, float zoom,
        int width, int height);

    /**
     * @brief Inverse Kinematics: Pixel to Angle.
     * @details Converts a 2D point on the fisheye image to its corresponding 3D spherical coordinates (Alpha/Beta).
     * Used for "Click-to-Move" functionality (e.g., clicking on the raw image to center the view there).
     *
     * @param x Horizontal pixel position.
     * @param y Vertical pixel position.
     * @return Pair of {Alpha (Tilt), Beta (Pan)} angles.
     */
    std::pair<float, float> getAlphaBeta(int x, int y);

    /**
     * @brief Gets the name of the active backend (e.g., "CUDA", "OpenCL", "CPU").
     * @return std::string The backend name.
     */
    std::string getBackendName() const;

    /**
     * @brief Provides direct access to the underlying Moildev engine instance.
     * @return Moildev& A reference to the internal `Moildev` object.
     */
    // Getter pointer ke engine OCL jika dibutuhkan akses langsung
    MoilEngine *getMoildev()
    {
        return moil.get();
    }

private:
    /** @brief Private constructor to enforce the singleton pattern. */
    MoildevApplicator();

    /** @brief Default destructor. */
    ~MoildevApplicator() = default;

    /// A flag to track if the Moildev instance has been created (not currently used).
    static bool moilCreated;
    /// The core Moildev engine instance.
    std::unique_ptr<MoilEngine> moil;

    std::recursive_mutex mtx;

    /** @brief Initializes the internal Moildev object with the loaded camera parameters. */
    void initializeMoil();
    /** @brief Loads camera parameters from the database via the `Constants` utility. */
    void loadCameraParametersFromDatabase();
    /** @brief Pre-computes the lookup tables for fast alpha-to-rho and rho-to-alpha conversions. */
    void initializeAlphaRhoTables();

    /// Lookup table for converting elevation angle (alpha) to pixel radius (rho).
    std::vector<double> alpha_to_rho_table;
    /// Lookup table for converting pixel radius (rho) to elevation angle (alpha).
    std::vector<int> rho_to_alpha_table;
};

#endif // MOILDEV_APPLICATOR_HPP