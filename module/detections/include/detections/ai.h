#pragma once

#include <memory>
#include <logger/logger.h>
#ifdef V2H
#include <constants/define_cup.h>
#include <constants/define_drpai.h>
#include <constants/define_midas.h>
// #include <constants/define_face.h>
#include <fcntl.h>
#include <linux/drpai.h>
#include <sys/ioctl.h>
#include <unistd.h>
#endif

// // =======================
// // Include Face Modules
// // =======================
// #include <detections/face/arcface.h>
// #include <detections/face/face_aligner.h>

// #ifdef V2H
// #include <detections/face/face_detector_v2h.h>
// #else
// #include <detections/face/face_detector.h>
// #endif

// =======================
// Include Cup Modules
// =======================
// #include <detections/cup/cup_size_estimator.h>

#if defined(V2H) && (DRP_AI_TVM_RUNTIME == 1)
#include <detections/cup/cup_detector_v2h.h>
#include <detections/midas/midas_estimator_v2h.h>
#else
#include <detections/cup/cup_detector.h>
#include <detections/midas/midas_estimator.h>
#endif

#ifdef V2H
uint64_t get_drpai_start_addr(int drpai_fd);
uint64_t init_drpai(int drpai_fd);
#endif

/**
 * @class AI
 * @brief A singleton class that initializes and manages all AI-related modules
 *        such as face detection, alignment, and recognition.
 *
 * This class ensures a single global instance for all AI subsystems,
 * preventing redundant model loading and ensuring consistent use across the
 * application.
 *
 * Example:
 * @code
 * AI* ai = AI::get_instance();
 * ai->face_detector->detect(img);
 * @endcode
 */

class AI {
public:
    /**
     * @brief Retrieves the singleton instance of the AI system.
     *
     * @return Pointer to the singleton AI instance.
     */
    static AI* get_instance() {
        static AI instance;  // Guaranteed to be created once (thread-safe in C++11+)
        return &instance;
    }

    // ===========================
    // Public AI modules
    // Managed using unique_ptr for safe auto-destruction
    // ===========================

#ifdef V2H
    // std::unique_ptr<FaceDetectorV2H> face_detector;       ///< Hardware-optimized detector (V2H)
    // std::unique_ptr<MeraDrpRuntimeWrapper> face_runtime;  ///< MERA DRPAI runtime
    // bool face_runtime_status = false;
#if (1) == DRP_AI_TVM_RUNTIME
    /**
     * @brief Lazy-load MiDaS hardware context.
     * @return true on success or if already loaded.
     */
    bool LoadMidas();
    void ReloadPre(const std::string& dir);

    bool IsMidasLoaded() const { return is_midas_loaded; }

    /* Track loaded pre-processing to avoid redundant reloads */
    std::string current_pre_dir = "";

    /* AI Module Accessors */
    std::unique_ptr<CupDetectorV2H>   cup_detector;
    std::unique_ptr<MidasEstimatorV2H> midas_estimator;

    /* DRP-AI TVM Runtimes */
    std::unique_ptr<MeraDrpRuntimeWrapper> cup_runtime;
    std::unique_ptr<MeraDrpRuntimeWrapper> midas_runtime;

    /* Shared Pre-processing hardware unit */
    std::unique_ptr<PreRuntime> shared_preruntime;

    bool cup_runtime_status   = false;
    bool midas_runtime_status = false;
    bool is_midas_loaded      = false;

    /* Physical memory base for DRP-AI */
    uint64_t drpaimem_addr_start = 0;
    int      drpai_fd            = -1;
#else
    std::unique_ptr<CupDetector> cup_detector;    ///< Default cup detector
    std::unique_ptr<MidasEstimatorV2H> midas_estimator;  ///< Placeholder (will be V2H only)
#endif

    int32_t drpai_freq = DRPAI_FREQ;

#else
    // std::unique_ptr<FaceDetector> face_detector;  ///< Default face detector
    std::unique_ptr<CupDetector>    cup_detector;      ///< Default cup detector
    std::unique_ptr<MidasEstimator> midas_estimator;   ///< TFLite MiDaS Depth Estimator
#endif

    // std::unique_ptr<FaceAligner> face_aligner;  ///< Face alignment engine
    // std::unique_ptr<ArcFace> arcface;           ///< Face embedding model

    // std::unique_ptr<CupSizeEstimator> cup_estimator; ///< Cup size estimator

    /**
     * @brief Destructor (default).
     * Unique_ptr members automatically free resources.
     */
    ~AI();

private:
    /**
     * @brief Private constructor (singleton pattern).
     *
     * Loads all necessary AI components (face detection, alignment,
     * recognition).
     */
    AI();

    /// Deleted copy constructor (singleton pattern)
    AI(const AI&) = delete;

    /// Deleted assignment operator (singleton pattern)
    AI& operator=(const AI&) = delete;
};
