#pragma once

#include <opencv2/opencv.hpp>
#include <camera_multiplexer.h>
#include <utils/logger.h>
#include <mutex>
#include <condition_variable>
#include <detections/ai.h>
#include <databases/database.h>
#include <chrono>
#include <atomic>

#include <detections/box.h>

#ifdef V2H
#include <constants/define_cup.h>
#include <constants/define_drpai.h>
#include <linux/drpai.h>
#include <fcntl.h>
// #include <corecrt_io.h>
#include <threads/drp_queue.h>
#endif


/**
 * @enum StateCupDetection
 * @brief Represents the various states of the cup detection pipeline.
 *
 * - IDLE: Waiting, no detection process running.
 * - RUNNING: Detection actively running.
 * - SUCCESS: Cup successfully detected and positioned.
 * - PAUSE: Temporarily paused (e.g., UI transition).
 */
enum class StateCupDetection {
    IDLE,
    RUNNING,
    SUCCESS,
    PAUSE,
};


/**
 * @struct ThreadCupLockData
 * @brief Synchronization primitives for cup detection threads.
 *
 * Contains mutexes and condition variables used to ensure
 * thread-safe access to shared cup data and to coordinate
 * state transitions between threads.
 */
struct ThreadCupLockData {
    std::mutex cup_data_mutex;           ///< Lock for cup data access
    std::condition_variable cup_data_cv; ///< CV to signal new data availability

    std::mutex state_mutex;               ///< Lock for state modifications
    std::condition_variable state_cv;     ///< CV to signal state changes
};


/**
 * @struct ThreadCupData
 * @brief Contains shared data used by the cup detection pipeline.
 *
 * This struct holds:
 * - Latest camera frame
 * - Detection and estimation results
 * - Cup position metadata
 * - Current pipeline state
 *
 * All members must be accessed only while holding
 * `ThreadCupLockData::cup_data_mutex`.
 */
struct ThreadCupData {
    cv::Mat current_frame;                   ///< Raw frame from camera

    std::vector<Detection> detections;  ///< Detected cup bounding boxes

    StateCupDetection state = StateCupDetection::IDLE; ///< Current pipeline state

    bool is_center_in_rim = false;            ///< Whether cup is positioned correctly

    cv::Point center_point;                  ///< Center point of detected cup

    CupMeasurement measurement;              ///< Estimated cup size and volume
};


namespace threads {

    /**
     * @brief Global shared cup detection data.
     *
     * Must be accessed under proper locking using `cup_lock_data`.
     */
    extern ThreadCupData cup_data;

    /**
     * @brief Global lock and synchronization data used by the cup pipeline.
     */
    extern ThreadCupLockData cup_lock_data;


    /**
     * @brief Main worker thread for cup detection.
     *
     * Steps performed:
     * - Wait until state becomes IDLE
     * - Capture frame from camera
     * - Run cup detection & measurement
     * - Update `cup_data`
     * - Transition to SUCCESS state when cup is correctly placed
     */
    void t_cup_detection();


    /**
     * @brief Resets internal shared cup detection data.
     *
     * Clears detection results and measurement data so that
     * the detection pipeline can restart with a clean state.
     */
    void reset_cup_data_();

    std::tuple<std::vector<Detection>, long, long> infer_drp_cup(cv::Mat &frame);

} // namespace threads
