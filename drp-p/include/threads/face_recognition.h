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
#include <linux/drpai.h>
#include <constants/define_face.h>
#include <constants/define_drpai.h>
#include <linux/drpai.h>
#include <fcntl.h>
// #include <corecrt_io.h>
#include <threads/drp_queue.h>
#endif


/**
 * @enum StateFaceRecognition
 * @brief Represents the various states of the face recognition pipeline.
 *
 * - IDLE: Waiting, no process running.
 * - RUNNING: Detection/recognition actively running.
 * - SUCCESS: Recognition completed successfully.
 * - PAUSE: Temporarily paused (e.g., changing camera, UI transition).
 */
enum class StateFaceRecognition {
    IDLE,
    RUNNING,
    SUCCESS,
    PAUSE,
};


/**
 * @struct ThreadFaceLockData
 * @brief Synchronization primitives for face recognition threads.
 *
 * This struct contains mutexes and condition variables used to guarantee
 * thread-safe access to shared face data and to coordinate state transitions.
 */
struct ThreadFaceLockData {
    std::mutex face_data_mutex;              ///< Lock for face data access
    std::condition_variable face_data_cv;    ///< CV to signal new data availability

    std::mutex state_mutex;                  ///< Lock for state modifications
    std::condition_variable state_cv;        ///< CV to signal state changes
};


/**
 * @struct ThreadFaceData
 * @brief Contains shared data used by face detection and recognition threads.
 *
 * This struct holds:
 * - The latest camera frames
 * - Detected faces and ROIs
 * - Recognition results
 * - Embedding vectors
 * - Recognition state and metadata
 *
 * All fields in this struct are meant to be accessed only when protected
 * via `ThreadFaceLockData::face_data_mutex` to ensure thread safety.
 */
struct ThreadFaceData {
    cv::Mat current_frame;                  ///< Raw frame from camera
    cv::Mat detected_frame;                 ///< Frame after drawing detection boxes

    std::vector<Detection> detections;      ///< Detection objects (face detector output)
    std::vector<cv::Rect> detected_face_rect; ///< Bounding boxes for faces
    int detected_face_index = -1;           ///< Index of selected/primary face

    StateFaceRecognition state = StateFaceRecognition::IDLE; ///< Current pipeline state

    bool is_face_in_position = false;       ///< Whether face is in the required ROI
    cv::Rect limit_rect;                    ///< Required ROI threshold rectangle

    int recognized_face_id = -1;            ///< ID of successfully recognized face
    int last_recognized_face_id = -1;       ///< Previously recognized face ID

    std::vector<float> new_face_embedding;  ///< Embedding calculated for new/unknown face

    /**
     * @brief Timestamp to track recognition failure timeout.
     *
     * Stored atomically to avoid data races when accessed across threads.
     */
    std::chrono::steady_clock::time_point fail_recognition_state_time;
};


/**
 * @brief Computes and sets a limit rectangle inside the given frame.
 *
 * @param frame Input image whose size determines the allowed face ROI region.
 */
void set_limit_rect(cv::Mat frame);


namespace threads {

    /**
     * @brief Global shared face detection/recognition data.
     *
     * Must be accessed under proper locking using face_lock_data.
     */
    extern ThreadFaceData face_data;

    /**
     * @brief Global lock and synchronization data used by the face pipeline.
     */
    extern ThreadFaceLockData face_lock_data;


    /**
     * @brief Main worker thread for face detection.
     *
     * Steps performed:
     * - Capture frame from camera
     * - Run face detection (DRP-AI or CPU)
     * - Update `face_data`
     * - Notify recognition thread if needed
     */
    void t_face_detection();


    /**
     * @brief Main worker thread for face recognition.
     *
     * Steps performed:
     * - Wait for new detection data
     * - Align face
     * - Generate embedding
     * - Compare with database
     * - Update recognition states in `face_data`
     */
    void t_face_recognition();


    /**
     * @brief Internal helper: checks whether required face data is available.
     *
     * @return true if face data exists and is valid.
     */
    static bool check_face_data_();


    /**
     * @brief Resets internal shared face data to initial conditions.
     *
     * Clears most fields inside `face_data` so that the detection pipeline
     * can restart with a clean state.
     */
    void reset_face_data_();

    std::tuple<std::vector<Detection>, long, long> infer_drp_face(cv::Mat &frame);

} // namespace threads

