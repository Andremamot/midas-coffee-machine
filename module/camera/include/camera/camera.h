/**
 * @file camera.h
 * @brief Declaration of the Camera class.
 *
 * Camera provides continuous frame capture from a camera source and exposes
 * the latest frame through a thread-safe API. Capture is performed on a dedicated
 * background thread and stored in an internal buffer.
 *
 * The camera source is represented as a variant:
 * - int: V4L2 device index (e.g., 0 for /dev/video0)
 * - std::string: stream path/URL (reserved for future/alternate backends)
 */

#pragma once

#include <camera/define.h>
#include <logger/logger.h>

#include <atomic>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <string>
#include <thread>
#include <variant>

/**
 * @brief Threaded camera capture wrapper.
 *
 * A Camera instance owns a capture thread that repeatedly opens the configured
 * source, reads frames, and stores the newest frame in memory.
 *
 * Typical usage:
 * @code
 * Camera cam(0, false);
 * cam.start_camera();
 * cv::Mat frame = cam.get_frame();
 * cam.stop_camera();
 * @endcode
 *
 * @note This class is designed to be used by CameraMultiplexer, but can also be used directly.
 */
class Camera {
public:
    /**
     * @brief Construct a Camera for the specified source.
     *
     * If @p autostart is false, capture may start automatically depending on implementation.
     * Otherwise, the user must call start_camera().
     *
     * @param source Camera source (device index or stream URL/path).
     * @param autostart If true, disables auto-start on construction.
     */
    Camera(std::variant<int, std::string> source, bool autostart);

    /**
     * @brief Destroy the Camera and release all resources.
     *
     * Stops the capture thread (if running) and releases VideoCapture and buffers.
     */
    ~Camera();

    /**
     * @brief Start the camera capture thread.
     *
     * Safe to call multiple times; has no effect if already running.
     */
    void start_camera();

    /**
     * @brief Stop the capture thread and release the camera device.
     *
     * Safe to call multiple times; has no effect if already stopped.
     */
    void stop_camera();

    /**
     * @brief Get the latest captured frame.
     *
     * The returned frame is a copy of the internal buffer.
     * If no frame is available, an empty cv::Mat is returned.
     *
     * @return Latest frame or empty cv::Mat.
     */
    cv::Mat get_frame();

private:
    /**
     * @brief Internal capture loop run by camera_thread.
     *
     * Continuously opens the camera source, reads frames, and updates the internal buffer
     * while @ref running is true.
     */
    void camera_loop();

    /** @brief Background thread responsible for continuous frame capture. */
    std::thread camera_thread;

    /** @brief Capture loop state flag (true while camera_loop should keep running). */
    std::atomic<bool> running;

    /** @brief OpenCV VideoCapture handle (GStreamer/V4L2 backend). */
    cv::VideoCapture cap;

    /** @brief Mutex protecting access to the frame buffer. */
    std::mutex frame_mutex;

    /** @brief Latest captured frame buffer. */
    cv::Mat frame;

    /** @brief Additional frame buffer/cup reserved for internal use. */
    cv::Mat frame_cup;

    /** @brief Configured camera source (device index or stream URL/path). */
    std::variant<int, std::string> camera_source;
};
