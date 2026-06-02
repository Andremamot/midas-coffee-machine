/**
 * @file frame_processor.hpp
 * @brief Defines the FrameProcessor namespace for handling video frame processing and display.
 * @details This header declares the structures, enums, global variables, and function signatures
 * used for asynchronous frame processing. It defines the `FunctionProcess` struct to bundle all
 * necessary data for a single processing task and outlines the public interface for the namespace.
 */
#ifndef FRAMEPROCESSOR_HPP
#define FRAMEPROCESSOR_HPP

#include <opencv2/opencv.hpp>
#include "constants/constants.hpp"
#include <gtk/gtk.h>
#include <vector>
#include <future>
#include "models/frame_model.hpp"
#include "models/function_model.hpp"
#include "helpers/thread_pool.hpp"
#include "helpers/zone_policy.hpp"

/**
 * @namespace FrameProcessor
 * @brief Async Rendering Engine for Fisheye Transformations.
 * @details This namespace manages the parallel processing pipeline that converts raw camera frames
 * into the various user-configured views (Panorama, Anypoint) for display.
 *
 * **Architecture**:
 * - **Producer**: The main loop captures a frame and spawns multiple `FunctionProcess` tasks.
 * - **Consumer**: The `ThreadPool` executes these tasks, performing `cv::remap` and resizing.
 * - **UI Update**: Completed frames are marshaled back to the Main Thread to update GTK widgets.
 */
namespace FrameProcessor
{
    /// The most recent raw frame captured from the camera or video file.
    extern cv::Mat originalFrame;
    
    extern cv::Mat originalFrameCam1;
    extern cv::Mat originalFrameCam2;

    /// A scaled version of the original frame, used for certain UI elements.
    extern cv::Mat scaledFrame;
    /// The current frame being processed from a playback video file.
    extern cv::Mat playbackFrame;
    /// The global OpenCV VideoCapture object for reading from cameras or files.
    extern cv::VideoCapture cap;

    /// Mutex to protect access to global frame variables (Issue 4)
    extern std::mutex frameMutex;

    /**
     * @enum ViewType
     * @brief Specifies the type of transformation to be applied to a frame.
     */
    enum class ViewType
    {
        ORIGINAL, ///< Display the frame with minimal processing (e.g., resizing, drawing overlays).
        PANORAMA, ///< Apply a panoramic remapping transformation.
        ANYPOINT, ///< Apply an anypoint (rectified) remapping transformation.
    };
    
    struct DetectionData {
    std::vector<cv::Point2f> centers;
    std::vector<cv::Rect> boxes;
    std::vector<cv::Scalar> colors;
};

    /**
     * @struct FunctionProcess
     * @brief Unit of Work for the Rendering Pipeline.
     * @details Encapsulates all context required to transform *one* generic frame into *one* specific
     * view for a specific UI widget.
     *
     * **Lifecycle**:
     * 1. Created in the Main Loop (Producer).
     * 2. Copied into the `ThreadPool` queue.
     * 3. Processed by a Worker Thread (Consumer).
     * 4. Resulting `cv::Mat` is pushed to the UI.
     */
   struct FunctionProcess
{
    cv::Mat frame;
    GtkImage *widget;
    GtkBox *borderBox = nullptr;
    ViewType viewType;
    FrameRecord frameRecord;
    FunctionRecord functionRecord;
    int64_t sequenceId = 0;
    
    // PERBAIKAN: Gunakan struct deteksi agar data sinkron dengan frame ini
    DetectionData detection; 
    
    ZonePolicy resolvedPolicy;

    FunctionProcess(const cv::Mat &f, GtkImage *w, ViewType vt,
                    const FrameRecord &fr, const FunctionRecord &func_r,
                    const DetectionData &det, const ZonePolicy &zp,
                    GtkBox *bb = nullptr)
        : frame(f), widget(w), viewType(vt), frameRecord(fr),
          functionRecord(func_r), detection(det), resolvedPolicy(zp), borderBox(bb) {}
};

    /// A map to maintain the current alert state of each border box to avoid redundant UI updates.
    static std::unordered_map<GtkBox *, bool> alertStateMap;
    /// A mutex to protect access to the `alertStateMap` from multiple threads.
    static std::mutex alertStateMutex;

    /**
     * @brief Dispatcher: Schedules Parallel Rendering Tasks.
     * @details Iterates through the list of `FunctionProcess` jobs and submits them to the `ThreadPool`.
     * This ensures that multi-view layouts are rendered concurrently, utilizing all CPU cores.
     *
     * @param fps Vector of tasks to execute.
     */
    void processMultipleTasksAsync(const std::vector<FunctionProcess> &fps);

    /** @brief Clears the internal GPU map cache. Called when camera parameters change. */
    void clearGpuMapCache();

    /**
     * @brief Worker: Executes a Single Rendering Task.
     * @details
     * 1. **Resolve**: Determines if the view needs remapping (Panorama/Anypoint) or just resizing (Original).
     * 2. **Transform**: Calls `remapFrame` or `resizeFrame` accordingly.
     * 3. **Analyze**: (Optional) Checks zone policies for border alerts.
     * 4. **Display**: Uses `g_idle_add` to safely update the `GtkImage` on the main thread.
     *
     * @param fp The job context.
     */
    void processFramesAsynchronously(const FunctionProcess &fp);

    /**
     * @brief Core Transformation: Applies Optical Dewarping.
     * @details Uses `cv::remap` with the pre-calculated LUTs (`mapX`, `mapY`) stored in the `FunctionRecord`.
     * This is the most computationally expensive operation in the pipeline.
     *
     * @param frame Raw fisheye input.
     * @param frameRecord Target display layout configuration.
     * @param functionRecord Optical transformation parameters.
     * @return The dewarped image, ready for display.
     */
    cv::Mat remapFrame(const cv::Mat &frame, const FrameRecord &frameRecord, const FunctionRecord &functionRecord);

    /**
     * @brief Resizes a frame to fit within specified dimensions while maintaining aspect ratio.
     * @param frame The source cv::Mat frame.
     * @param Dimension The target cv::Size.
     * @return cv::Mat The resized image.
     */
    cv::Mat resizeFrame(const cv::Mat &frame, const cv::Size &Dimension);

    cv::Mat fit_and_stretch_single_pass(const cv::Mat &source, int target_w, int target_h, bool stretchX2);
}
#endif // FRAMEPROCESSOR_HPP
