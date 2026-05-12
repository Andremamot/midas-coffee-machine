// #ifdef V2H
// #pragma once

// #include <opencv2/opencv.hpp>
// #include <camera_multiplexer.h>
// #include <utils/logger.h>
// #include <mutex>
// #include <condition_variable>
// #include <dmabuf.h>

// enum class StateCameraCapture
// {
//     STOPPED,
//     CAPTURING
// };

// enum class StateMemoryWrite
// {
//     READY,
//     WRITING
// };

// struct ThreadCapLockData
// {
//     std::mutex cap_data_mutex;           ///< Lock for cap data access
//     std::condition_variable cap_data_cv; ///< CV to signal new data availability

//     std::mutex state_mutex;           ///< Lock for state modifications
//     std::condition_variable state_cv; ///< CV to signal state changes
// };

// struct ThreadCapData
// {
//     cv::Mat current_frame; ///< Raw frame from camera
//     CameraMultiplexer multiplexer;
//     StateCameraCapture state = StateCameraCapture::STOPPED;

//     dma_buffer *drpai_buf_cup = nullptr;
//     dma_buffer *drpai_buf_face = nullptr;
// };

// namespace threads
// {
//     extern ThreadCapData cap_data;
//     extern ThreadCapLockData cap_lock_data;
//     void t_capture();
// }

// #endif