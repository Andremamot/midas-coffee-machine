// #ifdef V2H
// #include <threads/capture.h>

// namespace threads
// {
//     ThreadCapData cap_data;
//     ThreadCapLockData cap_lock_data;

//     void init_capture()
//     {
//         int ret = 0;
//         threads::cap_data.drpai_buf_face = (dma_buffer *)malloc(sizeof(dma_buffer));
//         ret = buffer_alloc_dmabuf(threads::cap_data.drpai_buf_face, IMAGE_WIDTH * IMAGE_HEIGHT * BGR_CHANNEL);
//         if (-1 == ret)
//         {
//             UTIL_LOG_ERROR("Failed to Allocate DMA buffer for the drpai_buf (Face)");
//             free(threads::cap_data.drpai_buf_face);
//         }
//         threads::cap_data.drpai_buf_cup = (dma_buffer *)malloc(sizeof(dma_buffer));
//         ret = buffer_alloc_dmabuf(threads::cap_data.drpai_buf_cup -, IMAGE_WIDTH * IMAGE_HEIGHT * BGR_CHANNEL);
//         if (-1 == ret)
//         {
//             UTIL_LOG_ERROR("Failed to Allocate DMA buffer for the drpai_buf (Cup)");
//             free(threads::cap_data.drpai_buf_cup);
//         }
//     }

//     void t_capture()
//     {
//         while (true)
//         {
//             {
//                 std::unique_lock<std::mutex> lock(threads::cap_lock_data.state_mutex);
//                 threads::cap_lock_data.state_cv.wait(
//                     lock,
//                     []
//                     { return threads::cap_data.state == StateCameraCapture::CAPTURING; });
//             }
//             cv::Mat frame_face = CameraMultiplexer::get_instance().get_frame("face_cam");
//             cv::Mat frame_cup = CameraMultiplexer::get_instance().get_frame("cup_cam");

//             if (frame_face.empty() || frame_cup.empty())
//             {
//                 UTIL_LOG_WARNING("Camera frame is empty.");
//                 std::this_thread::sleep_for(std::chrono::milliseconds(100));
//                 continue;
//             }

//             {
//                 std::unique_lock<std::mutex> lock(threads::cap_lock_data.cap_data_mutex);

//                 memcpy(threads::cap_data.drpai_buf_face->mem, frame_face.data, threads::cap_data.drpai_buf_face->size);
//                 memcpy(threads::cap_data.drpai_buf_cup->mem, frame_cup.data, threads::cap_data.drpai_buf_cup->size);

//                 // Flush Buffer
//                 int ret = 0;
//                 ret = buffer_flush_dmabuf(threads::cap_data.drpai_buf_face->idx, threads::cap_data.drpai_buf_face->size);
//                 if (0 != ret)
//                 {
//                     UTIL_LOG_ERROR("Capture Thread Terminated (Face)");
//                     pthread_exit(NULL);
//                 }
//                 ret = buffer_flush_dmabuf(threads::cap_data.drpai_buf_cup->idx, threads::cap_data.drpai_buf_cup->size);
//                 if (0 != ret)
//                 {
//                     UTIL_LOG_ERROR("Capture Thread Terminated (Cup)");
//                     pthread_exit(NULL);
//                 }
//             }
//         }
//     }
// }
// #endif