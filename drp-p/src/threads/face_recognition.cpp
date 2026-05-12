#include <threads/face_recognition.h>

using Clock = std::chrono::steady_clock;

void set_limit_rect(cv::Mat frame)
{
    std::lock_guard<std::mutex> lock(threads::face_lock_data.face_data_mutex);
    int frame_width = frame.cols;
    int frame_height = frame.rows;

    int box_width = static_cast<int>(frame_width * 0.7);
    int box_height = static_cast<int>(frame_height * 0.7);
    int box_x = (frame_width - box_width) / 2;
    int box_y = (frame_height - box_height) / 2;

    threads::face_data.limit_rect = cv::Rect(box_x, box_y, box_width, box_height);
}

namespace threads
{
    ThreadFaceData face_data;
    ThreadFaceLockData face_lock_data;

    void t_face_detection()
    {
        while (true)
        {
            AI *detector = AI::get_instance();
            // #ifdef V2H

            //             // get current drpai hardware status
            //             drpai_status_t drpai_status;
            //             ioctl(detector->drpai_fd, DRPAI_GET_STATUS, &drpai_status);

            //             // if drpai is still running, wait
            //             if (drpai_status.status == DRPAI_STATUS_RUN)
            //             {
            //                 std::this_thread::sleep_for(std::chrono::milliseconds(100));
            //                 continue;
            //             }
            // #endif
            {
                std::unique_lock<std::mutex> lock(threads::face_lock_data.state_mutex);
                threads::face_lock_data.state_cv.wait(
                    lock,
                    []
                    { return threads::face_data.state == StateFaceRecognition::IDLE; });
            }
            cv::Mat current_frame = CameraMultiplexer::get_instance().get_frame("face_cam");

            if (current_frame.empty())
            {
                UTIL_LOG_WARNING("Camera frame is empty.");
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                continue;
            }

            cv::Mat local_frame = current_frame.clone();

            cv::Rect local_limit_rect;
            {
                std::lock_guard<std::mutex> lock(threads::face_lock_data.face_data_mutex);
                local_limit_rect = threads::face_data.limit_rect;
            }

            if (local_limit_rect.width == 0 || local_limit_rect.height == 0)
            {
                set_limit_rect(local_frame);
                std::lock_guard<std::mutex> lock(threads::face_lock_data.face_data_mutex);
                local_limit_rect = threads::face_data.limit_rect;
            }

            // -------------- detect process -------------- //
            std::tuple<std::vector<Detection>, long, long> res;

#ifdef V2H
            res = infer_drp_face(local_frame);
#else
            res = detector->face_detector->detect(local_frame);
#endif
            // -------------- end detect process -------------- //

            // -------------- post process ---------------- //
            std::vector<Detection> detections = std::get<0>(res);
            bool is_in_pos = false;
            int detected_idx = -1;
            std::vector<cv::Rect> rects;

            int idx = 0;
            for (const auto &det : detections)
            {
#ifdef LINUX64
                cv::Rect face_rect(
                    static_cast<int>(det.box.x),
                    static_cast<int>(det.box.y),
                    static_cast<int>(det.box.w - det.box.x),
                    static_cast<int>(det.box.h - det.box.y));
#endif // LINUX64
#ifdef V2H

                int32_t x_min = (int)det.box.x - round((int)det.box.w / 2.);
                int32_t y_min = (int)det.box.y - round((int)det.box.h / 2.);
                int32_t width = (int)det.box.w;
                int32_t height = (int)det.box.h;

                x_min = std::max(0, std::min(x_min, DRPAI_IN_WIDTH - 1));
                y_min = std::max(0, std::min(y_min, DRPAI_IN_HEIGHT - 1));
                width = std::min(width, DRPAI_IN_WIDTH - x_min);
                height = std::min(height, DRPAI_IN_HEIGHT - y_min);

                cv::Rect face_rect(x_min, y_min, width, height);
#endif // V2H
                rects.push_back(face_rect);

                if ((local_limit_rect & face_rect) == face_rect)
                {
                    is_in_pos = true;
                    detected_idx = idx;
                    break;
                }
                idx++;
            }
            // -------------- end post process ---------------- //

            // -------------- output lock ---------------- //
            {
                std::lock_guard<std::mutex> lock(threads::face_lock_data.face_data_mutex);

                threads::face_data.current_frame = local_frame;
                threads::face_data.detections = detections;
                threads::face_data.detected_face_rect = rects;
                threads::face_data.is_face_in_position = is_in_pos;
                threads::face_data.detected_face_index = detected_idx;

                threads::face_lock_data.face_data_cv.notify_all();
            }
            // -------------- end output lock ---------------- //

            // -------------- transport to recognition ---------------- //
            if (is_in_pos)
            {
                std::lock_guard<std::mutex> lock(threads::face_lock_data.state_mutex);
                threads::face_data.state = StateFaceRecognition::RUNNING;
                threads::face_lock_data.state_cv.notify_all();
            }
            // -------------- end transport ---------------- //
        }
    }

    void t_face_recognition()
    {
        while (true)
        {
            // jalan kalo state running
            {
                std::unique_lock<std::mutex> lock(threads::face_lock_data.state_mutex);
                threads::face_lock_data.state_cv.wait(
                    lock,
                    []
                    { return threads::face_data.state == StateFaceRecognition::RUNNING; });
            }

            cv::Mat local_frame;
            cv::Rect local_face_rect;

            {
                std::unique_lock<std::mutex> lock(threads::face_lock_data.face_data_mutex);
                if (!check_face_data_())
                {
                    std::lock_guard<std::mutex> state_lock(threads::face_lock_data.state_mutex);
                    threads::face_data.state = StateFaceRecognition::IDLE;
                    threads::face_lock_data.state_cv.notify_all();
                    continue;
                }
                int face_idx = threads::face_data.detected_face_index;
                local_frame = threads::face_data.current_frame.clone();
                local_face_rect = threads::face_data.detected_face_rect[face_idx];
            }

            AI *detector = AI::get_instance();

            // -------------- align process ---------------- //
            cv::Mat aligned_frame;
            bool align_success = false;
            std::tuple<bool, long, long, long> res = detector->face_aligner->align(local_frame, local_face_rect, aligned_frame);

            align_success = std::get<0>(res);
            ;

            if (!align_success)
            {
                UTIL_LOG_WARNING("Face alignment failed, skipping frame.");
                {
                    std::lock_guard<std::mutex> lock(threads::face_lock_data.face_data_mutex);
                    threads::face_data.fail_recognition_state_time = Clock::now();
                }
                {
                    std::lock_guard<std::mutex> state_lock(threads::face_lock_data.state_mutex);
                    threads::face_data.state = StateFaceRecognition::IDLE;
                    threads::face_lock_data.state_cv.notify_all();
                }
                continue;
            }
            // -------------- end align process ---------------- //

            // -------------- embedding process ---------------- //
            std::vector<float> embedding;
            try
            {
                embedding = detector->arcface->calc_embedding(aligned_frame);
            }
            catch (std::exception &e)
            {
                UTIL_LOG_ERROR("ArcFace embedding failed: " + std::string(e.what()));
                {
                    std::lock_guard<std::mutex> lock(threads::face_lock_data.face_data_mutex);
                    threads::face_data.fail_recognition_state_time = Clock::now();
                }
                {
                    std::lock_guard<std::mutex> state_lock(threads::face_lock_data.state_mutex);
                    threads::face_data.state = StateFaceRecognition::IDLE;
                    threads::face_lock_data.state_cv.notify_all();
                }
                continue;
            }

            int face_id = Database::instance().faces().getFaceId(embedding, 0.7f);

            {
                std::lock_guard<std::mutex> lock(threads::face_lock_data.face_data_mutex);

                if (face_id != -1)
                { // if recognized
                    threads::face_data.recognized_face_id = face_id;
                    threads::face_data.last_recognized_face_id = face_id;
                }
                else
                {                                               // if not recognized or unknown
                    threads::face_data.recognized_face_id = -1; // Unknown
                    threads::face_data.new_face_embedding = embedding;
                    threads::face_data.last_recognized_face_id = -1;
                }
            }
            // -------------- end embedding process ---------------- //

            {
                std::lock_guard<std::mutex> state_lock(threads::face_lock_data.state_mutex);
                threads::face_data.state = StateFaceRecognition::SUCCESS;
                threads::face_lock_data.state_cv.notify_all();
            }
        }
    }

    bool check_face_data_()
    {
        // Must be called inside lock face data
        if (threads::face_data.detections.empty())
            return false;
        if (threads::face_data.detected_face_index == -1)
            return false;
        if (!threads::face_data.is_face_in_position)
            return false;
        if (threads::face_data.detected_face_rect.empty())
            return false;

        // Check index is valid
        if (threads::face_data.detected_face_index >= threads::face_data.detected_face_rect.size())
            return false;

        if (threads::face_data.current_frame.empty())
            return false;

        return true;
    }

    void reset_face_data_()
    {
        std::lock_guard<std::mutex> lock(threads::face_lock_data.face_data_mutex);
        threads::face_data.detections.clear();
        threads::face_data.detected_face_rect.clear();
        threads::face_data.is_face_in_position = false;
        threads::face_data.detected_face_index = -1;
    }

#ifdef V2H
    std::tuple<std::vector<Detection>, long, long> infer_drp_face(cv::Mat &frame)
    {
        DRPJob job;
        job.frame = frame.clone();
        job.inference_func = [](cv::Mat &frame) -> std::tuple<std::vector<Detection>, long, long>
        {
            return AI::get_instance()->face_detector->detect(frame);
        };

        auto future = job.completion_promise.get_future();

        {
            std::lock_guard<std::mutex> lock(threads::drp_queue_mutex);
            threads::drp_job_queue.push(std::move(job));
        }
        threads::drp_queue_cv.notify_one();

        return future.get();
    }
#endif
}