#include <threads/cup_detection.h>

using Clock = std::chrono::steady_clock;

namespace threads
{
    ThreadCupData cup_data;
    ThreadCupLockData cup_lock_data;

    void t_cup_detection()
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
                std::unique_lock<std::mutex> lock(threads::cup_lock_data.state_mutex);
                threads::cup_lock_data.state_cv.wait(
                    lock,
                    []
                    { return threads::cup_data.state == StateCupDetection::IDLE; });
            }
            cv::Mat current_frame = CameraMultiplexer::get_instance().get_frame("cup_cam");

            if (current_frame.empty())
            {
                UTIL_LOG_WARNING("Camera frame is empty.");
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                continue;
            }

            cv::Mat local_frame = current_frame.clone();

            // -------------- detect process -------------- //

            std::tuple<std::vector<Detection>, long, long> res;
#if defined(V2H) && (DRP_AI_TVM_RUNTIME == 1)
            res = infer_drp_cup(local_frame);
#else
            res = detector->cup_detector->detect(local_frame);
#endif
            std::vector<Detection> detections = std::get<0>(res);

            CupMeasurement measurement = detector->cup_estimator->estimate(detections);
            cv::Point center_point = cv::Point(local_frame.cols / 2, local_frame.rows / 2);
            bool is_center_in_rim = detector->cup_detector->check_cup_in_center(detections, center_point);

            // -------------- output lock ---------------- //
            {
                std::lock_guard<std::mutex> lock(threads::cup_lock_data.cup_data_mutex);

                threads::cup_data.current_frame = local_frame;
                threads::cup_data.detections = detections;
                threads::cup_data.is_center_in_rim = is_center_in_rim;
                threads::cup_data.center_point = center_point;
                threads::cup_data.measurement = measurement;

                threads::cup_lock_data.cup_data_cv.notify_all();
            }
            // -------------- end output lock ---------------- //

            // -------------- transport ---------------- //
            if (is_center_in_rim)
            {
                std::lock_guard<std::mutex> lock(threads::cup_lock_data.state_mutex);
                threads::cup_data.state = StateCupDetection::SUCCESS;
                threads::cup_lock_data.state_cv.notify_all();
            }
            // -------------- end transport ---------------- //
        }
    }

    void reset_cup_data_()
    {
        std::lock_guard<std::mutex> lock(threads::cup_lock_data.cup_data_mutex);
        threads::cup_data.detections.clear();
        threads::cup_data.is_center_in_rim = false;
        threads::cup_data.center_point = cv::Point2f(0, 0);
        threads::cup_data.measurement.size = CupSize::UNKNOWN;
        threads::cup_data.measurement.volume_ml = 0.0;
        threads::cup_data.measurement.size_str = "Unknown";
    }

#if defined(V2H) && (DRP_AI_TVM_RUNTIME == 1)
    std::tuple<std::vector<Detection>, long, long> infer_drp_cup(cv::Mat &frame)
    {
        DRPJob job;
        job.frame = frame.clone();
        job.inference_func = [](cv::Mat &frame) -> std::tuple<std::vector<Detection>, long, long>
        {
            return AI::get_instance()->cup_detector->detect(frame);
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