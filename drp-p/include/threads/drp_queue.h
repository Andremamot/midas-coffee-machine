#ifdef V2H
#pragma once

#include <queue>
#include <mutex>
#include <condition_variable>
#include <future>
#include <functional>
#include <detections/box.h>
#include <opencv2/opencv.hpp>
#include <vector>

struct DRPJob
{
    std::function<std::tuple<std::vector<Detection>, long, long>(cv::Mat &)> inference_func; ///< The task to be executed
    cv::Mat frame; ///< Input frame for the task

    std::promise<std::tuple<std::vector<Detection>, long, long>> completion_promise; ///< Promise to signal task completion
};

namespace threads{
    extern std::queue<DRPJob> drp_job_queue;
    extern std::mutex drp_queue_mutex;
    extern std::condition_variable drp_queue_cv;

    void t_drp_queue_worker();
}
#endif