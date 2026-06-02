#pragma once

#include <queue>
#include <mutex>
#include <condition_variable>
#include <future>
#include <functional>
#include <vector>
#include <opencv2/opencv.hpp>

/**
 * @struct DRPJob
 * @brief Represents a single inference task for the DRP-AI hardware.
 */
struct DRPJob {
    std::function<void()> work;
};

/**
 * @class DRPQueue
 * @brief Serializes all DRP-AI hardware calls to prevent concurrent access errors.
 */
class DRPQueue {
public:
    static DRPQueue& get_instance() {
        static DRPQueue instance;
        return instance;
    }

    /**
     * @brief Enqueues a task and waits for its completion.
     */
    template<typename F, typename... Args>
    auto enqueue(F&& f, Args&&... args) -> std::future<typename std::invoke_result<F, Args...>::type> {
        using return_type = typename std::invoke_result<F, Args...>::type;

        auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );

        std::future<return_type> res = task->get_future();
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            if (stop) throw std::runtime_error("DRPQueue is stopped");

            jobs.emplace([task]() { (*task)(); });
        }
        condition.notify_one();
        return res;
    }

    ~DRPQueue() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            stop = true;
        }
        condition.notify_all();
        if (worker.joinable()) worker.join();
    }

private:
    DRPQueue() : stop(false) {
        worker = std::thread([this]() {
            for (;;) {
                std::function<void()> job;
                {
                    std::unique_lock<std::mutex> lock(this->queue_mutex);
                    this->condition.wait(lock, [this] { return this->stop || !this->jobs.empty(); });
                    if (this->stop && this->jobs.empty()) return;
                    job = std::move(this->jobs.front());
                    this->jobs.pop();
                }
                job();
            }
        });
    }

    std::thread worker;
    std::queue<std::function<void()>> jobs;
    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop;
};
