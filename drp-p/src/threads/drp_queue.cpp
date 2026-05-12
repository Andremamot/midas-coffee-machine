#ifdef V2H
#include <threads/drp_queue.h>


namespace threads
{
    std::queue<DRPJob> drp_job_queue;
    std::mutex drp_queue_mutex;
    std::condition_variable drp_queue_cv;
    
    void t_drp_queue_worker()
    {
        while (true)
        {
            DRPJob job;

            {
                std::unique_lock<std::mutex> lock(drp_queue_mutex);
                drp_queue_cv.wait(lock, []
                                  { return !drp_job_queue.empty(); });

                job = std::move(drp_job_queue.front());
                drp_job_queue.pop();
            }

            // Execute the inference function
            try
            {
                auto result = job.inference_func(job.frame);

                // Set the result in the promise to signal completion
                job.completion_promise.set_value(result);
            }
            catch (const std::exception &e)
            {
                // In case of exception, set the exception in the promise
                job.completion_promise.set_exception(std::make_exception_ptr(e));
            }
        }
    }
}
#endif