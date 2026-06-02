/**
 * @file anypoint_controller.cpp
 * @brief Anypoint controller implementation.
 */

#include "anypoint_controller.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <cmath>
#include <sstream>
#include <iomanip>

namespace fusion {

AnypointController::AnypointController(double init_pitch, double init_yaw,
                                       double init_zoom)
    : pitch_(init_pitch), yaw_(init_yaw), zoom_(init_zoom),
      init_pitch_(init_pitch), init_yaw_(init_yaw), init_zoom_(init_zoom)
{}

void AnypointController::mouse_callback(int event, int x, int y, int flags) {
    (void)flags;
    if (event == cv::EVENT_LBUTTONDOWN) {
        dragging_ = true;
        drag_start_x_ = x;
        drag_start_y_ = y;
        drag_start_pitch_ = pitch_;
        drag_start_yaw_ = yaw_;
    } else if (event == cv::EVENT_MOUSEMOVE && dragging_) {
        double dx = x - drag_start_x_;
        double dy = y - drag_start_y_;
        pitch_ = drag_start_pitch_ + dy * 0.1;
        yaw_ = drag_start_yaw_ + dx * 0.1;
        pitch_ = std::clamp(pitch_, -75.0, 75.0);
        yaw_ = std::clamp(yaw_, -75.0, 75.0);
        changed_ = true;
    } else if (event == cv::EVENT_LBUTTONUP) {
        dragging_ = false;
    } else if (event == 10 /* cv::EVENT_MOUSEWHEEL */) {
        // Mocking cv::EVENT_MOUSEWHEEL since highgui is not linked
        // In actual GTK use case, we use scroll_event
        double delta = (flags > 0) ? 0.1 : -0.1;
        zoom_ = std::clamp(zoom_ + delta, 1.0, 8.0);
        changed_ = true;
    }
}

void AnypointController::attach(const std::string& window_name) {
    (void)window_name;
    // Removed cv::setMouseCallback to drop highgui dependency
}

void AnypointController::reset() {
    pitch_ = init_pitch_;
    yaw_ = init_yaw_;
    zoom_ = init_zoom_;
    changed_ = true;
}

void AnypointController::draw_overlay(cv::Mat& frame) const {
    int bw = 220, bh = 80;
    int bx = frame.cols - bw - 10, by = frame.rows - bh - 10;
    cv::rectangle(frame, cv::Point(bx, by), cv::Point(bx + bw, by + bh),
                  cv::Scalar(0, 0, 0), cv::FILLED);
    cv::rectangle(frame, cv::Point(bx, by), cv::Point(bx + bw, by + bh),
                  cv::Scalar(100, 100, 100), 1);

    std::ostringstream ss;
    ss << std::fixed << std::setprecision(1);
    ss << "Alpha: " << pitch_;
    cv::putText(frame, ss.str(), cv::Point(bx + 8, by + 22),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
    ss.str(""); ss << "Beta: " << yaw_;
    cv::putText(frame, ss.str(), cv::Point(bx + 8, by + 42),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
    ss.str(""); ss << "Zoom: " << zoom_ << "x";
    cv::putText(frame, ss.str(), cv::Point(bx + 8, by + 62),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
}

} // namespace fusion
