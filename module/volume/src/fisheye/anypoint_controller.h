/**
 * @file anypoint_controller.h
 * @brief Mouse-driven anypoint view controller for Moildev.
 * Port of Python core/anypoint_controller.py.
 */

#pragma once

#include <opencv2/core.hpp>
#include <string>

namespace fusion {

class AnypointController {
public:
    AnypointController(double init_pitch = 0, double init_yaw = 0,
                       double init_zoom = 1.0);

    /// OpenCV mouse callback handler
    void mouse_callback(int event, int x, int y, int flags);

    /// Attach to an OpenCV window
    void attach(const std::string& window_name);

    /// Reset to initial values
    void reset();

    /// Draw HUD overlay with current parameters
    void draw_overlay(cv::Mat& frame) const;

    // Getters
    double pitch() const { return pitch_; }
    double yaw() const { return yaw_; }
    double zoom() const { return zoom_; }
    bool changed() const { return changed_; }
    void clear_changed() { changed_ = false; }

private:
    double pitch_, yaw_, zoom_;
    double init_pitch_, init_yaw_, init_zoom_;
    bool dragging_ = false;
    int drag_start_x_ = 0, drag_start_y_ = 0;
    double drag_start_pitch_ = 0, drag_start_yaw_ = 0;
    bool changed_ = false;
};

} // namespace fusion
