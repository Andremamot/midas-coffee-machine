#include "camera_config.hpp"

CameraConfig::CameraConfig(int width, int height) : W(width), H(height) {
    BRIGHT_ROW_START = static_cast<int>(0.08 * H);
    BRIGHT_ROW_END   = static_cast<int>(0.33 * H);
    SEARCH_ROW_START = static_cast<int>(0.33 * H);
    SPLIT_COL        = static_cast<int>(0.50 * W);
}
