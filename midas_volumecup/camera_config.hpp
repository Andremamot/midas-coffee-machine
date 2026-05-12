#pragma once

class CameraConfig {
public:
    int W;
    int H;
    int BRIGHT_ROW_START;
    int BRIGHT_ROW_END;
    int SEARCH_ROW_START;
    int SPLIT_COL;

    CameraConfig() = default;
    CameraConfig(int width, int height);
};
