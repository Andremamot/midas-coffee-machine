#ifdef V2H
#pragma once

/*****************************************
 * includes
 ******************************************/
#include <array>
#include <cstdint>
#include <string>

namespace c {
class yolov3 {
public:
    /* Model Binary */
    inline static const std::string model_dir = "model/face/v2h/yolov3";
    inline static const std::string pre_dir = model_dir + "/preprocess";
    inline static const std::string label_list = "model/face/v2h/yolov3/labels.txt";

    /* DRP-AI */
    inline static constexpr uint32_t DRPAI_MEM_OFFSET = 0x10000000;

    /* YOLOv3 Parameters */
    inline static constexpr uint8_t NUM_CLASS = 1;
    inline static constexpr uint8_t NUM_BB = 3;
    inline static constexpr uint8_t NUM_INF_OUT_LAYER = 3;

    /* Grid sizes */
    inline static constexpr std::array<uint8_t, NUM_INF_OUT_LAYER> num_grids = {13, 26, 52};

    /* Output size */
    inline static constexpr uint32_t num_inf_out =
        (NUM_CLASS + 5) * NUM_BB *
        (num_grids[0] * num_grids[0] + num_grids[1] * num_grids[1] + num_grids[2] * num_grids[2]);

    /* Anchors */
    inline static constexpr std::array<double, 18> anchors = {
        5, 9, 9, 17, 16, 29, 26, 47, 42, 73, 72, 114, 118, 195, 209, 304, 353, 465};

    /* Thresholds */
    inline static constexpr float TH_PROB = 0.5f;
    inline static constexpr float TH_NMS = 0.3f;

    /* Model input size */
    inline static constexpr uint32_t MODEL_IN_W = 416;
    inline static constexpr uint32_t MODEL_IN_H = 416;
};

}  // namespace c

#endif  // V2H