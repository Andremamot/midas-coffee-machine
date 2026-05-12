#ifdef V2H
#pragma once

/*****************************************
 * includes
 ******************************************/
#include <array>
#include <cstdint>
#include <string>
#include <vector>

/* Padding input mode to maintain the aspect ratio of DRP-AI input image.
   This mode requires the DRP-AI object file having the squared input size CAM_IMAGE_WIDTH x
   CAM_IMAGE_WIDTH. 0: No padding 1: With padding (maintains the aspect ratio) */
#define DRPAI_INPUT_PADDING (1)

/* Tuning of sigmoid timing for acceleration of CPU DFL and post processing.
   0: Do sigmoid in DFL (Original implementation)
   1: Skip sigmoid in DFL and do sigmoid after argmax in post processing
   2: Skip sigmoid in DFL and do sigmoid after threshold processing in post processing
*/
#define CPU_DFL_SIGMOID_SKIP (0)

/* Enable or Disable the multi-threading for CPU DFL processing.
   0: Disable (single-thread)
   1: Enable (multi-thread)
*/
#define CPU_DFL_MULTI_THREAD (1)

/* Enable DRP-AI TVM Runtime or Using ONNX Runtime
   0: Use ONNX Runtime for CPU post-processing (for development and debugging purpose)
   1: Use DRP-AI TVM Runtime for post-processing on RZ/V2H (for actual deployment)
*/
#define DRP_AI_TVM_RUNTIME (1)

namespace c {
class yolov8 {
public:
    /* Model Binary */
    inline static const std::string model_dir = "model/cup/v2h/yolov8";
    inline static const std::string pre_dir = model_dir + "/preprocess";
    inline static const std::string label_list = "model/cup/v2h/yolov8/labels.txt";

    inline static uint32_t DRPAI_MEM_OFFSET = 0x0000000;

    /* Detection parameters */
    inline static constexpr int NUM_CLASS = 2;
    inline static constexpr int NUM_BB = 1;
    inline static constexpr int NUM_INF_OUT_LAYER = 3;
    inline static constexpr int REG_MAX = 16;

    /* Grid sizes */
    inline static constexpr std::array<uint8_t, NUM_INF_OUT_LAYER> num_grids = {80, 40, 20};

    /* Output sizes */
    inline static constexpr uint32_t num_dfl80_out = (REG_MAX * 4) * num_grids[0] * num_grids[0];
    inline static constexpr uint32_t num_dfl40_out = (REG_MAX * 4) * num_grids[1] * num_grids[1];
    inline static constexpr uint32_t num_dfl20_out = (REG_MAX * 4) * num_grids[2] * num_grids[2];

    inline static constexpr uint32_t num_class80_out = NUM_CLASS * num_grids[0] * num_grids[0];
    inline static constexpr uint32_t num_class40_out = NUM_CLASS * num_grids[1] * num_grids[1];
    inline static constexpr uint32_t num_class20_out = NUM_CLASS * num_grids[2] * num_grids[2];

    inline static float output_dfl80[num_dfl80_out];
    inline static float output_dfl40[num_dfl40_out];
    inline static float output_dfl20[num_dfl20_out];
    inline static float output_class80[num_class80_out];
    inline static float output_class40[num_class40_out];
    inline static float output_class20[num_class20_out];

    inline static constexpr uint32_t num_inf_out =
        (NUM_CLASS + 4) * NUM_BB *
        (num_grids[0] * num_grids[0] + num_grids[1] * num_grids[1] + num_grids[2] * num_grids[2]);

    inline static constexpr uint32_t num_grid_points = (num_grids[0] * num_grids[0]) +
                                                       (num_grids[1] * num_grids[1]) +
                                                       (num_grids[2] * num_grids[2]);

    /* Thresholds */
    inline static constexpr float TH_PROB = 0.2f;
    inline static constexpr float TH_NMS = 0.5f;

    /* Model input size */
    static constexpr uint32_t MODEL_IN_W = 640;
    static constexpr uint32_t MODEL_IN_H = 640;

    // static const std::vector<int> STRIDE_FPN {8, 16, 32};
};

}  // namespace c

#endif  // V2H