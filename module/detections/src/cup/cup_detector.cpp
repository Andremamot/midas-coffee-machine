#if defined(LINUX64) || DRP_AI_TVM_RUNTIME == 0
#include <detections/cup/cup_detector.h>

using Clock = std::chrono::steady_clock;
static inline long ms_between(const Clock::time_point& a, const Clock::time_point& b) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(b - a).count();
}

CupDetector::CupDetector() {
    // initialize onnx cup detector
    env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "CUP_DETECTOR");
    session_options = Ort::SessionOptions();
    session = std::make_unique<Ort::Session>(env, model_path.c_str(), session_options);

    memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
}

std::tuple<std::vector<Detection>> CupDetector::detect(cv::Mat& frame) {
    float* blop_ptr = nullptr;  // Pointer to hold preprocessed image data

    // Preprocess the image and obtain a pointer to the blob
    cv::Mat preprocessed_image = ultralytics::yolov8::preprocess(frame, blop_ptr, input_dims);

    // Compute the total number of elements in the input tensor
    size_t input_tensor_size = ultralytics::utils::vectorProduct(input_dims);

    // Create a vector from the blob data for ONNX Runtime input
    std::vector<float> input_tensor_value(blop_ptr, blop_ptr + input_tensor_size);

    delete[] blop_ptr;  // Free the allocated memory for the blob

    // Create an Ort memory info object (can be cached if used repeatedly)
    static Ort::MemoryInfo memoryInfo =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // Create input tensor object using the preprocessed data
    Ort::Value input_tensor =
        Ort::Value::CreateTensor<float>(memoryInfo, input_tensor_value.data(), input_tensor_size,
                                        input_dims.data(), input_dims.size());

    // Run the inference session with the input tensor and retrieve output tensors
    auto output_tensors =
        session->Run(Ort::RunOptions{nullptr}, &INPUT_NAME, &input_tensor, 1, &OUTPUT_NAME, 1);

    // Determine the resized image shape based on input tensor shape
    cv::Size resized_image_shape(static_cast<int>(input_dims[3]), static_cast<int>(input_dims[2]));

    // Postprocess the output tensors to obtain detections
    std::vector<Detection> detections = ultralytics::yolov8::postprocess(
        frame.size(), resized_image_shape, output_tensors, THRESHOLD_SCORE, THRESHOLD_NMS);

    return std::make_tuple(detections);
}

bool CupDetector::check_cup_in_center(const std::vector<Detection>& detections,
                                      const cv::Point& center_point) {
    // check is rim in center
    bool is_center_in_rim = false;
    std::vector<Box> cup_rim_boxes;
    std::vector<Box> cup_body_boxes;

    for (const auto& det : detections) {
        if (det.class_id == 0)  // "cup" (rim)
            cup_rim_boxes.push_back(det.box);

        else if (det.class_id == 1)  // "cup_body"
            cup_body_boxes.push_back(det.box);
    }

    for (const auto& rim_box : cup_rim_boxes) {
        int rim_center_x = rim_box.x + rim_box.w / 2;
        int rim_center_y = rim_box.y + rim_box.h / 2;

        bool associated_body_found = false;

        for (const auto& body_box : cup_body_boxes) {
            if (body_box.contains(rim_center_x, rim_center_y)) {
                associated_body_found = true;
                break;
            }
        }
        if (!associated_body_found) continue;

        float margin_x_r = rim_box.w * TOLERANCE_FACTOR;
        float margin_y_r = rim_box.h * TOLERANCE_FACTOR;

        float inner_x1_r = rim_box.x + margin_x_r;
        float inner_y1_r = rim_box.y + margin_y_r;
        float inner_x2_r = rim_box.x + rim_box.w - margin_x_r;
        float inner_y2_r = rim_box.y + rim_box.h - margin_y_r;

        if (center_point.x >= inner_x1_r && center_point.x <= inner_x2_r &&
            center_point.y >= inner_y1_r && center_point.y <= inner_y2_r) {
            is_center_in_rim = true;
            break;
        }
    }
    return is_center_in_rim;
}

CupDetector::~CupDetector() { delete session.get(); }
#endif