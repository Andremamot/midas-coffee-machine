#ifdef LINUX64
#include <detections/face/face_detector.h>

using Clock = std::chrono::steady_clock;
static inline long ms_between(const Clock::time_point& a, const Clock::time_point& b) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(b - a).count();
}

FaceDetector::FaceDetector() {
    // initialize onnx face detector
    env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "FACE_DETECTOR");
    session_options = Ort::SessionOptions();
    session = std::make_unique<Ort::Session>(env, model_path.c_str(), session_options);

    memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
}

std::tuple<std::vector<Detection>> FaceDetector::detect(cv::Mat& frame) {
    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(INPUT_WIDTH, INPUT_HEIGHT), 0, 0, cv::INTER_LINEAR);

    cv::Mat img_rgb;
    cv::cvtColor(resized, img_rgb, cv::COLOR_BGR2RGB);

    cv::Mat img_float;
    img_rgb.convertTo(img_float, CV_32F, 1.0 / 255.0);

    std::vector<cv::Mat> channels(3);
    cv::split(img_float, channels);

    std::vector<float> input_tensor_values(INPUT_WIDTH * INPUT_HEIGHT * 3);
    memcpy(input_tensor_values.data(), channels[0].data,
           INPUT_WIDTH * INPUT_HEIGHT * sizeof(float));
    memcpy(input_tensor_values.data() + INPUT_WIDTH * INPUT_HEIGHT, channels[1].data,
           INPUT_WIDTH * INPUT_HEIGHT * sizeof(float));
    memcpy(input_tensor_values.data() + 2 * INPUT_WIDTH * INPUT_HEIGHT, channels[2].data,
           INPUT_WIDTH * INPUT_HEIGHT * sizeof(float));

    std::vector<int64_t> input_tensor_shape = {1, 3, (int64_t)INPUT_WIDTH, (int64_t)INPUT_HEIGHT};

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_tensor_values.data(), input_tensor_values.size(),
        input_tensor_shape.data(), input_tensor_shape.size());

    auto output_tensors =
        session->Run(Ort::RunOptions{nullptr}, &INPUT_NAME, &input_tensor, 1, &OUTPUT_NAME, 1);

    auto detections = darknet::get_detections(std::move(output_tensors[0]), THRESHOLD_SCORE,
                                              THRESHOLD_NMS, NUM_CLASSES);

    float scale_x = (float)frame.cols;
    float scale_y = (float)frame.rows;

    for (auto& detection : detections) {
        detection.box.x *= scale_x;
        detection.box.y *= scale_y;
        detection.box.w *= scale_x;
        detection.box.h *= scale_y;
    }

    LOGR_DEBUG("Number of detections: " + std::to_string(detections.size()));
    return std::make_tuple(detections);
}

FaceDetector::~FaceDetector() { delete session.get(); }
#endif