#include <detections/face/face_aligner.h>

using Clock = std::chrono::steady_clock;
static inline long ms_between(const Clock::time_point& a, const Clock::time_point& b) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(b - a).count();
}

FaceAligner::FaceAligner()
    : env(ORT_LOGGING_LEVEL_WARNING, "face_aligner"),
      session(nullptr),
      session_options() {
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    std::string model_path = BASE_PATH + MODEL_FILE;
    session = std::make_unique<Ort::Session>(env, model_path.c_str(), session_options);
    memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
}

std::vector<cv::Point2f> FaceAligner::extract_landmarks(const std::vector<float>& output,
                                                        int face_w,
                                                        int face_h,
                                                        const cv::Rect& face_rect) {
    std::vector<cv::Point2f> landmarks;
    for (int i = 0; i < 106; ++i) {
        float x = (output[i * 2] + 1.0f) * 0.5f * face_w + face_rect.x;
        float y = (output[i * 2 + 1] + 1.0f) * 0.5f * face_h + face_rect.y;
        landmarks.emplace_back(x, y);
    }
    return landmarks;
}

bool FaceAligner::is_face_landmark_ratio_valid(const std::vector<cv::Point2f>& landmarks) {
    const auto& eye_left = landmarks[36];
    const auto& eye_right = landmarks[45];
    const auto& mouth_left = landmarks[48];
    const auto& mouth_right = landmarks[54];

    float eye_dist = cv::norm(eye_right - eye_left);
    float mouth_dist = cv::norm(mouth_right - mouth_left);

    float ratio = mouth_dist / eye_dist;
    if (ratio < 0.47f || ratio > 0.8f) {  // Sesuaikan batas wajar
        LOGR_ERROR("Mouth-eye distance ratio abnormal: " + std::to_string(ratio));
        return false;
    }

    return true;
}

std::tuple<bool> FaceAligner::align(const cv::Mat& frame,
                                    const cv::Rect& face_rect,
                                    cv::Mat& aligned_face) {
    if (face_rect.width <= 0 || face_rect.height <= 0 || face_rect.x < 0 || face_rect.y < 0 ||
        face_rect.x + face_rect.width > frame.cols || face_rect.y + face_rect.height > frame.rows) {
        LOGR_ERROR("Invalid face rect.");
        return std::make_tuple(false);
    }

    cv::Mat image = frame.clone();

    int feather = 0;
    int out_size = 56;
    int align_size = 112;

    int width = image.cols;
    int height = image.rows;
    int x1 = face_rect.x;
    int y1 = face_rect.y;
    int x2 = face_rect.x + face_rect.width;
    int y2 = face_rect.y + face_rect.height;

    x1 = x1 - feather;
    y1 = y1 - feather;
    x2 = x2 + feather;
    y2 = y2 + feather;

    int w = x2 - x1 + 1;
    int h = y2 - y1 + 1;

    int size = static_cast<int>(std::max(w, h) * 1.1f);

    int cx = x1 + w / 2;
    int cy = y1 + h / 2;

    x1 = cx - size / 2;
    x2 = x1 + size;
    y1 = cy - size / 2;
    y2 = y1 + size;

    int dx = std::max(0, -x1);
    int dy = std::max(0, -y1);

    x1 = std::max(0, x1);
    y1 = std::max(0, y1);

    int edx = std::max(0, x2 - width);
    int edy = std::max(0, y2 - height);

    x2 = std::min(width, x2);
    y2 = std::min(height, y2);

    cv::Rect crop_rect(x1, y1, x2 - x1, y2 - y1);
    if (crop_rect.width <= 0 || crop_rect.height <= 0) return std::make_tuple(false);

    cv::Mat cropped = image(crop_rect).clone();
    if (dx > 0 || dy > 0 || edx > 0 || edy > 0) {
        cv::copyMakeBorder(cropped, cropped, dy, edy, dx, edx, cv::BORDER_CONSTANT, 0);
    }

    cv::resize(cropped, cropped, cv::Size(out_size, out_size));

    if (cropped.empty()) return std::make_tuple(false);

    cv::Mat image_float;
    cropped.convertTo(image_float, CV_32F, 1.0 / 255.0);

    std::vector<cv::Mat> channels(3);
    cv::split(image_float, channels);

    std::vector<float> input_tensor_values(out_size * out_size * 3);
    memcpy(input_tensor_values.data(), channels[0].data, out_size * out_size * sizeof(float));
    memcpy(input_tensor_values.data() + out_size * out_size, channels[1].data,
           out_size * out_size * sizeof(float));
    memcpy(input_tensor_values.data() + 2 * out_size * out_size, channels[2].data,
           out_size * out_size * sizeof(float));

    std::array<int64_t, 4> input_shape{1, 3, out_size, out_size};

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_tensor_values.data(), input_tensor_values.size(), input_shape.data(),
        input_shape.size());

    auto output_tensors =
        session->Run(Ort::RunOptions{nullptr}, &INPUT_NAME, &input_tensor, 1, &OUTPUT_NAME, 1);

    float* output_data = output_tensors[0].GetTensorMutableData<float>();
    int landmark_num = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape()[1] / 2;

    std::vector<cv::Point2f> landmarks;
    for (int i = 0; i < landmark_num; ++i) {
        float x = output_data[2 * i] * crop_rect.width + crop_rect.x;
        float y = output_data[2 * i + 1] * crop_rect.height + crop_rect.y;
        landmarks.emplace_back(x, y);
    }

    std::vector<cv::Point2f> src_points = {
        landmarks[36],  // left eye corner
        landmarks[45],  // right eye corner
        landmarks[30],  // nose tip
        landmarks[48],  // left mouth
        landmarks[54]   // right mouth
    };

    std::vector<cv::Point2f> dst_points = {{38.2946f, 51.6963f},
                                           {73.5318f, 51.5014f},
                                           {56.0252f, 71.7366f},
                                           {41.5493f, 92.3655f},
                                           {70.7299f, 92.2041f}};
    cv::Mat debug = image.clone();
    /*for (const auto& pt : { landmarks[36], landmarks[45], landmarks[30],
    landmarks[48], landmarks[54] }) { cv::circle(debug, pt, 2, cv::Scalar(0,
    255, 0), -1);
    }*/
    for (const auto& pt : landmarks) {
        cv::circle(debug, pt, 2, cv::Scalar(0, 255, 0), -1);
    }

    if (!is_face_landmark_ratio_valid(landmarks)) {
        return std::make_tuple(false);
    }

    float scale = align_size / 112.0f;
    for (auto& p : dst_points) {
        p *= scale;
    }

    cv::Mat trans = cv::estimateAffinePartial2D(src_points, dst_points);
    if (trans.empty()) {
        LOGR_ERROR("Affine transform failed.");
        return std::make_tuple(false);
    }

    cv::warpAffine(frame, aligned_face, trans, cv::Size(align_size, align_size), cv::INTER_LINEAR);

    return std::make_tuple(true);
}