#include <detections/face/arcface.h>

ArcFace::ArcFace() {
    model = tflite::FlatBufferModel::BuildFromFile((BASE_PATH + MODEL_FILE).c_str());
    if (!model) {
        throw std::runtime_error("Model not found: " + BASE_PATH + MODEL_FILE);
    }

    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder builder(*model, resolver);

    builder(&interpreter);

    if (!interpreter) {
        throw std::runtime_error("Failed to construct interpreter.");
    }

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        throw std::runtime_error("Failed to allocate tensors.");
    }
}

std::vector<float> ArcFace::calc_embedding(const cv::Mat& img) {
    if (img.empty()) {
        throw std::invalid_argument("Input image is empty");
    }

    cv::Mat rgb, resized;
    cv::cvtColor(img, rgb, cv::COLOR_BGR2RGB);
    cv::resize(rgb, resized, cv::Size(112, 112));
    resized.convertTo(resized, CV_32FC3, 1.0 / 255.0);

    float* input_tensor = interpreter->typed_input_tensor<float>(0);
    std::memcpy(input_tensor, resized.data, 112 * 112 * 3 * sizeof(float));

    if (interpreter->Invoke() != kTfLiteOk) {
        throw std::runtime_error("Error during inference.");
    }

    const float* output = interpreter->typed_output_tensor<float>(0);
    int output_size = interpreter->tensor(interpreter->outputs()[0])->bytes / sizeof(float);

    std::vector<float> embedding(output, output + output_size);
    l2_norm(embedding);
    return embedding;
}

void ArcFace::l2_norm(std::vector<float>& vec) {
    float norm = 0.0f;
    for (float v : vec) norm += v * v;
    norm = std::sqrt(norm);
    if (norm > 0) {
        for (float& v : vec) v /= norm;
    }
}

float ArcFace::get_distance_embeddings(const std::vector<float>& emb1,
                                       const std::vector<float>& emb2) {
    if (emb1.size() != emb2.size()) {
        throw std::invalid_argument("Embeddings must have the same size");
    }

    float sum = 0.0f;
    for (size_t i = 0; i < emb1.size(); ++i) {
        float diff = emb1[i] - emb2[i];
        sum += diff * diff;
    }
    return sum;
}

float cosine_similarity(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size() || a.empty()) return 0.0f;

    float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    if (norm_a == 0.0f || norm_b == 0.0f) return 0.0f;

    return dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
}