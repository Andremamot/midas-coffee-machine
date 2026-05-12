#if defined(LINUX64) || DRP_AI_TVM_RUNTIME == 0
#include <detections/midas/midas_estimator.h>
#include <iostream>

MidasEstimator::MidasEstimator() : input_height(0), input_width(0), input_channels(0), output_height(0), output_width(0) {
    initializeModel(4);
}

MidasEstimator::~MidasEstimator() {}

void MidasEstimator::initializeModel(size_t thread_num) {
    model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str(), nullptr);
    if (!model) {
        std::cerr << "MidasEstimator Error: Model file not found or could not be loaded at " << model_path << "\n";
        return;
    }

    tflite::InterpreterBuilder interpreter_builder(*model, op_resolver);
    interpreter_builder(&interpreter);
    if (!interpreter) {
        std::cerr << "MidasEstimator Error: Failed to create interpreter.\n";
        return;
    }

    interpreter->SetNumThreads(thread_num);
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        std::cerr << "MidasEstimator Error: Failed to allocate tensors.\n";
        return;
    }

    getModelInputDetails();
    getModelOutputDetails();
    std::cout << "Midas TFLite initialized successfully.\n";
}

void MidasEstimator::getModelInputDetails() {
    const auto input_indices = interpreter->inputs();
    const auto input_tensor = interpreter->tensor(input_indices[0]);
    const auto input_dims = input_tensor->dims;
    input_height = input_dims->data[1];
    input_width = input_dims->data[2];
    input_channels = input_dims->data[3];
}

void MidasEstimator::getModelOutputDetails() {
    const auto output_indices = interpreter->outputs();
    const auto output_tensor = interpreter->tensor(output_indices[0]);
    const auto output_dims = output_tensor->dims;
    output_height = output_dims->data[1];
    output_width = output_dims->data[2];
}

void MidasEstimator::prepareInputForInference(cv::Mat &img) {
    cv::Mat lab;
    cv::cvtColor(img, lab, cv::COLOR_BGR2Lab);
    std::vector<cv::Mat> lab_planes(3);
    cv::split(lab, lab_planes);
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
    clahe->apply(lab_planes[0], lab_planes[0]);
    cv::merge(lab_planes, lab);
    cv::Mat clahe_image;
    cv::cvtColor(lab, clahe_image, cv::COLOR_Lab2BGR);
    
    cv::cvtColor(clahe_image, clahe_image, cv::COLOR_BGR2RGB);
    cv::resize(clahe_image, clahe_image, cv::Size(input_width, input_height), 0, 0, cv::INTER_CUBIC);
    clahe_image.convertTo(clahe_image, CV_32FC3);

    cv::Mat mean_image(cv::Size(input_width, input_height), CV_32FC3, cv::Scalar(0.485, 0.456, 0.406));
    cv::Mat std_image(cv::Size(input_width, input_height), CV_32FC3, cv::Scalar(0.229, 0.224, 0.225));

    clahe_image = (clahe_image / 255.0 - mean_image) / std_image;

    std::memcpy(interpreter->typed_input_tensor<float>(0), clahe_image.ptr<float>(0), input_height * input_width * 3 * sizeof(float));
}

cv::Mat MidasEstimator::inference(const cv::Mat& frame) {
    if (!interpreter) return cv::Mat::zeros(frame.size(), CV_32F);

    cv::Mat frame_clone = frame.clone();
    prepareInputForInference(frame_clone);

    if (interpreter->Invoke() != kTfLiteOk) {
        std::cerr << "MidasEstimator Error: Failed to invoke interpreter.\n";
        return cv::Mat::zeros(frame.size(), CV_32F);
    }

    cv::Mat raw_disparity(output_height, output_width, CV_32F, interpreter->typed_output_tensor<float>(0));

    cv::Mat resized;
    cv::resize(raw_disparity, resized, frame.size(), 0, 0, cv::INTER_CUBIC);

    if (prev_prediction.empty() || prev_prediction.size() != resized.size()) {
        prev_prediction = resized.clone();
    } else {
        resized = (ema_alpha * resized) + ((1.0f - ema_alpha) * prev_prediction);
        prev_prediction = resized.clone();
    }
    return resized.clone();
}

cv::Mat MidasEstimator::get_standardized_depth(const cv::Mat& depth_map) {
    float SATURATION_POINT = 2500.0f;
    float scale = 1000.0f / SATURATION_POINT;
    
    cv::Mat standardized;
    depth_map.convertTo(standardized, CV_32F, scale);
    
    cv::threshold(standardized, standardized, 1000.0, 1000.0, cv::THRESH_TRUNC);
    cv::Mat zeros = cv::Mat::zeros(standardized.size(), CV_32F);
    cv::max(standardized, zeros, standardized);
    
    return standardized;
}

float MidasEstimator::get_tray_depth(const cv::Mat& depth_map, const cv::Rect& roi_coords) {
    cv::Mat standardized = depth_map;
    double minVal, maxVal;
    cv::minMaxLoc(depth_map, &minVal, &maxVal);
    if (maxVal > 1000) {
        standardized = get_standardized_depth(depth_map);
    }
    
    cv::Mat roi = standardized(roi_coords);
    if (roi.empty()) return 0.0f;
    
    std::vector<float> vals;
    vals.assign((float*)roi.datastart, (float*)roi.dataend);
    std::nth_element(vals.begin(), vals.begin() + vals.size() / 2, vals.end());
    return vals[vals.size() / 2];
}

float MidasEstimator::get_rim_depth(const cv::Mat& depth_map, const cv::Rect& bbox) {
    cv::Mat standardized = depth_map;
    double minVal, maxVal;
    cv::minMaxLoc(depth_map, &minVal, &maxVal);
    if (maxVal > 1000) {
        standardized = get_standardized_depth(depth_map);
    }
    
    int thickness_inward = std::max(4, bbox.height / 10);
    int px1 = std::max(bbox.x, bbox.x + bbox.width / 4);
    int px2 = std::min(bbox.x + bbox.width, bbox.x + bbox.width - bbox.width / 4);
    
    int py1 = bbox.y;
    int py2 = std::min(bbox.y + bbox.height, bbox.y + thickness_inward);
    
    cv::Rect patch_rect(px1, py1, px2 - px1, py2 - py1);
    cv::Mat patch = standardized(patch_rect);
    if (patch.empty()) return 0.0f;
    
    std::vector<float> vals;
    vals.assign((float*)patch.datastart, (float*)patch.dataend);
    std::nth_element(vals.begin(), vals.begin() + vals.size() / 2, vals.end());
    return vals[vals.size() / 2];
}
#endif
