#include "detector.hpp"
#include <iostream>
#include <algorithm>

YoloDetector::YoloDetector(const std::string& weights_path) {
    std::cout << "Loading YOLO ONNX model from " << weights_path << "..." << std::endl;
    try {
        net = cv::dnn::readNetFromONNX(weights_path);
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        std::cout << "Successfully loaded with OpenCV DNN." << std::endl;
    } catch (const cv::Exception& e) {
        std::cerr << "Exception loading ONNX: " << e.what() << "\nMake sure you have converted your YOLO model to ONNX using `yolo export model=... format=onnx`!" << std::endl;
    }
}

std::vector<BBox> YoloDetector::detect(const cv::Mat& frame) {
    std::vector<BBox> boxes;
    if (net.empty()) return boxes;
    
    cv::Mat blob;
    // YOLOv8 default is typically 640, but python says imgsz=320. Try 320.
    float input_width = 320.0f;
    float input_height = 320.0f;
    cv::dnn::blobFromImage(frame, blob, 1.0/255.0, cv::Size(input_width, input_height), cv::Scalar(), true, false);
    net.setInput(blob);
    
    std::vector<cv::Mat> outputs;
    try {
        net.forward(outputs, net.getUnconnectedOutLayersNames());
    } catch (const cv::Exception& e) {
        std::cerr << "Warning: YOLO network forward pass failed: " << e.what() << "\nReturning empty detection.\n";
        return boxes;
    }
    
    // Parse YOLOv8 ONNX output shape [1, num_classes + 4, num_anchors]
    if (outputs.empty()) return boxes;
    cv::Mat output = outputs[0];
    
    int dimensions = output.size[1]; 
    int rows = output.size[2]; 
    
    float x_factor = frame.cols / input_width;
    float y_factor = frame.rows / input_height;
    
    // Output is typically Float32. Shape [1, 84, 8400] for COCO, but for custom maybe [1, 5, 2100].
    float *data = (float *)output.data;
    output = output.reshape(0, dimensions);
    cv::Mat transposed;
    cv::transpose(output, transposed);
    float *tdata = (float *)transposed.data;
    
    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> bboxes;
    
    for (int i = 0; i < rows; ++i) {
        float *row = tdata + i * dimensions;
        cv::Mat scores(1, dimensions - 4, CV_32F, row + 4);
        cv::Point class_id_point;
        double max_class_score;
        cv::minMaxLoc(scores, 0, &max_class_score, 0, &class_id_point);
        
        // We only care about class 0 (cup rim)
        if (class_id_point.x == 0 && max_class_score > 0.25) { // threshold
            float cx = row[0];
            float cy = row[1];
            float w = row[2];
            float h = row[3];
            
            int left = int((cx - 0.5 * w) * x_factor);
            int top = int((cy - 0.5 * h) * y_factor);
            int width = int(w * x_factor);
            int height = int(h * y_factor);
            
            bboxes.push_back(cv::Rect(left, top, width, height));
            class_ids.push_back(class_id_point.x);
            confidences.push_back(static_cast<float>(max_class_score));
        }
    }
    
    std::vector<int> indices;
    cv::dnn::NMSBoxes(bboxes, confidences, 0.25f, 0.45f, indices);
    
    for (int i = 0; i < indices.size(); ++i) {
        int idx = indices[i];
        BBox b;
        b.x1 = bboxes[idx].x;
        b.y1 = bboxes[idx].y;
        b.x2 = bboxes[idx].x + bboxes[idx].width;
        b.y2 = bboxes[idx].y + bboxes[idx].height;
        b.conf = confidences[idx];
        boxes.push_back(b);
    }
    
    // Sort descending
    std::sort(boxes.begin(), boxes.end(), [](const BBox& a, const BBox& b) {
        return a.conf > b.conf;
    });
    
    return boxes;
}
