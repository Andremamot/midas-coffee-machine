// #pragma once

// #include <opencv2/opencv.hpp>
// #include <vector>

// struct BoundingBoxCup {
//     cv::Rect box;
//     float score;
//     int class_id;
// };

// enum CupSize {
//     UNKNOWN,
//     S,
//     M,
//     L,
//     XL
// };

// struct CupMeasurement {
//     double volume_ml = 0.0;
//     CupSize size = CupSize::UNKNOWN;
//     std::string size_str = "Unknown";
// };

// struct DetectionCup {
//     std::vector<BoundingBoxCup> detections;
//     bool is_center_in_rim;
//     cv::Point center_point;
//     CupMeasurement measurement;
// };
