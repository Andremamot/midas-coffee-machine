#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <detections/ai.h>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " -i <input_image> [-o <output_image>]\n";
        return 1;
    }
    
    std::string input_path;
    std::string output_path = "output_depth_test.png";
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-i" || arg == "--input") {
            if (i + 1 < argc) input_path = argv[++i];
        } else if (arg == "-o" || arg == "--output") {
            if (i + 1 < argc) output_path = argv[++i];
        }
    }
    
    if (input_path.empty()) {
        std::cerr << "Error: Input image is required.\n";
        return 1;
    }

    std::cout << "Loading AI Framework (MiDaS module via ONNXRuntime)...\n";
    AI* ai = AI::get_instance();
    
    cv::Mat img = cv::imread(input_path);
    if (img.empty()) {
        std::cerr << "Error: Could not read image at " << input_path << "\n";
        return 1;
    }
    
    std::cout << "Processing " << input_path << "...\n";
    cv::Mat depth = ai->midas_estimator->inference(img);
    
    cv::Mat depth_norm;
    cv::normalize(depth, depth_norm, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::Mat heatmap;
    cv::applyColorMap(depth_norm, heatmap, cv::COLORMAP_INFERNO);
    
    cv::Mat combined;
    cv::hconcat(img, heatmap, combined);
    
    cv::imwrite(output_path, combined);
    std::cout << "Success! Side-by-side result saved to: " << output_path << "\n";
    
    return 0;
}
