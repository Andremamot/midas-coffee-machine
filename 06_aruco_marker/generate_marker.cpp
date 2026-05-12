#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <sys/stat.h>

#ifdef _WIN32
#include <direct.h>
#define MKDIR(x) _mkdir(x)
#else
#define MKDIR(x) mkdir(x, 0777)
#endif

int main(int argc, char** argv) {
    int marker_id = 0;
    int size_px = 400;
    std::string dictionary_name = "DICT_4X4_50";
    int border_bits = 1;
    std::string output_path = "";

    std::map<std::string, cv::aruco::PREDEFINED_DICTIONARY_NAME> dict_map = {
        {"DICT_4X4_50", cv::aruco::DICT_4X4_50},
        {"DICT_4X4_100", cv::aruco::DICT_4X4_100},
        {"DICT_4X4_250", cv::aruco::DICT_4X4_250},
        {"DICT_5X5_50", cv::aruco::DICT_5X5_50},
        {"DICT_5X5_100", cv::aruco::DICT_5X5_100},
        {"DICT_5X5_250", cv::aruco::DICT_5X5_250},
        {"DICT_6X6_50", cv::aruco::DICT_6X6_50},
        {"DICT_6X6_100", cv::aruco::DICT_6X6_100},
        {"DICT_6X6_250", cv::aruco::DICT_6X6_250},
        {"DICT_7X7_50", cv::aruco::DICT_7X7_50},
        {"DICT_7X7_100", cv::aruco::DICT_7X7_100},
        {"DICT_7X7_250", cv::aruco::DICT_7X7_250}
    };

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--id" && i + 1 < argc) marker_id = std::stoi(argv[++i]);
        else if (arg == "--size" && i + 1 < argc) size_px = std::stoi(argv[++i]);
        else if (arg == "--dictionary" && i + 1 < argc) dictionary_name = argv[++i];
        else if (arg == "--border-bits" && i + 1 < argc) border_bits = std::stoi(argv[++i]);
        else if (arg == "--output" && i + 1 < argc) output_path = argv[++i];
    }

    if (dict_map.find(dictionary_name) == dict_map.end()) {
        std::cerr << "Unknown dictionary: " << dictionary_name << "\n";
        return 1;
    }

    cv::Ptr<cv::aruco::Dictionary> aruco_dict = cv::aruco::getPredefinedDictionary(dict_map[dictionary_name]);
    cv::Mat marker_img;
    cv::aruco::drawMarker(aruco_dict, marker_id, size_px, marker_img, border_bits);

    int border_px = size_px / 4;
    cv::Mat canvas(size_px + 2 * border_px, size_px + 2 * border_px, CV_8UC1, cv::Scalar(255));
    marker_img.copyTo(canvas(cv::Rect(border_px, border_px, size_px, size_px)));

    if (output_path.empty()) {
        MKDIR("markers");
        output_path = "markers/aruco_" + dictionary_name + "_id" + std::to_string(marker_id) + "_" + std::to_string(size_px) + "px.png";
    }

    cv::imwrite(output_path, canvas);

    std::cout << "Marker generated:\n";
    std::cout << "   Dictionary : " << dictionary_name << "\n";
    std::cout << "   ID         : " << marker_id << "\n";
    std::cout << "   Size       : " << size_px << "px (+ " << border_px << "px white border each side)\n";
    std::cout << "   Saved to   : " << output_path << "\n";

    return 0;
}
