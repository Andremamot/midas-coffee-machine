#include "07_midas_aruco_fusion/core/moil_undistorter.hpp"
#include <iostream>
int main() {
    try {
        MoilUndistorter moil("07_midas_aruco_fusion/camera_parameters.json", "syue_7730v1_6", 0.0, 0.0, 0.0, 1.6, 2);
        std::cout << "SUCCESS" << std::endl;
    } catch(std::exception& e) {
        std::cerr << "EXCEPTION: " << e.what() << std::endl;
    }
    return 0;
}
