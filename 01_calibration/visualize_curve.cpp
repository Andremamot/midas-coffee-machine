#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <yaml-cpp/yaml.h>
#include <matplot/matplot.h>

int main() {
    std::string config_path = "../midas_calibration.yaml";
    std::string output_path = "calibration_curve_cpp.png";

    try {
        YAML::Node config = YAML::LoadFile(config_path);
        double a = config["a"] ? config["a"].as<double>() : 0.0;
        double b = config["b"] ? config["b"].as<double>() : 0.0;
        double c = config["c"] ? config["c"].as<double>() : 0.0;

        std::vector<double> r_values = matplot::linspace(0.5, 3.0, 500);
        std::vector<double> z_values;
        for (double r : r_values) {
            z_values.push_back((a / (r + b)) + c);
        }

        auto f = matplot::figure(true);
        f->size(1200, 720);
        
        // Due to Matplot++ style limitations compared to matplotlib, 
        // we approximate the styling.
        matplot::plot(r_values, z_values, "-b")->line_width(3).display_name("Inverse Fit");
        
        matplot::title("MiDaS Calibration: Distance vs Ratio (Inverse Model)");
        matplot::xlabel("AI Ratio (M_rim / M_tray)");
        matplot::ylabel("Distance Z (mm)");
        
        matplot::grid(matplot::on);
        matplot::legend();
        
        matplot::save(output_path);
        std::cout << "Plot saved to " << output_path << std::endl;
        
    } catch (const YAML::Exception& e) {
        std::cerr << "YAML Error (" << config_path << "): " << e.what() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    return 0;
}
