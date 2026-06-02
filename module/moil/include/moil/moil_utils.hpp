/**
 * @file moil_utils.hpp
 * @brief Declares utility functions for Moildev-related OpenCV operations.
 * @details This header defines the `MoilUtils` namespace, which contains helper functions
 * for tasks such as visualizing remapping areas as polygons and performing point-in-polygon tests.
 */
#ifndef MOIL_UTILS_HPP
#define MOIL_UTILS_HPP

#include <opencv2/opencv.hpp>
#include <vector>

/**
 * @namespace MoilUtils
 * @brief Geometric Utilities for Fisheye Transformations.
 * @details Provides helper algorithms to analyze the spatial properties of the remapping process,
 * such as determining "Where is this Panorama on the original image?".
 */
namespace MoilUtils
{
    /**
     * @brief Visualization: Projects the Remap Boundary onto the Source Image.
     * @details Draws the outline of the target view (e.g., the rectangular area of an Anypoint view)
     * back onto the circular fisheye image. This aids in debugging and calibration by showing
     * exactly what part of the lens is being used.
     *
     * @param image The raw fisheye image (drawn upon in-place).
     * @param mapX The X-Lookup Table of the view.
     * @param mapY The Y-Lookup Table of the view.
     */
    void drawPolygon(cv::Mat &image, const cv::Mat &mapX, const cv::Mat &mapY);

    /**
     * @brief Extracts the boundary points from remap matrices into a single contour.
     * @details This function traces the edges of the valid (non-zero) areas in `mapX` and `mapY`
     * and combines them into a single, ordered vector of `cv::Point`. The resulting vector
     * can be used to represent the remapped area as a closed polygon.
     *
     * @param mapX The x-coordinate map from `cv::remap` (type CV_32FC1).
     * @param mapY The y-coordinate map from `cv::remap` (type CV_32FC1).
     * @return std::vector<cv::Point> A vector of `cv::Point` representing the vertices of the polygon.
     */
    std::vector<cv::Point> getPolygonFromMaps(const cv::Mat &mapX, const cv::Mat &mapY);

    /**
     * @brief Hit-Testing: Check if a Cartesian Point is inside the View.
     * @details Determines if a specific pixel on the raw image falls within the active area of
     * the remapped view.
     *
     * **Algorithm**:
     * 1. Extract boundary contour from MapX/MapY.
     * 2. Use `cv::pointPolygonTest` to check intersection.
     *
     * @param mapX The view's X-map.
     * @param mapY The view's Y-map.
     * @param point The point to test (e.g., object centroid).
     * @return `true` if the point is visible in this view.
     */
    bool isPointInPolygon(const cv::Mat &mapX, const cv::Mat &mapY, const cv::Point2f &point);

    /**
     * @brief Checks if any point from a given list is inside a specified polygon.
     * @details This function efficiently checks a list of points against a polygon. It stops and
     * returns `true` as soon as the first point inside the polygon is found, making it
     * efficient for checking multiple objects (e.g., detection centers).
     *
     * @param polygon A vector of `cv::Point` representing the vertices of the polygon.
     * @param points A vector of `cv::Point2f` to be tested against the polygon.
     * @return bool Returns `true` if at least one point is inside or on the boundary of the polygon, `false` otherwise.
     */
    bool isAnyPointInPolygon(const std::vector<cv::Point> &polygon, const std::vector<cv::Point2f> &points);

}

#endif // MOIL_UTILS_HPP