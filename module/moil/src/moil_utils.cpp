/**
 * @file moil_utils.cpp
 * @brief Implements utility functions for Moildev-related OpenCV operations.
 *
 * @details This file contains the definitions for the functions declared in `moil_utils.hpp`, 
 * including drawing remapping polygons and performing point-in-polygon collision detection.
 */
#include "utils/moil_utils.hpp"
#include <vector>

/**
 * @brief Draws the boundary of a remapped area onto a source image.
 * @details This function extracts the edge points from `mapX` and `mapY` and
 * draws them as four distinct polylines (top, bottom, left, right) onto the
 * `image`. It is useful for visualizing the field of view of an Anypoint or
 * Panorama view on the original source image. The line thickness is dynamically
 * adjusted based on the image width.
 *
 * @param image The input source image to be drawn upon (modified in-place).
 * Must be in BGR format.
 * @param mapX The x-coordinate map from `cv::remap` (type CV_32FC1).
 * @param mapY The y-coordinate map from `cv::remap` (type CV_32FC1).
 */
void MoilUtils::drawPolygon(cv::Mat &image, const cv::Mat &mapX,
                            const cv::Mat &mapY) {
  if (mapX.empty() || mapY.empty() || image.empty()) {
    return;
  }

  std::vector<cv::Point> top_points, bottom_points, left_points, right_points;

  // Extract points from the edges of the remap maps
  // Top edge
  for (int x = 0; x < mapX.cols; ++x) {
    float mx = mapX.at<float>(0, x);
    float my = mapY.at<float>(0, x);
    if (mx != 0 && my != 0) {
      top_points.push_back(
          cv::Point(static_cast<int>(mx), static_cast<int>(my)));
      cv::circle(image, cv::Point(static_cast<int>(mx), static_cast<int>(my)),
                 3, cv::Scalar(0, 0, 255), -1); // Gambar titik merah
    }
  }

  // Bottom edge
  for (int x = 0; x < mapX.cols; ++x) {
    float mx = mapX.at<float>(mapX.rows - 1, x);
    float my = mapY.at<float>(mapY.rows - 1, x);
    if (mx != 0 && my != 0) {
      bottom_points.push_back(
          cv::Point(static_cast<int>(mx), static_cast<int>(my)));
      cv::circle(image, cv::Point(static_cast<int>(mx), static_cast<int>(my)),
                 3, cv::Scalar(0, 255, 0), -1); // Gambar titik hijau
    }
  }

  // Left edge
  for (int y = 0; y < mapX.rows; ++y) {
    float mx = mapX.at<float>(y, 0);
    float my = mapY.at<float>(y, 0);
    if (mx != 0 && my != 0) {
      left_points.push_back(
          cv::Point(static_cast<int>(mx), static_cast<int>(my)));
      cv::circle(image, cv::Point(static_cast<int>(mx), static_cast<int>(my)),
                 3, cv::Scalar(255, 0, 0), -1); // Gambar titik biru
    }
  }

  // Right edge
  for (int y = 0; y < mapX.rows; ++y) {
    float mx = mapX.at<float>(y, mapX.cols - 1);
    float my = mapY.at<float>(y, mapY.cols - 1);
    if (mx != 0 && my != 0) {
      right_points.push_back(
          cv::Point(static_cast<int>(mx), static_cast<int>(my)));
      cv::circle(image, cv::Point(static_cast<int>(mx), static_cast<int>(my)),
                 3, cv::Scalar(255, 255, 0), -1); // Gambar titik kuning
    }
  }

  // Dynamically determine line thickness based on source image width
  int line_thickness = 5;
  if (image.cols > 1944)
    line_thickness = 10;
  else if (image.cols >= 1300)
    line_thickness = 6;
  else if (image.cols >= 800)
    line_thickness = 3;

  // Draw the polylines onto the input image
  if (!top_points.empty())
    cv::polylines(image, top_points, false, cv::Scalar(0, 0, 255),
                  line_thickness); // Merah
  if (!left_points.empty())
    cv::polylines(image, left_points, false, cv::Scalar(255, 0, 0),
                  line_thickness); // Biru
  if (!bottom_points.empty())
    cv::polylines(image, bottom_points, false, cv::Scalar(0, 255, 0),
                  line_thickness); // Hijau
  if (!right_points.empty())
    cv::polylines(image, right_points, false, cv::Scalar(0, 255, 0),
                  line_thickness); // Hijau
}

/**
 * @brief Extracts the boundary points from remap matrices into a single
 * contour.
 * @details This function traces the edges of the valid (non-zero) areas in
 * `mapX` and `mapY` and combines them into a single, ordered vector of
 * `cv::Point`. The resulting vector can be used to represent the remapped area
 * as a closed polygon.
 *
 * @param mapX The x-coordinate map from `cv::remap` (type CV_32FC1).
 * @param mapY The y-coordinate map from `cv::remap` (type CV_32FC1).
 * @return std::vector<cv::Point> A vector of `cv::Point` representing the
 * vertices of the polygon.
 */
std::vector<cv::Point> MoilUtils::getPolygonFromMaps(const cv::Mat &mapX,
                                                     const cv::Mat &mapY) {
  std::vector<cv::Point> polygon_points;
  if (mapX.empty() || mapY.empty()) {
    return polygon_points;
  }

  std::vector<cv::Point> top_points, bottom_points, left_points, right_points;

  // Extract Top Edge (left to right)
  for (int x = 0; x < mapX.cols; ++x) {
    float mx = mapX.at<float>(0, x);
    float my = mapY.at<float>(0, x);
    if (mx != 0 || my != 0) {
      top_points.push_back(
          cv::Point(static_cast<int>(mx), static_cast<int>(my)));
    }
  }

  // Extract Right Edge (top to bottom)
  for (int y = 0; y < mapX.rows; ++y) {
    float mx = mapX.at<float>(y, mapX.cols - 1);
    float my = mapY.at<float>(y, mapY.cols - 1);
    if (mx != 0 || my != 0) {
      right_points.push_back(
          cv::Point(static_cast<int>(mx), static_cast<int>(my)));
    }
  }

  // Extract Bottom Edge (left to right)
  for (int x = 0; x < mapX.cols; ++x) {
    float mx = mapX.at<float>(mapX.rows - 1, x);
    float my = mapY.at<float>(mapY.rows - 1, x);
    if (mx != 0 || my != 0) {
      bottom_points.push_back(
          cv::Point(static_cast<int>(mx), static_cast<int>(my)));
    }
  }

  // Extract Left Edge (top to bottom)
  for (int y = 0; y < mapX.rows; ++y) {
    float mx = mapX.at<float>(y, 0);
    float my = mapY.at<float>(y, 0);
    if (mx != 0 || my != 0) {
      left_points.push_back(
          cv::Point(static_cast<int>(mx), static_cast<int>(my)));
    }
  }

  // Combine all points in clockwise order to form a closed polygon
  polygon_points.insert(polygon_points.end(), top_points.begin(),
                        top_points.end());
  polygon_points.insert(polygon_points.end(), right_points.begin(),
                        right_points.end());

  std::reverse(bottom_points.begin(), bottom_points.end());
  polygon_points.insert(polygon_points.end(), bottom_points.begin(),
                        bottom_points.end());

  std::reverse(left_points.begin(), left_points.end());
  polygon_points.insert(polygon_points.end(), left_points.begin(),
                        left_points.end());

  return polygon_points;
}

/**
 * @brief Checks if a given point is inside the area defined by remap matrices.
 * @details This is a convenience function that first calls `getPolygonFromMaps`
 * to generate the boundary and then uses `cv::pointPolygonTest` to determine if
 * the specified point lies inside or on the edge of that polygon.
 *
 * @param mapX The x-coordinate map from `cv::remap`.
 * @param mapY The y-coordinate map from `cv::remap`.
 * @param point The `cv::Point2f` to test.
 * @return bool Returns `true` if the point is inside or on the boundary of the
 * polygon, `false` otherwise.
 */
bool MoilUtils::isPointInPolygon(const cv::Mat &mapX, const cv::Mat &mapY,
                                 const cv::Point2f &point) {
  // 1. Get the polygon corner points from the maps.
  std::vector<cv::Point> polygon_points =
      MoilUtils::getPolygonFromMaps(mapX, mapY);

  // 2. If the polygon is empty (e.g., invalid maps), the point cannot be
  // inside.
  if (polygon_points.empty()) {
    return false;
  }

  // 3. Use OpenCV's built-in function for the test.
  double distance = cv::pointPolygonTest(polygon_points, point, false);

  // 4. Return 'true' if the distance is >= 0 (meaning inside or exactly on the
  // edge).
  return distance >= 0;
}

/**
 * @brief Checks if any point from a given list is inside a specified polygon.
 * @details This function efficiently checks a list of points against a polygon.
 * It stops and returns `true` as soon as the first point inside the polygon is
 * found, making it efficient for checking multiple objects (e.g., detection
 * centers).
 *
 * @param polygon A vector of `cv::Point` representing the vertices of the
 * polygon.
 * @param points A vector of `cv::Point2f` to be tested against the polygon.
 * @return bool Returns `true` if at least one point is inside or on the
 * boundary of the polygon, `false` otherwise.
 */
bool MoilUtils::isAnyPointInPolygon(const std::vector<cv::Point> &polygon,
                                    const std::vector<cv::Point2f> &points) {
  // If there's no polygon or no points to check, return false immediately.
  if (polygon.empty() || points.empty()) {
    return false;
  }

  // Loop through each point to be checked
  for (const auto &point : points) {
    // Use OpenCV's efficient point-in-polygon test.
    // A result >= 0 means the point is inside or on the boundary.
    if (cv::pointPolygonTest(polygon, point, false) >= 0) {
      // As soon as we find one point inside, we know the answer is true.
      // No need to check the rest, so we return immediately.
      return true;
    }
  }

  // If the loop completes, it means no points were found inside.
  return false;
}