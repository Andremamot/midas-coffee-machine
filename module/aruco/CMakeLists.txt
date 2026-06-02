# CMakeLists.txt for module/aruco
# Builds mod_aruco as a shared library for ArUco marker detection.

file(GLOB_RECURSE ARUCO_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
)

add_library(mod_aruco SHARED ${ARUCO_SOURCES})

target_include_directories(mod_aruco PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

find_package(PkgConfig REQUIRED)
pkg_check_modules(OPENCV REQUIRED opencv4)

target_include_directories(mod_aruco PUBLIC ${OPENCV_INCLUDE_DIRS})
target_link_libraries(mod_aruco PUBLIC ${OPENCV_LIBRARIES} yaml-cpp)
