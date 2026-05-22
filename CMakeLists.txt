cmake_minimum_required(VERSION 3.10)
project(MidasCoffeeMachine CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# =========================================================
# === Detect Architecture =================================
# =========================================================
if(CMAKE_SYSTEM_PROCESSOR)
    set(ARCH_NAME ${CMAKE_SYSTEM_PROCESSOR})
else()
    execute_process(
        COMMAND uname -m
        OUTPUT_VARIABLE ARCH_NAME
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
endif()

message(STATUS "Target architecture: ${ARCH_NAME}")

# =========================================================
# === Cross compile safety (Poky) =========================
# =========================================================
set(IS_AARCH64 FALSE)

if(ARCH_NAME STREQUAL "aarch64")
    message(STATUS "Configuring for AARCH64 (cross compile / embedded)")
    set(IS_AARCH64 TRUE)

    # Global definitions for V2H platform
    add_compile_definitions(V2H)
    add_compile_definitions(DRP_AI_TVM_RUNTIME=1)

    set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
    set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
    set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
    set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

else()
    message(STATUS "Configuring for x86 native build")
endif()

# =========================================================
# === Packages ============================================
# =========================================================
find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)

find_package(PkgConfig REQUIRED)
pkg_check_modules(GTK3 REQUIRED gtk+-3.0)

# =========================================================
# === Third Party: Matplot++ (LOCAL) ======================
# =========================================================
option(MATPLOTPP_BUILD_EXAMPLES OFF)
option(MATPLOTPP_BUILD_TESTS OFF)
option(MATPLOTPP_BUILD_SHARED_LIBS ON)

add_subdirectory(third_party/matplotplusplus EXCLUDE_FROM_ALL)

# =========================================================
# === Third Party: nlohmann_json (LOCAL HEADER ONLY) ======
# =========================================================
set(NLOHMANN_JSON_INCLUDE_DIR
    ${CMAKE_SOURCE_DIR}/third_party/nlohmann_json/include
)

include_directories(${NLOHMANN_JSON_INCLUDE_DIR})

# =========================================================
# === TVM / ARM Compute (ONLY FOR AARCH64) ================
# =========================================================
if(IS_AARCH64)

    message(STATUS "Enabling TVM / V2H runtime libs")

    set(TVM_LIB_DIR
        /home/user/Documents/TVM/drp-ai_tvm/obj/build_runtime/v2h/lib
    )

    link_directories(${TVM_LIB_DIR})

    set(TVM_LIBS
        mera2_runtime
        arm_compute
        arm_compute_core
        arm_compute_graph
        acl_rt
        mmngr
        mmngrbuf
    )

else()

    message(STATUS "Skipping TVM / V2H libs for x86")

    set(TVM_LIBS "")
endif()

# =========================================================
# === Includes ============================================
# =========================================================
include_directories(
    ${CMAKE_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${YAML_CPP_INCLUDE_DIR}
)

# =========================================================
# === Core Library ========================================
# =========================================================
set(CORE_SRC
    midas_volumecup/camera_config.cpp
    midas_volumecup/volume_math.cpp
    midas_volumecup/detector.cpp
    midas_volumecup/depth.cpp
)

add_library(midas_core ${CORE_SRC})

target_link_libraries(midas_core
    ${OpenCV_LIBS}
    yaml-cpp
    pthread
    dl
)

# =========================================================
# === Modules =============================================
# =========================================================
add_subdirectory(module/detections)
add_subdirectory(module/logger)
add_subdirectory(module/camera)

target_include_directories(midas_core PUBLIC
    ${CMAKE_SOURCE_DIR}/module/detections/include
    ${CMAKE_SOURCE_DIR}/module/logger/include
    ${CMAKE_SOURCE_DIR}/module/camera/include
    ${CMAKE_SOURCE_DIR}/module/detections/constants
    ${CMAKE_SOURCE_DIR}/module/detections/lib/yolo/include
)

target_link_libraries(midas_core
    mod_detect
    mod_logger
    mod_camera
    ${TVM_LIBS}
)

# =========================================================
# === Executables =========================================
# =========================================================

add_executable(calibrate_depth_multivariate 01_calibration/calibrate_depth_multivariate.cpp)
target_link_libraries(calibrate_depth_multivariate midas_core matplot)

add_executable(calibrate_midas_polynomial 01_calibration/calibrate_midas_polynomial.cpp)
target_link_libraries(calibrate_midas_polynomial midas_core)

add_executable(visualize_curve 01_calibration/visualize_curve.cpp)
target_link_libraries(visualize_curve midas_core matplot)

add_executable(evaluate_depth_multivariate 02_evaluation/evaluate_depth_multivariate.cpp)
target_link_libraries(evaluate_depth_multivariate midas_core matplot)

add_executable(evaluate_test_data 02_evaluation/evaluate_test_data.cpp)
target_link_libraries(evaluate_test_data midas_core)

add_executable(detect_camera_height_midas 03_diagnostics/detect_camera_height_midas.cpp)
target_link_libraries(detect_camera_height_midas midas_core)

add_executable(test_single_image 03_diagnostics/test_single_image.cpp)
target_link_libraries(test_single_image midas_core)

add_executable(collect_test_data 04_dataset/collect_test_data.cpp)
target_link_libraries(collect_test_data midas_core)

add_executable(run_volumecup_midas 05_production/run_volumecup_midas.cpp)
target_link_libraries(run_volumecup_midas midas_core)

add_executable(tray_detector_cpp 05_tray_pattern_recog/tray_detector.cpp)
target_link_libraries(tray_detector_cpp midas_core)

add_executable(run_aruco
    06_aruco_marker/run_aruco.cpp
    06_aruco_marker/aruco_detector.cpp
)
target_link_libraries(run_aruco midas_core matplot)

add_executable(generate_marker 06_aruco_marker/generate_marker.cpp)
target_link_libraries(generate_marker ${OpenCV_LIBS})

# =========================================================
# === 07: ArUco + MiDaS + YOLOv8 Fusion (C++ / RZ-V2H) ==
# =========================================================
add_executable(run_fusion_cpp
    07_midas_aruco_fusion/run_fusion.cpp
    07_midas_aruco_fusion/core/height_math.cpp
    07_midas_aruco_fusion/core/calibration_storage.cpp
    07_midas_aruco_fusion/core/calibration_routines.cpp
    07_midas_aruco_fusion/core/live_pipeline.cpp
    07_midas_aruco_fusion/core/session_reporter.cpp
    07_midas_aruco_fusion/core/moil_undistorter.cpp
    07_midas_aruco_fusion/core/gui_fusion.cpp
    07_midas_aruco_fusion/core/volume_math.cpp
    06_aruco_marker/aruco_detector.cpp
)
target_include_directories(run_fusion_cpp PRIVATE
    ${CMAKE_SOURCE_DIR}/07_midas_aruco_fusion
    ${CMAKE_SOURCE_DIR}/07_midas_aruco_fusion/lib/include
    ${CMAKE_SOURCE_DIR}/06_aruco_marker
    ${GTK3_INCLUDE_DIRS}
)
target_link_libraries(run_fusion_cpp
    midas_core
    stdc++fs
    ${CMAKE_SOURCE_DIR}/module/moil/lib/libmoildevren.a
    ${GTK3_LIBRARIES}
)