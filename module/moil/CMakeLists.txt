# CMakeLists.txt for module/moil

file(GLOB_RECURSE MOIL_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
)

add_library(mod_moil STATIC ${MOIL_SOURCES})

# Include directories for this module
target_include_directories(mod_moil PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/nlohmann_json/single_include
)

# Detect Target Architecture
if(CMAKE_SYSTEM_PROCESSOR)
    set(ARCH_NAME ${CMAKE_SYSTEM_PROCESSOR})
else()
    execute_process(COMMAND uname -m OUTPUT_VARIABLE ARCH_NAME OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()

# We need OpenCV for cv::Mat
find_package(PkgConfig REQUIRED)
pkg_check_modules(OPENCV REQUIRED opencv4)

target_include_directories(mod_moil PUBLIC ${OPENCV_INCLUDE_DIRS})
target_link_libraries(mod_moil PUBLIC ${OPENCV_LIBRARIES})

# Link libmoildevren.a if on aarch64
if(ARCH_NAME MATCHES "aarch64")
    # Link against the static library in 07_midas_aruco_fusion/lib
    target_link_libraries(mod_moil PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/lib/libmoildev_cpu.so
    )
    # Headers for moildev_applicator, frame_processor, moil_utils
    target_include_directories(mod_moil PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/lib/include
    )
endif()
