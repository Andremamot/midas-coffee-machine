# ─────────────────────────────────────────────────────────────────────────────
# mod_aruco — ArUco Marker Detection & Pose Estimation Module
#
# Mengikuti pola backup_module/:
#   - SHARED library (mod_aruco)
#   - include/aruco/ → public headers (diakses sebagai <aruco/aruco_detector.h>)
#   - src/           → implementasi
#
# Dependencies:
#   - OpenCV (core, aruco, imgproc, calib3d)
#   - yaml-cpp (untuk load calibration .yml)
#   - mod_logger (opsional, untuk logging internal)
#
# Gunakan via add_subdirectory() dari proyek induk.
# ─────────────────────────────────────────────────────────────────────────────

# Kumpulkan semua .cpp secara rekursif (sama seperti pola mod_camera)
file(GLOB_RECURSE ARUCO_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
)

# Buat shared library
add_library(mod_aruco SHARED ${ARUCO_SOURCES})

# Suppress deprecation warnings (OpenCV aruco API lama)
target_compile_options(mod_aruco PRIVATE
    -Wno-deprecated-enum-enum-conversion
    -Wno-deprecated-declarations
)

# ── Include Directories ──────────────────────────────────────────────────────
target_include_directories(mod_aruco PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include     # public: <aruco/...>
)

# ── OpenCV ───────────────────────────────────────────────────────────────────
find_package(PkgConfig REQUIRED)
pkg_check_modules(OPENCV REQUIRED opencv4)

target_include_directories(mod_aruco PUBLIC ${OPENCV_INCLUDE_DIRS})
target_link_libraries(mod_aruco PUBLIC ${OPENCV_LIBRARIES})

# ── yaml-cpp ─────────────────────────────────────────────────────────────────
# Digunakan untuk load calibration_params.yml
find_package(yaml-cpp QUIET)
if(yaml-cpp_FOUND)
    target_link_libraries(mod_aruco PUBLIC yaml-cpp)
else()
    # Fallback: cari manual
    find_library(YAML_CPP_LIB yaml-cpp)
    if(YAML_CPP_LIB)
        target_link_libraries(mod_aruco PUBLIC ${YAML_CPP_LIB})
    else()
        message(WARNING "[mod_aruco] yaml-cpp tidak ditemukan. "
                        "Link manual: target_link_libraries(your_app PRIVATE yaml-cpp)")
    endif()
endif()

# ── pthread ───────────────────────────────────────────────────────────────────
target_link_libraries(mod_aruco PUBLIC pthread)

# Export path ke parent (root)
set(MODARUCO_RUNTIME_LIBS
    $<TARGET_FILE:mod_aruco>
    PARENT_SCOPE
)

message(STATUS "[mod_aruco] Module configured")
