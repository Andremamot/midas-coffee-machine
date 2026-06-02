cmake_minimum_required(VERSION 3.15)
project(mod_volume)

# ─────────────────────────────────────────────────────────────────────────────
# mod_volume — Cup Volume Estimator Module
#
# Mengikuti pola Model_kakip/module/:
#   - SHARED library (mod_volume)
#   - include/ → public headers (diakses sebagai <volume/cup_volume_estimator.h>)
#   - src/      → implementasi + internal headers
#
# Gunakan via add_subdirectory() dari proyek induk.
# ─────────────────────────────────────────────────────────────────────────────

# Kumpulkan semua .cpp secara rekursif (sama seperti pola mod_camera)
file(GLOB_RECURSE VOLUME_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
)

# Buat shared library
add_library(mod_volume SHARED ${VOLUME_SOURCES})

# Suppress Moildev compatibility warnings
target_compile_options(mod_volume PRIVATE
    -Wno-deprecated-enum-enum-conversion
)

# ── Include Directories ──────────────────────────────────────────────────────
target_include_directories(mod_volume PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include     # public: <volume/...>
)
target_include_directories(mod_volume PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src         # internal headers (moil_undistorter.h, dll)
    ${CMAKE_CURRENT_SOURCE_DIR}/src/fisheye # internal fisheye headers
)

# ── OpenCV ───────────────────────────────────────────────────────────────────
find_package(PkgConfig REQUIRED)
pkg_check_modules(OPENCV REQUIRED opencv4)
target_include_directories(mod_volume PUBLIC ${OPENCV_INCLUDE_DIRS})
target_link_libraries(mod_volume PUBLIC ${OPENCV_LIBRARIES})

# ── nlohmann_json ────────────────────────────────────────────────────────────
# Coba find_package dulu, jika tidak ada gunakan FetchContent
find_package(nlohmann_json QUIET)
if(nlohmann_json_FOUND)
    target_link_libraries(mod_volume PUBLIC nlohmann_json::nlohmann_json)
else()
    include(FetchContent)
    FetchContent_Declare(nlohmann_json
        URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    FetchContent_MakeAvailable(nlohmann_json)
    target_link_libraries(mod_volume PUBLIC nlohmann_json::nlohmann_json)
endif()

# ── Moildev (libmoildev.a) ───────────────────────────────────────────────────
# libmoildev.a TIDAK di-compile dengan -fPIC sehingga TIDAK BISA di-embed ke .so.
# Solusi: gunakan INTERFACE — artinya proyek pemanggil yang link libmoildev.a.
#
# Cara penggunaan di proyek induk:
#   add_subdirectory(module/volume)
#   target_link_libraries(my_app PRIVATE mod_volume /path/to/libmoildev.a)
#
set(MOILDEV_SEARCH_PATHS
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../weights/moil     # dari Model_kakip/module/volume/
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../weights/moil  # fallback satu level lebih dalam
)
find_library(MOILDEV_LIB moildev
    PATHS ${MOILDEV_SEARCH_PATHS}
    NO_DEFAULT_PATH
)
if(MOILDEV_LIB)
    # INTERFACE = tidak di-embed ke .so, tapi caller harus link juga
    target_link_libraries(mod_volume INTERFACE ${MOILDEV_LIB})
    message(STATUS "[mod_volume] Moildev found: ${MOILDEV_LIB}")
else()
    message(WARNING "[mod_volume] libmoildev.a tidak ditemukan. "
                    "Kalian harus link manual: target_link_libraries(your_app PRIVATE /path/to/libmoildev.a)")
endif()

# pthread
target_link_libraries(mod_volume PUBLIC pthread)

message(STATUS "[mod_volume] Module configured")
