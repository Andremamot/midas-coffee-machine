# ─────────────────────────────────────────────────────────────────────────────
# mod_volume — Cup Volume Estimator Module
#
# Mengikuti pola backup_module/:
#   - SHARED library (mod_volume)
#   - include/volume/ → public headers (diakses sebagai <volume/...>)
#   - src/            → implementasi
#
# Public API:
#   #include <volume/height_math.h>   → BBox, polyval, 7 mode kalkulasi tinggi
#   #include <volume/volume_math.h>   → measure_rim_width_px, calc_diameter, calc_volume
#   #include <volume/volume_config.h> → konstanta konfigurasi (MOIL_ZOOM, dll.)
#
# Dependencies:
#   - OpenCV (imgproc, core)
#   - (opsional) mod_moil jika menggunakan MoilUndistorter di proyek induk
#
# Gunakan via add_subdirectory() dari proyek induk.
# ─────────────────────────────────────────────────────────────────────────────

# Kumpulkan semua .cpp secara rekursif (sama seperti pola mod_camera)
file(GLOB_RECURSE VOLUME_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
)

# Buat shared library
add_library(mod_volume SHARED ${VOLUME_SOURCES})

# Suppress compatibility warnings
target_compile_options(mod_volume PRIVATE
    -Wno-deprecated-enum-enum-conversion
)

# ── Include Directories ──────────────────────────────────────────────────────
target_include_directories(mod_volume PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include     # public: <volume/...>
)

# ── OpenCV ───────────────────────────────────────────────────────────────────
find_package(PkgConfig REQUIRED)
pkg_check_modules(OPENCV REQUIRED opencv4)

target_include_directories(mod_volume PUBLIC ${OPENCV_INCLUDE_DIRS})
target_link_libraries(mod_volume PUBLIC ${OPENCV_LIBRARIES})

# ── pthread ───────────────────────────────────────────────────────────────────
target_link_libraries(mod_volume PUBLIC pthread)

# Export path ke parent (root)
set(MODVOLUME_RUNTIME_LIBS
    $<TARGET_FILE:mod_volume>
    PARENT_SCOPE
)

message(STATUS "[mod_volume] Module configured")
