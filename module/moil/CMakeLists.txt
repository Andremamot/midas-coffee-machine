# ─────────────────────────────────────────────────────────────────────────────
# mod_moil — Fisheye Undistortion Module (via libmoildevren.a)
#
# Mengikuti pola backup_module/:
#   - SHARED library (mod_moil)
#   - include/moil/ → public headers (diakses sebagai <moil/moil_undistorter.h>)
#   - src/          → implementasi
#
# CATATAN libmoildevren.a:
#   libmoildevren.a TIDAK dikompilasi dengan -fPIC sehingga TIDAK BISA di-embed
#   ke .so. Gunakan INTERFACE — proyek pemanggil yang harus link libmoildevren.a.
#
#   Cara penggunaan di proyek induk:
#     add_subdirectory(module/moil)
#     target_link_libraries(my_app PRIVATE mod_moil /path/to/libmoildevren.a)
#
# Gunakan via add_subdirectory() dari proyek induk.
# ─────────────────────────────────────────────────────────────────────────────

# Kumpulkan semua .cpp secara rekursif
file(GLOB_RECURSE MOIL_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
)

# Buat shared library
add_library(mod_moil SHARED ${MOIL_SOURCES})

# Suppress Moildev compatibility warnings
target_compile_options(mod_moil PRIVATE
    -Wno-deprecated-enum-enum-conversion
)

# ── Include Directories ──────────────────────────────────────────────────────
target_include_directories(mod_moil PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include     # public: <moil/...>
)

# ── OpenCV ───────────────────────────────────────────────────────────────────
find_package(PkgConfig REQUIRED)
pkg_check_modules(OPENCV REQUIRED opencv4)

target_include_directories(mod_moil PUBLIC ${OPENCV_INCLUDE_DIRS})
target_link_libraries(mod_moil PUBLIC ${OPENCV_LIBRARIES})

# ── nlohmann_json ────────────────────────────────────────────────────────────
# Digunakan untuk parse camera_parameters.json
find_package(nlohmann_json QUIET)
if(nlohmann_json_FOUND)
    target_link_libraries(mod_moil PUBLIC nlohmann_json::nlohmann_json)
else()
    include(FetchContent)
    FetchContent_Declare(nlohmann_json
        URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    FetchContent_MakeAvailable(nlohmann_json)
    target_link_libraries(mod_moil PUBLIC nlohmann_json::nlohmann_json)
endif()

# ── Moildev (libmoildevren.a) ─────────────────────────────────────────────────
# libmoildevren.a TIDAK di-compile dengan -fPIC sehingga TIDAK BISA di-embed ke .so.
# Solusi: gunakan INTERFACE — artinya proyek pemanggil yang link libmoildevren.a.
#
# Cari libmoildevren.a di lokasi yang umum (relatif dari proyek root)
set(MOILDEV_SEARCH_PATHS
    ${CMAKE_CURRENT_SOURCE_DIR}/lib                        # dari module/moil/lib/
    ${CMAKE_CURRENT_SOURCE_DIR}/../../module/moil/lib      # fallback relatif
    ${CMAKE_SOURCE_DIR}/module/moil/lib                    # dari root proyek
    ${CMAKE_SOURCE_DIR}/weights/moil                       # dari root/weights/moil/
)
find_library(MOILDEV_LIB moildevren
    PATHS ${MOILDEV_SEARCH_PATHS}
    NO_DEFAULT_PATH
)
if(MOILDEV_LIB)
    # INTERFACE = tidak di-embed ke .so, tapi caller harus link juga
    target_link_libraries(mod_moil INTERFACE ${MOILDEV_LIB})
    message(STATUS "[mod_moil] Moildev found: ${MOILDEV_LIB}")
else()
    message(WARNING "[mod_moil] libmoildevren.a tidak ditemukan. "
                    "Link manual: target_link_libraries(your_app PRIVATE /path/to/libmoildevren.a)")
endif()

# ── pthread ───────────────────────────────────────────────────────────────────
target_link_libraries(mod_moil PUBLIC pthread)

# Export path ke parent (root)
set(MODMOIL_RUNTIME_LIBS
    $<TARGET_FILE:mod_moil>
    PARENT_SCOPE
)

message(STATUS "[mod_moil] Module configured")
