##############################################################################
# CMakeLists.txt — 07_midas_aruco_fusion / moildev_applicator (C++ library)
#
# Mengkompilasi core/moildev_applicator.cpp menjadi shared library
# libmoildev_applicator.so yang dapat dipakai oleh Python (ctypes) atau
# binary C++ standalone.
#
# nlohmann_json telah terinstall via:
#   sudo pacman -S nlohmann-json
#
# Cara build (library):
#   cd 07_midas_aruco_fusion
#   cmake -B build/cmake -DCMAKE_BUILD_TYPE=Release
#   cmake --build build/cmake -j$(nproc)
#
# Cara build + test (x86, tanpa hardware ARM):
#   cmake -B build/cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBUILD_TESTS=ON
#   cmake --build build/cmake -j$(nproc)
#   ctest --test-dir build/cmake --output-on-failure
##############################################################################

cmake_minimum_required(VERSION 3.18)
project(moildev_applicator VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── Tentukan path root proyek ───────────────────────────────────────────────
set(PROJ_ROOT     "${CMAKE_CURRENT_SOURCE_DIR}")
set(MODULE_ROOT   "${PROJ_ROOT}/../module")

# ── OpenCV ──────────────────────────────────────────────────────────────────
find_package(OpenCV REQUIRED
    COMPONENTS core imgproc calib3d highgui
)
message(STATUS "OpenCV: ${OpenCV_VERSION} (${OpenCV_INCLUDE_DIRS})")

# ── nlohmann/json — terinstall via pacman -S nlohmann-json ──────────────────
find_package(nlohmann_json REQUIRED)
message(STATUS "nlohmann_json: ditemukan")

# ── Deteksi arsitektur target ───────────────────────────────────────────────
# libmoildev_cpu.so adalah binary ARM aarch64 (untuk Renesas RZ/V2H).
# Jika build di x86_64 (PC development), kita tidak bisa link ke ARM binary.
# Dalam mode ini kita hanya compile object file (untuk syntax/type checking).

include(CheckSymbolExists)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm")
    set(IS_ARM_TARGET TRUE)
else()
    set(IS_ARM_TARGET FALSE)
    message(STATUS "[MOIL] Build di PC x86_64 — moildev_cpu akan di-stub (tidak di-link)")
endif()

if(IS_ARM_TARGET)
    if(NOT EXISTS "${MOILDEV_LIB}")
        message(FATAL_ERROR
            "libmoildev_cpu.so tidak ditemukan di: ${MOILDEV_LIB}\n"
            "Pastikan module/moil/lib/libmoildev_cpu.so tersedia.")
    endif()
    message(STATUS "libmoildev_cpu.so [ARM]: ${MOILDEV_LIB}")
    add_library(moildev_cpu SHARED IMPORTED)
    set_target_properties(moildev_cpu PROPERTIES
        IMPORTED_LOCATION "${MOILDEV_LIB}"
        INTERFACE_INCLUDE_DIRECTORIES "${MOILDEV_INC}"
    )
else()
    # PC mode: buat interface library hanya untuk header
    add_library(moildev_cpu INTERFACE)
    target_include_directories(moildev_cpu INTERFACE "${MOILDEV_INC}")
    message(STATUS "libmoildev_cpu.so [PC stub]: header-only dari ${MOILDEV_INC}")
endif()

# ── Target: libmoildev_applicator.so ──────────────────────────────────────────────
add_library(moildev_applicator SHARED
    core/moildev_applicator.cpp
)

target_include_directories(moildev_applicator
    PUBLIC
        "${PROJ_ROOT}"
        "${PROJ_ROOT}/core"
        "${MOILDEV_INC}"
)

if(IS_ARM_TARGET)
    target_link_libraries(moildev_applicator
        PUBLIC
            ${OpenCV_LIBS}
            nlohmann_json::nlohmann_json
            moildev_cpu
    )
else()
    # PC: link ke OpenCV + nlohmann saja, TANPA ARM binary
    target_link_libraries(moildev_applicator
        PUBLIC
            ${OpenCV_LIBS}
            nlohmann_json::nlohmann_json
    )
    target_include_directories(moildev_applicator PRIVATE "${MOILDEV_INC}")
endif()


# Compile options
target_compile_options(moildev_applicator PRIVATE
    -Wall -Wextra -Wpedantic
    -O3
    $<$<CONFIG:Debug>:-g -fsanitize=address>
)

# Set RPATH agar libmoildev_cpu.so ditemukan saat runtime tanpa LD_LIBRARY_PATH
set_target_properties(moildev_applicator PROPERTIES
    INSTALL_RPATH "${MOILDEV_DIR}/lib"
    BUILD_WITH_INSTALL_RPATH TRUE
    VERSION "${PROJECT_VERSION}"
    SOVERSION 1
)

# ── Output ke folder lib/ ───────────────────────────────────────────────────
set_target_properties(moildev_applicator PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${PROJ_ROOT}/lib"
    ARCHIVE_OUTPUT_DIRECTORY "${PROJ_ROOT}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${PROJ_ROOT}/bin"
)

# ── Install ─────────────────────────────────────────────────────────────────
install(TARGETS moildev_applicator
    LIBRARY  DESTINATION lib
    ARCHIVE  DESTINATION lib
    RUNTIME  DESTINATION bin
)
install(FILES core/moildev_applicator.hpp
    DESTINATION include/moildev_applicator
)

# ── Summary ─────────────────────────────────────────────────────────────────
message(STATUS "")
message(STATUS "====== moildev_applicator build summary ======")
message(STATUS "  Build type    : ${CMAKE_BUILD_TYPE}")
message(STATUS "  OpenCV        : ${OpenCV_VERSION}")
message(STATUS "  nlohmann_json : OK (pacman)")
message(STATUS "  moildev_cpu   : ${MOILDEV_LIB}")
message(STATUS "  Output dir    : ${PROJ_ROOT}/lib")
message(STATUS "")

# ── Unit Tests (x86 only, tidak butuh ARM hardware) ─────────────────────────
# Aktifkan dengan: cmake -DBUILD_TESTS=ON
# Hanya berjalan di x86_64 — pure math logic tanpa DRP-AI / moildev_cpu
option(BUILD_TESTS "Build standalone C++ unit tests (x86 only)" OFF)

if(BUILD_TESTS)
    if(IS_ARM_TARGET)
        message(WARNING "[TEST] BUILD_TESTS hanya untuk x86 development. Dinonaktifkan di ARM.")
    else()
        enable_testing()
        message(STATUS "[TEST] Membangun unit tests (x86 mode)...")

        # ── Common compile flags untuk semua test ─────────────────────────
        set(TEST_COMPILE_FLAGS -Wall -Wextra -O2 -DNDEBUG)

        # ── test_height_math ─────────────────────────────────────────────
        add_executable(test_height_math
            tests/test_height_math.cpp
            core/height_math.cpp
        )
        target_include_directories(test_height_math PRIVATE
            "${PROJ_ROOT}"
            "${PROJ_ROOT}/core"
        )
        target_link_libraries(test_height_math PRIVATE ${OpenCV_LIBS})
        target_compile_options(test_height_math PRIVATE ${TEST_COMPILE_FLAGS})
        set_target_properties(test_height_math PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${PROJ_ROOT}/build/tests"
        )
        add_test(NAME HeightMath COMMAND test_height_math)
        message(STATUS "  [TEST] test_height_math: OK")

        # ── test_volume_math ─────────────────────────────────────────────
        add_executable(test_volume_math
            tests/test_volume_math.cpp
            core/volume_math.cpp
        )
        target_include_directories(test_volume_math PRIVATE
            "${PROJ_ROOT}"
            "${PROJ_ROOT}/core"
        )
        target_link_libraries(test_volume_math PRIVATE ${OpenCV_LIBS})
        target_compile_options(test_volume_math PRIVATE ${TEST_COMPILE_FLAGS})
        set_target_properties(test_volume_math PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${PROJ_ROOT}/build/tests"
        )
        add_test(NAME VolumeMath COMMAND test_volume_math)
        message(STATUS "  [TEST] test_volume_math: OK")

        # ── test_calib_poly (header-only poly_fit.hpp) ────────────────────
        # Tidak butuh .cpp tambahan — poly_fit.hpp adalah header-only
        add_executable(test_calib_poly
            tests/test_calib_poly.cpp
        )
        target_include_directories(test_calib_poly PRIVATE
            "${PROJ_ROOT}"
            "${PROJ_ROOT}/core"
        )
        target_compile_options(test_calib_poly PRIVATE ${TEST_COMPILE_FLAGS})
        set_target_properties(test_calib_poly PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${PROJ_ROOT}/build/tests"
        )
        add_test(NAME CalibPoly COMMAND test_calib_poly)
        message(STATUS "  [TEST] test_calib_poly: OK")

        # ── test_moildev_math (standalone math, tanpa moildev_cpu.so) ────
        add_executable(test_moildev_math
            tests/test_moildev_math.cpp
        )
        target_include_directories(test_moildev_math PRIVATE
            "${PROJ_ROOT}"
            "${PROJ_ROOT}/core"
        )
        target_link_libraries(test_moildev_math PRIVATE ${OpenCV_LIBS})
        target_compile_options(test_moildev_math PRIVATE ${TEST_COMPILE_FLAGS})
        set_target_properties(test_moildev_math PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${PROJ_ROOT}/build/tests"
        )
        add_test(NAME MoildevMath COMMAND test_moildev_math)
        message(STATUS "  [TEST] test_moildev_math: OK")

        message(STATUS "")
        message(STATUS "  Jalankan semua test: ctest --test-dir build/cmake --output-on-failure")
        message(STATUS "")
    endif()
endif()

