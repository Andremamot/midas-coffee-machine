# Ambil semua source cpp di src/ recursively
file(GLOB_RECURSE CAM_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
)

# Buat library shared
add_library(mod_camera SHARED ${CAM_SOURCES})

target_compile_options(mod_camera PRIVATE
    -Wno-deprecated-enum-enum-conversion
)

target_include_directories(mod_camera PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# =========================
# Link OpenCV
# =========================
find_package(PkgConfig REQUIRED)
pkg_check_modules(OPENCV REQUIRED opencv4)

target_include_directories(mod_camera PUBLIC ${OPENCV_INCLUDE_DIRS})
target_link_libraries(mod_camera PUBLIC ${OPENCV_LIBRARIES})

target_link_libraries(mod_camera PRIVATE mod_logger)