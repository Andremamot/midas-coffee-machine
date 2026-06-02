# CMakeLists.txt for module/cup_height

file(GLOB_RECURSE CUP_HEIGHT_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
)

add_library(mod_cup_height SHARED ${CUP_HEIGHT_SOURCES})

target_include_directories(mod_cup_height PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
