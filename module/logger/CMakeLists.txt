add_library(mod_logger SHARED
    src/logger.cpp
)

target_include_directories(mod_logger PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/spdlog/include
)

set(SPDLOG_LIB ${CMAKE_CURRENT_SOURCE_DIR}/lib/spdlog/libspdlog.a)

target_link_libraries(mod_logger PUBLIC ${SPDLOG_LIB})

# DEFINE __DEBUG (optional toggle)
target_compile_definitions(mod_logger PUBLIC __DEBUG)

# Export path ke parent (root)
set(MODLOGGER_RUNTIME_LIBS
    ${SPDLOG_LIB}
    $<TARGET_FILE:mod_logger>
    PARENT_SCOPE
)