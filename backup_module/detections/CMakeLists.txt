# Ambil semua source cpp di src/ recursively
file(GLOB_RECURSE DETECTION_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/drp_queue.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/yolo/src/*.cpp
)

# Buat library shared
add_library(mod_detect SHARED ${DETECTION_SOURCES})

# target_compile_options(mod_detect PRIVATE
#     -Wno-deprecated-enum-enum-conversion
# )

# =========================
# Detect target architecture
# =========================
if(CMAKE_SYSTEM_PROCESSOR)
    set(ARCH_NAME ${CMAKE_SYSTEM_PROCESSOR})
else()
    execute_process(COMMAND uname -m OUTPUT_VARIABLE ARCH_NAME OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()
message(STATUS "Module detections target architecture: ${ARCH_NAME}")

# =========================
# Tentukan folder ONNX dan TFLite berdasarkan arsitektur
# =========================
if(ARCH_NAME MATCHES "aarch64")
    set(ONNXRUNTIME_LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/lib/onnxruntime-linux-aarch64-1.20.1)
    set(TFLITE_LIB ${CMAKE_CURRENT_SOURCE_DIR}/lib/tflite/tensorflowlite_aarch64)
    set(TVM_ROOT $ENV{TVM_ROOT})

    target_include_directories(mod_detect PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/lib/drp-ai_tvm_v230
        ${TVM_ROOT}/tvm/include
        ${TVM_ROOT}/setup/include
        ${TVM_ROOT}/tvm/3rdparty/dlpack/include
        ${TVM_ROOT}/tvm/3rdparty/dmlc-core/include
        ${TVM_ROOT}/tvm/3rdparty/compiler-rt
        ${TVM_ROOT}/3rdparty/spdlog/include
        ${TVM_ROOT}/3rdparty/asio/include
    )

    set (LIBMERA_RT_PATH ${TVM_ROOT}/obj/build_runtime/v2h/lib)
    add_definitions(-DMERA_DRP_RUNTIME)

    file(GLOB_RECURSE V2H_SRC ${CMAKE_CURRENT_SOURCE_DIR}/lib/drp-ai_tvm_v230/*.cpp)

    target_sources(mod_detect PUBLIC ${V2H_SRC})

    add_library(mera2_runtime SHARED IMPORTED)
    set_target_properties(mera2_runtime PROPERTIES
        IMPORTED_LOCATION ${LIBMERA_RT_PATH}/libmera2_runtime.so)

    add_library(mera2_plan_io SHARED IMPORTED)
    set_target_properties(mera2_plan_io PROPERTIES
        IMPORTED_LOCATION ${LIBMERA_RT_PATH}/libmera2_plan_io.so)

    add_library(drp_rt SHARED IMPORTED)
    set_target_properties(drp_rt PROPERTIES
        IMPORTED_LOCATION ${LIBMERA_RT_PATH}/libdrp_rt.so)

    add_library(drp_tvm_rt SHARED IMPORTED)
    set_target_properties(drp_tvm_rt PROPERTIES
        IMPORTED_LOCATION ${LIBMERA_RT_PATH}/libdrp_tvm_rt.so)

    target_link_libraries(mod_detect PUBLIC
        mera2_runtime
        mera2_plan_io
        drp_rt
        drp_tvm_rt
        pthread
        rt
        dl
        gpiodcxx
        mmngrbuf
    )

    add_compile_definitions(V2H)
    add_compile_definitions(DRP_AI_TVM_RUNTIME=1)
else()
    set(ONNXRUNTIME_LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/lib/onnxruntime-linux-x64-1.20.1)
    set(TFLITE_LIB ${CMAKE_CURRENT_SOURCE_DIR}/lib/tflite/tensorflowlite_x86_64)
    add_compile_definitions(LINUX64)
endif()

# =========================
# Include semua folder internal
# =========================
target_include_directories(mod_detect PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/tflite
    ${ONNXRUNTIME_LIB_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/flatbuffers
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/yolo/include
)

# =========================
# Link OpenCV
# =========================
find_package(PkgConfig REQUIRED)
pkg_check_modules(OPENCV REQUIRED opencv4)

target_include_directories(mod_detect PUBLIC ${OPENCV_INCLUDE_DIRS})
target_link_libraries(mod_detect PUBLIC ${OPENCV_LIBRARIES})

# =========================
# Link ONNX Runtime dan TFLite sesuai arsitektur
# =========================
target_link_libraries(mod_detect PUBLIC
    ${TFLITE_LIB}/lib/libtensorflowlite.so
    ${ONNXRUNTIME_LIB_DIR}/lib/libonnxruntime.so
    mod_logger
)

# Export path ke parent (root)
set(MODDETECT_RUNTIME_LIBS
    ${TFLITE_LIB}
    ${ONNXRUNTIME_LIB_DIR}/lib/libonnxruntime.so
    $<TARGET_FILE:mod_detect>
    PARENT_SCOPE
)