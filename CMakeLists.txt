cmake_minimum_required(VERSION 3.18)

if(WIN32 AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    if(DEFINED ENV{VCPKG_ROOT} AND EXISTS "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake")
        set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")
    else()
        message(FATAL_ERROR "CMAKE_TOOLCHAIN_FILE is not set. Define it explicitly or set VCPKG_ROOT.")
    endif()
endif()

project(CameraCVService)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# =========================
# ONNXRUNTIME CONFIG
# =========================
if(WIN32)
    set(DEFAULT_ONNXRUNTIME_ROOT "E:/tools/onnxruntime1_18")
else()
    set(DEFAULT_ONNXRUNTIME_ROOT "/opt/onnxruntime")
endif()

if(NOT DEFINED ONNXRUNTIME_ROOT OR ONNXRUNTIME_ROOT STREQUAL "")
    set(ONNXRUNTIME_ROOT "${DEFAULT_ONNXRUNTIME_ROOT}" CACHE PATH "ONNX Runtime root directory" FORCE)
else()
    set(ONNXRUNTIME_ROOT "${ONNXRUNTIME_ROOT}" CACHE PATH "ONNX Runtime root directory")
endif()

set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_ROOT}/include")

if(WIN32)
    set(ONNXRUNTIME_LIBRARY "${ONNXRUNTIME_ROOT}/lib/onnxruntime.lib")
    set(ONNXRUNTIME_RUNTIME_DIR "${ONNXRUNTIME_ROOT}/lib")
else()
    find_library(ONNXRUNTIME_LIBRARY
        NAMES onnxruntime
        PATHS "${ONNXRUNTIME_ROOT}/lib" "${ONNXRUNTIME_ROOT}/lib64"
        NO_DEFAULT_PATH
    )
    if(NOT ONNXRUNTIME_LIBRARY)
        find_library(ONNXRUNTIME_LIBRARY NAMES onnxruntime)
    endif()
    if(NOT ONNXRUNTIME_LIBRARY)
        message(FATAL_ERROR "ONNX Runtime library was not found")
    endif()
    get_filename_component(ONNXRUNTIME_RUNTIME_DIR "${ONNXRUNTIME_LIBRARY}" DIRECTORY)
endif()

# =========================
# OpenH264
# =========================
set(OPENH264_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/openh264-2.6.0/codec/api/wels")
set(OPENH264_RUNTIME_DLL "${CMAKE_CURRENT_SOURCE_DIR}/third_party/openh264-2.6.0-win64.dll")

# =========================
# OPTIONS
# =========================
option(USE_BYTETRACK "Enable ByteTrack tracking" ON)
option(USE_TENSORRT "Enable TensorRT runtime backend" OFF)

# =========================
# DEPENDENCIES
# =========================
include(CheckIPOSupported)
check_ipo_supported(RESULT IPO_SUPPORTED OUTPUT IPO_ERROR)

find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)
find_package(LibDataChannel CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)
find_package(OpenSSL REQUIRED)

# =========================
# BYTE TRACK
# =========================
set(BYTETRACK_SOURCES "")

if(USE_BYTETRACK)
    set(BYTETRACK_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ByteTrack-cpp)

    if(EXISTS "${BYTETRACK_DIR}/src/BYTETracker.cpp")
        message(STATUS "ByteTrack found → enabling tracking module")

        set(BYTETRACK_SOURCES
            ${BYTETRACK_DIR}/src/BYTETracker.cpp
            ${BYTETRACK_DIR}/src/KalmanFilter.cpp
            ${BYTETRACK_DIR}/src/Object.cpp
            ${BYTETRACK_DIR}/src/Rect.cpp
            ${BYTETRACK_DIR}/src/STrack.cpp
            ${BYTETRACK_DIR}/src/lapjv.cpp
        )
    else()
        message(WARNING "ByteTrack missing → DISABLED (build will continue)")
        set(USE_BYTETRACK OFF)
    endif()
endif()

# =========================
# EXECUTABLE
# =========================
add_executable(camera_cv_service
    src/main.cpp
    src/core/pipeline/runtime_config.cpp
    src/capture/opencv_video_source.cpp
    src/camera_manager.cpp
    src/inference_engine.cpp
    src/inference_backend_factory.cpp
    src/onnx_runtime_backend.cpp
    src/openh264_encoder.cpp
    src/tracking_manager.cpp
    src/webrtc_service.cpp
    src/webrtc_service_signaling.cpp
    src/webrtc_service_peer.cpp
    src/webrtc_service_media.cpp
    ${BYTETRACK_SOURCES}
)

# Platform sources
if(WIN32)
    target_sources(camera_cv_service PRIVATE src/platform/windows/windows_platform_services.cpp)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_sources(camera_cv_service PRIVATE src/platform/linux/linux_platform_services.cpp)
else()
    message(FATAL_ERROR "Unsupported platform")
endif()

# =========================
# TENSORRT CONFIG
# =========================
if(USE_TENSORRT)
    find_package(CUDA REQUIRED)
    find_path(TENSORRT_INCLUDE_DIR NvInfer.h REQUIRED)
    find_library(TENSORRT_LIBRARY NAMES nvinfer REQUIRED)
    find_library(TENSORRT_PLUGIN_LIBRARY NAMES nvinfer_plugin REQUIRED)
    
    message(STATUS "TensorRT enabled")
    message(STATUS "TensorRT include: ${TENSORRT_INCLUDE_DIR}")
    message(STATUS "TensorRT library: ${TENSORRT_LIBRARY}")
    
    target_sources(camera_cv_service PRIVATE src/tensorrt_runtime_backend.cpp)
    target_include_directories(camera_cv_service PRIVATE ${TENSORRT_INCLUDE_DIR} ${CUDA_INCLUDE_DIRS})
    target_compile_definitions(camera_cv_service PRIVATE USE_TENSORRT=1)
    target_link_libraries(camera_cv_service PRIVATE ${TENSORRT_LIBRARY} ${TENSORRT_PLUGIN_LIBRARY} ${CUDA_LIBRARIES})
else()
    message(STATUS "TensorRT disabled (set USE_TENSORRT=ON to enable)")
endif()

# =========================
# INCLUDE DIRS
# =========================
target_include_directories(camera_cv_service PRIVATE
    include
    include/models
    ${ONNXRUNTIME_INCLUDE_DIR}
    ${OPENH264_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

if(USE_BYTETRACK AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ByteTrack-cpp/include")
    target_include_directories(camera_cv_service PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ByteTrack-cpp/include
    )
endif()

# =========================
# DEFINITIONS
# =========================
target_compile_definitions(camera_cv_service PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:_CRT_SECURE_NO_WARNINGS>
)

if(USE_BYTETRACK)
    target_compile_definitions(camera_cv_service PRIVATE USE_BYTETRACK=1)
else()
    target_compile_definitions(camera_cv_service PRIVATE USE_BYTETRACK=0)
endif()

# =========================
# COMPILER FLAGS
# =========================
if(MSVC)
    target_compile_options(camera_cv_service PRIVATE /W4 /EHsc)
else()
    target_compile_options(camera_cv_service PRIVATE -Wall -Wextra -Wpedantic)
endif()

# =========================
# IPO/LTO
# =========================
if(IPO_SUPPORTED)
    set_property(TARGET camera_cv_service PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
endif()

# =========================
# LINKING
# =========================
target_link_libraries(camera_cv_service PRIVATE
    ${OpenCV_LIBS}
    Threads::Threads
    LibDataChannel::LibDataChannel
    Eigen3::Eigen
    OpenSSL::Crypto
    ${ONNXRUNTIME_LIBRARY}
)

if(WIN32)
    target_link_libraries(camera_cv_service PRIVATE Version)
elseif(UNIX)
    target_link_libraries(camera_cv_service PRIVATE ${CMAKE_DL_LIBS})
endif()

# =========================
# OUTPUT
# =========================
set_target_properties(camera_cv_service PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

install(TARGETS camera_cv_service
    RUNTIME DESTINATION bin
)

# =========================
# POST BUILD (WIN ONLY)
# =========================
if(WIN32)
    set(ONNXRUNTIME_RUNTIME_DLLS
        onnxruntime.dll
        onnxruntime_providers_shared.dll
        onnxruntime_providers_cuda.dll
        onnxruntime_providers_tensorrt.dll
    )

    foreach(ORT_RUNTIME_DLL IN LISTS ONNXRUNTIME_RUNTIME_DLLS)
        if(EXISTS "${ONNXRUNTIME_RUNTIME_DIR}/${ORT_RUNTIME_DLL}")
            add_custom_command(TARGET camera_cv_service POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${ONNXRUNTIME_RUNTIME_DIR}/${ORT_RUNTIME_DLL}"
                "$<TARGET_FILE_DIR:camera_cv_service>/${ORT_RUNTIME_DLL}"
            )
        endif()
    endforeach()

    if(EXISTS "${OPENH264_RUNTIME_DLL}")
        add_custom_command(TARGET camera_cv_service POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${OPENH264_RUNTIME_DLL}"
            "$<TARGET_FILE_DIR:camera_cv_service>/openh264.dll"
        )
    endif()
endif()

# =========================
# DEBUG INFO
# =========================
message(STATUS "OpenCV: ${OpenCV_VERSION}")
message(STATUS "ONNX Runtime: ${ONNXRUNTIME_INCLUDE_DIR}")
message(STATUS "ByteTrack enabled: ${USE_BYTETRACK}")
