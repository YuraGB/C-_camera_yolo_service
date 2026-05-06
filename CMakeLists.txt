cmake_minimum_required(VERSION 3.18)

if(NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    set(CMAKE_TOOLCHAIN_FILE "E:/tools/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")
endif()

project(CameraCVService)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(DEFAULT_ONNXRUNTIME_ROOT "E:/tools/onnxruntime1_18")
if(NOT DEFINED ONNXRUNTIME_ROOT OR ONNXRUNTIME_ROOT STREQUAL "")
    set(ONNXRUNTIME_ROOT "${DEFAULT_ONNXRUNTIME_ROOT}" CACHE PATH "ONNX Runtime root directory" FORCE)
else()
    set(ONNXRUNTIME_ROOT "${ONNXRUNTIME_ROOT}" CACHE PATH "ONNX Runtime root directory")
endif()
set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_ROOT}/include")
set(ONNXRUNTIME_LIBRARY "${ONNXRUNTIME_ROOT}/lib/onnxruntime.lib")
set(ONNXRUNTIME_RUNTIME_DIR "${ONNXRUNTIME_ROOT}/lib")
set(OPENH264_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/openh264-2.6.0/codec/api/wels")
set(OPENH264_RUNTIME_DLL "${CMAKE_CURRENT_SOURCE_DIR}/third_party/openh264-2.6.0-win64.dll")

include(CheckIPOSupported)
check_ipo_supported(RESULT IPO_SUPPORTED OUTPUT IPO_ERROR)

find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)
find_package(LibDataChannel CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)

add_executable(camera_cv_service
    src/main.cpp
    src/camera_manager.cpp
    src/inference_engine.cpp
    src/openh264_encoder.cpp
    src/tracking_manager.cpp
    src/webrtc_service.cpp
    src/webrtc_service_signaling.cpp
    src/webrtc_service_peer.cpp
    src/webrtc_service_media.cpp
    third_party/ByteTrack-cpp/src/BYTETracker.cpp
    third_party/ByteTrack-cpp/src/KalmanFilter.cpp
    third_party/ByteTrack-cpp/src/Object.cpp
    third_party/ByteTrack-cpp/src/Rect.cpp
    third_party/ByteTrack-cpp/src/STrack.cpp
    third_party/ByteTrack-cpp/src/lapjv.cpp
)

set(BYTETRACK_SOURCES
    third_party/ByteTrack-cpp/src/BYTETracker.cpp
    third_party/ByteTrack-cpp/src/KalmanFilter.cpp
    third_party/ByteTrack-cpp/src/Object.cpp
    third_party/ByteTrack-cpp/src/Rect.cpp
    third_party/ByteTrack-cpp/src/STrack.cpp
    third_party/ByteTrack-cpp/src/lapjv.cpp
)

target_include_directories(camera_cv_service PRIVATE
    include
    include/models
    ${ONNXRUNTIME_INCLUDE_DIR}
    ${OPENH264_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ByteTrack-cpp/include
    ${OpenCV_INCLUDE_DIRS}/opencv4
    E:/tools/vcpkg/installed/x64-windows/include
)

target_compile_definitions(camera_cv_service PRIVATE
    NOMINMAX
    WIN32_LEAN_AND_MEAN
    $<$<CXX_COMPILER_ID:MSVC>:_CRT_SECURE_NO_WARNINGS>
)

if(MSVC)
    target_compile_options(camera_cv_service PRIVATE
        /permissive-
        /W4
        /EHsc
        $<$<CONFIG:Release>:/O2>
        $<$<CONFIG:Release>:/Ob2>
        $<$<CONFIG:Release>:/Oi>
        $<$<CONFIG:Release>:/Ot>
        $<$<CONFIG:Release>:/GL>
    )
    target_link_options(camera_cv_service PRIVATE
        $<$<CONFIG:Release>:/LTCG>
        $<$<CONFIG:Release>:/OPT:REF>
        $<$<CONFIG:Release>:/OPT:ICF>
    )
else()
    target_compile_options(camera_cv_service PRIVATE
        -Wall
        -Wextra
        -Wpedantic
        $<$<CONFIG:Release>:-O3>
        $<$<CONFIG:Release>:-DNDEBUG>
    )
endif()

if(MSVC)
    set_source_files_properties(${BYTETRACK_SOURCES} PROPERTIES
        COMPILE_OPTIONS "/wd4100;/wd4244;/wd4267;/wd4305"
    )
else()
    set_source_files_properties(${BYTETRACK_SOURCES} PROPERTIES COMPILE_OPTIONS -w)
endif()

if(IPO_SUPPORTED)
    set_property(TARGET camera_cv_service PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
else()
    message(STATUS "IPO/LTO is not supported: ${IPO_ERROR}")
endif()

target_link_libraries(camera_cv_service PRIVATE
    ${OpenCV_LIBS}
    Threads::Threads
    LibDataChannel::LibDataChannel
    Eigen3::Eigen
    ${ONNXRUNTIME_LIBRARY}
    Version
)

set_target_properties(camera_cv_service PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

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
            COMMENT "Copying ${ORT_RUNTIME_DLL} to runtime output"
        )
    endif()
endforeach()

if(EXISTS "${OPENH264_RUNTIME_DLL}")
    add_custom_command(TARGET camera_cv_service POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${OPENH264_RUNTIME_DLL}"
            "$<TARGET_FILE_DIR:camera_cv_service>/openh264-2.6.0-win64.dll"
        COMMENT "Copying OpenH264 runtime DLL to runtime output"
    )
endif()

message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "ONNX Runtime include dir: ${ONNXRUNTIME_INCLUDE_DIR}")
message(STATUS "ONNX Runtime runtime dir: ${ONNXRUNTIME_RUNTIME_DIR}")
