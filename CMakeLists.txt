cmake_minimum_required(VERSION 3.18)
project(CameraCVService)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_TOOLCHAIN_FILE "E:/tools/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")

set(ONNXRUNTIME_ROOT "E:/tools/onnxruntime1_18")
set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_ROOT}/include")
set(ONNXRUNTIME_LIBRARY "${ONNXRUNTIME_ROOT}/lib/onnxruntime.lib")
set(ONNXRUNTIME_RUNTIME_DIR "${ONNXRUNTIME_ROOT}/lib")
set(OPENH264_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/openh264-2.6.0/codec/api/wels")
set(OPENH264_RUNTIME_DLL "${CMAKE_CURRENT_SOURCE_DIR}/third_party/openh264-2.6.0-win64.dll")

find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)
find_package(LibDataChannel CONFIG REQUIRED)

add_executable(camera_cv_service
    src/main.cpp
    src/camera_manager.cpp
    src/inference_engine.cpp
    src/openh264_encoder.cpp
    src/webrtc_service.cpp
)

target_include_directories(camera_cv_service PRIVATE
    include
    include/models
    ${ONNXRUNTIME_INCLUDE_DIR}
    ${OPENH264_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}/opencv4
    E:/tools/vcpkg/installed/x64-windows/include
)

target_link_libraries(camera_cv_service PRIVATE
    ${OpenCV_LIBS}
    Threads::Threads
    LibDataChannel::LibDataChannel
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
