cmake_minimum_required(VERSION 3.18)
project(CameraCVService)

# ------------------------------
# C++ стандарт
# ------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


# ------------------------------
# ONNX Runtime (розпакований вручну)
# ------------------------------
set(ONNXRUNTIME_INCLUDE_DIR "E:/tools/onnxruntime/include")
set(ONNXRUNTIME_LIBRARY "E:/tools/onnxruntime/lib/onnxruntime.lib")

# Вказати шляхи до include та lib для OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Вказати шляхи до Protobuf та gRPC
include_directories(E:/Progects/test/camera_cv_service/include)
include_directories(E:/Progects/test/camera_cv_service/include/generated_models)
link_directories(E:/Progects/test/camera_cv_service/lib)

# ------------------------------
# Toolchain vcpkg
# ------------------------------
set(CMAKE_TOOLCHAIN_FILE "E:/tools/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")


# ------------------------------
# Пакети через vcpkg
# ------------------------------
find_package(OpenCV REQUIRED)
find_package(Protobuf REQUIRED)
find_package(gRPC CONFIG REQUIRED)
find_package(Threads REQUIRED)

# ------------------------------
# Генерація protobuf / gRPC файлів
# ------------------------------
set(GENERATED_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include/generated_models)
file(MAKE_DIRECTORY ${GENERATED_DIR})

set(PROTO_FILE ${CMAKE_CURRENT_SOURCE_DIR}/models/detection.proto)

# Повні шляхи до protoc та grpc plugin
set(PROTOC_EXE E:/tools/vcpkg/installed/x64-windows/tools/protobuf/protoc.exe)
set(GRPC_CPP_PLUGIN E:/tools/vcpkg/installed/x64-windows/tools/grpc/grpc_cpp_plugin.exe)
set(PROTO_INCLUDE_DIR E:/tools/vcpkg/installed/x64-windows/include)

# .pb.cc / .pb.h
add_custom_command(
    OUTPUT ${GENERATED_DIR}/detection.pb.cc ${GENERATED_DIR}/detection.pb.h
    COMMAND ${PROTOC_EXE}
            --proto_path=${CMAKE_CURRENT_SOURCE_DIR}/models
            --proto_path=${PROTO_INCLUDE_DIR}
            --cpp_out=${GENERATED_DIR}
            ${PROTO_FILE}
    DEPENDS ${PROTO_FILE}
    COMMENT "Generating protobuf files"
)

# .grpc.pb.cc / .grpc.pb.h
add_custom_command(
    OUTPUT ${GENERATED_DIR}/detection.grpc.pb.cc ${GENERATED_DIR}/detection.grpc.pb.h
    COMMAND ${PROTOC_EXE}
            --proto_path=${CMAKE_CURRENT_SOURCE_DIR}/models
            --proto_path=${PROTO_INCLUDE_DIR}
            --grpc_out=${GENERATED_DIR}
            --plugin=protoc-gen-grpc=${GRPC_CPP_PLUGIN}
            ${PROTO_FILE}
    DEPENDS ${PROTO_FILE}
    COMMENT "Generating gRPC files"
)

set(GENERATED_SRC
    ${GENERATED_DIR}/detection.pb.cc
    ${GENERATED_DIR}/detection.grpc.pb.cc
)

# ------------------------------
# Executable
# ------------------------------
add_executable(camera_cv_service
    src/main.cpp
    src/byte_tracker.cpp
    src/byte_tracker_lapjv.cpp
    src/camera_manager.cpp
    src/inference_engine.cpp
    src/grpc_server.cpp
    ${GENERATED_SRC}
)

# ------------------------------
# Include directories
# ------------------------------
target_include_directories(camera_cv_service PRIVATE
    include
    include/models
    ${GENERATED_DIR}
    ${ONNXRUNTIME_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}/opencv4
    ${Protobuf_INCLUDE_DIRS}
    E:/tools/vcpkg/installed/x64-windows/include
    E:/tools/vcpkg/installed/x64-windows/include/grpc
    E:/tools/vcpkg/installed/x64-windows/include/grpcpp
    E:/tools/onnxruntime/include
)

# ------------------------------
# Link libraries
# ------------------------------
target_link_libraries(camera_cv_service PRIVATE
    ${OpenCV_LIBS}
    gRPC::grpc++
    gRPC::grpc
    protobuf::libprotobuf
    Threads::Threads
    ${ONNXRUNTIME_LIBRARY}
    Version
)

# ------------------------------
# Output directory
# ------------------------------
set_target_properties(camera_cv_service PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# ------------------------------
# Статусні повідомлення
# ------------------------------
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "ONNX Runtime include dir: ${ONNXRUNTIME_INCLUDE_DIR}")
message(STATUS "Generated proto files dir: ${GENERATED_DIR}")
message(STATUS "Protobuf include dir: ${PROTO_INCLUDE_DIR}")
