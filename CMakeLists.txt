cmake_minimum_required(VERSION 3.18)
project(CameraCVService)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(GENERATED_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include/generated_models)
file(MAKE_DIRECTORY ${GENERATED_DIR})
set(PROTO_FILE ${CMAKE_CURRENT_SOURCE_DIR}/models/detection.proto)

set(ONNXRUNTIME_ROOT "" CACHE PATH "Root directory of the ONNX Runtime installation")
set(PROTOC_EXE "" CACHE FILEPATH "Path to protoc executable")
set(GRPC_CPP_PLUGIN "" CACHE FILEPATH "Path to grpc_cpp_plugin executable")

find_package(OpenCV REQUIRED)
find_package(Protobuf REQUIRED)
find_package(gRPC CONFIG REQUIRED)
find_package(Threads REQUIRED)

find_path(ONNXRUNTIME_INCLUDE_DIR
    NAMES onnxruntime_cxx_api.h
    HINTS
        ${ONNXRUNTIME_ROOT}
        ${ONNXRUNTIME_ROOT}/include
    PATH_SUFFIXES include
    REQUIRED
)

find_library(ONNXRUNTIME_LIBRARY
    NAMES onnxruntime
    HINTS
        ${ONNXRUNTIME_ROOT}
        ${ONNXRUNTIME_ROOT}/lib
    PATH_SUFFIXES lib
    REQUIRED
)

if(NOT PROTOC_EXE)
    set(PROTOC_EXE $<TARGET_FILE:protobuf::protoc>)
endif()

if(NOT GRPC_CPP_PLUGIN)
    get_target_property(GRPC_CPP_PLUGIN gRPC::grpc_cpp_plugin LOCATION)
endif()

add_custom_command(
    OUTPUT ${GENERATED_DIR}/detection.pb.cc ${GENERATED_DIR}/detection.pb.h
    COMMAND ${PROTOC_EXE}
            --proto_path=${CMAKE_CURRENT_SOURCE_DIR}/models
            --cpp_out=${GENERATED_DIR}
            ${PROTO_FILE}
    DEPENDS ${PROTO_FILE}
    COMMENT "Generating protobuf files"
)

add_custom_command(
    OUTPUT ${GENERATED_DIR}/detection.grpc.pb.cc ${GENERATED_DIR}/detection.grpc.pb.h
    COMMAND ${PROTOC_EXE}
            --proto_path=${CMAKE_CURRENT_SOURCE_DIR}/models
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

add_executable(camera_cv_service
    src/main.cpp
    src/camera_manager.cpp
    src/inference_engine.cpp
    src/grpc_server.cpp
    ${GENERATED_SRC}
)

target_include_directories(camera_cv_service PRIVATE
    include
    ${GENERATED_DIR}
    ${ONNXRUNTIME_INCLUDE_DIR}
    ${Protobuf_INCLUDE_DIRS}
)

target_link_libraries(camera_cv_service PRIVATE
    ${OpenCV_LIBS}
    gRPC::grpc++
    gRPC::grpc
    protobuf::libprotobuf
    Threads::Threads
    ${ONNXRUNTIME_LIBRARY}
    Version
)

set_target_properties(camera_cv_service PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "ONNX Runtime include dir: ${ONNXRUNTIME_INCLUDE_DIR}")
message(STATUS "ONNX Runtime library: ${ONNXRUNTIME_LIBRARY}")
message(STATUS "Generated proto files dir: ${GENERATED_DIR}")
message(STATUS "Using protoc executable: ${PROTOC_EXE}")
message(STATUS "Using grpc_cpp_plugin: ${GRPC_CPP_PLUGIN}")
