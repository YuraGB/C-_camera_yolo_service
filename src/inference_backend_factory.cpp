#include "inference_backend.h"
#include "onnx_runtime_backend.h"
#ifdef USE_TENSORRT
#include "tensorrt_runtime_backend.h"
#endif
#include <cstdlib>
#include <iostream>
#include <string>

std::unique_ptr<InferenceBackend> InferenceBackendFactory::createBackend() {
    const char* runtime_env = std::getenv("RUNTIME_BACKEND");
    std::string runtime_backend = runtime_env ? runtime_env : "onnx";

    for (auto& c : runtime_backend) {
        c = std::tolower(static_cast<unsigned char>(c));
    }

    std::cout << "[Factory] Requested runtime backend: " << runtime_backend << std::endl;

#ifdef USE_TENSORRT
    if (runtime_backend == "tensorrt") {
        std::cout << "[Factory] Creating TensorRT backend" << std::endl;
        try {
            return std::make_unique<TensorRTBackend>();
        } catch (const std::exception& e) {
            std::cerr << "[Factory] TensorRT initialization failed: " << e.what() << std::endl;
            std::cout << "[Factory] Falling back to ONNX Runtime" << std::endl;
        }
    }
#endif

    std::cout << "[Factory] Creating ONNX Runtime backend" << std::endl;
    return std::make_unique<ONNXRuntimeBackend>();
}
