#include "tensorrt_runtime_backend.h"

#ifdef USE_TENSORRT
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <cuda_runtime.h>

namespace {
const std::vector<std::string> kCocoLabels = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed",
    "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
};

std::string classIdToLabel(int class_id) {
    if (class_id >= 0 && class_id < static_cast<int>(kCocoLabels.size())) {
        return kCocoLabels[class_id];
    }
    return std::to_string(class_id);
}

bool readEnvBool(const char* name, bool fallback) {
    if (const char* raw = std::getenv(name)) {
        const std::string value(raw);
        return value == "1" || value == "true" || value == "TRUE" || value == "yes" || value == "YES";
    }
    return fallback;
}
}

void TensorRTBackend::Logger::log(Severity severity, const char* msg) noexcept {
    switch (severity) {
        case Severity::kINTERNAL_ERROR:
        case Severity::kERROR:
            std::cerr << "[TensorRT] ERROR: " << msg << std::endl;
            break;
        case Severity::kWARNING:
            std::cout << "[TensorRT] WARNING: " << msg << std::endl;
            break;
        case Severity::kINFO:
        case Severity::kVERBOSE:
            std::cout << "[TensorRT] INFO: " << msg << std::endl;
            break;
    }
}

TensorRTBackend::TensorRTBackend()
    : verbose_logging_(false) {
}

TensorRTBackend::~TensorRTBackend() {
    for (auto& binding : device_bindings_) {
        if (binding) {
            cudaFree(binding);
        }
    }
    device_bindings_.clear();
    context_.reset();
    engine_.reset();
    runtime_.reset();
}

void TensorRTBackend::initialize(const std::string& model_path) {
    verbose_logging_ = readEnvBool("CAMERA_VERBOSE_LOGS", false);

    std::ifstream engine_file(model_path, std::ios::binary);
    if (!engine_file.good()) {
        throw std::runtime_error("[TensorRT] Failed to open model file: " + model_path);
    }

    engine_file.seekg(0, std::end);
    size_t size = engine_file.tellg();
    engine_file.seekg(0, std::beg);

    std::vector<char> engine_data(size);
    engine_file.read(engine_data.data(), size);
    engine_file.close();

    runtime_ = std::shared_ptr<nvinfer1::IRuntime>(
        nvinfer1::createInferRuntime(logger_),
        [](nvinfer1::IRuntime* ptr) { ptr->destroy(); }
    );

    if (!runtime_) {
        throw std::runtime_error("[TensorRT] Failed to create runtime");
    }

    engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(
        runtime_->deserializeCudaEngine(engine_data.data(), size),
        [](nvinfer1::ICudaEngine* ptr) { ptr->destroy(); }
    );

    if (!engine_) {
        throw std::runtime_error("[TensorRT] Failed to deserialize CUDA engine");
    }

    context_ = std::shared_ptr<nvinfer1::IExecutionContext>(
        engine_->createExecutionContext(),
        [](nvinfer1::IExecutionContext* ptr) { ptr->destroy(); }
    );

    if (!context_) {
        throw std::runtime_error("[TensorRT] Failed to create execution context");
    }

    int num_bindings = engine_->getNbBindings();
    device_bindings_.resize(num_bindings);

    for (int i = 0; i < num_bindings; ++i) {
        nvinfer1::Dims dims = engine_->getBindingDimensions(i);
        size_t volume = 1;
        for (int j = 0; j < dims.nbDims; ++j) {
            volume *= dims.d[j];
        }

        size_t binding_size = volume * sizeof(float);
        cudaError_t cuda_status = cudaMalloc(&device_bindings_[i], binding_size);
        if (cuda_status != cudaSuccess) {
            throw std::runtime_error("[TensorRT] CUDA memory allocation failed: " + 
                                    std::string(cudaGetErrorString(cuda_status)));
        }
    }

    std::cout << "[TensorRT] Engine initialized with " << num_bindings << " bindings" << std::endl;
}

bool TensorRTBackend::isReady() const {
    return context_ != nullptr && engine_ != nullptr;
}

std::vector<Detection> TensorRTBackend::runInference(
    const float* input_data,
    const std::vector<int64_t>& input_shape,
    int frame_width,
    int frame_height,
    float confidence_threshold,
    int input_width,
    int input_height) {

    if (!context_ || !engine_) {
        return {};
    }

    try {
        size_t input_size = input_shape[1] * input_shape[2] * input_shape[3] * sizeof(float);

        cudaError_t cuda_status = cudaMemcpy(device_bindings_[0], input_data, input_size, cudaMemcpyHostToDevice);
        if (cuda_status != cudaSuccess) {
            std::cerr << "[TensorRT] CUDA copy to device failed: " << cudaGetErrorString(cuda_status) << std::endl;
            return {};
        }

        if (!context_->executeV2(device_bindings_.data())) {
            std::cerr << "[TensorRT] Inference failed" << std::endl;
            return {};
        }

        std::vector<float> output_data;
        nvinfer1::Dims output_dims = engine_->getBindingDimensions(1);
        size_t output_volume = 1;
        for (int i = 0; i < output_dims.nbDims; ++i) {
            output_volume *= output_dims.d[i];
        }
        output_data.resize(output_volume);

        cuda_status = cudaMemcpy(output_data.data(), device_bindings_[1], 
                                output_volume * sizeof(float), cudaMemcpyDeviceToHost);
        if (cuda_status != cudaSuccess) {
            std::cerr << "[TensorRT] CUDA copy from device failed: " << cudaGetErrorString(cuda_status) << std::endl;
            return {};
        }

        std::vector<int64_t> output_shape;
        output_shape.resize(output_dims.nbDims);
        for (int i = 0; i < output_dims.nbDims; ++i) {
            output_shape[i] = output_dims.d[i];
        }

        return parseYOLO(output_data.data(), output_data.size(), frame_width, frame_height,
                        confidence_threshold, input_width, input_height);

    } catch (const std::exception& e) {
        std::cerr << "[TensorRT] Inference exception: " << e.what() << std::endl;
    }

    return {};
}

std::string TensorRTBackend::getBackendName() const {
    return "TensorRT";
}

std::vector<Detection> TensorRTBackend::parseYOLO(
    const float* data,
    size_t output_size,
    int frame_width,
    int frame_height,
    float confidence_threshold,
    int input_width,
    int input_height) {

    std::vector<Detection> detections;

    if (output_size < 6) {
        return detections;
    }

    const float scale_x = static_cast<float>(frame_width) / static_cast<float>(input_width);
    const float scale_y = static_cast<float>(frame_height) / static_cast<float>(input_height);

    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    std::vector<int> class_ids;

    size_t stride = 6;
    for (size_t i = 0; i < output_size; i += stride) {
        if (i + stride > output_size) break;

        float x = data[i + 0];
        float y = data[i + 1];
        float w = data[i + 2];
        float h = data[i + 3];
        float conf = data[i + 4];
        int class_id = static_cast<int>(data[i + 5]);

        if (conf <= confidence_threshold) {
            continue;
        }

        float left_f = (x - (w * 0.5f)) * scale_x;
        float top_f = (y - (h * 0.5f)) * scale_y;
        float right_f = (x + (w * 0.5f)) * scale_x;
        float bottom_f = (y + (h * 0.5f)) * scale_y;

        const int left = std::clamp(static_cast<int>(std::round(left_f)), 0, frame_width);
        const int top = std::clamp(static_cast<int>(std::round(top_f)), 0, frame_height);
        const int right = std::clamp(static_cast<int>(std::round(right_f)), 0, frame_width);
        const int bottom = std::clamp(static_cast<int>(std::round(bottom_f)), 0, frame_height);
        const int box_width = std::max(0, right - left);
        const int box_height = std::max(0, bottom - top);

        if (box_width == 0 || box_height == 0) {
            continue;
        }

        boxes.emplace_back(left, top, box_width, box_height);
        scores.push_back(conf);
        class_ids.push_back(class_id);
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, scores, confidence_threshold, 0.45f, indices);

    for (int idx : indices) {
        detections.push_back({classIdToLabel(class_ids[idx]), scores[idx], BBox(boxes[idx])});
    }

    return detections;
}

#endif
