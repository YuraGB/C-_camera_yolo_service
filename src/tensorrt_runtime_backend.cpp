#include "tensorrt_runtime_backend.h"

#ifdef USE_TENSORRT
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <stdexcept>
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

size_t dataTypeSize(nvinfer1::DataType type) {
    switch (type) {
        case nvinfer1::DataType::kFLOAT:
            return 4;
        case nvinfer1::DataType::kHALF:
            return 2;
        case nvinfer1::DataType::kINT8:
            return 1;
        case nvinfer1::DataType::kINT32:
            return 4;
        case nvinfer1::DataType::kBOOL:
            return 1;
        default:
            throw std::runtime_error("[TensorRT] Unsupported binding data type");
    }
}

size_t dimsVolume(const nvinfer1::Dims& dims) {
    if (dims.nbDims <= 0) {
        throw std::runtime_error("[TensorRT] Binding dimensions are empty");
    }

    size_t volume = 1;
    for (int i = 0; i < dims.nbDims; ++i) {
        if (dims.d[i] <= 0) {
            throw std::runtime_error("[TensorRT] Binding dimensions are not fully resolved");
        }
        volume *= static_cast<size_t>(dims.d[i]);
    }
    return volume;
}

bool hasDynamicDim(const nvinfer1::Dims& dims) {
    for (int i = 0; i < dims.nbDims; ++i) {
        if (dims.d[i] < 0) {
            return true;
        }
    }
    return false;
}

nvinfer1::Dims shapeToDims(const std::vector<int64_t>& shape, int target_rank) {
    if (shape.empty() || shape.size() > 8 || target_rank <= 0 || target_rank > 8) {
        throw std::runtime_error("[TensorRT] Unsupported input shape rank");
    }

    size_t source_offset = 0;
    if (static_cast<size_t>(target_rank) + 1 == shape.size()) {
        source_offset = 1;
    } else if (static_cast<size_t>(target_rank) != shape.size()) {
        throw std::runtime_error("[TensorRT] Input shape rank does not match engine binding rank");
    }

    nvinfer1::Dims dims{};
    dims.nbDims = target_rank;
    for (int i = 0; i < dims.nbDims; ++i) {
        const int64_t dim = shape[source_offset + static_cast<size_t>(i)];
        if (dim <= 0) {
            throw std::runtime_error("[TensorRT] Input shape contains a non-positive dimension");
        }
        dims.d[i] = static_cast<int>(dim);
    }
    return dims;
}

void ensureDeviceBuffer(std::vector<void*>& bindings, std::vector<size_t>& sizes, int index, size_t bytes) {
    if (bytes == 0) {
        throw std::runtime_error("[TensorRT] Refusing to allocate an empty binding buffer");
    }

    if (index < 0 || static_cast<size_t>(index) >= bindings.size()) {
        throw std::runtime_error("[TensorRT] Binding index is out of range");
    }

    if (sizes[static_cast<size_t>(index)] >= bytes && bindings[static_cast<size_t>(index)] != nullptr) {
        return;
    }

    if (bindings[static_cast<size_t>(index)] != nullptr) {
        cudaFree(bindings[static_cast<size_t>(index)]);
        bindings[static_cast<size_t>(index)] = nullptr;
        sizes[static_cast<size_t>(index)] = 0;
    }

    cudaError_t cuda_status = cudaMalloc(&bindings[static_cast<size_t>(index)], bytes);
    if (cuda_status != cudaSuccess) {
        throw std::runtime_error("[TensorRT] CUDA memory allocation failed: " +
                                 std::string(cudaGetErrorString(cuda_status)));
    }
    sizes[static_cast<size_t>(index)] = bytes;
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
    : input_binding_index_(-1),
      output_binding_index_(-1),
      verbose_logging_(false) {
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
    binding_sizes_.resize(num_bindings, 0);

    for (int i = 0; i < num_bindings; ++i) {
        const nvinfer1::DataType type = engine_->getBindingDataType(i);
        if (type != nvinfer1::DataType::kFLOAT) {
            throw std::runtime_error("[TensorRT] Only FP32 TensorRT bindings are supported");
        }

        if (engine_->bindingIsInput(i)) {
            if (input_binding_index_ != -1) {
                throw std::runtime_error("[TensorRT] Multiple input bindings are not supported");
            }
            input_binding_index_ = i;
        } else {
            if (output_binding_index_ != -1) {
                throw std::runtime_error("[TensorRT] Multiple output bindings are not supported");
            }
            output_binding_index_ = i;
        }

        nvinfer1::Dims dims = engine_->getBindingDimensions(i);
        if (!hasDynamicDim(dims)) {
            ensureDeviceBuffer(device_bindings_, binding_sizes_, i, dimsVolume(dims) * dataTypeSize(type));
        }
    }

    if (input_binding_index_ == -1 || output_binding_index_ == -1) {
        throw std::runtime_error("[TensorRT] Engine must expose exactly one input and one output binding");
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
        if (input_shape.size() < 4) {
            throw std::runtime_error("[TensorRT] Input shape must have at least 4 dimensions");
        }

        nvinfer1::Dims model_input_dims = engine_->getBindingDimensions(input_binding_index_);
        if (hasDynamicDim(model_input_dims)) {
            if (!context_->setBindingDimensions(input_binding_index_, shapeToDims(input_shape, model_input_dims.nbDims))) {
                throw std::runtime_error("[TensorRT] Failed to set dynamic input dimensions");
            }
            model_input_dims = context_->getBindingDimensions(input_binding_index_);
        } else if (model_input_dims.nbDims == static_cast<int>(input_shape.size())) {
            for (int i = 0; i < model_input_dims.nbDims; ++i) {
                if (model_input_dims.d[i] != static_cast<int>(input_shape[static_cast<size_t>(i)])) {
                    throw std::runtime_error("[TensorRT] Input shape does not match engine binding dimensions");
                }
            }
        } else if (model_input_dims.nbDims + 1 == static_cast<int>(input_shape.size())) {
            for (int i = 0; i < model_input_dims.nbDims; ++i) {
                if (model_input_dims.d[i] != static_cast<int>(input_shape[static_cast<size_t>(i + 1)])) {
                    throw std::runtime_error("[TensorRT] Input shape does not match engine binding dimensions");
                }
            }
        } else {
            throw std::runtime_error("[TensorRT] Input shape rank does not match engine binding rank");
        }

        if (!context_->allInputDimensionsSpecified()) {
            throw std::runtime_error("[TensorRT] Dynamic input dimensions are not fully specified");
        }

        const size_t input_size = dimsVolume(model_input_dims) * dataTypeSize(engine_->getBindingDataType(input_binding_index_));
        ensureDeviceBuffer(device_bindings_, binding_sizes_, input_binding_index_, input_size);

        nvinfer1::Dims output_dims = context_->getBindingDimensions(output_binding_index_);
        const size_t output_size = dimsVolume(output_dims) * dataTypeSize(engine_->getBindingDataType(output_binding_index_));
        ensureDeviceBuffer(device_bindings_, binding_sizes_, output_binding_index_, output_size);

        cudaError_t cuda_status = cudaMemcpy(device_bindings_[static_cast<size_t>(input_binding_index_)], input_data, input_size, cudaMemcpyHostToDevice);
        if (cuda_status != cudaSuccess) {
            std::cerr << "[TensorRT] CUDA copy to device failed: " << cudaGetErrorString(cuda_status) << std::endl;
            return {};
        }

        if (!context_->executeV2(device_bindings_.data())) {
            std::cerr << "[TensorRT] Inference failed" << std::endl;
            return {};
        }

        std::vector<float> output_data;
        const size_t output_volume = output_size / sizeof(float);
        output_data.resize(output_volume);

        cuda_status = cudaMemcpy(output_data.data(), device_bindings_[static_cast<size_t>(output_binding_index_)],
                                output_size, cudaMemcpyDeviceToHost);
        if (cuda_status != cudaSuccess) {
            std::cerr << "[TensorRT] CUDA copy from device failed: " << cudaGetErrorString(cuda_status) << std::endl;
            return {};
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
