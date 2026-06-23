#include "onnx_runtime_backend.h"
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <filesystem>
#include <algorithm>
#include <atomic>
#include <thread>
#include <stdexcept>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <winver.h>
#ifdef max
#undef max
#endif
#ifdef min
#undef min
#endif
#endif

#if defined(__has_include)
#if __has_include(<onnxruntime/core/providers/cuda/cuda_provider_factory.h>)
#include <onnxruntime/core/providers/cuda/cuda_provider_factory.h>
#define HAS_ORT_CUDA_PROVIDER 1
#endif
#endif

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

bool hasProvider(const std::vector<std::string>& providers, const char* provider_name) {
    return std::find(providers.begin(), providers.end(), provider_name) != providers.end();
}

bool readEnvBool(const char* name, bool fallback) {
    if (const char* raw = std::getenv(name)) {
        const std::string value(raw);
        return value == "1" || value == "true" || value == "TRUE" || value == "yes" || value == "YES";
    }
    return fallback;
}

void logProviders(const std::vector<std::string>& providers) {
    std::cout << "[ONNX] Runtime version: " << Ort::GetVersionString() << std::endl;
    if (providers.empty()) {
        std::cout << "[ONNX] Available execution providers: none reported" << std::endl;
        return;
    }
    std::cout << "[ONNX] Available execution providers:";
    for (const auto& provider : providers) {
        std::cout << " " << provider;
    }
    std::cout << std::endl;
}
}

ONNXRuntimeBackend::ONNXRuntimeBackend()
    : env_(ORT_LOGGING_LEVEL_WARNING, "ONNXBackend"),
      session_options_(),
      memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)),
      selected_execution_provider_("CPUExecutionProvider"),
      verbose_logging_(false) {
}

ONNXRuntimeBackend::~ONNXRuntimeBackend() {
    session_.reset();
}

void ONNXRuntimeBackend::initialize(const std::string& model_path) {
    verbose_logging_ = readEnvBool("CAMERA_VERBOSE_LOGS", false);

    const auto cpu_threads = std::max(1u, std::thread::hardware_concurrency());
    session_options_.SetIntraOpNumThreads(static_cast<int>(cpu_threads));
    session_options_.SetExecutionMode(ExecutionMode::ORT_PARALLEL);
    session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    bool gpu_enabled = false;
    const auto providers = Ort::GetAvailableProviders();
    logProviders(providers);

#if defined(HAS_ORT_CUDA_PROVIDER)
    if (!gpu_enabled && hasProvider(providers, "CUDAExecutionProvider")) {
        try {
            std::cout << "[ONNX] Attempting to enable CUDAExecutionProvider" << std::endl;
            OrtCUDAProviderOptions cuda_options{};
            cuda_options.device_id = 0;
            session_options_.AppendExecutionProvider_CUDA(cuda_options);
            selected_execution_provider_ = "CUDAExecutionProvider";
            std::cout << "[ONNX] CUDAExecutionProvider enabled" << std::endl;
            gpu_enabled = true;
        } catch (const Ort::Exception& e) {
            std::cerr << "[ONNX] Failed to enable CUDAExecutionProvider: " << e.what() << std::endl;
        }
    }
#endif

    if (!gpu_enabled) {
        selected_execution_provider_ = "CPUExecutionProvider";
        std::cout << "[ONNX] CUDAExecutionProvider not available, using CPUExecutionProvider fallback" << std::endl;
    }

    std::filesystem::path path_fs(model_path);
#ifdef _WIN32
    std::wstring ort_model_path = path_fs.wstring();
#else
    std::string ort_model_path = path_fs.string();
#endif

    try {
        std::cout << "[ONNX] Creating session with provider preference: "
                  << selected_execution_provider_ << std::endl;
        session_ = std::make_unique<Ort::Session>(env_, ort_model_path.c_str(), session_options_);
        std::cout << "[ONNX] Session created successfully" << std::endl;

        Ort::AllocatorWithDefaultOptions allocator;
        size_t num_inputs = session_->GetInputCount();
        input_names_str_.resize(num_inputs);
        input_names_.resize(num_inputs);

        for (size_t i = 0; i < num_inputs; ++i) {
            auto name = session_->GetInputNameAllocated(i, allocator);
            input_names_str_[i] = (!name || std::strlen(name.get()) == 0) ? "images" : std::string(name.get());
            input_names_[i] = input_names_str_[i].c_str();
            std::cout << "[ONNX] Input " << i << ": " << input_names_[i] << std::endl;
        }

        size_t num_outputs = session_->GetOutputCount();
        output_names_str_.resize(num_outputs);
        output_names_.resize(num_outputs);

        for (size_t i = 0; i < num_outputs; ++i) {
            auto name = session_->GetOutputNameAllocated(i, allocator);
            output_names_str_[i] = (!name || std::strlen(name.get()) == 0) ? "output0" : std::string(name.get());
            output_names_[i] = output_names_str_[i].c_str();
            std::cout << "[ONNX] Output " << i << ": " << output_names_[i] << std::endl;
        }

    } catch (const Ort::Exception& e) {
        std::cerr << "[ONNX] Session creation failed: " << e.what() << std::endl;
        throw;
    }
}

bool ONNXRuntimeBackend::isReady() const {
    return session_ != nullptr;
}

std::vector<Detection> ONNXRuntimeBackend::runInference(
    const float* input_data,
    const std::vector<int64_t>& input_shape,
    int frame_width,
    int frame_height,
    float confidence_threshold,
    int input_width,
    int input_height) {

    if (!session_) return {};

    try {
        if (input_shape.size() < 4) {
            throw std::runtime_error("[ONNX] Input shape must have at least 4 dimensions");
        }
        if (input_names_.size() != 1) {
            throw std::runtime_error("[ONNX] Current backend supports exactly one model input");
        }

        size_t input_tensor_size = 1;
        for (int64_t dim : input_shape) {
            if (dim <= 0) {
                throw std::runtime_error("[ONNX] Input shape contains a non-positive dimension");
            }
            input_tensor_size *= static_cast<size_t>(dim);
        }

        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info_,
            const_cast<float*>(input_data),
            input_tensor_size,
            input_shape.data(),
            input_shape.size());

        auto output_tensors = session_->Run(
            Ort::RunOptions{nullptr},
            input_names_.data(),
            &input_tensor,
            1,
            output_names_.data(),
            output_names_.size());

        if (!output_tensors.empty() && output_tensors[0].IsTensor()) {
            auto& tensor = output_tensors[0];
            float* output_data = tensor.GetTensorMutableData<float>();
            const auto tensor_info = tensor.GetTensorTypeAndShapeInfo();
            const auto output_shape = tensor_info.GetShape();

            return parseYOLO(output_data, output_shape, frame_width, frame_height, 
                           confidence_threshold, input_width, input_height);
        }
    } catch (const Ort::Exception& e) {
        std::cerr << "[ONNX] Inference error: " << e.what() << std::endl;
    }

    return {};
}

std::string ONNXRuntimeBackend::getBackendName() const {
    return "ONNXRuntime";
}

std::vector<Detection> ONNXRuntimeBackend::parseYOLO(
    const float* data,
    const std::vector<int64_t>& output_shape,
    int frame_width,
    int frame_height,
    float confidence_threshold,
    int input_width,
    int input_height) {

    std::vector<Detection> detections;

    if (output_shape.size() != 3 || output_shape[1] <= 0 || output_shape[2] <= 0) {
        return detections;
    }

    const int64_t dim1 = output_shape[1];
    const int64_t dim2 = output_shape[2];
    const bool feature_major = dim1 <= 256 && dim1 < dim2;
    const int64_t num_features = feature_major ? dim1 : dim2;
    const int64_t num_predictions = feature_major ? dim2 : dim1;

    if (num_features < 6 || num_predictions <= 0) {
        return detections;
    }

    auto at = [&](int64_t pred, int64_t feature) -> float {
        return feature_major
            ? data[(feature * num_predictions) + pred]
            : data[(pred * num_features) + feature];
    };

    const float scale_x = static_cast<float>(frame_width) / static_cast<float>(input_width);
    const float scale_y = static_cast<float>(frame_height) / static_cast<float>(input_height);

    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    std::vector<int> class_ids;
    boxes.reserve(static_cast<size_t>(num_predictions / 8));

    for (int64_t pred = 0; pred < num_predictions; ++pred) {
        const float x = at(pred, 0);
        const float y = at(pred, 1);
        const float w_or_x2 = at(pred, 2);
        const float h_or_y2 = at(pred, 3);

        float max_conf = 0.0f;
        int class_id = -1;
        const float maybe_class_id = num_features == 6 ? at(pred, 5) : -1.0f;
        const bool nms_output =
            num_features == 6 &&
            maybe_class_id >= 0.0f &&
            maybe_class_id < static_cast<float>(kCocoLabels.size()) &&
            std::abs(maybe_class_id - std::round(maybe_class_id)) < 0.001f;

        if (nms_output) {
            max_conf = at(pred, 4);
            class_id = static_cast<int>(std::round(maybe_class_id));
        } else {
            const bool has_objectness = num_features == 85 || num_features == 6 + static_cast<int64_t>(kCocoLabels.size());
            const float objectness = has_objectness ? at(pred, 4) : 1.0f;
            const int class_offset = has_objectness ? 5 : 4;
            const int num_classes = static_cast<int>(num_features - class_offset);

            for (int cls = 0; cls < num_classes; ++cls) {
                const float class_conf = at(pred, class_offset + cls);
                const float conf = objectness * class_conf;
                if (conf > max_conf) {
                    max_conf = conf;
                    class_id = cls;
                }
            }
        }

        if (!std::isfinite(max_conf) || max_conf <= confidence_threshold) {
            continue;
        }

        float left_f = 0.0f;
        float top_f = 0.0f;
        float right_f = 0.0f;
        float bottom_f = 0.0f;

        const bool normalized_coords =
            std::max({std::abs(x), std::abs(y), std::abs(w_or_x2), std::abs(h_or_y2)}) <= 2.0f;
        const float coord_scale_x = normalized_coords ? static_cast<float>(frame_width) : scale_x;
        const float coord_scale_y = normalized_coords ? static_cast<float>(frame_height) : scale_y;

        if (nms_output && w_or_x2 > x && h_or_y2 > y) {
            left_f = x * coord_scale_x;
            top_f = y * coord_scale_y;
            right_f = w_or_x2 * coord_scale_x;
            bottom_f = h_or_y2 * coord_scale_y;
        } else {
            left_f = (x - (w_or_x2 * 0.5f)) * coord_scale_x;
            top_f = (y - (h_or_y2 * 0.5f)) * coord_scale_y;
            right_f = (x + (w_or_x2 * 0.5f)) * coord_scale_x;
            bottom_f = (y + (h_or_y2 * 0.5f)) * coord_scale_y;
        }

        const int left = std::clamp(static_cast<int>(std::round(left_f)), 0, frame_width);
        const int top = std::clamp(static_cast<int>(std::round(top_f)), 0, frame_height);
        const int right = std::clamp(static_cast<int>(std::round(right_f)), 0, frame_width);
        const int bottom = std::clamp(static_cast<int>(std::round(bottom_f)), 0, frame_height);
        const int width = std::max(0, right - left);
        const int height = std::max(0, bottom - top);

        if (width == 0 || height == 0) {
            continue;
        }

        boxes.emplace_back(left, top, width, height);
        scores.push_back(max_conf);
        class_ids.push_back(class_id);
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, scores, confidence_threshold, 0.45f, indices);

    for (int idx : indices) {
        detections.push_back({classIdToLabel(class_ids[idx]), scores[idx], BBox(boxes[idx])});
    }

    return detections;
}
