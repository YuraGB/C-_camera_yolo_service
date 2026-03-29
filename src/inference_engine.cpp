#include "inference_engine.h"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <opencv2/opencv.hpp>
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
#if __has_include(<onnxruntime/core/providers/dml/dml_provider_factory.h>)
#include <onnxruntime/core/providers/dml/dml_provider_factory.h>
#define HAS_ORT_DML_PROVIDER 1
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

#ifdef _WIN32
std::string narrow(const std::wstring& value) {
    if (value.empty()) {
        return {};
    }

    const int size = WideCharToMultiByte(CP_UTF8, 0, value.c_str(), -1, nullptr, 0, nullptr, nullptr);
    if (size <= 1) {
        return {};
    }

    std::string result(static_cast<size_t>(size - 1), '\0');
    WideCharToMultiByte(CP_UTF8, 0, value.c_str(), -1, result.data(), size, nullptr, nullptr);
    return result;
}

std::string getFileVersionString(const std::wstring& dll_path) {
    DWORD handle = 0;
    const DWORD version_size = GetFileVersionInfoSizeW(dll_path.c_str(), &handle);
    if (version_size == 0) {
        return {};
    }

    std::vector<char> version_data(version_size);
    if (!GetFileVersionInfoW(dll_path.c_str(), 0, version_size, version_data.data())) {
        return {};
    }

    VS_FIXEDFILEINFO* file_info = nullptr;
    UINT file_info_len = 0;
    if (!VerQueryValueW(version_data.data(), L"\\", reinterpret_cast<LPVOID*>(&file_info), &file_info_len) ||
        file_info == nullptr) {
        return {};
    }

    return std::to_string(HIWORD(file_info->dwFileVersionMS)) + "." +
           std::to_string(LOWORD(file_info->dwFileVersionMS)) + "." +
           std::to_string(HIWORD(file_info->dwFileVersionLS)) + "." +
           std::to_string(LOWORD(file_info->dwFileVersionLS));
}

void logLoadedModuleVersion(const wchar_t* module_name, const char* label) {
    HMODULE module = GetModuleHandleW(module_name);
    if (!module) {
        std::cout << "[ONNX] " << label << " module is not currently loaded" << std::endl;
        return;
    }

    std::wstring module_path(MAX_PATH, L'\0');
    const DWORD length = GetModuleFileNameW(module, module_path.data(), static_cast<DWORD>(module_path.size()));
    if (length == 0) {
        std::cout << "[ONNX] " << label << " module is loaded, but its path could not be resolved" << std::endl;
        return;
    }

    module_path.resize(length);
    const auto version = getFileVersionString(module_path);
    std::cout << "[ONNX] " << label << " module path: " << narrow(module_path) << std::endl;
    if (version.empty()) {
        std::cout << "[ONNX] " << label << " module version could not be determined" << std::endl;
    } else {
        std::cout << "[ONNX] " << label << " module version: " << version << std::endl;
    }
}

using AppendExecutionProviderDeviceFn = OrtStatus*(ORT_API_CALL*)(OrtSessionOptions*, int);

bool appendProviderBySymbol(HMODULE ort_module,
                            const char* symbol_name,
                            const char* provider_label,
                            OrtSessionOptions* session_options,
                            int device_id,
                            std::string& error_message) {
    if (!ort_module) {
        error_message = "onnxruntime.dll is not loaded";
        return false;
    }

    auto append_fn = reinterpret_cast<AppendExecutionProviderDeviceFn>(
        GetProcAddress(ort_module, symbol_name)
    );
    if (!append_fn) {
        error_message = std::string(symbol_name) + " export is not available";
        return false;
    }

    OrtStatus* status = append_fn(session_options, device_id);
    if (!status) {
        return true;
    }

    error_message = Ort::GetApi().GetErrorMessage(status);
    Ort::GetApi().ReleaseStatus(status);
    return false;
}
#endif

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

#ifdef _WIN32
    if (hasProvider(providers, "CUDAExecutionProvider")) {
        logLoadedModuleVersion(L"onnxruntime_providers_cuda.dll", "CUDAExecutionProvider");
    }
    if (hasProvider(providers, "TensorrtExecutionProvider")) {
        logLoadedModuleVersion(L"onnxruntime_providers_tensorrt.dll", "TensorrtExecutionProvider");
    }
#endif
}
}

InferenceEngine::InferenceEngine(const std::string& model_path)
    : model_path_(model_path),
      env_(ORT_LOGGING_LEVEL_WARNING, "InferenceEngine"),
      session_options_(),
      running_(false)
{
    const auto cpu_threads = std::max(1u, std::thread::hardware_concurrency());
    auto configureBaseOptions = [cpu_threads](Ort::SessionOptions& options) {
        options.SetIntraOpNumThreads(static_cast<int>(cpu_threads));
        options.SetExecutionMode(ExecutionMode::ORT_PARALLEL);
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    };

    configureBaseOptions(session_options_);
    configureExecutionProvider();

    std::filesystem::path path_fs(model_path);
    std::wstring wmodel_path = path_fs.wstring();

    try {
        std::cout << "[ONNX] Creating session with provider preference: "
                  << selected_execution_provider_ << std::endl;

        session_ = std::make_unique<Ort::Session>(env_, wmodel_path.c_str(), session_options_);
        std::cout << "[ONNX] Session created successfully with provider preference: "
                  << selected_execution_provider_ << std::endl;

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
        std::cerr << "[ONNX] Session creation failed with provider preference "
                  << selected_execution_provider_ << ": " << e.what() << std::endl;

        if (selected_execution_provider_ != "CPUExecutionProvider") {
            try {
                std::cout << "[ONNX] Retrying session creation with CPUExecutionProvider fallback" << std::endl;
                Ort::SessionOptions cpu_session_options;
                configureBaseOptions(cpu_session_options);
                selected_execution_provider_ = "CPUExecutionProvider";
                session_ = std::make_unique<Ort::Session>(env_, wmodel_path.c_str(), cpu_session_options);
                std::cout << "[ONNX] Session created successfully with CPUExecutionProvider fallback" << std::endl;

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
            } catch (const Ort::Exception& fallback_error) {
                std::cerr << "[ONNX] CPU fallback session creation failed: "
                          << fallback_error.what() << std::endl;
            } catch (const std::exception& fallback_error) {
                std::cerr << "[ONNX] CPU fallback init error: "
                          << fallback_error.what() << std::endl;
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "[ONNX] Init error with provider preference "
                  << selected_execution_provider_ << ": " << e.what() << std::endl;
    }
}

InferenceEngine::~InferenceEngine() {
    stop();
}

void InferenceEngine::configureExecutionProvider() {
    bool gpu_enabled = false;
    const auto providers = Ort::GetAvailableProviders();
    logProviders(providers);

#if !defined(HAS_ORT_CUDA_PROVIDER)
    std::cout << "[ONNX] CUDA provider factory header was not available at compile time; "
                 "using runtime symbol lookup fallback if possible." << std::endl;
#endif
#if !defined(HAS_ORT_DML_PROVIDER)
    std::cout << "[ONNX] DML provider factory header was not available at compile time; "
                 "using runtime symbol lookup fallback if possible." << std::endl;
#endif

#ifdef _WIN32
    HMODULE ort_module = GetModuleHandleW(L"onnxruntime.dll");
    if (!ort_module) {
        std::cout << "[ONNX] onnxruntime.dll module is not currently loaded during provider configuration" << std::endl;
    }
#endif

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
            std::cerr << "[ONNX] Failed to enable CUDAExecutionProvider: "
                      << e.what() << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "[ONNX] Failed to enable CUDAExecutionProvider: "
                      << e.what() << std::endl;
        }
    }
#endif

#ifdef _WIN32
    if (!gpu_enabled && hasProvider(providers, "CUDAExecutionProvider")) {
        std::string error_message;
        std::cout << "[ONNX] Attempting to enable CUDAExecutionProvider via runtime symbol lookup" << std::endl;
        if (appendProviderBySymbol(
                ort_module,
                "OrtSessionOptionsAppendExecutionProvider_CUDA",
                "CUDAExecutionProvider",
                session_options_,
                0,
                error_message)) {
            selected_execution_provider_ = "CUDAExecutionProvider";
            std::cout << "[ONNX] CUDAExecutionProvider enabled via runtime symbol lookup" << std::endl;
            gpu_enabled = true;
        } else {
            std::cerr << "[ONNX] Failed to enable CUDAExecutionProvider via runtime symbol lookup: "
                      << error_message << std::endl;
        }
    }
#endif

#if defined(HAS_ORT_DML_PROVIDER)
    if (!gpu_enabled && hasProvider(providers, "DmlExecutionProvider")) {
        try {
            std::cout << "[ONNX] Attempting to enable DmlExecutionProvider" << std::endl;
            session_options_.AppendExecutionProvider_DML(0);
            selected_execution_provider_ = "DmlExecutionProvider";
            std::cout << "[ONNX] DmlExecutionProvider enabled" << std::endl;
            gpu_enabled = true;
        } catch (const Ort::Exception& e) {
            std::cerr << "[ONNX] Failed to enable DmlExecutionProvider: "
                      << e.what() << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "[ONNX] Failed to enable DmlExecutionProvider: "
                      << e.what() << std::endl;
        }
    }
#endif

#ifdef _WIN32
    if (!gpu_enabled && hasProvider(providers, "DmlExecutionProvider")) {
        std::string error_message;
        std::cout << "[ONNX] Attempting to enable DmlExecutionProvider via runtime symbol lookup" << std::endl;
        if (appendProviderBySymbol(
                ort_module,
                "OrtSessionOptionsAppendExecutionProvider_DML",
                "DmlExecutionProvider",
                session_options_,
                0,
                error_message)) {
            selected_execution_provider_ = "DmlExecutionProvider";
            std::cout << "[ONNX] DmlExecutionProvider enabled via runtime symbol lookup" << std::endl;
            gpu_enabled = true;
        } else {
            std::cerr << "[ONNX] Failed to enable DmlExecutionProvider via runtime symbol lookup: "
                      << error_message << std::endl;
        }
    }
#endif

    if (!gpu_enabled) {
        selected_execution_provider_ = "CPUExecutionProvider";
        std::cout << "[ONNX] GPU execution provider not available, using CPUExecutionProvider" << std::endl;
    }

    std::cout << "[ONNX] Selected execution provider preference: "
              << selected_execution_provider_ << std::endl;
    std::cout << "[ONNX] Provider version details are not exposed by the C++ runtime API; "
                 "the runtime version above is the active ONNX Runtime build version." << std::endl;
}

void InferenceEngine::start() {
    running_ = true;
    inference_thread_ = std::thread(&InferenceEngine::inferenceLoop, this);
}

void InferenceEngine::stop() {
    running_ = false;
    queue_cv_.notify_all();

    if (inference_thread_.joinable())
        inference_thread_.join();
}

void InferenceEngine::processFrame(std::shared_ptr<Frame> frame) {
    if (!frame) return;

    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        while (!input_queue_.empty()) {
            input_queue_.pop();
        }
        input_queue_.push(frame);
    }

    queue_cv_.notify_one();
}

std::shared_ptr<Frame> InferenceEngine::getResult() {
    std::lock_guard<std::mutex> lock(queue_mutex_);

    if (output_queue_.empty())
        return nullptr;

    while (output_queue_.size() > 1) {
        output_queue_.pop();
    }

    auto frame = output_queue_.front();
    output_queue_.pop();

    return frame;
}

void InferenceEngine::inferenceLoop() {
    while (running_) {
        std::shared_ptr<Frame> frame;

        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            queue_cv_.wait(lock, [this]() {
                return !input_queue_.empty() || !running_;
            });

            if (!running_) break;

            frame = input_queue_.front();
            input_queue_.pop();
        }

        processFrameImpl(frame);

        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            output_queue_.push(frame);
        }
    }
}

void InferenceEngine::processFrameImpl(std::shared_ptr<Frame> frame) {
    if (!frame || frame->mat.empty()) return;

    frame->detections.clear();

    cv::Mat send_mat = frame->mat;
    std::vector<int> jpeg_params = {cv::IMWRITE_JPEG_QUALITY, 90};
    cv::imencode(".jpg", send_mat, frame->jpeg, jpeg_params);

    cv::Mat infer_mat;
    cv::resize(frame->mat, infer_mat, cv::Size(640, 640));
    infer_mat.convertTo(infer_mat, CV_32F, 1.0 / 255.0);
    cv::cvtColor(infer_mat, infer_mat, cv::COLOR_BGR2RGB);

    Frame tmp_frame("", 0, 0, infer_mat);
    std::vector<float> input_tensor_values = tmp_frame.getDataCHW();
    std::vector<int64_t> input_shape = {1, infer_mat.channels(), infer_mat.rows, infer_mat.cols};

    try {
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            input_tensor_values.data(),
            input_tensor_values.size(),
            input_shape.data(),
            input_shape.size());

        auto output_tensors = session_->Run(
            Ort::RunOptions{nullptr},
            input_names_.data(),
            &input_tensor,
            input_names_.size(),
            output_names_.data(),
            output_names_.size());

        if (!output_tensors.empty() && output_tensors[0].IsTensor()) {
            auto& tensor = output_tensors[0];
            float* output_data = tensor.GetTensorMutableData<float>();
            const auto tensor_info = tensor.GetTensorTypeAndShapeInfo();
            const auto output_shape = tensor_info.GetShape();

            frame->inference_result.assign(output_data, output_data + tensor_info.GetElementCount());
            parseYOLO(frame, output_shape);
        }

    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime inference error: " << e.what() << std::endl;
    }
}

void InferenceEngine::parseYOLO(std::shared_ptr<Frame> frame, const std::vector<int64_t>& output_shape) {
    const float conf_threshold = 0.25f;
    const float iou_threshold = 0.45f;
    frame->detections.clear();

    if (output_shape.size() != 3 || output_shape[1] < 5 || output_shape[2] <= 0) {
        std::cerr << "[YOLO] Unexpected output shape:";
        for (auto dim : output_shape) {
            std::cerr << " " << dim;
        }
        std::cerr << std::endl;
        return;
    }

    const int64_t num_features = output_shape[1];
    const int64_t num_predictions = output_shape[2];
    const int num_classes = static_cast<int>(num_features - 4);
    const float* data = frame->inference_result.data();

    const float scale_x = static_cast<float>(frame->mat.cols) / 640.0f;
    const float scale_y = static_cast<float>(frame->mat.rows) / 640.0f;

    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    std::vector<int> class_ids;

    for (int64_t pred = 0; pred < num_predictions; ++pred) {
        const float x = data[pred];
        const float y = data[num_predictions + pred];
        const float w = data[(2 * num_predictions) + pred];
        const float h = data[(3 * num_predictions) + pred];

        float max_conf = 0.0f;
        int class_id = -1;
        for (int cls = 0; cls < num_classes; ++cls) {
            const float conf = data[((4 + cls) * num_predictions) + pred];
            if (conf > max_conf) {
                max_conf = conf;
                class_id = cls;
            }
        }

        if (max_conf <= conf_threshold) {
            continue;
        }

        const int left = std::max(0, static_cast<int>((x - (w * 0.5f)) * scale_x));
        const int top = std::max(0, static_cast<int>((y - (h * 0.5f)) * scale_y));
        const int width = std::max(0, static_cast<int>(w * scale_x));
        const int height = std::max(0, static_cast<int>(h * scale_y));

        if (width == 0 || height == 0) {
            continue;
        }

        boxes.emplace_back(left, top, width, height);
        scores.push_back(max_conf);
        class_ids.push_back(class_id);
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, scores, conf_threshold, iou_threshold, indices);

    for (int idx : indices) {
        frame->detections.push_back({classIdToLabel(class_ids[idx]), scores[idx], BBox(boxes[idx])});
    }

    std::cout << "[DEBUG] detections: " << frame->detections.size() << std::endl;
}
