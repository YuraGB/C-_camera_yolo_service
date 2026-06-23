#pragma once

#include "inference_backend.h"
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <memory>

class ONNXRuntimeBackend : public InferenceBackend {
public:
    ONNXRuntimeBackend();
    ~ONNXRuntimeBackend() override;

    void initialize(const std::string& model_path) override;
    bool isReady() const override;
    std::vector<Detection> runInference(
        const float* input_data,
        const std::vector<int64_t>& input_shape,
        int frame_width,
        int frame_height,
        float confidence_threshold,
        int input_width,
        int input_height) override;
    std::string getBackendName() const override;

private:
    std::vector<Detection> parseYOLO(
        const float* data,
        const std::vector<int64_t>& output_shape,
        int frame_width,
        int frame_height,
        float confidence_threshold,
        int input_width,
        int input_height);

    Ort::Env env_;
    Ort::SessionOptions session_options_;
    std::unique_ptr<Ort::Session> session_;
    Ort::MemoryInfo memory_info_;

    std::vector<std::string> input_names_str_;
    std::vector<const char*> input_names_;
    std::vector<std::string> output_names_str_;
    std::vector<const char*> output_names_;
    std::string selected_execution_provider_;
    bool verbose_logging_;
};
