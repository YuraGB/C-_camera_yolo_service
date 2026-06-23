#pragma once

#include "inference_backend.h"
#include <memory>
#include <string>
#include <vector>

#ifdef USE_TENSORRT
#include <NvInfer.h>
#include <NvOnnxParser.h>

class TensorRTBackend : public InferenceBackend {
public:
    TensorRTBackend();
    ~TensorRTBackend() override;

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
        size_t output_size,
        int frame_width,
        int frame_height,
        float confidence_threshold,
        int input_width,
        int input_height);

    class Logger : public nvinfer1::ILogger {
    public:
        void log(Severity severity, const char* msg) noexcept override;
    };

    Logger logger_;
    std::shared_ptr<nvinfer1::IRuntime> runtime_;
    std::shared_ptr<nvinfer1::ICudaEngine> engine_;
    std::shared_ptr<nvinfer1::IExecutionContext> context_;
    std::vector<void*> device_bindings_;
    std::vector<size_t> binding_sizes_;
    int input_binding_index_;
    int output_binding_index_;
    bool verbose_logging_;
};

#endif
