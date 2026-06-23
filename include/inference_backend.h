#pragma once

#include <memory>
#include <string>
#include <vector>
#include <cstdint>
#include "frame.h"

class InferenceBackend {
public:
    virtual ~InferenceBackend() = default;

    virtual void initialize(const std::string& model_path) = 0;
    virtual bool isReady() const = 0;

    virtual std::vector<Detection> runInference(
        const float* input_data,
        const std::vector<int64_t>& input_shape,
        int frame_width,
        int frame_height,
        float confidence_threshold,
        int input_width,
        int input_height) = 0;

    virtual std::string getBackendName() const = 0;
};

class InferenceBackendFactory {
public:
    static std::unique_ptr<InferenceBackend> createBackend();
    static std::unique_ptr<InferenceBackend> createBackend(const std::string& runtime_backend);
};
