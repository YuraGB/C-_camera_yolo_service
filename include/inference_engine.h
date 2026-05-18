#pragma once

#include <string>
#include <vector>
#include <memory>
#include <thread>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <atomic>
#include <cstdint>
#include <onnxruntime_cxx_api.h>
#include "frame.h"

struct InferenceMetricsSnapshot {
    int64_t submitted_frames = 0;
    int64_t dropped_pending_frames = 0;
    int64_t processed_frames = 0;
    int64_t total_detections = 0;
    double avg_inference_ms = 0.0;
    double max_inference_ms = 0.0;
};

class InferenceEngine {
public:
    explicit InferenceEngine(const std::string& model_path);
    ~InferenceEngine();

    InferenceEngine(const InferenceEngine&) = delete;
    InferenceEngine& operator=(const InferenceEngine&) = delete;

    void start();
    void stop();
    bool isReady() const;

    void processFrame(std::shared_ptr<Frame> frame);
    std::shared_ptr<Frame> getResult();
    InferenceMetricsSnapshot consumeMetricsSnapshot();

private:
    void inferenceLoop();
    std::shared_ptr<Frame> processFrameImpl(const std::shared_ptr<Frame>& frame);
    void prepareInputTensor(const cv::Mat& frame);
    std::vector<Detection> parseYOLO(
        const float* data,
        const std::vector<int64_t>& output_shape,
        int frame_width,
        int frame_height);
    void configureExecutionProvider();

    std::string model_path_;
    std::string selected_execution_provider_ = "CPUExecutionProvider";
    Ort::Env env_;
    Ort::SessionOptions session_options_;
    std::unique_ptr<Ort::Session> session_;

    std::vector<std::string> input_names_str_;
    std::vector<const char*> input_names_;
    std::vector<std::string> output_names_str_;
    std::vector<const char*> output_names_;

    std::thread inference_thread_;
    std::atomic<bool> running_{false};

    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::queue<std::shared_ptr<Frame>> input_queue_;
    std::queue<std::shared_ptr<Frame>> output_queue_;

    int input_width_ = 640;
    int input_height_ = 640;
    float confidence_threshold_ = 0.25f;
    float iou_threshold_ = 0.45f;
    bool verbose_logging_ = false;
    Ort::MemoryInfo memory_info_;
    std::vector<float> input_tensor_values_;
    std::vector<int64_t> input_shape_;

    std::atomic<int64_t> submitted_frames_{0};
    std::atomic<int64_t> dropped_pending_frames_{0};
    std::atomic<int64_t> processed_frames_{0};
    std::atomic<int64_t> total_detections_{0};
    std::atomic<int64_t> total_inference_us_{0};
    std::atomic<int64_t> max_inference_us_{0};
};
