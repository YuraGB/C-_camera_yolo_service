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
#include <chrono>
#include "frame.h"
#include "inference_backend.h"

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
    void waitForDetectionInterval(std::chrono::steady_clock::time_point inference_started);

    std::string model_path_;
    std::unique_ptr<InferenceBackend> backend_;

    std::thread inference_thread_;
    std::atomic<bool> running_{false};

    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::queue<std::shared_ptr<Frame>> input_queue_;
    std::queue<std::shared_ptr<Frame>> output_queue_;

    int input_width_ = 640;
    int input_height_ = 640;
    int min_detection_interval_ms_ = 0;
    float confidence_threshold_ = 0.25f;
    float iou_threshold_ = 0.45f;
    bool verbose_logging_ = false;
    std::vector<float> input_tensor_values_;
    std::vector<int64_t> input_shape_;

    std::atomic<int64_t> submitted_frames_{0};
    std::atomic<int64_t> dropped_pending_frames_{0};
    std::atomic<int64_t> processed_frames_{0};
    std::atomic<int64_t> total_detections_{0};
    std::atomic<int64_t> total_inference_us_{0};
    std::atomic<int64_t> max_inference_us_{0};
};
