#include "inference_engine.h"
#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <cctype>
#include <cmath>
#include <iostream>
#include <chrono>
#include <filesystem>
#include <string>
#include <opencv2/opencv.hpp>

namespace {
std::string readEnvString(const char* name, const std::string& fallback) {
    if (const char* raw = std::getenv(name)) {
        return raw;
    }
    return fallback;
}

int readEnvInt(const char* name, int fallback) {
    if (const char* raw = std::getenv(name)) {
        try {
            return std::max(1, std::stoi(raw));
        } catch (...) {
        }
    }
    return fallback;
}

int readEnvNonNegativeInt(const char* name, int fallback) {
    if (const char* raw = std::getenv(name)) {
        try {
            return std::max(0, std::stoi(raw));
        } catch (...) {
        }
    }
    return fallback;
}

float readEnvFloat(const char* name, float fallback) {
    if (const char* raw = std::getenv(name)) {
        try {
            return std::stof(raw);
        } catch (...) {
        }
    }
    return fallback;
}

bool readEnvBool(const char* name, bool fallback) {
    if (const char* raw = std::getenv(name)) {
        const std::string value(raw);
        return value == "1" || value == "true" || value == "TRUE" || value == "yes" || value == "YES";
    }
    return fallback;
}

std::string resolveOnnxFallbackModelPath(const std::string& model_path) {
    const std::string configured = readEnvString("CAMERA_ONNX_FALLBACK_MODEL_PATH", "");
    if (!configured.empty()) {
        return configured;
    }

    std::filesystem::path fallback_path(model_path);
    const std::string extension = fallback_path.extension().string();
    if (extension == ".engine" || extension == ".plan" || extension == ".trt" || extension == ".tensorrt") {
        fallback_path.replace_extension(".onnx");
        return fallback_path.string();
    }

    return model_path;
}
}

InferenceEngine::InferenceEngine(const std::string& model_path)
    : model_path_(model_path),
      running_(false) {
    input_width_ = readEnvInt("CAMERA_INFERENCE_WIDTH", 640);
    input_height_ = readEnvInt("CAMERA_INFERENCE_HEIGHT", 640);
    min_detection_interval_ms_ = readEnvNonNegativeInt("CAMERA_MIN_DETECTION_INTERVAL_MS", 0);
    confidence_threshold_ = std::clamp(readEnvFloat("CAMERA_CONF_THRESHOLD", 0.25f), 0.0f, 1.0f);
    iou_threshold_ = std::clamp(readEnvFloat("CAMERA_IOU_THRESHOLD", 0.45f), 0.0f, 1.0f);
    verbose_logging_ = readEnvBool("CAMERA_VERBOSE_LOGS", false);
    input_shape_ = {1, 3, input_height_, input_width_};
    input_tensor_values_.resize(static_cast<size_t>(3 * input_height_ * input_width_));

    try {
        const std::string requested_backend = readEnvString("RUNTIME_BACKEND", "onnx");
        backend_ = InferenceBackendFactory::createBackend(requested_backend);
        backend_->initialize(model_path);
        std::cout << "[Engine] Backend initialized: " << backend_->getBackendName() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "[Engine] Backend initialization failed: " << e.what() << std::endl;
        const std::string requested_backend = readEnvString("RUNTIME_BACKEND", "onnx");
        std::string lowered_backend = requested_backend;
        std::transform(lowered_backend.begin(), lowered_backend.end(), lowered_backend.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (lowered_backend != "tensorrt") {
            throw;
        }

        std::cerr << "[Engine] Falling back to ONNX Runtime" << std::endl;
        backend_ = InferenceBackendFactory::createBackend("onnx");
        backend_->initialize(resolveOnnxFallbackModelPath(model_path));
        std::cout << "[Engine] Backend initialized: " << backend_->getBackendName() << std::endl;
    }
}

InferenceEngine::~InferenceEngine() {
    stop();
}

void InferenceEngine::start() {
    if (!backend_ || !backend_->isReady()) {
        throw std::runtime_error("InferenceEngine is not ready: Backend initialization failed");
    }

    running_ = true;
    inference_thread_ = std::thread(&InferenceEngine::inferenceLoop, this);
}

void InferenceEngine::stop() {
    running_ = false;
    queue_cv_.notify_all();

    if (inference_thread_.joinable())
        inference_thread_.join();

    std::lock_guard<std::mutex> lock(queue_mutex_);
    while (!input_queue_.empty()) {
        input_queue_.pop();
    }
    while (!output_queue_.empty()) {
        output_queue_.pop();
    }
}

bool InferenceEngine::isReady() const {
    return backend_ && backend_->isReady();
}

void InferenceEngine::processFrame(std::shared_ptr<Frame> frame) {
    if (!frame || !running_ || !backend_) return;

    submitted_frames_.fetch_add(1);
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        while (!input_queue_.empty()) {
            input_queue_.pop();
            dropped_pending_frames_.fetch_add(1);
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

InferenceMetricsSnapshot InferenceEngine::consumeMetricsSnapshot() {
    InferenceMetricsSnapshot snapshot;
    snapshot.submitted_frames = submitted_frames_.exchange(0);
    snapshot.dropped_pending_frames = dropped_pending_frames_.exchange(0);
    snapshot.processed_frames = processed_frames_.exchange(0);
    snapshot.total_detections = total_detections_.exchange(0);

    const int64_t total_us = total_inference_us_.exchange(0);
    const int64_t max_us = max_inference_us_.exchange(0);
    if (snapshot.processed_frames > 0) {
        snapshot.avg_inference_ms =
            static_cast<double>(total_us) / static_cast<double>(snapshot.processed_frames) / 1000.0;
        snapshot.max_inference_ms = static_cast<double>(max_us) / 1000.0;
    }

    return snapshot;
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

        const auto inference_started = std::chrono::steady_clock::now();
        auto result_frame = processFrameImpl(frame);
        if (!result_frame) {
            waitForDetectionInterval(inference_started);
            continue;
        }

        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            while (!output_queue_.empty()) {
                output_queue_.pop();
            }
            output_queue_.push(std::move(result_frame));
        }

        waitForDetectionInterval(inference_started);
    }
}

void InferenceEngine::waitForDetectionInterval(std::chrono::steady_clock::time_point inference_started) {
    if (min_detection_interval_ms_ <= 0 || !running_) {
        return;
    }

    const auto next_allowed_inference =
        inference_started + std::chrono::milliseconds(min_detection_interval_ms_);
    const auto now = std::chrono::steady_clock::now();
    if (now >= next_allowed_inference) {
        return;
    }

    std::unique_lock<std::mutex> lock(queue_mutex_);
    queue_cv_.wait_until(lock, next_allowed_inference, [this]() {
        return !running_;
    });
}

std::shared_ptr<Frame> InferenceEngine::processFrameImpl(const std::shared_ptr<Frame>& frame) {
    if (!frame || frame->mat.empty() || !backend_) return nullptr;

    const auto inference_started = std::chrono::steady_clock::now();
    prepareInputTensor(frame->mat);

    try {
        auto detections = backend_->runInference(
            input_tensor_values_.data(),
            input_shape_,
            frame->mat.cols,
            frame->mat.rows,
            confidence_threshold_,
            input_width_,
            input_height_);

        auto result_frame = std::make_shared<Frame>();
        result_frame->camera_id = frame->camera_id;
        result_frame->frame_id = frame->frame_id;
        result_frame->timestamp = frame->timestamp;
        result_frame->frame_width = frame->width();
        result_frame->frame_height = frame->height();
        result_frame->detections = detections;

        const auto inference_us = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - inference_started).count();
        processed_frames_.fetch_add(1);
        total_detections_.fetch_add(static_cast<int64_t>(result_frame->detections.size()));
        total_inference_us_.fetch_add(inference_us);

        int64_t previous_max = max_inference_us_.load();
        while (inference_us > previous_max &&
               !max_inference_us_.compare_exchange_weak(previous_max, inference_us)) {
        }
        return result_frame;

    } catch (const std::exception& e) {
        std::cerr << "[Engine] Inference error: " << e.what() << std::endl;
    }

    return nullptr;
}

void InferenceEngine::prepareInputTensor(const cv::Mat& frame) {
    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(input_width_, input_height_), 0.0, 0.0, cv::INTER_LINEAR);

    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

    std::vector<cv::Mat> channels(3);
    cv::split(rgb, channels);

    const size_t plane_size = static_cast<size_t>(input_width_ * input_height_);
    for (int channel = 0; channel < 3; ++channel) {
        channels[channel].convertTo(
            cv::Mat(input_height_, input_width_, CV_32F, input_tensor_values_.data() + (plane_size * channel)),
            CV_32F,
            1.0 / 255.0);
    }
}
