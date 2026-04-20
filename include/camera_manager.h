#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <thread>
#include <deque>
#include <memory>
#include <atomic>
#include <chrono>
#include <mutex>
#include <condition_variable>
#include <unordered_map>
#include "models/frame.h"
#include "inference_engine.h"

class CameraManager {
public:
    explicit CameraManager(InferenceEngine* engine);
    ~CameraManager();

    // Публічні методи
    bool addCamera(const std::string& camera_id, const std::string& source);
    bool removeCamera(const std::string& camera_id);
    void startAllCameras();
    void stopAllCameras();
    std::shared_ptr<Frame> getLatestFrame(const std::string& camera_id);

private:
    struct CameraInfo {
        std::string camera_id;
        std::string source;
        cv::VideoCapture cap;
        std::thread capture_thread;
        std::atomic<bool> running{false};
        std::deque<std::shared_ptr<Frame>> frame_queue;
        std::mutex queue_mutex;
        std::condition_variable queue_cv;
        bool is_file_source = false;
        double source_fps = 0.0;
        int64_t first_source_timestamp_ms = -1;
        int64_t playback_start_wallclock_ms = 0;
        std::chrono::steady_clock::time_point playback_start_time{};

        CameraInfo(const std::string& id, const std::string& src)
            : camera_id(id), source(src) {}
    };

    InferenceEngine* inferenceEngine_ = nullptr;
    std::unordered_map<std::string, std::unique_ptr<CameraInfo>> cameras_;
    std::mutex cameras_mutex_;

    // Приватні методи
    void captureLoop(const std::string& camera_id);
    void enqueueFrame(std::shared_ptr<Frame> frame, CameraInfo& camInfo);
    std::shared_ptr<Frame> dequeueFrame(CameraInfo& camInfo);
};
