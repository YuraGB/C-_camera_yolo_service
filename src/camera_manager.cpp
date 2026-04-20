#include "camera_manager.h"
#include <iostream>
#include <chrono>
#include <thread>
#include <opencv2/opencv.hpp>

namespace {
int64_t currentWallClockMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

bool isNumericSource(const std::string& source) {
    if (source.empty()) {
        return false;
    }

    size_t start = (source[0] == '-' || source[0] == '+') ? 1 : 0;
    if (start >= source.size()) {
        return false;
    }

    for (size_t i = start; i < source.size(); ++i) {
        if (!std::isdigit(static_cast<unsigned char>(source[i]))) {
            return false;
        }
    }

    return true;
}
}  // namespace

CameraManager::CameraManager(InferenceEngine* engine)
    : inferenceEngine_(engine) {}

CameraManager::~CameraManager() {
    stopAllCameras();
}

bool CameraManager::addCamera(const std::string& camera_id, const std::string& source) {
    std::lock_guard<std::mutex> lock(cameras_mutex_);
    if (cameras_.find(camera_id) != cameras_.end()) return false;

    cameras_[camera_id] = std::make_unique<CameraInfo>(camera_id, source);
    return true;
}

bool CameraManager::removeCamera(const std::string& camera_id) {
    std::lock_guard<std::mutex> lock(cameras_mutex_);
    auto it = cameras_.find(camera_id);
    if (it == cameras_.end()) return false;

    it->second->running = false;
    if (it->second->capture_thread.joinable())
        it->second->capture_thread.join();

    cameras_.erase(it);
    return true;
}

void CameraManager::startAllCameras() {
    std::lock_guard<std::mutex> lock(cameras_mutex_);
    for (auto& [id, cam] : cameras_) {
        if (!cam->running) {
            cam->running = true;
            cam->capture_thread = std::thread(&CameraManager::captureLoop, this, id);
        }
    }
}

void CameraManager::stopAllCameras() {
    std::lock_guard<std::mutex> lock(cameras_mutex_);
    for (auto& [id, cam] : cameras_) {
        cam->running = false;
        if (cam->capture_thread.joinable())
            cam->capture_thread.join();
    }
}

std::shared_ptr<Frame> CameraManager::getLatestFrame(const std::string& camera_id) {
    std::lock_guard<std::mutex> lock(cameras_mutex_);
    auto it = cameras_.find(camera_id);
    if (it == cameras_.end()) return nullptr;
    return dequeueFrame(*it->second);
}

void CameraManager::captureLoop(const std::string& camera_id) {
    CameraInfo* cam = nullptr;
    {
        std::lock_guard<std::mutex> lock(cameras_mutex_);
        auto it = cameras_.find(camera_id);
        if (it == cameras_.end()) return;
        cam = it->second.get();
    }

    bool opened = false;
    cam->is_file_source = !isNumericSource(cam->source);

    try {
        int index = std::stoi(cam->source);
        opened = cam->cap.open(index);
        cam->is_file_source = false;
    } catch (...) {
        opened = cam->cap.open(cam->source);
        if (!opened) {
            std::cerr << "[WARN] Failed to open source: " << cam->source
                      << " (check path and format)" << std::endl;
            return;
        }
    }

    if (!opened) {
        std::cerr << "[WARN] Failed to open source: " << cam->source << std::endl;
        return;
    }

    cam->source_fps = cam->cap.get(cv::CAP_PROP_FPS);
    cam->first_source_timestamp_ms = -1;
    cam->playback_start_wallclock_ms = 0;
    cam->playback_start_time = std::chrono::steady_clock::now();

    int64_t frame_id = 0;
    auto frameTimestampForSource = [&]() {
        const int64_t fallback_timestamp_ms = currentWallClockMs();
        if (!cam->is_file_source) {
            return fallback_timestamp_ms;
        }

        const double pos_msec = cam->cap.get(cv::CAP_PROP_POS_MSEC);
        if (pos_msec < 0.0) {
            return fallback_timestamp_ms;
        }

        const int64_t source_timestamp_ms = static_cast<int64_t>(pos_msec);
        if (cam->first_source_timestamp_ms < 0) {
            cam->first_source_timestamp_ms = source_timestamp_ms;
            cam->playback_start_time = std::chrono::steady_clock::now();
            cam->playback_start_wallclock_ms = fallback_timestamp_ms;
        }

        return cam->playback_start_wallclock_ms +
               std::max<int64_t>(0, source_timestamp_ms - cam->first_source_timestamp_ms);
    };
    auto paceVideoFilePlayback = [&]() {
        if (!cam->is_file_source || cam->first_source_timestamp_ms < 0) {
            return;
        }

        const double pos_msec = cam->cap.get(cv::CAP_PROP_POS_MSEC);
        if (pos_msec < 0.0) {
            return;
        }

        const auto target_offset = std::chrono::milliseconds(
            std::max<int64_t>(0, static_cast<int64_t>(pos_msec) - cam->first_source_timestamp_ms));
        const auto target_time = cam->playback_start_time + target_offset;
        const auto now = std::chrono::steady_clock::now();

        if (target_time > now) {
            std::this_thread::sleep_until(target_time);
        }
    };
    auto catchUpVideoFilePlayback = [&]() {
        if (!cam->is_file_source || cam->first_source_timestamp_ms < 0) {
            return;
        }

        const double frame_interval_ms =
            (cam->source_fps > 1.0) ? (1000.0 / cam->source_fps) : 33.333;
        const double allowed_lag_ms = frame_interval_ms * 1.5;

        while (cam->running) {
            const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - cam->playback_start_time).count();
            const double target_source_ms =
                static_cast<double>(cam->first_source_timestamp_ms + std::max<int64_t>(0, elapsed_ms));
            const double current_source_ms = cam->cap.get(cv::CAP_PROP_POS_MSEC);

            if (current_source_ms < 0.0 || current_source_ms + allowed_lag_ms >= target_source_ms) {
                break;
            }

            if (!cam->cap.grab()) {
                break;
            }

            ++frame_id;
        }
    };

    while (cam->running) {
        cv::Mat mat;
        if (!cam->cap.read(mat) || mat.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            continue;
        }

        auto frame = std::make_shared<Frame>();
        frame->camera_id = camera_id;
        frame->frame_id = frame_id++;
        frame->timestamp = frameTimestampForSource();
        frame->mat = std::move(mat);

        enqueueFrame(frame, *cam);
        paceVideoFilePlayback();
        catchUpVideoFilePlayback();
    }

    cam->cap.release();
}

void CameraManager::enqueueFrame(std::shared_ptr<Frame> frame, CameraInfo& cam) {
    std::lock_guard<std::mutex> lock(cam.queue_mutex);
    const size_t MAX_QUEUE = 1;
    if (cam.frame_queue.size() >= MAX_QUEUE)
        cam.frame_queue.pop_front();
    cam.frame_queue.push_back(frame);
    cam.queue_cv.notify_one();
}

std::shared_ptr<Frame> CameraManager::dequeueFrame(CameraInfo& cam) {
    std::lock_guard<std::mutex> lock(cam.queue_mutex);
    if (cam.frame_queue.empty()) return nullptr;
    auto frame = cam.frame_queue.back();
    cam.frame_queue.clear();
    return frame;
}
