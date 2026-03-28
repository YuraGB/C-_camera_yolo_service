#include "camera_manager.h"
#include <iostream>
#include <chrono>
#include <thread>
#include <opencv2/opencv.hpp>

CameraManager::CameraManager(InferenceEngine* engine)
    : inferenceEngine_(engine) {}

CameraManager::~CameraManager() {
    stopAllCameras();
}

// --------------------------
// Додаємо камеру або відео-джерело
// --------------------------
bool CameraManager::addCamera(const std::string& camera_id, const std::string& source) {
    std::lock_guard<std::mutex> lock(cameras_mutex_);
    if (cameras_.find(camera_id) != cameras_.end()) return false;

    cameras_[camera_id] = std::make_unique<CameraInfo>(camera_id, source);
    return true;
}

// --------------------------
// Видаляємо камеру
// --------------------------
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

// --------------------------
// Запуск усіх камер
// --------------------------
void CameraManager::startAllCameras() {
    std::lock_guard<std::mutex> lock(cameras_mutex_);
    for (auto& [id, cam] : cameras_) {
        if (!cam->running) {
            cam->running = true;
            cam->capture_thread = std::thread(&CameraManager::captureLoop, this, id);
        }
    }
}

// --------------------------
// Зупинка усіх камер
// --------------------------
void CameraManager::stopAllCameras() {
    std::lock_guard<std::mutex> lock(cameras_mutex_);
    for (auto& [id, cam] : cameras_) {
        cam->running = false;
        if (cam->capture_thread.joinable())
            cam->capture_thread.join();
    }
}

// --------------------------
// Отримати останній кадр з камери
// --------------------------
std::shared_ptr<Frame> CameraManager::getLatestFrame(const std::string& camera_id) {
    std::lock_guard<std::mutex> lock(cameras_mutex_);
    auto it = cameras_.find(camera_id);
    if (it == cameras_.end()) return nullptr;
    return dequeueFrame(*it->second);
}

// --------------------------
// Основний цикл захоплення кадрів
// --------------------------
void CameraManager::captureLoop(const std::string& camera_id) {
    CameraInfo* cam = nullptr;
    {
        std::lock_guard<std::mutex> lock(cameras_mutex_);
        auto it = cameras_.find(camera_id);
        if (it == cameras_.end()) return;
        cam = it->second.get();
    }

    bool opened = false;
    try { 
        int index = std::stoi(cam->source); // спроба як індекс камери
        opened = cam->cap.open(index);
    } catch (...) {
        opened = cam->cap.open(cam->source); // або як відеофайл
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

    int64_t frame_id = 0;
    auto fps_delay = std::chrono::milliseconds(200); // ~5 FPS

    while (cam->running) {
        auto start_time = std::chrono::high_resolution_clock::now();

        cv::Mat mat;
        if (!cam->cap.read(mat) || mat.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            continue;
        }

        cv::cvtColor(mat, mat, cv::COLOR_BGR2RGB); // для ONNX/Yolo

        auto frame = std::make_shared<Frame>(
            camera_id,
            frame_id++,
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()
            ).count(),
            mat
        );

        enqueueFrame(frame, *cam);

        if (inferenceEngine_)
            inferenceEngine_->processFrame(frame); // одразу відправляємо на інференс

        // розрахунок часу сну для стабільного FPS
        auto elapsed = std::chrono::high_resolution_clock::now() - start_time;
        auto sleep_time = fps_delay - std::chrono::duration_cast<std::chrono::milliseconds>(elapsed);
        if (sleep_time.count() > 0)
            std::this_thread::sleep_for(sleep_time);
    }

    cam->cap.release();
}

// --------------------------
// Приватні методи для черги
// --------------------------
void CameraManager::enqueueFrame(std::shared_ptr<Frame> frame, CameraInfo& cam) {
    std::lock_guard<std::mutex> lock(cam.queue_mutex);
    const size_t MAX_QUEUE = 10;
    if (cam.frame_queue.size() >= MAX_QUEUE)
        cam.frame_queue.pop_front();
    cam.frame_queue.push_back(frame);
    cam.queue_cv.notify_one();
}

std::shared_ptr<Frame> CameraManager::dequeueFrame(CameraInfo& cam) {
    std::lock_guard<std::mutex> lock(cam.queue_mutex);
    if (cam.frame_queue.empty()) return nullptr;
    return cam.frame_queue.back(); // завжди беремо останній
}