#pragma once
#include <iostream>
#include <thread>
#include <chrono>
#include <atomic>
#include <csignal>
#include <vector>
#include <string>
#include <unordered_map>
#include <filesystem>
#include <opencv2/opencv.hpp>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>

#undef min
#undef max
#elif defined(__linux__)
#include <unistd.h>
#endif

#include "inference_engine.h"
#include "camera_manager.h"
#include "grpc_server.h"

// ------------------------------
// Глобальний прапорець для завершення сервісу
// ------------------------------
std::atomic<bool> g_running{true};

namespace {
std::filesystem::path getExecutableDir() {
#ifdef _WIN32
    std::wstring buffer(MAX_PATH, L'\0');
    const DWORD length = GetModuleFileNameW(nullptr, buffer.data(), static_cast<DWORD>(buffer.size()));
    if (length == 0) {
        return {};
    }

    buffer.resize(length);
    return std::filesystem::path(buffer).parent_path();
#elif defined(__linux__)
    std::vector<char> buffer(4096, '\0');
    const ssize_t length = readlink("/proc/self/exe", buffer.data(), buffer.size());
    if (length <= 0) {
        return {};
    }

    return std::filesystem::path(std::string(buffer.data(), static_cast<size_t>(length))).parent_path();
#else
    return {};
#endif
}

std::filesystem::path sourceRootHint() {
    return std::filesystem::path(__FILE__).parent_path().parent_path();
}

std::filesystem::path resolveExistingPath(const std::string& raw_path) {
    const std::filesystem::path input(raw_path);
    if (input.is_absolute() && std::filesystem::exists(input)) {
        return input;
    }

    std::vector<std::filesystem::path> candidates;
    candidates.push_back(input);

    const auto cwd = std::filesystem::current_path();
    candidates.push_back(cwd / input);

    const auto exe_dir = getExecutableDir();
    if (!exe_dir.empty()) {
        candidates.push_back(exe_dir / input);
        candidates.push_back(exe_dir.parent_path() / input);
        candidates.push_back(exe_dir.parent_path().parent_path() / input);
    }

    const auto source_root = sourceRootHint();
    candidates.push_back(source_root / input);

    for (const auto& candidate : candidates) {
        std::error_code ec;
        if (std::filesystem::exists(candidate, ec) && !ec) {
            return std::filesystem::weakly_canonical(candidate, ec);
        }
    }

    return input;
}
}

// ------------------------------
// Обробник сигналу Ctrl+C
// ------------------------------
void signalHandler(int signum) {
    std::cout << "\n[INFO] Interrupt signal (" << signum << ") received. Stopping service..." << std::endl;
    g_running = false;
}

// ------------------------------
// Динамічне виявлення підключених камер
// ------------------------------
std::vector<std::string> detectConnectedCameras(CameraManager& camera_manager, int max_cams = 10) {
    std::vector<std::string> camera_ids;

    for (int i = 0; i < max_cams; ++i) {
        cv::VideoCapture cap(i);
        if (cap.isOpened()) {
            cap.release();
            std::string cam_id = "camera_" + std::to_string(i);
            if (camera_manager.addCamera(cam_id, std::to_string(i))) {
                std::cout << "[INFO] Detected and added camera " << i << std::endl;
                camera_ids.push_back(cam_id);
            }
        }
    }

    if (camera_ids.empty()) {
        std::cerr << "[WARN] No cameras detected!" << std::endl;
    }

    return camera_ids;
}

// ------------------------------
// Додавання відео-файлів як джерел
// ------------------------------
std::vector<std::string> addVideoSources(CameraManager& camera_manager, const std::vector<std::string>& video_paths) {
    std::vector<std::string> video_ids;
    int idx = 0;

    for (const auto& path : video_paths) {
        std::string vid_id = "video_" + std::to_string(idx++);
        if (camera_manager.addCamera(vid_id, path)) {
            std::cout << "[INFO] Added video source: " << path << std::endl;
            video_ids.push_back(vid_id);
        }
    }

    return video_ids;
}

// ------------------------------
// Головна функція
// ------------------------------
int main() {
    std::cout << "[INFO] Starting Camera CV Service..." << std::endl;

    // Встановлюємо обробник Ctrl+C
    std::signal(SIGINT, signalHandler);

    // ------------------------------
    // Ініціалізація компонентів
    // ------------------------------
    const auto model_path = resolveExistingPath("models/yolov8x.onnx");
    const auto test_video_path = resolveExistingPath("test_video.mp4");

    std::cout << "[INFO] Model path: " << model_path.string() << std::endl;
    std::cout << "[INFO] Test video path: " << test_video_path.string() << std::endl;

    InferenceEngine inference_engine(model_path.string());
    CameraManager camera_manager(&inference_engine);
    GRPCServer grpc_server("0.0.0.0:50051");

    // Виявлення камер
    std::vector<std::string> camera_ids = detectConnectedCameras(camera_manager, 10);

    // Додавання відео-файлів
    std::vector<std::string> video_files = {test_video_path.string()};
    std::vector<std::string> video_ids = addVideoSources(camera_manager, video_files);

    if (camera_ids.empty() && video_ids.empty()) {
        std::cerr << "[ERROR] No cameras or video sources available. Exiting." << std::endl;
        return 1;
    }

    // Запуск камер та сервісів
    camera_manager.startAllCameras();
    inference_engine.start();
    grpc_server.start();

    std::cout << "[INFO] Service started. Processing frames.." << std::endl;

    // ------------------------------
    // Основний цикл обробки
    // ------------------------------
    int64_t global_frame_counter = 0; // числовий лічильник кадрів
    std::unordered_map<std::string, std::vector<Detection>> latest_detections;


    // auto fake_frame = std::make_shared<Frame>();
    // fake_frame->frame_id = 0;
    // fake_frame->timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
    //     std::chrono::system_clock::now().time_since_epoch()
    // ).count();

    // // Мінімальний JPEG, щоб поле image не було порожнім
    // fake_frame->jpeg = std::vector<uint8_t>{0xFF, 0xD8, 0xFF, 0xD9};

    // // Можна залишити пусті detections або додати тестовий
    // grpc_server.sendDetectionResult(fake_frame);
    // std::cout << "[DEBUG] Фейковий кадр додано у чергу" << std::endl;


    while (g_running) {
        auto processFrames = [&](const std::vector<std::string>& ids) {
            for (const auto& id : ids) {
                if (auto result = inference_engine.getResult()) {
                    latest_detections[result->camera_id] = result->detections;
                }

                auto frame = camera_manager.getLatestFrame(id);
                if (!frame) continue;

                // Присвоюємо джерело кадру
                frame->camera_id = id;  
                // Встановлюємо frame_id та timestamp
                frame->frame_id = global_frame_counter++;
                frame->timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::system_clock::now().time_since_epoch()
                ).count();

                inference_engine.processFrame(frame);

                auto stream_frame = std::make_shared<Frame>(
                    frame->camera_id,
                    frame->frame_id,
                    frame->timestamp,
                    frame->mat
                );

                auto it = latest_detections.find(id);
                if (it != latest_detections.end()) {
                    stream_frame->detections = it->second;
                }

                grpc_server.sendDetectionResult(stream_frame);
            }
        };

        processFrames(camera_ids);
        processFrames(video_ids);

        // Коротка пауза для стабільності FPS
    }

    // ------------------------------
    // Зупинка сервісів
    // ------------------------------
    std::cout << "[INFO] Stopping services..." << std::endl;
    camera_manager.stopAllCameras();
    inference_engine.stop();
    grpc_server.stop();
    std::cout << "[INFO] Service stopped." << std::endl;

    return 0;
}
