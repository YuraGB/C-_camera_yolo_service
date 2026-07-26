#include <iostream>
#include <thread>
#include <chrono>
#include <atomic>
#include <csignal>
#include <vector>
#include <string>
#include <filesystem>

#include <nlohmann/json.hpp>

#include "inference_engine.h"
#include "camera_manager.h"
#include "tracking_manager.h"
#include "webrtc_service.h"
#include "core/pipeline/runtime_config.h"
#include "platform/platform_services.h"

std::atomic<bool> g_running{true};

namespace {
void drainDetectionResults(
    InferenceEngine& inference_engine,
    TrackingManager& tracking_manager,
    WebRTCService& webrtc_service)
{
    while (auto result = inference_engine.getResult()) {
        if (!result->detections.empty()) {
            webrtc_service.sendDetectionResult(result);
        }
        tracking_manager.submitDetections(result);
    }
}

void detectionPublishLoop(
    InferenceEngine& inference_engine,
    TrackingManager& tracking_manager,
    WebRTCService& webrtc_service)
{
    while (g_running) {
        drainDetectionResults(inference_engine, tracking_manager, webrtc_service);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    drainDetectionResults(inference_engine, tracking_manager, webrtc_service);
}

void inferenceMetricsPublishLoop(
    InferenceEngine& inference_engine,
    WebRTCService& webrtc_service,
    int interval_ms)
{
    if (interval_ms <= 0) {
        return;
    }

    while (g_running) {
        std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms));
        const auto snapshot = inference_engine.consumeMetricsSnapshot();
        if (snapshot.submitted_frames == 0 &&
            snapshot.processed_frames == 0 &&
            snapshot.dropped_pending_frames == 0) {
            continue;
        }

        nlohmann::json payload = {
            {"type", "pipeline_metrics"},
            {"scope", "inference"},
            {"camera_id", "all"},
            {"interval_ms", interval_ms},
            {"submitted_frames", snapshot.submitted_frames},
            {"dropped_pending_frames", snapshot.dropped_pending_frames},
            {"processed_frames", snapshot.processed_frames},
            {"total_detections", snapshot.total_detections},
            {"avg_inference_ms", snapshot.avg_inference_ms},
            {"max_inference_ms", snapshot.max_inference_ms},
        };
        webrtc_service.sendPipelineMetrics(payload);
    }
}
}

void signalHandler(int signum) {
    std::cout << "\n[INFO] Interrupt signal (" << signum << ") received. Stopping service..." << std::endl;
    g_running = false;
}

std::vector<std::string> detectConnectedCameras(
    CameraManager& camera_manager,
    const platform::PlatformServices& platform_services,
    int max_cams = 10) {
    std::vector<std::string> camera_ids;

    for (const int camera_index : platform_services.enumerateCameraIndices(max_cams)) {
        std::string cam_id = "camera_" + std::to_string(camera_index);
        if (camera_manager.addCamera(cam_id, std::to_string(camera_index))) {
            std::cout << "[INFO] Detected and added camera " << camera_index << std::endl;
            camera_ids.push_back(cam_id);
        }
    }

    if (camera_ids.empty()) {
        std::cerr << "[WARN] No cameras detected!" << std::endl;
    }

    return camera_ids;
}

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

int main() {
    std::cout << "[INFO] Starting Camera CV Service..." << std::endl;
    std::signal(SIGINT, signalHandler);

    auto platform_services = platform::createPlatformServices();
    const auto runtime_config = core::pipeline::loadRuntimeConfig(*platform_services);
    const auto& model_path = runtime_config.model_path;
    const auto& test_video_path = runtime_config.test_video_path;

    std::cout << "[INFO] Platform: " << platform_services->name() << std::endl;
    std::cout << "[INFO] Model path: " << model_path.string() << std::endl;
    if (!test_video_path.empty()) {
        std::cout << "[INFO] Test video path: " << test_video_path.string() << std::endl;
    } else {
        std::cout << "[INFO] Test video path: not configured" << std::endl;
    }

    if (!std::filesystem::exists(model_path)) {
        std::cerr << "[ERROR] Model file was not found: " << model_path.string() << std::endl;
        return 1;
    }

    try {
        InferenceEngine inference_engine(model_path.string());
        if (!inference_engine.isReady()) {
            std::cerr << "[ERROR] Inference engine could not initialize the ONNX session." << std::endl;
            return 1;
        }

        CameraManager camera_manager;
        TrackingManager tracking_manager;
        WebRTCService webrtc_service(runtime_config.webrtc);

        std::vector<std::string> camera_ids = detectConnectedCameras(
            camera_manager,
            *platform_services,
            runtime_config.max_camera_scan);

        std::vector<std::string> video_files;
        if (!test_video_path.empty() && std::filesystem::is_regular_file(test_video_path)) {
            video_files.push_back(test_video_path.string());
        } else if (!test_video_path.empty()) {
            std::cout << "[INFO] Optional test video was not found, skipping: "
                      << test_video_path.string() << std::endl;
        }

        std::vector<std::string> video_ids = addVideoSources(camera_manager, video_files);

        if (camera_ids.empty() && video_ids.empty()) {
            std::cerr << "[ERROR] No cameras or video sources available. Exiting." << std::endl;
            return 1;
        }

        for (const auto& id : camera_ids) {
            webrtc_service.addVideoSource(id);
        }
        for (const auto& id : video_ids) {
            webrtc_service.addVideoSource(id);
        }

        webrtc_service.start();
        std::thread detection_thread(
            detectionPublishLoop,
            std::ref(inference_engine),
            std::ref(tracking_manager),
            std::ref(webrtc_service));
        std::thread metrics_thread(
            inferenceMetricsPublishLoop,
            std::ref(inference_engine),
            std::ref(webrtc_service),
            runtime_config.webrtc.pipeline_metrics_interval_ms);

        std::cout << "[INFO] Service started. Waiting for a WebRTC viewer before processing frames..." << std::endl;

        bool processing_active = false;

        while (g_running) {
            const bool has_viewer = webrtc_service.hasActivePeerConnections();
            if (has_viewer && !processing_active) {
                std::cout << "[INFO] WebRTC viewer connected. Starting capture and inference." << std::endl;
                camera_manager.startAllCameras();
                inference_engine.start();
                processing_active = true;
            } else if (!has_viewer && processing_active) {
                std::cout << "[INFO] No WebRTC viewers remain. Stopping capture and inference." << std::endl;
                camera_manager.stopAllCameras();
                inference_engine.stop();
                processing_active = false;
            }

            if (!processing_active) {
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
                continue;
            }

            bool captured_any_frame = false;

            auto processFrames = [&](const std::vector<std::string>& ids) {
                for (const auto& id : ids) {
                    auto frame = camera_manager.getLatestFrame(id);
                    if (!frame) {
                        continue;
                    }

                    captured_any_frame = true;
                    if (frame->camera_id.empty()) {
                        frame->camera_id = id;
                    }

                    webrtc_service.sendFrame(frame->camera_id, frame);
                    inference_engine.processFrame(frame);
                    drainDetectionResults(inference_engine, tracking_manager, webrtc_service);
                    if (auto tracked_frame = tracking_manager.buildTrackedFrame(frame);
                        tracked_frame && !tracked_frame->detections.empty()) {
                        webrtc_service.sendDetectionResult(tracked_frame);
                    }
                }
            };

            processFrames(camera_ids);
            processFrames(video_ids);

            if (!captured_any_frame) {
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }
        }

        std::cout << "[INFO] Stopping services..." << std::endl;
        if (detection_thread.joinable()) {
            detection_thread.join();
        }
        if (metrics_thread.joinable()) {
            metrics_thread.join();
        }
        if (processing_active) {
            camera_manager.stopAllCameras();
            inference_engine.stop();
        }
        webrtc_service.stop();
        std::cout << "[INFO] Service stopped." << std::endl;
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "[ERROR] Service failed: " << ex.what() << std::endl;
        return 1;
    }
}
