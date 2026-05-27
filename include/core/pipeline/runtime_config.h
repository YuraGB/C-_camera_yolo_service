#pragma once

#include <filesystem>
#include <string>

#include "platform/platform_services.h"
#include "webrtc_service.h"

namespace core::pipeline {

struct RuntimeConfig {
  std::filesystem::path model_path;
  std::filesystem::path test_video_path;
  WebRTCServiceConfig webrtc;
  int max_camera_scan = 10;
};

RuntimeConfig loadRuntimeConfig(const platform::PlatformServices& platform_services);

int readEnvInt(const char* name, int fallback);
bool readEnvBool(const char* name, bool fallback);
std::filesystem::path resolveExistingPath(
    const std::string& raw_path,
    const platform::PlatformServices& platform_services);

}  // namespace core::pipeline
