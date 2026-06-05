#include "core/pipeline/runtime_config.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <vector>

namespace core::pipeline {
namespace {

std::string readEnvString(const char* name, const std::string& fallback) {
  const char* raw = std::getenv(name);
  if (!raw || std::strlen(raw) == 0) {
    return fallback;
  }

  return raw;
}

std::optional<std::string> readOptionalEnvString(const char* name) {
  const char* raw = std::getenv(name);
  if (!raw || std::strlen(raw) == 0) {
    return std::nullopt;
  }

  return std::string(raw);
}

}  // namespace

int readEnvInt(const char* name, int fallback) {
  if (const char* raw = std::getenv(name)) {
    try {
      return std::stoi(raw);
    } catch (...) {
    }
  }

  return fallback;
}

bool readEnvBool(const char* name, bool fallback) {
  if (const char* raw = std::getenv(name)) {
    const std::string value(raw);
    return value == "1" || value == "true" || value == "TRUE" ||
           value == "yes" || value == "YES";
  }

  return fallback;
}

std::filesystem::path resolveExistingPath(
    const std::string& raw_path,
    const platform::PlatformServices& platform_services) {
  if (raw_path.empty()) {
    return {};
  }

  const std::filesystem::path input(raw_path);
  if (input.is_absolute() && std::filesystem::exists(input)) {
    return input;
  }

  std::vector<std::filesystem::path> candidates;
  candidates.push_back(input);

  const auto cwd = std::filesystem::current_path();
  candidates.push_back(cwd / input);

  const auto exe_dir = platform_services.executableDir();
  if (!exe_dir.empty()) {
    candidates.push_back(exe_dir / input);
    candidates.push_back(exe_dir.parent_path() / input);
    candidates.push_back(exe_dir.parent_path().parent_path() / input);
  }

  const auto source_root = platform_services.sourceRootHint();
  if (!source_root.empty()) {
    candidates.push_back(source_root / input);
  }

  for (const auto& candidate : candidates) {
    std::error_code ec;
    if (std::filesystem::exists(candidate, ec) && !ec) {
      return std::filesystem::weakly_canonical(candidate, ec);
    }
  }

  return input;
}

RuntimeConfig loadRuntimeConfig(const platform::PlatformServices& platform_services) {
  RuntimeConfig config;
  config.model_path = resolveExistingPath(
      readEnvString("CAMERA_MODEL_PATH", "models/yolov8x.onnx"),
      platform_services);
  if (const auto test_video_env = readOptionalEnvString("CAMERA_TEST_VIDEO_PATH")) {
    config.test_video_path = resolveExistingPath(*test_video_env, platform_services);
  } else {
    config.test_video_path = resolveExistingPath("media/test_video.mp4", platform_services);
    if (!std::filesystem::exists(config.test_video_path)) {
      config.test_video_path = resolveExistingPath("test_video.mp4", platform_services);
    }
  }

  config.max_camera_scan = std::clamp(readEnvInt("CAMERA_MAX_CAMERA_SCAN", 10), 0, 64);

  config.webrtc.signaling_url = readEnvString("CAMERA_SIGNALING_URL", platform_services.defaultSignalingUrl());
  config.webrtc.local_peer_id = readEnvString("CAMERA_PEER_ID", "camera-cv-service");
  config.webrtc.remote_peer_id = readOptionalEnvString("CAMERA_REMOTE_PEER_ID");
  config.webrtc.auth_jwt_secret = readEnvString("CAMERA_AUTH_JWT_SECRET", "");
  config.webrtc.auth_jwt_issuer = readEnvString("CAMERA_AUTH_JWT_ISSUER", "camera-cv-service");
  config.webrtc.auth_jwt_audience = readEnvString("CAMERA_AUTH_JWT_AUDIENCE", "signaling");
  config.webrtc.auth_jwt_role = readEnvString("CAMERA_AUTH_JWT_ROLE", "service");
  config.webrtc.auth_jwt_email = readOptionalEnvString("CAMERA_AUTH_JWT_EMAIL");
  config.webrtc.auth_jwt_ttl_seconds = std::clamp(
      readEnvInt("CAMERA_AUTH_JWT_TTL_SECONDS", 300),
      30,
      24 * 60 * 60);
  config.webrtc.ice_servers = {"stun:stun.l.google.com:19302"};
  config.webrtc.max_live_latency_ms = readEnvInt("CAMERA_MAX_LIVE_LATENCY_MS", 150);
  config.webrtc.max_live_width = readEnvInt("CAMERA_MAX_LIVE_WIDTH", 1280);
  config.webrtc.max_live_height = readEnvInt("CAMERA_MAX_LIVE_HEIGHT", 720);
  config.webrtc.video_latency_sample_interval_ms = std::clamp(
      readEnvInt("CAMERA_VIDEO_LATENCY_SAMPLE_INTERVAL_MS", 1000), 100, 60000);
  config.webrtc.pipeline_metrics_interval_ms = std::clamp(
      readEnvInt("CAMERA_PIPELINE_METRICS_INTERVAL_MS", 1000), 100, 60000);
  config.webrtc.max_detection_buffered_bytes =
      static_cast<size_t>(readEnvInt("CAMERA_MAX_DETECTION_BUFFERED_BYTES", 128 * 1024));
  config.webrtc.verbose_logging = readEnvBool("CAMERA_VERBOSE_LOGS", false);

  const std::string openh264_library = readEnvString(
      "CAMERA_OPENH264_LIBRARY",
      platform_services.defaultOpenH264LibraryName());
  config.webrtc.openh264_dll_path = resolveExistingPath(openh264_library, platform_services).string();

  return config;
}

}  // namespace core::pipeline
