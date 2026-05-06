#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cctype>
#include <random>
#include <string>

#include <opencv2/opencv.hpp>
#include <rtc/rtc.hpp>

namespace webrtc_service_internal {

inline int64_t currentTimestampMs() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

inline double toRelativeSeconds(int64_t start_ms, int64_t frame_ms) {
  if (frame_ms <= 0 || start_ms <= 0) {
    return 0.0;
  }

  const double seconds =
      static_cast<double>(std::max<int64_t>(0, frame_ms - start_ms)) / 1000.0;
  return std::round(seconds * 1000.0) / 1000.0;
}

inline rtc::DataChannelInit makeDetectionChannelInit() {
  rtc::DataChannelInit init;
  // Prefer the browser-default reliable/ordered mode for widest interop.
  // Detection messages are small, and compatibility is more important here
  // than partial-reliability tuning.
  return init;
}

inline bool isPeerTerminal(rtc::PeerConnection::State state) {
  return state == rtc::PeerConnection::State::Disconnected ||
         state == rtc::PeerConnection::State::Failed ||
         state == rtc::PeerConnection::State::Closed;
}

inline uint32_t randomSsrc() {
  static std::mt19937 generator{std::random_device{}()};
  static std::uniform_int_distribution<uint32_t> distribution;
  return distribution(generator);
}

inline std::string sanitizeMid(const std::string& camera_id) {
  std::string mid = "cam_";
  mid.reserve(camera_id.size() + 4);
  for (unsigned char ch : camera_id) {
    if (std::isalnum(ch) || ch == '_' || ch == '-') {
      mid.push_back(static_cast<char>(ch));
    } else {
      mid.push_back('_');
    }
  }
  return mid;
}

inline double estimateLiveFps(double current_fps, int64_t previous_timestamp_ms,
                              int64_t current_timestamp_ms) {
  if (previous_timestamp_ms <= 0 || current_timestamp_ms <= previous_timestamp_ms) {
    return current_fps;
  }

  const double delta_ms =
      static_cast<double>(current_timestamp_ms - previous_timestamp_ms);
  if (delta_ms < 1.0) {
    return current_fps;
  }

  const double instant_fps = std::clamp(1000.0 / delta_ms, 1.0, 60.0);
  if (current_fps <= 0.0) {
    return instant_fps;
  }

  constexpr double kAlpha = 0.2;
  return (current_fps * (1.0 - kAlpha)) + (instant_fps * kAlpha);
}

inline bool shouldForceKeyframe(int64_t encoded_frame_count, double live_fps) {
  const int64_t keyframe_interval_frames = std::max<int64_t>(
      15, static_cast<int64_t>(std::llround(std::max(10.0, live_fps) * 2.0)));
  return keyframe_interval_frames > 0 &&
         (encoded_frame_count % keyframe_interval_frames) == 0;
}

inline cv::Mat prepareLiveFrameForEncoding(const cv::Mat& input, int max_width,
                                           int max_height) {
  if (input.empty() || max_width <= 0 || max_height <= 0) {
    return input;
  }

  if (input.cols <= max_width && input.rows <= max_height) {
    return input;
  }

  const double scale_x =
      static_cast<double>(max_width) / static_cast<double>(input.cols);
  const double scale_y =
      static_cast<double>(max_height) / static_cast<double>(input.rows);
  const double scale = std::min(scale_x, scale_y);
  if (scale >= 1.0) {
    return input;
  }

  int target_width = std::max(2, static_cast<int>(std::round(input.cols * scale)));
  int target_height =
      std::max(2, static_cast<int>(std::round(input.rows * scale)));
  if ((target_width % 2) != 0) {
    --target_width;
  }
  if ((target_height % 2) != 0) {
    --target_height;
  }

  cv::Mat resized;
  cv::resize(input, resized, cv::Size(target_width, target_height), 0.0, 0.0,
             cv::INTER_AREA);
  return resized;
}

constexpr auto kReconnectBaseDelay = std::chrono::milliseconds(500);
constexpr auto kReconnectMaxDelay = std::chrono::milliseconds(10000);

}  // namespace webrtc_service_internal
