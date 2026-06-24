#include "webrtc_service.h"

#include <algorithm>
#include <cmath>
#include <iostream>

#include <nlohmann/json.hpp>
#include <rtc/h264rtppacketizer.hpp>

#include "webrtc_service_internal.h"

using namespace webrtc_service_internal;

void WebRTCService::sendFrame(const std::string& camera_id,
                              const std::shared_ptr<Frame>& frame) {
  if (!frame || frame->mat.empty() || !running_ || camera_id.empty()) {
    return;
  }

  std::shared_ptr<SourceStreamState> source_state;
  {
    std::lock_guard<std::mutex> lock(sources_mutex_);
    auto it = sources_.find(camera_id);
    if (it == sources_.end()) {
      return;
    }
    source_state = it->second;
  }

  {
    std::lock_guard<std::mutex> lock(source_state->frame_mutex);
    source_state->latest_frame = frame;
  }

  const int64_t now_ms = currentTimestampMs();
  const int64_t capture_delay_ms =
      (frame->timestamp > 0) ? std::max<int64_t>(0, now_ms - frame->timestamp) : 0;
  {
    std::lock_guard<std::mutex> lock(source_state->timeline_mutex);
    ++source_state->metrics_received_frames;
    source_state->metrics_capture_delay_sum_ms += capture_delay_ms;
    source_state->metrics_capture_delay_max_ms =
        std::max(source_state->metrics_capture_delay_max_ms, capture_delay_ms);
  }
  source_state->frame_cv.notify_one();
}

void WebRTCService::sendDetectionResult(const std::shared_ptr<Frame>& frame) {
  if (!frame || !running_) {
    return;
  }

  static std::atomic<int64_t> detection_log_counter{0};
  const auto log_index = ++detection_log_counter;
  if (config_.verbose_logging && (log_index % 30) == 1) {
    std::cout << "[WebRTC] Queueing detection_frame for camera_id="
              << frame->camera_id << ", detections=" << frame->detections.size()
              << ", timestamp=" << frame->timestamp << std::endl;
  }

  int target_width = 0;
  int target_height = 0;
  {
    std::lock_guard<std::mutex> lock(sources_mutex_);
    auto it = sources_.find(frame->camera_id);
    if (it != sources_.end() && it->second) {
      std::lock_guard<std::mutex> timeline_lock(it->second->timeline_mutex);
      target_width = it->second->latest_encoded_width;
      target_height = it->second->latest_encoded_height;
    }
  }

  broadcastDetectionMessage(buildDetectionMessage(*frame, target_width, target_height));
}

void WebRTCService::sendPipelineMetrics(const nlohmann::json& payload) {
  if (!running_) {
    return;
  }

  broadcastDetectionMessage(payload.dump());
}

std::string WebRTCService::buildDetectionMessage(
    const Frame& frame,
    int target_width,
    int target_height) const {
  nlohmann::json payload = {
      {"type", "detection_frame"},
      {"camera_id", frame.camera_id},
      {"timestamp", toRelativeSeconds(service_start_timestamp_ms_, frame.timestamp)},
      {"detections", nlohmann::json::array()},
  };

  const int source_width = frame.width();
  const int source_height = frame.height();
  const bool should_scale =
      source_width > 0 && source_height > 0 &&
      target_width > 0 && target_height > 0 &&
      (source_width != target_width || source_height != target_height);
  const double scale_x =
      should_scale ? static_cast<double>(target_width) / source_width : 1.0;
  const double scale_y =
      should_scale ? static_cast<double>(target_height) / source_height : 1.0;

  for (const auto& detection : frame.detections) {
    const int output_width = should_scale ? target_width : source_width;
    const int output_height = should_scale ? target_height : source_height;
    const int x = std::clamp(
        static_cast<int>(std::round(detection.bbox.x * scale_x)),
        0,
        output_width > 0 ? output_width : detection.bbox.x);
    const int y = std::clamp(
        static_cast<int>(std::round(detection.bbox.y * scale_y)),
        0,
        output_height > 0 ? output_height : detection.bbox.y);
    const int width = output_width > 0
        ? std::clamp(
              static_cast<int>(std::round(detection.bbox.width * scale_x)),
              0,
              std::max(0, output_width - x))
        : std::max(
              0,
              static_cast<int>(std::round(detection.bbox.width * scale_x)));
    const int height = output_height > 0
        ? std::clamp(
              static_cast<int>(std::round(detection.bbox.height * scale_y)),
              0,
              std::max(0, output_height - y))
        : std::max(
              0,
              static_cast<int>(std::round(detection.bbox.height * scale_y)));

    payload["detections"].push_back({
        {"label", detection.label},
        {"confidence", detection.confidence},
        {"bbox",
         {
             {"x", x},
             {"y", y},
             {"width", width},
             {"height", height},
         }},
    });
  }

  return payload.dump();
}

std::string WebRTCService::buildVideoPipelineMetricsMessage(
    const SourceStreamState& source_state,
    int64_t now_ms) const {
  const int64_t interval_ms = std::max<int64_t>(
      1, source_state.last_pipeline_metrics_sent_ms > 0
             ? now_ms - source_state.last_pipeline_metrics_sent_ms
             : config_.pipeline_metrics_interval_ms);
  const double interval_seconds = static_cast<double>(interval_ms) / 1000.0;
  const double avg_capture_delay_ms =
      source_state.metrics_received_frames > 0
          ? static_cast<double>(source_state.metrics_capture_delay_sum_ms) /
                static_cast<double>(source_state.metrics_received_frames)
          : 0.0;
  const double avg_encode_ms =
      source_state.metrics_encoded_frames > 0
          ? static_cast<double>(source_state.metrics_encode_sum_us) /
                static_cast<double>(source_state.metrics_encoded_frames) / 1000.0
          : 0.0;

  nlohmann::json payload = {
      {"type", "pipeline_metrics"},
      {"scope", "video"},
      {"camera_id", source_state.camera_id},
      {"interval_ms", interval_ms},
      {"capture_fps", static_cast<double>(source_state.metrics_received_frames) / interval_seconds},
      {"encode_fps", static_cast<double>(source_state.metrics_encoded_frames) / interval_seconds},
      {"avg_capture_delay_ms", avg_capture_delay_ms},
      {"max_capture_delay_ms", source_state.metrics_capture_delay_max_ms},
      {"avg_h264_encode_ms", avg_encode_ms},
      {"max_h264_encode_ms", static_cast<double>(source_state.metrics_encode_max_us) / 1000.0},
      {"dropped_stale_frames", source_state.metrics_dropped_stale_frames},
      {"total_dropped_stale_frames", source_state.dropped_stale_live_frames},
      {"estimated_live_fps", source_state.smoothed_live_fps},
  };

  return payload.dump();
}

std::string WebRTCService::buildVideoLatencySampleMessage(
    const SourceStreamState& source_state,
    const Frame& frame,
    int64_t encoded_timestamp_ms) const {
  nlohmann::json payload = {
      {"type", "video_latency_sample"},
      {"camera_id", frame.camera_id.empty() ? source_state.camera_id : frame.camera_id},
      {"frame_id", frame.frame_id},
      {"capture_timestamp_ms", frame.timestamp},
      {"encoded_timestamp_ms", encoded_timestamp_ms},
      {"sample_interval_ms", config_.video_latency_sample_interval_ms},
  };

  return payload.dump();
}

void WebRTCService::broadcastDetectionMessage(const std::string& message) {
  std::vector<std::shared_ptr<PeerSession>> sessions;
  {
    std::lock_guard<std::mutex> lock(sessions_mutex_);
    for (const auto& [_, session] : sessions_) {
      sessions.push_back(session);
    }
  }

  for (const auto& session : sessions) {
    if (!session || session->closing || session->closed ||
        !session->detection_channel || !session->detection_channel->isOpen()) {
      if (session && !session->logged_detection_channel_not_open) {
        std::cout << "[WebRTC] Dropping detection_frame for peer " << session->peer_id
                  << ": detection channel is not open yet" << std::endl;
        session->logged_detection_channel_not_open = true;
      }
      continue;
    }

    session->logged_detection_channel_not_open = false;
    if (session->detection_channel->bufferedAmount() >
        config_.max_detection_buffered_bytes) {
      std::cout << "[WebRTC] Skipping detection_frame for peer " << session->peer_id
                << ": bufferedAmount="
                << session->detection_channel->bufferedAmount() << std::endl;
      continue;
    }

    if (config_.verbose_logging) {
      std::cout << "[WebRTC] Sending detection_frame to peer " << session->peer_id
                << std::endl;
    }
    session->detection_channel->send(message);
  }
}

void WebRTCService::maybeSendVideoLatencySample(
    const std::shared_ptr<SourceStreamState>& source_state,
    const std::shared_ptr<Frame>& frame,
    int64_t encoded_timestamp_ms) {
  if (!source_state || !frame || config_.video_latency_sample_interval_ms <= 0) {
    return;
  }

  bool should_send = false;
  {
    std::lock_guard<std::mutex> lock(source_state->timeline_mutex);
    if (source_state->last_latency_sample_sent_ms < 0 ||
        encoded_timestamp_ms - source_state->last_latency_sample_sent_ms >=
            config_.video_latency_sample_interval_ms) {
      source_state->last_latency_sample_sent_ms = encoded_timestamp_ms;
      should_send = true;
    }
  }

  if (!should_send) {
    return;
  }

  broadcastDetectionMessage(
      buildVideoLatencySampleMessage(*source_state, *frame, encoded_timestamp_ms));
}

void WebRTCService::maybeSendVideoPipelineMetrics(
    const std::shared_ptr<SourceStreamState>& source_state,
    int64_t now_ms) {
  if (!source_state || config_.pipeline_metrics_interval_ms <= 0) {
    return;
  }

  std::string message;
  {
    std::lock_guard<std::mutex> lock(source_state->timeline_mutex);
    if (source_state->last_pipeline_metrics_sent_ms < 0) {
      source_state->last_pipeline_metrics_sent_ms = now_ms;
      return;
    }

    if (now_ms - source_state->last_pipeline_metrics_sent_ms <
        config_.pipeline_metrics_interval_ms) {
      return;
    }

    message = buildVideoPipelineMetricsMessage(*source_state, now_ms);
    source_state->last_pipeline_metrics_sent_ms = now_ms;
    source_state->metrics_received_frames = 0;
    source_state->metrics_encoded_frames = 0;
    source_state->metrics_dropped_stale_frames = 0;
    source_state->metrics_capture_delay_sum_ms = 0;
    source_state->metrics_capture_delay_max_ms = 0;
    source_state->metrics_encode_sum_us = 0;
    source_state->metrics_encode_max_us = 0;
  }

  broadcastDetectionMessage(message);
}

void WebRTCService::sourceLoop(
    const std::shared_ptr<SourceStreamState>& source_state) {
  while (running_ && source_state->running) {
    std::shared_ptr<Frame> frame;
    {
      std::unique_lock<std::mutex> lock(source_state->frame_mutex);
      source_state->frame_cv.wait(lock, [&]() {
        return !running_ || !source_state->running ||
               source_state->latest_frame != nullptr;
      });
      if (!running_ || !source_state->running) {
        break;
      }
      frame = source_state->latest_frame;
      source_state->latest_frame.reset();
    }

    if (!frame || frame->mat.empty()) {
      continue;
    }

    try {
      encodeAndBroadcastVideo(source_state, frame);
    } catch (const std::exception& error) {
      std::cerr << "[WebRTC] Video encode/send failed for camera_id="
                << source_state->camera_id << ": " << error.what() << std::endl;
    }
  }
}

void WebRTCService::encodeAndBroadcastVideo(
    const std::shared_ptr<SourceStreamState>& source_state,
    const std::shared_ptr<Frame>& frame) {
  if (!source_state || !source_state->encoder || !frame) {
    return;
  }

  const int64_t now_ms = currentTimestampMs();
  const int64_t live_lag_ms =
      (frame->timestamp > 0) ? std::max<int64_t>(0, now_ms - frame->timestamp) : 0;
  if (config_.max_live_latency_ms > 0 &&
      live_lag_ms > config_.max_live_latency_ms) {
    ++source_state->dropped_stale_live_frames;
    {
      std::lock_guard<std::mutex> lock(source_state->timeline_mutex);
      ++source_state->metrics_dropped_stale_frames;
    }
    if ((source_state->dropped_stale_live_frames % 30) == 1) {
      std::cout << "[WebRTC] Dropping stale frame for camera_id="
                << source_state->camera_id << ", lag=" << live_lag_ms
                << "ms threshold=" << config_.max_live_latency_ms << "ms"
                << std::endl;
    }
    return;
  }

  source_state->smoothed_live_fps =
      estimateLiveFps(source_state->smoothed_live_fps,
                      source_state->last_encoded_frame_timestamp_ms,
                      frame->timestamp);
  if (source_state->smoothed_live_fps > 0.0) {
    source_state->encoder->setTargetFrameRate(source_state->smoothed_live_fps);
  }

  cv::Mat live_mat = prepareLiveFrameForEncoding(
      frame->mat, config_.max_live_width, config_.max_live_height);
  {
    std::lock_guard<std::mutex> lock(source_state->timeline_mutex);
    source_state->latest_encoded_width = live_mat.cols;
    source_state->latest_encoded_height = live_mat.rows;
  }

  const bool force_idr =
      shouldForceKeyframe(source_state->encoded_frame_count,
                          source_state->smoothed_live_fps);
  const auto encode_started = std::chrono::steady_clock::now();
  auto bitstream = source_state->encoder->encode(live_mat, frame->timestamp, force_idr);
  const auto encode_us = std::chrono::duration_cast<std::chrono::microseconds>(
      std::chrono::steady_clock::now() - encode_started).count();
  if (bitstream.empty()) {
    return;
  }

  {
    std::lock_guard<std::mutex> lock(source_state->timeline_mutex);
    ++source_state->metrics_encoded_frames;
    source_state->metrics_encode_sum_us += encode_us;
    source_state->metrics_encode_max_us =
        std::max<int64_t>(source_state->metrics_encode_max_us, encode_us);
  }

  int64_t first_live_timestamp_ms = 0;
  {
    std::lock_guard<std::mutex> lock(source_state->timeline_mutex);
    if (source_state->first_live_timestamp_ms < 0) {
      source_state->first_live_timestamp_ms = frame->timestamp;
    }
    first_live_timestamp_ms = source_state->first_live_timestamp_ms;
  }

  const double seconds = toRelativeSeconds(first_live_timestamp_ms, frame->timestamp);
  const uint32_t rtp_timestamp = rtc::RtpPacketizationConfig::getTimestampFromSeconds(
      seconds, rtc::H264RtpPacketizer::defaultClockRate);

  struct TrackSender {
    std::shared_ptr<rtc::Track> track;
    std::shared_ptr<rtc::RtpPacketizationConfig> rtp_config;
  };

  std::vector<TrackSender> tracks;
  {
    std::lock_guard<std::mutex> lock(sessions_mutex_);
    for (const auto& [_, session] : sessions_) {
      auto it = session->video_tracks.find(source_state->camera_id);
      if (it != session->video_tracks.end() && it->second && it->second->isOpen()) {
        auto config_it = session->video_rtp_configs.find(source_state->camera_id);
        if (config_it != session->video_rtp_configs.end() && config_it->second) {
          tracks.push_back({it->second, config_it->second});
        }
      }
    }
  }

  for (const auto& sender : tracks) {
    sender.rtp_config->timestamp = rtp_timestamp;
    sender.track->send(reinterpret_cast<const rtc::byte*>(bitstream.data()),
                       bitstream.size());
  }

  maybeSendVideoLatencySample(source_state, frame, currentTimestampMs());
  maybeSendVideoPipelineMetrics(source_state, currentTimestampMs());
  source_state->last_encoded_frame_timestamp_ms = frame->timestamp;
  ++source_state->encoded_frame_count;
}

void WebRTCService::startSourceWorker(
    const std::shared_ptr<SourceStreamState>& source_state) {
  if (!source_state || source_state->running) {
    return;
  }

  source_state->encoder =
      std::make_unique<OpenH264Encoder>(config_.openh264_dll_path);
  if (!source_state->encoder->isReady()) {
    throw std::runtime_error("OpenH264 encoder is not ready for camera_id=" +
                             source_state->camera_id);
  }

  source_state->running = true;
  source_state->worker_thread =
      std::thread(&WebRTCService::sourceLoop, this, source_state);
}

void WebRTCService::stopSourceWorker(
    const std::shared_ptr<SourceStreamState>& source_state) {
  if (!source_state || !source_state->running) {
    return;
  }

  source_state->running = false;
  source_state->frame_cv.notify_all();
  if (source_state->worker_thread.joinable()) {
    source_state->worker_thread.join();
  }
  source_state->encoder.reset();
  {
    std::lock_guard<std::mutex> lock(source_state->timeline_mutex);
    source_state->first_live_timestamp_ms = -1;
    source_state->dropped_stale_live_frames = 0;
    source_state->last_encoded_frame_timestamp_ms = -1;
    source_state->last_latency_sample_sent_ms = -1;
    source_state->last_pipeline_metrics_sent_ms = -1;
    source_state->latest_encoded_width = 0;
    source_state->latest_encoded_height = 0;
    source_state->smoothed_live_fps = 0.0;
    source_state->encoded_frame_count = 0;
    source_state->metrics_received_frames = 0;
    source_state->metrics_encoded_frames = 0;
    source_state->metrics_dropped_stale_frames = 0;
    source_state->metrics_capture_delay_sum_ms = 0;
    source_state->metrics_capture_delay_max_ms = 0;
    source_state->metrics_encode_sum_us = 0;
    source_state->metrics_encode_max_us = 0;
  }
}
