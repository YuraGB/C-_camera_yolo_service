#include "webrtc_service.h"

#include <iostream>

#include <nlohmann/json.hpp>

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
  source_state->frame_cv.notify_one();
}

void WebRTCService::sendDetectionResult(const std::shared_ptr<Frame>& frame) {
  if (!frame || !running_) {
    return;
  }

  static std::atomic<int64_t> detection_log_counter{0};
  const auto log_index = ++detection_log_counter;
  if ((log_index % 30) == 1) {
    std::cout << "[WebRTC] Queueing detection_frame for camera_id="
              << frame->camera_id << ", detections=" << frame->detections.size()
              << ", timestamp=" << frame->timestamp << std::endl;
  }
  broadcastDetectionMessage(buildDetectionMessage(*frame));
}

std::string WebRTCService::buildDetectionMessage(const Frame& frame) const {
  nlohmann::json payload = {
      {"type", "detection_frame"},
      {"camera_id", frame.camera_id},
      {"timestamp", toRelativeSeconds(service_start_timestamp_ms_, frame.timestamp)},
      {"detections", nlohmann::json::array()},
  };

  for (const auto& detection : frame.detections) {
    payload["detections"].push_back({
        {"label", detection.label},
        {"confidence", detection.confidence},
        {"bbox",
         {
             {"x", detection.bbox.x},
             {"y", detection.bbox.y},
             {"width", detection.bbox.width},
             {"height", detection.bbox.height},
         }},
    });
  }

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
    if (!session || !session->detection_channel) {
      continue;
    }
    if (!session->detection_channel->isOpen()) {
      std::cout << "[WebRTC] Skipping detection_frame for peer " << session->peer_id
                << ": detection channel is not open" << std::endl;
      continue;
    }
    if (session->detection_channel->bufferedAmount() >
        config_.max_detection_buffered_bytes) {
      std::cout << "[WebRTC] Skipping detection_frame for peer " << session->peer_id
                << ": bufferedAmount="
                << session->detection_channel->bufferedAmount() << std::endl;
      continue;
    }

    std::cout << "[WebRTC] Sending detection_frame to peer " << session->peer_id
              << std::endl;
    session->detection_channel->send(message);
  }
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

  const bool force_idr =
      shouldForceKeyframe(source_state->encoded_frame_count,
                          source_state->smoothed_live_fps);
  auto bitstream = source_state->encoder->encode(live_mat, frame->timestamp, force_idr);
  if (bitstream.empty()) {
    return;
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
  rtc::FrameInfo info{std::chrono::duration<double>(seconds)};

  std::vector<std::shared_ptr<rtc::Track>> tracks;
  {
    std::lock_guard<std::mutex> lock(sessions_mutex_);
    for (const auto& [_, session] : sessions_) {
      auto it = session->video_tracks.find(source_state->camera_id);
      if (it != session->video_tracks.end() && it->second && it->second->isOpen()) {
        tracks.push_back(it->second);
      }
    }
  }

  for (const auto& track : tracks) {
    track->sendFrame(reinterpret_cast<const rtc::byte*>(bitstream.data()),
                     bitstream.size(), info);
  }

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
    source_state->smoothed_live_fps = 0.0;
    source_state->encoded_frame_count = 0;
  }
}
