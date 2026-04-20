#include "webrtc_service.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <utility>

#include <nlohmann/json.hpp>
#include <rtc/h264rtppacketizer.hpp>
#include <rtc/rtcpnackresponder.hpp>
#include <rtc/rtcpsrreporter.hpp>

namespace {
int64_t currentTimestampMs() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

double toRelativeSeconds(int64_t start_ms, int64_t frame_ms) {
  if (frame_ms <= 0 || start_ms <= 0) {
    return 0.0;
  }

  const double seconds = static_cast<double>(std::max<int64_t>(0, frame_ms - start_ms)) / 1000.0;
  return std::round(seconds * 1000.0) / 1000.0;
}

rtc::DataChannelInit makeDetectionChannelInit() {
  rtc::DataChannelInit init;
  init.reliability.unordered = true;
  init.reliability.maxPacketLifeTime = std::chrono::milliseconds(1500);
  return init;
}

bool isPeerTerminal(rtc::PeerConnection::State state) {
  return state == rtc::PeerConnection::State::Disconnected ||
         state == rtc::PeerConnection::State::Failed ||
         state == rtc::PeerConnection::State::Closed;
}

uint32_t randomSsrc() {
  static std::mt19937 generator{std::random_device{}()};
  static std::uniform_int_distribution<uint32_t> distribution;
  return distribution(generator);
}

double estimateLiveFps(double current_fps, int64_t previous_timestamp_ms, int64_t current_timestamp_ms) {
  if (previous_timestamp_ms <= 0 || current_timestamp_ms <= previous_timestamp_ms) {
    return current_fps;
  }

  const double delta_ms = static_cast<double>(current_timestamp_ms - previous_timestamp_ms);
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

bool shouldForceKeyframe(int64_t encoded_frame_count, double live_fps) {
  const int64_t keyframe_interval_frames =
      std::max<int64_t>(15, static_cast<int64_t>(std::llround(std::max(10.0, live_fps) * 2.0)));
  return keyframe_interval_frames > 0 && (encoded_frame_count % keyframe_interval_frames) == 0;
}

cv::Mat prepareLiveFrameForEncoding(
    const cv::Mat& input,
    int max_width,
    int max_height) {
  if (input.empty() || max_width <= 0 || max_height <= 0) {
    return input;
  }

  if (input.cols <= max_width && input.rows <= max_height) {
    return input;
  }

  const double scale_x = static_cast<double>(max_width) / static_cast<double>(input.cols);
  const double scale_y = static_cast<double>(max_height) / static_cast<double>(input.rows);
  const double scale = std::min(scale_x, scale_y);

  if (scale >= 1.0) {
    return input;
  }

  int target_width = std::max(2, static_cast<int>(std::round(input.cols * scale)));
  int target_height = std::max(2, static_cast<int>(std::round(input.rows * scale)));
  if ((target_width % 2) != 0) {
    --target_width;
  }
  if ((target_height % 2) != 0) {
    --target_height;
  }

  cv::Mat resized;
  cv::resize(input, resized, cv::Size(target_width, target_height), 0.0, 0.0, cv::INTER_AREA);
  return resized;
}
}  // namespace

WebRTCService::WebRTCService(WebRTCServiceConfig config) : config_(std::move(config)) {}

WebRTCService::~WebRTCService() {
  stop();
}

void WebRTCService::start() {
  if (running_) {
    return;
  }

  if (config_.signaling_url.empty()) {
    throw std::runtime_error("WebRTC signaling URL is required");
  }
  if (config_.local_peer_id.empty()) {
    throw std::runtime_error("WebRTC local peer id is required");
  }

  service_start_timestamp_ms_ = currentTimestampMs();
  running_ = true;
  video_encoder_ = std::make_unique<OpenH264Encoder>(config_.openh264_dll_path);
  if (!video_encoder_->isReady()) {
    throw std::runtime_error("OpenH264 encoder is not ready");
  }

  rtc::WebSocket::Configuration ws_config;
  ws_config.connectionTimeout = std::chrono::seconds(5);
  ws_config.pingInterval = std::chrono::seconds(10);

  signaling_socket_ = std::make_shared<rtc::WebSocket>(ws_config);
  signaling_socket_->onOpen([this]() {
    std::cout << "[WebRTC] Signaling websocket connected: " << config_.signaling_url << std::endl;

    nlohmann::json register_message = {
        {"type", "register"},
        {"peerId", config_.local_peer_id},
    };
    sendSignalingJson(register_message.dump());
    flushPendingSignalingMessages();

    if (config_.remote_peer_id && !config_.remote_peer_id->empty()) {
      createOfferForPeer(*config_.remote_peer_id);
    }
  });

  signaling_socket_->onClosed([this]() {
    std::cout << "[WebRTC] Signaling websocket closed" << std::endl;
  });

  signaling_socket_->onError([](const std::string& error) {
    std::cerr << "[WebRTC] Signaling websocket error: " << error << std::endl;
  });

  signaling_socket_->onMessage([this](rtc::message_variant data) {
    if (!std::holds_alternative<std::string>(data)) {
      return;
    }
    handleSignalingMessage(std::get<std::string>(data));
  });

  std::cout << "[WebRTC] Connecting signaling websocket to " << config_.signaling_url << std::endl;
  signaling_socket_->open(config_.signaling_url);

  video_thread_ = std::thread(&WebRTCService::videoLoop, this);
}

void WebRTCService::stop() {
  if (!running_) {
    return;
  }

  running_ = false;
  live_frame_cv_.notify_all();

  if (signaling_socket_) {
    signaling_socket_->close();
    signaling_socket_.reset();
  }

  std::vector<std::shared_ptr<PeerSession>> sessions;
  {
    std::lock_guard<std::mutex> lock(sessions_mutex_);
    for (auto& [_, session] : sessions_) {
      sessions.push_back(session);
    }
    sessions_.clear();
  }

  for (const auto& session : sessions) {
    if (!session) {
      continue;
    }
    if (session->detection_channel) {
      session->detection_channel->close();
    }
    if (session->live_track) {
      session->live_track->close();
    }
    if (session->peer_connection) {
      session->peer_connection->close();
    }
  }

  if (video_thread_.joinable()) {
    video_thread_.join();
  }

  video_encoder_.reset();
  {
    std::lock_guard<std::mutex> lock(live_timeline_mutex_);
    first_live_timestamp_ms_ = -1;
    dropped_stale_live_frames_ = 0;
    last_encoded_frame_timestamp_ms_ = -1;
    smoothed_live_fps_ = 0.0;
  }

  std::cout << "[WebRTC] Service stopped" << std::endl;
}

void WebRTCService::sendLiveFrame(const std::shared_ptr<Frame>& frame) {
  if (!frame || frame->mat.empty() || !running_) {
    return;
  }

  {
    std::lock_guard<std::mutex> lock(live_frame_mutex_);
    latest_live_frame_ = frame;
  }
  live_frame_cv_.notify_one();
}

void WebRTCService::sendDetectionResult(const std::shared_ptr<Frame>& frame) {
  if (!frame || !running_) {
    return;
  }

  broadcastDetectionMessage(buildDetectionMessage(*frame));
}

void WebRTCService::createOfferForPeer(const std::string& peer_id) {
  if (peer_id.empty()) {
    return;
  }

  auto session = createPeerSession(peer_id);
  configurePeerSession(session, true);
  session->peer_connection->setLocalDescription(rtc::Description::Type::Offer);
  std::cout << "[WebRTC] Creating offer for peer " << peer_id << std::endl;
}

void WebRTCService::handleSignalingMessage(const std::string& message) {
  nlohmann::json json;
  try {
    json = nlohmann::json::parse(message);
  } catch (const std::exception& error) {
    std::cerr << "[WebRTC] Failed to parse signaling message: " << error.what() << std::endl;
    return;
  }

  const std::string type = json.value("type", "");
  const std::string target_peer_id = json.value("targetPeerId", "");
  if (!target_peer_id.empty() && target_peer_id != config_.local_peer_id) {
    return;
  }

  auto peer_id = extractPeerId(json);
  if (!peer_id || peer_id->empty()) {
    std::cerr << "[WebRTC] Signaling message is missing peer id: " << message << std::endl;
    return;
  }

  if (type == "offer-request" || type == "viewer-join" || type == "connect") {
      std::cout << "[WebRTC] Trigger createOfferForPeer, type: " << type
            << ", peer_id: " << *peer_id << std::endl;
            
    createOfferForPeer(*peer_id);
    return;
  }

  std::shared_ptr<PeerSession> session;
  {
    std::lock_guard<std::mutex> lock(sessions_mutex_);
    auto it = sessions_.find(*peer_id);
    if (it != sessions_.end()) {
      session = it->second;
    }
  }

  if (!session) {
    session = createPeerSession(*peer_id);
    configurePeerSession(session, false);
  }

  if (type == "answer") {
    try {
      session->peer_connection->setRemoteDescription(
          rtc::Description(json.at("sdp").get<std::string>(), "answer"));
      std::cout << "[WebRTC] Applied remote answer from " << *peer_id << std::endl;
    } catch (const std::exception& error) {
      std::cerr << "[WebRTC] Failed to apply remote answer from " << *peer_id
                << ": " << error.what() << std::endl;
    }
    return;
  }

  if (type == "offer") {
    try {
      session->peer_connection->setRemoteDescription(
          rtc::Description(json.at("sdp").get<std::string>(), "offer"));
      session->peer_connection->setLocalDescription(rtc::Description::Type::Answer);
      std::cout << "[WebRTC] Received remote offer from " << *peer_id << std::endl;
    } catch (const std::exception& error) {
      std::cerr << "[WebRTC] Failed to handle remote offer from " << *peer_id
                << ": " << error.what() << std::endl;
    }
    return;
  }

  if (type == "ice-candidate") {
    try {
      const std::string candidate = json.at("candidate").get<std::string>();
      const std::string mid = json.value("mid", "");
      session->peer_connection->addRemoteCandidate(rtc::Candidate(candidate, mid));
    } catch (const std::exception& error) {
      std::cerr << "[WebRTC] Failed to add remote ICE candidate from " << *peer_id
                << ": " << error.what() << std::endl;
    }
  }
}

std::shared_ptr<WebRTCService::PeerSession> WebRTCService::createPeerSession(const std::string& peer_id) {
  std::lock_guard<std::mutex> lock(sessions_mutex_);
  auto it = sessions_.find(peer_id);
  if (it != sessions_.end()) {
    return it->second;
  }

  rtc::Configuration configuration;
  configuration.disableAutoNegotiation = false;
  configuration.maxMessageSize = 2 * 1024 * 1024;
  for (const auto& ice_server : config_.ice_servers) {
    if (!ice_server.empty()) {
      configuration.iceServers.emplace_back(ice_server);
    }
  }

  auto session = std::make_shared<PeerSession>();
  session->peer_id = peer_id;
  session->peer_connection = std::make_shared<rtc::PeerConnection>(configuration);
  sessions_.emplace(peer_id, session);
  return session;
}

void WebRTCService::configurePeerSession(const std::shared_ptr<PeerSession>& session, bool create_local_channels) {
  if (!session || !session->peer_connection) {
    return;
  }

  bool expected = false;
  if (!session->configured.compare_exchange_strong(expected, true)) {
    return;
  }

  session->peer_connection->onStateChange([this, peer_id = session->peer_id](rtc::PeerConnection::State state) {
    std::cout << "[WebRTC] Peer " << peer_id << " state: " << state << std::endl;
    if (isPeerTerminal(state)) {
      cleanupPeerSession(peer_id);
    }
  });

  session->peer_connection->onLocalDescription([this, peer_id = session->peer_id](rtc::Description description) {
    nlohmann::json message = {
        {"type", description.typeString()},
        {"peerId", config_.local_peer_id},
        {"targetPeerId", peer_id},
        {"sdp", std::string(description)},
    };
    sendSignalingJson(message.dump());
  });

  session->peer_connection->onLocalCandidate([this, peer_id = session->peer_id](rtc::Candidate candidate) {
    nlohmann::json message = {
        {"type", "ice-candidate"},
        {"peerId", config_.local_peer_id},
        {"targetPeerId", peer_id},
        {"candidate", candidate.candidate()},
        {"mid", candidate.mid()},
    };
    sendSignalingJson(message.dump());
  });

  session->peer_connection->onDataChannel(
      [this, weak_session = std::weak_ptr<PeerSession>(session)](std::shared_ptr<rtc::DataChannel> channel) {
        if (auto locked = weak_session.lock()) {
          attachDataChannel(locked, channel);
        }
      });

  attachVideoTrack(session);

  if (!session->detection_channel) {
    session->detection_channel =
        session->peer_connection->createDataChannel(config_.detection_channel_label, makeDetectionChannelInit());
    attachDataChannel(session, session->detection_channel);
  }

  if (!create_local_channels) {
    std::cout << "[WebRTC] Prepared answerer session for peer " << session->peer_id
              << " with local video track and detection channel" << std::endl;
  }
}

void WebRTCService::attachDataChannel(
    const std::shared_ptr<PeerSession>& session,
    const std::shared_ptr<rtc::DataChannel>& channel) {
  if (!session || !channel) {
    return;
  }

  if (channel->label() == config_.detection_channel_label) {
    session->detection_channel = channel;
  }

  channel->onOpen([peer_id = session->peer_id, label = channel->label()]() {
    std::cout << "[WebRTC] DataChannel opened for " << peer_id << ": " << label << std::endl;
  });

  channel->onClosed([peer_id = session->peer_id, label = channel->label()]() {
    std::cout << "[WebRTC] DataChannel closed for " << peer_id << ": " << label << std::endl;
  });

  channel->onError([peer_id = session->peer_id, label = channel->label()](std::string error) {
    std::cerr << "[WebRTC] DataChannel error for " << peer_id << " (" << label << "): " << error << std::endl;
  });

  channel->onMessage([peer_id = session->peer_id, label = channel->label()](rtc::message_variant data) {
    if (!std::holds_alternative<std::string>(data)) {
      return;
    }
    std::cout << "[WebRTC] Message from " << peer_id << " on " << label
              << ": " << std::get<std::string>(data) << std::endl;
  });
}

void WebRTCService::attachVideoTrack(const std::shared_ptr<PeerSession>& session) {
  if (!session || !session->peer_connection) {
    return;
  }

  session->video_ssrc = randomSsrc();
  const std::string cname = "camera-cv-video";
  const std::string msid = "camera-cv-stream";
  constexpr uint8_t payload_type = 102;

  auto video = rtc::Description::Video(config_.live_track_label);
  video.addH264Codec(payload_type);
  video.addSSRC(session->video_ssrc, cname, msid, config_.live_track_label);

  auto track = session->peer_connection->addTrack(video);
  auto rtp_config = std::make_shared<rtc::RtpPacketizationConfig>(
      session->video_ssrc,
      cname,
      payload_type,
      rtc::H264RtpPacketizer::ClockRate);
  auto packetizer = std::make_shared<rtc::H264RtpPacketizer>(
      rtc::NalUnit::Separator::StartSequence,
      rtp_config);
  auto sr_reporter = std::make_shared<rtc::RtcpSrReporter>(rtp_config);
  auto nack_responder = std::make_shared<rtc::RtcpNackResponder>();
  packetizer->addToChain(sr_reporter);
  packetizer->addToChain(nack_responder);
  track->setMediaHandler(packetizer);

  track->onOpen([peer_id = session->peer_id]() {
    std::cout << "[WebRTC] Video track opened for " << peer_id << std::endl;
  });
  track->onClosed([peer_id = session->peer_id]() {
    std::cout << "[WebRTC] Video track closed for " << peer_id << std::endl;
  });
  track->onError([peer_id = session->peer_id](std::string error) {
    std::cerr << "[WebRTC] Video track error for " << peer_id << ": " << error << std::endl;
  });

  session->live_track = track;
}

void WebRTCService::cleanupPeerSession(const std::string& peer_id) {
  std::shared_ptr<PeerSession> session;
  {
    std::lock_guard<std::mutex> lock(sessions_mutex_);
    auto it = sessions_.find(peer_id);
    if (it == sessions_.end()) {
      return;
    }
    session = it->second;
    sessions_.erase(it);
  }

  if (session->detection_channel) {
    session->detection_channel->close();
  }
  if (session->live_track) {
    session->live_track->close();
  }
  if (session->peer_connection) {
    session->peer_connection->close();
  }
}

std::string WebRTCService::buildDetectionMessage(const Frame& frame) const {
  nlohmann::json payload = {
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

void WebRTCService::videoLoop() {
  while (running_) {
    std::shared_ptr<Frame> frame;
    {
      std::unique_lock<std::mutex> lock(live_frame_mutex_);
      live_frame_cv_.wait(lock, [this]() {
        return !running_ || latest_live_frame_ != nullptr;
      });
      if (!running_) {
        break;
      }
      frame = latest_live_frame_;
      latest_live_frame_.reset();
    }

    if (!frame || frame->mat.empty()) {
      continue;
    }

    try {
      encodeAndBroadcastVideo(frame);
    } catch (const std::exception& error) {
      std::cerr << "[WebRTC] Video encode/send failed: " << error.what() << std::endl;
    }
  }
}

void WebRTCService::encodeAndBroadcastVideo(const std::shared_ptr<Frame>& frame) {
  if (!video_encoder_ || !frame) {
    return;
  }

  const int64_t now_ms = currentTimestampMs();
  const int64_t live_lag_ms = (frame->timestamp > 0) ? std::max<int64_t>(0, now_ms - frame->timestamp) : 0;
  if (config_.max_live_latency_ms > 0 && live_lag_ms > config_.max_live_latency_ms) {
    ++dropped_stale_live_frames_;
    if ((dropped_stale_live_frames_ % 30) == 1) {
      std::cout << "[WebRTC] Dropping stale live frame, lag=" << live_lag_ms
                << "ms threshold=" << config_.max_live_latency_ms << "ms" << std::endl;
    }
    return;
  }

  smoothed_live_fps_ =
      estimateLiveFps(smoothed_live_fps_, last_encoded_frame_timestamp_ms_, frame->timestamp);
  if (smoothed_live_fps_ > 0.0) {
    video_encoder_->setTargetFrameRate(smoothed_live_fps_);
  }

  cv::Mat live_mat = prepareLiveFrameForEncoding(
      frame->mat,
      config_.max_live_width,
      config_.max_live_height);

  const bool force_idr = shouldForceKeyframe(encoded_frame_count_, smoothed_live_fps_);
  auto bitstream = video_encoder_->encode(live_mat, frame->timestamp, force_idr);
  if (bitstream.empty()) {
    return;
  }

  int64_t first_live_timestamp_ms = 0;
  {
    std::lock_guard<std::mutex> lock(live_timeline_mutex_);
    if (first_live_timestamp_ms_ < 0) {
      first_live_timestamp_ms_ = frame->timestamp;
    }
    first_live_timestamp_ms = first_live_timestamp_ms_;
  }

  const double seconds = toRelativeSeconds(first_live_timestamp_ms, frame->timestamp);
  rtc::FrameInfo info{std::chrono::duration<double>(seconds)};

  std::vector<std::shared_ptr<PeerSession>> sessions;
  {
    std::lock_guard<std::mutex> lock(sessions_mutex_);
    for (const auto& [_, session] : sessions_) {
      sessions.push_back(session);
    }
  }

  for (const auto& session : sessions) {
    if (!session || !session->live_track || !session->live_track->isOpen()) {
      continue;
    }
    session->live_track->sendFrame(
        reinterpret_cast<const rtc::byte*>(bitstream.data()),
        bitstream.size(),
        info);
  }

  last_encoded_frame_timestamp_ms_ = frame->timestamp;
  ++encoded_frame_count_;
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
    if (!session || !session->detection_channel || !session->detection_channel->isOpen()) {
      continue;
    }
    if (session->detection_channel->bufferedAmount() > config_.max_detection_buffered_bytes) {
      continue;
    }

    session->detection_channel->send(message);
  }
}

void WebRTCService::sendSignalingJson(const std::string& payload) {
  std::lock_guard<std::mutex> lock(signaling_mutex_);
  if (signaling_socket_ && signaling_socket_->isOpen()) {
    signaling_socket_->send(payload);
    return;
  }
  pending_signaling_messages_.push_back(payload);
}

void WebRTCService::flushPendingSignalingMessages() {
  std::vector<std::string> messages;
  {
    std::lock_guard<std::mutex> lock(signaling_mutex_);
    if (!signaling_socket_ || !signaling_socket_->isOpen()) {
      return;
    }
    messages.swap(pending_signaling_messages_);
  }

  for (const auto& message : messages) {
    signaling_socket_->send(message);
  }
}

std::optional<std::string> WebRTCService::extractPeerId(const nlohmann::json& message) const {
  if (message.contains("peerId")) {
    return message.at("peerId").get<std::string>();
  }
  if (message.contains("sourcePeerId")) {
    return message.at("sourcePeerId").get<std::string>();
  }
  if (message.contains("from")) {
    return message.at("from").get<std::string>();
  }
  return std::nullopt;
}
