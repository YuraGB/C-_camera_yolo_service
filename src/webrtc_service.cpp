#include "webrtc_service.h"

#include <iostream>
#include <stdexcept>
#include <utility>

#include <nlohmann/json.hpp>

#include "webrtc_service_internal.h"

using namespace webrtc_service_internal;

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
  reconnect_attempt_ = 0;
  signaling_connected_ = false;
  signaling_connect_in_progress_ = false;

  {
    std::lock_guard<std::mutex> lock(sources_mutex_);
    for (const auto& [_, source_state] : sources_) {
      startSourceWorker(source_state);
    }
  }

  reconnect_thread_ = std::thread(&WebRTCService::signalingReconnectLoop, this);
  scheduleSignalingReconnect(std::chrono::milliseconds(0), "initial start");
}

void WebRTCService::stop() {
  if (!running_) {
    return;
  }

  running_ = false;
  signaling_connected_ = false;
  signaling_connect_in_progress_ = false;

  {
    std::lock_guard<std::mutex> lock(reconnect_mutex_);
    reconnect_requested_ = false;
  }
  reconnect_cv_.notify_all();

  closeSignalingSocket();

  if (reconnect_thread_.joinable()) {
    reconnect_thread_.join();
  }

  std::vector<std::shared_ptr<PeerSession>> sessions;
  {
    std::lock_guard<std::mutex> lock(sessions_mutex_);
    for (const auto& [_, session] : sessions_) {
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
    for (const auto& [_, track] : session->video_tracks) {
      if (track) {
        track->close();
      }
    }
    if (session->peer_connection) {
      session->peer_connection->close();
    }
  }

  std::vector<std::shared_ptr<SourceStreamState>> sources;
  {
    std::lock_guard<std::mutex> lock(sources_mutex_);
    for (const auto& [_, source_state] : sources_) {
      sources.push_back(source_state);
    }
  }
  for (const auto& source_state : sources) {
    stopSourceWorker(source_state);
  }

  std::cout << "[WebRTC] Service stopped" << std::endl;
}

void WebRTCService::addVideoSource(const std::string& camera_id) {
  if (camera_id.empty()) {
    return;
  }

  std::shared_ptr<SourceStreamState> source_state;
  {
    std::lock_guard<std::mutex> lock(sources_mutex_);
    auto it = sources_.find(camera_id);
    if (it != sources_.end()) {
      return;
    }

    source_state = std::make_shared<SourceStreamState>();
    source_state->camera_id = camera_id;
    source_state->track_mid = sanitizeMid(camera_id);
    sources_.emplace(camera_id, source_state);
  }

  if (running_) {
    startSourceWorker(source_state);
  }

  std::vector<std::shared_ptr<PeerSession>> sessions;
  {
    std::lock_guard<std::mutex> lock(sessions_mutex_);
    for (const auto& [_, session] : sessions_) {
      sessions.push_back(session);
    }
  }

  for (const auto& session : sessions) {
    attachVideoTrack(session, source_state);
    if (running_) {
      session->peer_connection->setLocalDescription(rtc::Description::Type::Offer);
    }
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

std::optional<std::string> WebRTCService::extractPeerId(
    const nlohmann::json& message) const {
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

void WebRTCService::connectSignalingSocket() {
  rtc::WebSocket::Configuration ws_config;
  ws_config.connectionTimeout = std::chrono::seconds(5);
  ws_config.pingInterval = std::chrono::seconds(10);

  auto socket = std::make_shared<rtc::WebSocket>(ws_config);
  uint64_t generation = 0;
  {
    std::lock_guard<std::mutex> lock(signaling_mutex_);
    generation = ++signaling_generation_;
    signaling_socket_ = socket;
  }

  socket->onOpen([this, generation]() {
    if (!isActiveSignalingGeneration(generation)) {
      return;
    }

    signaling_connect_in_progress_ = false;
    signaling_connected_ = true;
    reconnect_attempt_ = 0;
    std::cout << "[WebRTC] Signaling websocket connected: " << config_.signaling_url
              << " (generation " << generation << ")" << std::endl;

    nlohmann::json register_message = {
        {"type", "register"},
        {"peerId", config_.local_peer_id},
    };
    sendSignalingJson(register_message.dump());
    flushPendingSignalingMessages();

    if (config_.remote_peer_id && !config_.remote_peer_id->empty() &&
        *config_.remote_peer_id != config_.local_peer_id) {
      createOfferForPeer(*config_.remote_peer_id);
    }
  });

  socket->onClosed([this, generation]() {
    if (!isActiveSignalingGeneration(generation)) {
      return;
    }

    signaling_connect_in_progress_ = false;
    signaling_connected_ = false;
    std::cout << "[WebRTC] Signaling websocket closed (generation " << generation
              << ")" << std::endl;
    if (running_) {
      const int next_attempt = reconnect_attempt_ + 1;
      const auto delay = std::min(
          kReconnectMaxDelay,
          kReconnectBaseDelay * (1 << std::min(next_attempt - 1, 4)));
      scheduleSignalingReconnect(delay, "socket closed");
    }
  });

  socket->onError([this, generation](const std::string& error) {
    if (!isActiveSignalingGeneration(generation)) {
      return;
    }

    signaling_connect_in_progress_ = false;
    signaling_connected_ = false;
    std::cerr << "[WebRTC] Signaling websocket error (generation " << generation
              << "): " << error << std::endl;
    if (running_) {
      const int next_attempt = reconnect_attempt_ + 1;
      const auto delay = std::min(
          kReconnectMaxDelay,
          kReconnectBaseDelay * (1 << std::min(next_attempt - 1, 4)));
      scheduleSignalingReconnect(delay, "socket error");
    }
  });

  socket->onMessage([this, generation](rtc::message_variant data) {
    if (!isActiveSignalingGeneration(generation)) {
      return;
    }

    if (std::holds_alternative<std::string>(data)) {
      handleSignalingMessage(std::get<std::string>(data));
    }
  });

  signaling_connect_in_progress_ = true;
  std::cout << "[WebRTC] Connecting signaling websocket to " << config_.signaling_url
            << " (generation " << generation << ")" << std::endl;
  socket->open(config_.signaling_url);
}

void WebRTCService::closeSignalingSocket() {
  std::shared_ptr<rtc::WebSocket> socket;
  {
    std::lock_guard<std::mutex> lock(signaling_mutex_);
    socket = signaling_socket_;
    signaling_socket_.reset();
    ++signaling_generation_;
  }

  if (socket) {
    socket->close();
  }
}

void WebRTCService::scheduleSignalingReconnect(std::chrono::milliseconds delay,
                                               const std::string& reason) {
  if (!running_) {
    return;
  }

  {
    std::lock_guard<std::mutex> lock(reconnect_mutex_);
    reconnect_requested_ = true;
    reconnect_delay_ = delay;
    ++reconnect_attempt_;
    std::cout << "[WebRTC] Scheduling signaling reconnect attempt "
              << reconnect_attempt_ << " in " << delay.count() << "ms (" << reason
              << ")" << std::endl;
  }
  reconnect_cv_.notify_one();
}

void WebRTCService::signalingReconnectLoop() {
  std::unique_lock<std::mutex> lock(reconnect_mutex_);
  while (running_) {
    reconnect_cv_.wait(lock, [this]() { return !running_ || reconnect_requested_; });
    if (!running_) {
      break;
    }

    const auto delay = reconnect_delay_;
    reconnect_requested_ = false;
    lock.unlock();

    if (delay.count() > 0) {
      std::this_thread::sleep_for(delay);
    }

    if (running_ && !signaling_connected_ && !signaling_connect_in_progress_) {
      closeSignalingSocket();
      connectSignalingSocket();
    }

    lock.lock();
  }
}

bool WebRTCService::isActiveSignalingGeneration(uint64_t generation) {
  std::lock_guard<std::mutex> lock(signaling_mutex_);
  return signaling_socket_ != nullptr && generation == signaling_generation_;
}
