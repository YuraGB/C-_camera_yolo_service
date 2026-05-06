#include "webrtc_service.h"

#include <iostream>

#include <nlohmann/json.hpp>

void WebRTCService::createOfferForPeer(const std::string& peer_id) {
  if (peer_id.empty() || peer_id == config_.local_peer_id) {
    return;
  }

  cleanupFrontendSessionsExcept(peer_id);

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
    std::cerr << "[WebRTC] Failed to parse signaling message: " << error.what()
              << std::endl;
    return;
  }

  const std::string type = json.value("type", "");
  const std::string target_peer_id = json.value("targetPeerId", "");
  if (!target_peer_id.empty() && target_peer_id != config_.local_peer_id) {
    return;
  }

  auto peer_id = extractPeerId(json);
  if (!peer_id || peer_id->empty()) {
    std::cerr << "[WebRTC] Signaling message is missing peer id: " << message
              << std::endl;
    return;
  }
  if (*peer_id == config_.local_peer_id) {
    return;
  }

  cleanupFrontendSessionsExcept(*peer_id);

  if (type == "offer-request" || type == "viewer-join" || type == "connect") {
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
      sendTrackMap(session);
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
