#include "webrtc_service.h"

#include <iostream>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

void WebRTCService::createOfferForPeer(const std::string& peer_id) {
  if (peer_id.empty() || peer_id == config_.local_peer_id) {
    return;
  }

  cleanupFrontendSessionsExcept(peer_id);

  std::shared_ptr<PeerSession> session;
  {
    std::lock_guard<std::mutex> lock(sessions_mutex_);
    auto it = sessions_.find(peer_id);
    if (it != sessions_.end()) {
      session = it->second;
    }
  }

  if (session && (session->closing || session->closed)) {
    cleanupPeerSession(peer_id);
    session.reset();
  }

  if (!session) {
    session = createPeerSession(peer_id);
  } else if (session->connected || session->offer_in_progress) {
    return;
  }

  configurePeerSession(session, true);

  bool expected = false;
  if (!session->offer_in_progress.compare_exchange_strong(expected, true)) {
    return;
  }

  try {
    session->peer_connection->setLocalDescription(rtc::Description::Type::Offer);
    std::cout << "[WebRTC] Creating offer for peer " << peer_id << std::endl;
  } catch (const std::exception& error) {
    session->offer_in_progress = false;
    std::cerr << "[WebRTC] Failed to create offer for peer " << peer_id
              << ": " << error.what() << std::endl;
  }
}

void WebRTCService::flushPendingRemoteCandidates(
    const std::shared_ptr<PeerSession>& session) {
  if (!session || !session->peer_connection || !session->remote_description_applied) {
    return;
  }

  std::vector<std::pair<std::string, std::string>> candidates;
  {
    std::lock_guard<std::mutex> lock(session->pending_ice_mutex);
    candidates.swap(session->pending_remote_candidates);
  }

  for (const auto& [candidate, mid] : candidates) {
    try {
      session->peer_connection->addRemoteCandidate(rtc::Candidate(candidate, mid));
    } catch (const std::exception& error) {
      std::cerr << "[WebRTC] Failed to add queued remote ICE candidate from "
                << session->peer_id << ": " << error.what() << std::endl;
    }
  }
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
  if (type.empty() || type == "connected" || type == "registered" ||
      type == "ping" || type == "pong") {
    if (type == "ping") {
      nlohmann::json pong = {
          {"type", "pong"},
          {"peerId", config_.local_peer_id},
      };
      sendSignalingJson(pong.dump());
    }
    return;
  }

  if (type == "error") {
    std::cerr << "[WebRTC] Signaling error: " << message << std::endl;
    return;
  }

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

  if (type == "answer") {
    if (!session || session->closing || session->closed) {
      return;
    }

    try {
      session->peer_connection->setRemoteDescription(
          rtc::Description(json.at("sdp").get<std::string>(), "answer"));
      session->remote_description_applied = true;
      session->offer_in_progress = false;
      std::cout << "[WebRTC] Applied remote answer from " << *peer_id << std::endl;
      flushPendingRemoteCandidates(session);
      sendTrackMap(session);
    } catch (const std::exception& error) {
      session->offer_in_progress = false;
      std::cerr << "[WebRTC] Failed to apply remote answer from " << *peer_id
                << ": " << error.what() << std::endl;
    }
    return;
  }

  if (type == "offer") {
    if (session && (session->closing || session->closed)) {
      cleanupPeerSession(*peer_id);
      session.reset();
    }
    if (session && (session->connected || session->offer_in_progress)) {
      return;
    }
    if (!session) {
      session = createPeerSession(*peer_id);
    }
    configurePeerSession(session, false);

    try {
      session->peer_connection->setRemoteDescription(
          rtc::Description(json.at("sdp").get<std::string>(), "offer"));
      session->remote_description_applied = true;
      session->peer_connection->setLocalDescription(rtc::Description::Type::Answer);
      std::cout << "[WebRTC] Received remote offer from " << *peer_id << std::endl;
      flushPendingRemoteCandidates(session);
    } catch (const std::exception& error) {
      std::cerr << "[WebRTC] Failed to handle remote offer from " << *peer_id
                << ": " << error.what() << std::endl;
    }
    return;
  }

  if (type == "ice-candidate") {
    if (!session || session->closing || session->closed) {
      return;
    }

    try {
      const std::string candidate = json.at("candidate").get<std::string>();
      const std::string mid = json.value("mid", "");
      if (!session->remote_description_applied) {
        std::lock_guard<std::mutex> lock(session->pending_ice_mutex);
        session->pending_remote_candidates.emplace_back(candidate, mid);
        return;
      }
      session->peer_connection->addRemoteCandidate(rtc::Candidate(candidate, mid));
    } catch (const std::exception& error) {
      std::cerr << "[WebRTC] Failed to add remote ICE candidate from " << *peer_id
                << ": " << error.what() << std::endl;
    }
  }
}
