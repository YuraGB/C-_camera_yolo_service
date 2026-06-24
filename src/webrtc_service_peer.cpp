#include "webrtc_service.h"

#include <iostream>

#include <nlohmann/json.hpp>
#include <rtc/h264rtppacketizer.hpp>
#include <rtc/rtcpnackresponder.hpp>
#include <rtc/rtcpsrreporter.hpp>

#include "webrtc_service_internal.h"

using namespace webrtc_service_internal;

std::shared_ptr<WebRTCService::PeerSession> WebRTCService::createPeerSession(
    const std::string& peer_id) {
  std::lock_guard<std::mutex> lock(sessions_mutex_);
  auto it = sessions_.find(peer_id);
  if (it != sessions_.end()) {
    return it->second;
  }

  rtc::Configuration configuration;
  configuration.disableAutoNegotiation = true;
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

void WebRTCService::configurePeerSession(const std::shared_ptr<PeerSession>& session,
                                         bool create_local_channels) {
  if (!session || !session->peer_connection) {
    return;
  }

  bool expected = false;
  if (!session->configured.compare_exchange_strong(expected, true)) {
    return;
  }

  session->peer_connection->onStateChange(
      [this, weak_session = std::weak_ptr<PeerSession>(session),
       peer_id = session->peer_id](rtc::PeerConnection::State state) {
        std::cout << "[WebRTC] Peer " << peer_id << " state: " << state
                  << std::endl;
        auto locked = weak_session.lock();
        if (locked) {
          locked->connected = state == rtc::PeerConnection::State::Connected;
          if (locked->connected) {
            locked->offer_in_progress = false;
          }
          if (isPeerTerminal(state)) {
            locked->closed = true;
          }
        }
        if (isPeerTerminal(state)) {
          cleanupPeerSession(peer_id);
        }
      });

  session->peer_connection->onLocalDescription(
      [this, weak_session = std::weak_ptr<PeerSession>(session)](
          rtc::Description description) {
        auto locked = weak_session.lock();
        if (!locked) {
          return;
        }

        const std::string description_type = description.typeString();
        if (description_type == "offer") {
          locked->offer_in_progress = true;
        }
        nlohmann::json message = {
            {"type", description_type},
            {"peerId", config_.local_peer_id},
            {"targetPeerId", locked->peer_id},
            {"sdp", std::string(description)},
        };
        sendSignalingJson(message.dump());
        sendTrackMap(locked);
      });

  session->peer_connection->onLocalCandidate(
      [this, peer_id = session->peer_id](rtc::Candidate candidate) {
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
      [this, weak_session = std::weak_ptr<PeerSession>(session)](
          std::shared_ptr<rtc::DataChannel> channel) {
        if (auto locked = weak_session.lock()) {
          attachDataChannel(locked, channel);
        }
      });

  std::vector<std::shared_ptr<SourceStreamState>> sources;
  {
    std::lock_guard<std::mutex> lock(sources_mutex_);
    for (const auto& [_, source_state] : sources_) {
      sources.push_back(source_state);
    }
  }
  for (const auto& source_state : sources) {
    attachVideoTrack(session, source_state);
  }

  if (create_local_channels && !session->detection_channel) {
    session->detection_channel = session->peer_connection->createDataChannel(
        config_.detection_channel_label, makeDetectionChannelInit());
    std::cout << "[WebRTC] Created local detection data channel for peer "
              << session->peer_id << " with label "
              << config_.detection_channel_label << std::endl;
    attachDataChannel(session, session->detection_channel);
  }

  if (!create_local_channels) {
    std::cout << "[WebRTC] Prepared answerer session for peer " << session->peer_id
              << " with multi-track video and detection channel" << std::endl;
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

  std::cout << "[WebRTC] Attaching data channel for peer " << session->peer_id
            << ": label=" << channel->label() << std::endl;

  channel->onOpen(
      [this, weak_session = std::weak_ptr<PeerSession>(session),
       label = channel->label()]() {
        if (auto locked = weak_session.lock()) {
          std::cout << "[WebRTC] DataChannel opened for " << locked->peer_id << ": "
                    << label << std::endl;
          if (label == config_.detection_channel_label) {
            locked->detection_channel_open = true;
            sendTrackMap(locked);
          }
        }
      });

  channel->onClosed(
      [weak_session = std::weak_ptr<PeerSession>(session),
       expected_label = config_.detection_channel_label,
       peer_id = session->peer_id, label = channel->label()]() {
        if (auto locked = weak_session.lock()) {
          if (label == expected_label) {
            locked->detection_channel_open = false;
          }
        }
        std::cout << "[WebRTC] DataChannel closed for " << peer_id << ": " << label
                  << std::endl;
      });

  channel->onError(
      [peer_id = session->peer_id, label = channel->label()](std::string error) {
        std::cerr << "[WebRTC] DataChannel error for " << peer_id << " (" << label
                  << "): " << error << std::endl;
      });

  channel->onMessage(
      [peer_id = session->peer_id, label = channel->label()](
          rtc::message_variant data) {
        if (std::holds_alternative<std::string>(data)) {
          std::cout << "[WebRTC] Message from " << peer_id << " on " << label
                    << ": " << std::get<std::string>(data) << std::endl;
        }
      });
}

void WebRTCService::attachVideoTrack(
    const std::shared_ptr<PeerSession>& session,
    const std::shared_ptr<SourceStreamState>& source_state) {
  if (!session || !session->peer_connection || !source_state) {
    return;
  }

  if (session->video_tracks.find(source_state->camera_id) !=
      session->video_tracks.end()) {
    return;
  }

  constexpr uint8_t payload_type = 102;
  const uint32_t ssrc = randomSsrc();
  const std::string cname = "camera-cv-video";

  auto video = rtc::Description::Video(source_state->track_mid);
  video.addH264Codec(payload_type);
  video.addSSRC(ssrc, cname, source_state->camera_id, source_state->camera_id);

  auto track = session->peer_connection->addTrack(video);
  auto rtp_config = std::make_shared<rtc::RtpPacketizationConfig>(
      ssrc, cname, payload_type, rtc::H264RtpPacketizer::defaultClockRate);
  auto packetizer = std::make_shared<rtc::H264RtpPacketizer>(
      rtc::NalUnit::Separator::StartSequence, rtp_config);
  auto sr_reporter = std::make_shared<rtc::RtcpSrReporter>(rtp_config);
  auto nack_responder = std::make_shared<rtc::RtcpNackResponder>();
  packetizer->addToChain(sr_reporter);
  packetizer->addToChain(nack_responder);
  track->setMediaHandler(packetizer);

  track->onOpen([peer_id = session->peer_id, camera_id = source_state->camera_id,
                 mid = source_state->track_mid]() {
    std::cout << "[WebRTC] Video track opened for " << peer_id
              << ", camera_id=" << camera_id << ", mid=" << mid << std::endl;
  });
  track->onClosed([peer_id = session->peer_id, camera_id = source_state->camera_id]() {
    std::cout << "[WebRTC] Video track closed for " << peer_id
              << ", camera_id=" << camera_id << std::endl;
  });
  track->onError(
      [peer_id = session->peer_id, camera_id = source_state->camera_id](
          std::string error) {
        std::cerr << "[WebRTC] Video track error for " << peer_id
                  << ", camera_id=" << camera_id << ": " << error << std::endl;
      });

  session->video_tracks[source_state->camera_id] = track;
  session->video_rtp_configs[source_state->camera_id] = rtp_config;
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

  if (session->closing.exchange(true)) {
    return;
  }
  session->closed = true;
  session->detection_channel_open = false;

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

void WebRTCService::cleanupFrontendSessionsExcept(const std::string& active_peer_id) {
  std::vector<std::string> stale_peer_ids;
  {
    std::lock_guard<std::mutex> lock(sessions_mutex_);
    for (const auto& [peer_id, _] : sessions_) {
      if (peer_id == active_peer_id) {
        continue;
      }
      if (peer_id.rfind("frontend-", 0) == 0) {
        stale_peer_ids.push_back(peer_id);
      }
    }
  }

  for (const auto& stale_peer_id : stale_peer_ids) {
    std::cout << "[WebRTC] Cleaning up stale frontend peer session: "
              << stale_peer_id << std::endl;
    cleanupPeerSession(stale_peer_id);
  }
}

void WebRTCService::sendTrackMap(const std::shared_ptr<PeerSession>& session) {
  if (!session || !session->detection_channel || !session->detection_channel->isOpen()) {
    return;
  }

  nlohmann::json payload = {
      {"type", "track_map"},
      {"tracks", nlohmann::json::array()},
  };

  for (const auto& [camera_id, track] : session->video_tracks) {
    if (!track) {
      continue;
    }

    payload["tracks"].push_back({
        {"mid", track->mid()},
        {"camera_id", camera_id},
    });
  }

  session->detection_channel->send(payload.dump());
}
