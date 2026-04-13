#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <condition_variable>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>
#include <rtc/rtc.hpp>

#include "frame.h"
#include "openh264_encoder.h"

struct WebRTCServiceConfig {
  std::string signaling_url;
  std::string local_peer_id;
  std::optional<std::string> remote_peer_id;
  std::vector<std::string> ice_servers;
  std::string live_track_label = "liveStream";
  std::string detection_channel_label = "detectionStream";
  size_t max_detection_buffered_bytes = 128 * 1024;
  std::string openh264_dll_path = "third_party/openh264-2.6.0-win64.dll";
};

class WebRTCService {
 public:
  explicit WebRTCService(WebRTCServiceConfig config);
  ~WebRTCService();

  WebRTCService(const WebRTCService&) = delete;
  WebRTCService& operator=(const WebRTCService&) = delete;

  void start();
  void stop();

  void sendLiveFrame(const std::shared_ptr<Frame>& frame);
  void sendDetectionResult(const std::shared_ptr<Frame>& frame);

  void createOfferForPeer(const std::string& peer_id);
  void handleSignalingMessage(const std::string& message);

 private:
  struct PeerSession {
    std::string peer_id;
    std::shared_ptr<rtc::PeerConnection> peer_connection;
    std::shared_ptr<rtc::Track> live_track;
    std::shared_ptr<rtc::DataChannel> detection_channel;
    std::atomic<bool> connected{false};
    std::atomic<bool> configured{false};
    uint32_t video_ssrc = 0;
  };

  std::shared_ptr<PeerSession> createPeerSession(const std::string& peer_id);
  void configurePeerSession(const std::shared_ptr<PeerSession>& session, bool create_local_channels);
  void attachDataChannel(const std::shared_ptr<PeerSession>& session, const std::shared_ptr<rtc::DataChannel>& channel);
  void attachVideoTrack(const std::shared_ptr<PeerSession>& session);
  void cleanupPeerSession(const std::string& peer_id);

  std::string buildDetectionMessage(const Frame& frame) const;
  void videoLoop();
  void encodeAndBroadcastVideo(const std::shared_ptr<Frame>& frame);
  void broadcastDetectionMessage(const std::string& message);

  void sendSignalingJson(const std::string& payload);
  void flushPendingSignalingMessages();
  std::optional<std::string> extractPeerId(const nlohmann::json& message) const;

  WebRTCServiceConfig config_;
  std::mutex sessions_mutex_;
  std::unordered_map<std::string, std::shared_ptr<PeerSession>> sessions_;

  std::mutex live_frame_mutex_;
  std::condition_variable live_frame_cv_;
  std::shared_ptr<Frame> latest_live_frame_;
  std::thread video_thread_;

  std::mutex signaling_mutex_;
  std::vector<std::string> pending_signaling_messages_;
  std::shared_ptr<rtc::WebSocket> signaling_socket_;
  std::atomic<bool> running_{false};
  int64_t service_start_timestamp_ms_ = 0;
  std::unique_ptr<OpenH264Encoder> video_encoder_;
  int64_t encoded_frame_count_ = 0;
};
