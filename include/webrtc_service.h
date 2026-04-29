#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#include <chrono>

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
  int max_live_latency_ms = 150;
  int max_live_width = 1280;
  int max_live_height = 720;
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

  void addVideoSource(const std::string& camera_id);
  void sendFrame(const std::string& camera_id, const std::shared_ptr<Frame>& frame);
  void sendDetectionResult(const std::shared_ptr<Frame>& frame);

  void createOfferForPeer(const std::string& peer_id);
  void handleSignalingMessage(const std::string& message);

 private:
  struct SourceStreamState {
    std::string camera_id;
    std::string track_mid;
    std::mutex frame_mutex;
    std::condition_variable frame_cv;
    std::shared_ptr<Frame> latest_frame;
    std::thread worker_thread;
    std::unique_ptr<OpenH264Encoder> encoder;
    std::atomic<bool> running{false};
    std::mutex timeline_mutex;
    int64_t first_live_timestamp_ms = -1;
    int64_t dropped_stale_live_frames = 0;
    int64_t last_encoded_frame_timestamp_ms = -1;
    double smoothed_live_fps = 0.0;
    int64_t encoded_frame_count = 0;
  };

  struct PeerSession {
    std::string peer_id;
    std::shared_ptr<rtc::PeerConnection> peer_connection;
    std::unordered_map<std::string, std::shared_ptr<rtc::Track>> video_tracks;
    std::unordered_map<std::string, uint32_t> video_ssrcs;
    std::shared_ptr<rtc::DataChannel> detection_channel;
    std::atomic<bool> configured{false};
  };

  std::shared_ptr<PeerSession> createPeerSession(const std::string& peer_id);
  void configurePeerSession(const std::shared_ptr<PeerSession>& session, bool create_local_channels);
  void attachDataChannel(const std::shared_ptr<PeerSession>& session, const std::shared_ptr<rtc::DataChannel>& channel);
  void attachVideoTrack(const std::shared_ptr<PeerSession>& session, const std::shared_ptr<SourceStreamState>& source_state);
  void cleanupPeerSession(const std::string& peer_id);
  void cleanupFrontendSessionsExcept(const std::string& active_peer_id);

  std::string buildDetectionMessage(const Frame& frame) const;
  void broadcastDetectionMessage(const std::string& message);
  void sendTrackMap(const std::shared_ptr<PeerSession>& session);

  void sourceLoop(const std::shared_ptr<SourceStreamState>& source_state);
  void encodeAndBroadcastVideo(
      const std::shared_ptr<SourceStreamState>& source_state,
      const std::shared_ptr<Frame>& frame);

  void startSourceWorker(const std::shared_ptr<SourceStreamState>& source_state);
  void stopSourceWorker(const std::shared_ptr<SourceStreamState>& source_state);

  void sendSignalingJson(const std::string& payload);
  void flushPendingSignalingMessages();
  std::optional<std::string> extractPeerId(const nlohmann::json& message) const;
  void connectSignalingSocket();
  void closeSignalingSocket();
  void scheduleSignalingReconnect(std::chrono::milliseconds delay, const std::string& reason);
  void signalingReconnectLoop();
  bool isActiveSignalingGeneration(uint64_t generation);

  WebRTCServiceConfig config_;

  std::mutex sessions_mutex_;
  std::unordered_map<std::string, std::shared_ptr<PeerSession>> sessions_;

  std::mutex sources_mutex_;
  std::unordered_map<std::string, std::shared_ptr<SourceStreamState>> sources_;

  std::mutex signaling_mutex_;
  std::vector<std::string> pending_signaling_messages_;
  std::shared_ptr<rtc::WebSocket> signaling_socket_;
  std::atomic<bool> signaling_connected_{false};
  std::atomic<bool> signaling_connect_in_progress_{false};
  std::mutex reconnect_mutex_;
  std::condition_variable reconnect_cv_;
  std::thread reconnect_thread_;
  bool reconnect_requested_ = false;
  std::chrono::milliseconds reconnect_delay_{0};
  int reconnect_attempt_ = 0;
  uint64_t signaling_generation_ = 0;
  std::atomic<bool> running_{false};
  int64_t service_start_timestamp_ms_ = 0;
};
