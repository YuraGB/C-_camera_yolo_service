#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "ByteTrack/BYTETracker.h"
#include "frame.h"

struct TrackingManagerConfig {
  int assumed_detection_fps = 10;
  int track_buffer = 30;
  float track_thresh = 0.25f;
  float high_thresh = 0.5f;
  float match_thresh = 0.8f;
  int max_prediction_gap_ms = 500;
  int max_prediction_frames = 12;
};

class TrackingManager {
 public:
  explicit TrackingManager(TrackingManagerConfig config = {});

  void submitDetections(const std::shared_ptr<Frame>& detection_frame);
  std::shared_ptr<Frame> buildTrackedFrame(const std::shared_ptr<Frame>& live_frame);

 private:
  struct PredictedTrack {
    std::shared_ptr<byte_track::STrack> track;
    std::string label;
    float confidence = 0.0f;
    int64_t last_detection_timestamp_ms = 0;
    int64_t last_published_frame_id = -1;
    int predicted_frames_since_update = 0;
  };

  struct CameraState {
    std::shared_ptr<Frame> pending_detection_frame;
    std::shared_ptr<Frame> latest_detection_frame;
    std::vector<PredictedTrack> active_tracks;
  };

  static std::vector<Detection> buildDetectionsForFrame(
      const std::vector<PredictedTrack>& active_tracks,
      const Frame& live_frame);
  static byte_track::Object toByteTrackObject(const Detection& detection);
  static BBox clampBBox(const byte_track::Rect<float>& rect, const Frame& frame);

  void applyDetectionUpdate(CameraState& state, const std::shared_ptr<Frame>& detection_frame);
  std::vector<PredictedTrack> updateTracksForLabel(
      const std::string& camera_id,
      const std::string& label,
      const std::vector<Detection>& detections,
      int64_t detection_timestamp_ms,
      int64_t detection_frame_id);
  byte_track::BYTETracker& getOrCreateTracker(const std::string& camera_id, const std::string& label);

  TrackingManagerConfig config_;
  std::mutex mutex_;
  std::unordered_map<std::string, CameraState> camera_states_;
  std::unordered_map<std::string, std::unordered_map<std::string, std::unique_ptr<byte_track::BYTETracker>>> trackers_;
};
