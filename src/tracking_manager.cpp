#include "tracking_manager.h"

#include <algorithm>
#include <cmath>

TrackingManager::TrackingManager(TrackingManagerConfig config) : config_(std::move(config)) {}

void TrackingManager::submitDetections(const std::shared_ptr<Frame>& detection_frame) {
  if (!detection_frame || detection_frame->camera_id.empty()) {
    return;
  }

  std::lock_guard<std::mutex> lock(mutex_);
  camera_states_[detection_frame->camera_id].pending_detection_frame = detection_frame;
}

std::shared_ptr<Frame> TrackingManager::buildTrackedFrame(const std::shared_ptr<Frame>& live_frame) {
  if (!live_frame || live_frame->camera_id.empty()) {
    return nullptr;
  }

  std::lock_guard<std::mutex> lock(mutex_);
  auto& state = camera_states_[live_frame->camera_id];

  if (state.pending_detection_frame &&
      state.pending_detection_frame->timestamp <= live_frame->timestamp) {
    applyDetectionUpdate(state, state.pending_detection_frame);
    state.latest_detection_frame = state.pending_detection_frame;
    state.pending_detection_frame.reset();
  }

  if (state.active_tracks.empty() && !state.pending_detection_frame) {
    return nullptr;
  }

  std::vector<PredictedTrack> retained_tracks;
  retained_tracks.reserve(state.active_tracks.size());
  for (auto track_state : state.active_tracks) {
    const int64_t prediction_age_ms =
        std::max<int64_t>(0, live_frame->timestamp - track_state.last_detection_timestamp_ms);
    if (prediction_age_ms > config_.max_prediction_gap_ms ||
        track_state.predicted_frames_since_update > config_.max_prediction_frames) {
      continue;
    }

    if (live_frame->frame_id > track_state.last_published_frame_id && live_frame->timestamp > track_state.last_detection_timestamp_ms) {
      track_state.track->predict();
      track_state.last_published_frame_id = live_frame->frame_id;
      ++track_state.predicted_frames_since_update;
    }

    retained_tracks.push_back(track_state);
  }

  state.active_tracks = retained_tracks;

  auto tracked_frame = std::make_shared<Frame>();
  tracked_frame->camera_id = live_frame->camera_id;
  tracked_frame->frame_id = live_frame->frame_id;
  tracked_frame->timestamp = live_frame->timestamp;
  tracked_frame->frame_width = live_frame->width();
  tracked_frame->frame_height = live_frame->height();
  tracked_frame->detections = buildDetectionsForFrame(state.active_tracks, *live_frame);
  if (tracked_frame->detections.empty() &&
      state.latest_detection_frame &&
      !state.latest_detection_frame->detections.empty()) {
    const int64_t detection_age_ms =
        std::max<int64_t>(0, live_frame->timestamp - state.latest_detection_frame->timestamp);
    if (detection_age_ms <= config_.max_prediction_gap_ms) {
      tracked_frame->detections = state.latest_detection_frame->detections;
    }
  }
  return tracked_frame;
}

std::vector<Detection> TrackingManager::buildDetectionsForFrame(
    const std::vector<PredictedTrack>& active_tracks,
    const Frame& live_frame) {
  std::vector<Detection> detections;
  detections.reserve(active_tracks.size());

  for (const auto& track_state : active_tracks) {
    detections.emplace_back(
        track_state.label,
        track_state.confidence,
        clampBBox(track_state.track->getRect(), live_frame));
  }

  return detections;
}

byte_track::Object TrackingManager::toByteTrackObject(const Detection& detection) {
  return {
      byte_track::Rect<float>(
          static_cast<float>(detection.bbox.x),
          static_cast<float>(detection.bbox.y),
          static_cast<float>(detection.bbox.width),
          static_cast<float>(detection.bbox.height)),
      0,
      detection.confidence};
}

BBox TrackingManager::clampBBox(const byte_track::Rect<float>& rect, const Frame& frame) {
  const int left = std::max(0, static_cast<int>(std::round(rect.x())));
  const int top = std::max(0, static_cast<int>(std::round(rect.y())));
  const int right = std::min(frame.width(), static_cast<int>(std::round(rect.x() + rect.width())));
  const int bottom = std::min(frame.height(), static_cast<int>(std::round(rect.y() + rect.height())));

  return {
      left,
      top,
      std::max(0, right - left),
      std::max(0, bottom - top)};
}

void TrackingManager::applyDetectionUpdate(CameraState& state, const std::shared_ptr<Frame>& detection_frame) {
  std::unordered_map<std::string, std::vector<Detection>> detections_by_label;
  for (const auto& detection : detection_frame->detections) {
    detections_by_label[detection.label].push_back(detection);
  }

  std::vector<PredictedTrack> updated_tracks;
  for (const auto& [label, detections] : detections_by_label) {
    auto label_tracks = updateTracksForLabel(
        detection_frame->camera_id,
        label,
        detections,
        detection_frame->timestamp,
        detection_frame->frame_id);
    updated_tracks.insert(updated_tracks.end(), label_tracks.begin(), label_tracks.end());
  }

  // Keep short-lived predictions for labels missing in the latest YOLO result so tracker gaps stay minimal.
  for (const auto& existing_track : state.active_tracks) {
    if (detections_by_label.find(existing_track.label) != detections_by_label.end()) {
      continue;
    }

    const int64_t prediction_age_ms =
        std::max<int64_t>(0, detection_frame->timestamp - existing_track.last_detection_timestamp_ms);
    if (prediction_age_ms <= config_.max_prediction_gap_ms &&
        existing_track.predicted_frames_since_update <= config_.max_prediction_frames) {
      updated_tracks.push_back(existing_track);
    }
  }

  state.active_tracks = std::move(updated_tracks);
}

std::vector<TrackingManager::PredictedTrack> TrackingManager::updateTracksForLabel(
    const std::string& camera_id,
    const std::string& label,
    const std::vector<Detection>& detections,
    int64_t detection_timestamp_ms,
    int64_t detection_frame_id) {
  std::vector<byte_track::Object> objects;
  objects.reserve(detections.size());
  for (const auto& detection : detections) {
    objects.push_back(toByteTrackObject(detection));
  }

  auto& tracker = getOrCreateTracker(camera_id, label);
  auto tracks = tracker.update(objects);

  std::vector<PredictedTrack> predicted_tracks;
  predicted_tracks.reserve(tracks.size());
  for (const auto& track : tracks) {
    predicted_tracks.push_back({
        std::make_shared<byte_track::STrack>(*track),
        label,
        track->getScore(),
        detection_timestamp_ms,
        detection_frame_id,
        0});
  }

  return predicted_tracks;
}

byte_track::BYTETracker& TrackingManager::getOrCreateTracker(const std::string& camera_id, const std::string& label) {
  auto& camera_trackers = trackers_[camera_id];
  auto it = camera_trackers.find(label);
  if (it == camera_trackers.end()) {
    auto tracker = std::make_unique<byte_track::BYTETracker>(
        config_.assumed_detection_fps,
        config_.track_buffer,
        config_.track_thresh,
        config_.high_thresh,
        config_.match_thresh);
    it = camera_trackers.emplace(label, std::move(tracker)).first;
  }

  return *it->second;
}
