#include "byte_tracker.h"

#include <algorithm>

namespace {
cv::Point2f rectCenter(const cv::Rect2f& rect) {
    return {rect.x + (rect.width * 0.5f), rect.y + (rect.height * 0.5f)};
}
}

ByteTracker::ByteTracker(Config config)
    : config_(config) {}

void ByteTracker::advanceTo(int64_t timestamp_ms) {
    for (auto& track : tracks_) {
        predictTrack(track, timestamp_ms);
    }
}

void ByteTracker::updateWithDetections(const std::vector<Detection>& detections, int64_t timestamp_ms) {
    advanceTo(timestamp_ms);

    std::vector<int> high_confidence_indices;
    std::vector<int> low_confidence_indices;
    for (int i = 0; i < static_cast<int>(detections.size()); ++i) {
        if (detections[i].confidence >= config_.high_confidence_threshold) {
            high_confidence_indices.push_back(i);
        } else if (detections[i].confidence >= config_.low_confidence_threshold) {
            low_confidence_indices.push_back(i);
        }
    }

    std::vector<int> unmatched_tracks(tracks_.size());
    for (int i = 0; i < static_cast<int>(tracks_.size()); ++i) {
        unmatched_tracks[i] = i;
    }

    std::vector<int> unmatched_high = high_confidence_indices;
    matchDetections(detections, high_confidence_indices, unmatched_tracks, unmatched_high, timestamp_ms);

    std::vector<int> unmatched_low = low_confidence_indices;
    matchDetections(detections, low_confidence_indices, unmatched_tracks, unmatched_low, timestamp_ms);

    ageUnmatchedTracks(unmatched_tracks);
    createTracks(detections, unmatched_high, timestamp_ms);

    tracks_.erase(
        std::remove_if(tracks_.begin(), tracks_.end(), [this](const Track& track) {
            return track.missed_frames > config_.max_missed_frames;
        }),
        tracks_.end());
}

std::vector<Detection> ByteTracker::getTrackedDetections() const {
    std::vector<Detection> detections;
    detections.reserve(tracks_.size());

    for (const auto& track : tracks_) {
        if (track.missed_frames > config_.max_missed_frames) {
            continue;
        }

        detections.emplace_back(
            track.label,
            track.confidence,
            BBox(
                static_cast<int>(track.bbox.x),
                static_cast<int>(track.bbox.y),
                static_cast<int>(track.bbox.width),
                static_cast<int>(track.bbox.height)),
            track.id);
    }

    return detections;
}

void ByteTracker::predictTrack(Track& track, int64_t timestamp_ms) const {
    if (timestamp_ms <= track.last_timestamp_ms) {
        return;
    }

    const float dt_seconds = static_cast<float>(timestamp_ms - track.last_timestamp_ms) / 1000.0f;
    track.bbox.x += track.velocity.x * dt_seconds;
    track.bbox.y += track.velocity.y * dt_seconds;
    track.last_timestamp_ms = timestamp_ms;
}

void ByteTracker::ageUnmatchedTracks(const std::vector<int>& track_indices) {
    for (int track_index : track_indices) {
        if (track_index >= 0 && track_index < static_cast<int>(tracks_.size())) {
            tracks_[track_index].missed_frames += 1;
        }
    }
}

void ByteTracker::createTracks(const std::vector<Detection>& detections,
                               const std::vector<int>& detection_indices,
                               int64_t timestamp_ms) {
    for (int detection_index : detection_indices) {
        const auto& detection = detections[detection_index];
        Track track;
        track.id = next_track_id_++;
        track.label = detection.label;
        track.confidence = detection.confidence;
        track.bbox = cv::Rect2f(
            static_cast<float>(detection.bbox.x),
            static_cast<float>(detection.bbox.y),
            static_cast<float>(detection.bbox.width),
            static_cast<float>(detection.bbox.height));
        track.last_timestamp_ms = timestamp_ms;
        tracks_.push_back(track);
    }
}

void ByteTracker::matchDetections(const std::vector<Detection>& detections,
                                  const std::vector<int>& detection_indices,
                                  std::vector<int>& unmatched_tracks,
                                  std::vector<int>& unmatched_detections,
                                  int64_t timestamp_ms) {
    if (unmatched_tracks.empty() || detection_indices.empty()) {
        unmatched_detections = detection_indices;
        return;
    }

    unmatched_detections = detection_indices;
    std::vector<bool> detection_used(detections.size(), false);
    std::vector<int> still_unmatched_tracks;

    for (int track_index : unmatched_tracks) {
        float best_iou = 0.0f;
        int best_detection_index = -1;

        for (int detection_index : detection_indices) {
            if (detection_used[detection_index]) {
                continue;
            }

            const auto& detection = detections[detection_index];
            const cv::Rect2f detection_bbox(
                static_cast<float>(detection.bbox.x),
                static_cast<float>(detection.bbox.y),
                static_cast<float>(detection.bbox.width),
                static_cast<float>(detection.bbox.height));

            const float iou = computeIoU(tracks_[track_index].bbox, detection_bbox);
            if (iou > best_iou) {
                best_iou = iou;
                best_detection_index = detection_index;
            }
        }

        if (best_detection_index >= 0 && best_iou >= config_.match_iou_threshold) {
            detection_used[best_detection_index] = true;
            updateMatchedTrack(tracks_[track_index], detections[best_detection_index], timestamp_ms);
        } else {
            still_unmatched_tracks.push_back(track_index);
        }
    }

    unmatched_tracks = std::move(still_unmatched_tracks);

    unmatched_detections.clear();
    for (int detection_index : detection_indices) {
        if (!detection_used[detection_index]) {
            unmatched_detections.push_back(detection_index);
        }
    }
}

void ByteTracker::updateMatchedTrack(Track& track, const Detection& detection, int64_t timestamp_ms) {
    const cv::Rect2f previous_bbox = track.bbox;
    const cv::Point2f previous_center = rectCenter(previous_bbox);

    const cv::Rect2f detection_bbox(
        static_cast<float>(detection.bbox.x),
        static_cast<float>(detection.bbox.y),
        static_cast<float>(detection.bbox.width),
        static_cast<float>(detection.bbox.height));

    const cv::Point2f detection_center = rectCenter(detection_bbox);
    const float dt_seconds = std::max(0.001f, static_cast<float>(timestamp_ms - track.last_timestamp_ms) / 1000.0f);

    track.velocity = {
        (detection_center.x - previous_center.x) / dt_seconds,
        (detection_center.y - previous_center.y) / dt_seconds
    };
    track.label = detection.label;
    track.confidence = detection.confidence;
    track.bbox = detection_bbox;
    track.last_timestamp_ms = timestamp_ms;
    track.missed_frames = 0;
}

float ByteTracker::computeIoU(const cv::Rect2f& lhs, const cv::Rect2f& rhs) {
    const float intersection_area = (lhs & rhs).area();
    if (intersection_area <= 0.0f) {
        return 0.0f;
    }

    const float union_area = lhs.area() + rhs.area() - intersection_area;
    if (union_area <= 0.0f) {
        return 0.0f;
    }

    return intersection_area / union_area;
}
