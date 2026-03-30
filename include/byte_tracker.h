#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <opencv2/core/types.hpp>

#include "models/frame.h"

class ByteTracker {
public:
    struct Config {
        float high_confidence_threshold = 0.5f;
        float low_confidence_threshold = 0.1f;
        float match_iou_threshold = 0.3f;
        int max_missed_frames = 15;
    };

    explicit ByteTracker(Config config = {});

    void advanceTo(int64_t timestamp_ms);
    void updateWithDetections(const std::vector<Detection>& detections, int64_t timestamp_ms);
    std::vector<Detection> getTrackedDetections() const;

private:
    struct Track {
        int id = -1;
        std::string label;
        float confidence = 0.0f;
        cv::Rect2f bbox;
        cv::Point2f velocity{0.0f, 0.0f};
        int64_t last_timestamp_ms = 0;
        int missed_frames = 0;
    };

    void predictTrack(Track& track, int64_t timestamp_ms) const;
    void ageUnmatchedTracks(const std::vector<int>& track_indices);
    void createTracks(const std::vector<Detection>& detections,
                      const std::vector<int>& detection_indices,
                      int64_t timestamp_ms);
    void matchDetections(const std::vector<Detection>& detections,
                         const std::vector<int>& detection_indices,
                         std::vector<int>& unmatched_tracks,
                         std::vector<int>& unmatched_detections,
                         int64_t timestamp_ms);
    void updateMatchedTrack(Track& track, const Detection& detection, int64_t timestamp_ms);
    static float computeIoU(const cv::Rect2f& lhs, const cv::Rect2f& rhs);

    Config config_;
    int next_track_id_ = 1;
    std::vector<Track> tracks_;
};
