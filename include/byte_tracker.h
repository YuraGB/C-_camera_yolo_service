#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/video/tracking.hpp>

#include "models/frame.h"

class ByteTracker {
public:
    struct Config {
        int frame_rate = 30;
        int track_buffer = 30;
        float track_threshold = 0.5f;
        float high_threshold = 0.6f;
        float match_threshold = 0.8f;
        float low_match_threshold = 0.5f;
        float unconfirmed_match_threshold = 0.7f;
        float min_box_area = 10.0f;
        float max_aspect_ratio = 8.0f;
        float max_display_prediction_steps = 2.0f;
        float display_smoothing_alpha = 0.65f;
        int optical_flow_max_corners = 32;
        double optical_flow_quality_level = 0.01;
        double optical_flow_min_distance = 5.0;
        float optical_flow_max_error = 12.0f;
        int optical_flow_min_points = 8;
        float optical_flow_max_step_ratio = 0.18f;
    };

    explicit ByteTracker(Config config = {});

    void advanceTo(int64_t timestamp_ms);
    void observeFrame(const cv::Mat& frame, int64_t timestamp_ms);
    void updateWithDetections(const std::vector<Detection>& detections, int64_t timestamp_ms);
    std::vector<Detection> getTrackedDetections() const;

private:
    enum class TrackState {
        New = 0,
        Tracked = 1,
        Lost = 2,
        Removed = 3,
    };

    struct Track {
        int id = -1;
        std::string label;
        float score = 0.0f;
        cv::Rect2f rect;
        cv::Rect2f display_rect;
        TrackState state = TrackState::New;
        bool is_activated = false;
        size_t frame_id = 0;
        size_t start_frame_id = 0;
        size_t tracklet_length = 0;
        int64_t last_detection_timestamp_ms = 0;
        float average_detection_interval_ms = 33.0f;
        cv::KalmanFilter kalman_filter;
    };

    using TrackPtr = std::shared_ptr<Track>;

    TrackPtr createDetectionTrack(const Detection& detection) const;
    void activateTrack(const TrackPtr& track, size_t frame_id, int64_t timestamp_ms);
    void predictTrack(const TrackPtr& track) const;
    void updateTrack(const TrackPtr& track, const TrackPtr& detection, size_t frame_id, int64_t timestamp_ms);
    void reactivateTrack(const TrackPtr& track,
                         const TrackPtr& detection,
                         size_t frame_id,
                         int64_t timestamp_ms,
                         bool assign_new_id);
    void markLost(const TrackPtr& track) const;
    void markRemoved(const TrackPtr& track) const;

    std::vector<TrackPtr> jointTracks(const std::vector<TrackPtr>& lhs, const std::vector<TrackPtr>& rhs) const;
    std::vector<TrackPtr> subtractTracks(const std::vector<TrackPtr>& lhs, const std::vector<TrackPtr>& rhs) const;
    void removeDuplicateTracks(const std::vector<TrackPtr>& lhs,
                               const std::vector<TrackPtr>& rhs,
                               std::vector<TrackPtr>& lhs_result,
                               std::vector<TrackPtr>& rhs_result) const;

    void linearAssignment(const std::vector<std::vector<float>>& cost_matrix,
                          int row_count,
                          int column_count,
                          float threshold,
                          std::vector<std::vector<int>>& matches,
                          std::vector<int>& unmatched_rows,
                          std::vector<int>& unmatched_columns) const;
    std::vector<std::vector<float>> calcIouDistance(const std::vector<TrackPtr>& lhs,
                                                    const std::vector<TrackPtr>& rhs) const;
    static float computeIoU(const cv::Rect2f& lhs, const cv::Rect2f& rhs);
    static cv::Rect2f sanitizeRect(const cv::Rect2f& rect);
    static cv::Rect2f measurementToRect(const cv::Mat& measurement);
    static cv::Mat rectToMeasurement(const cv::Rect2f& rect);
    static void configureKalmanFilter(cv::KalmanFilter& filter);
    static void updateProcessNoise(cv::KalmanFilter& filter, float height);
    static void updateMeasurementNoise(cv::KalmanFilter& filter, float height);
    static cv::Rect2f predictedRectForDisplay(const Track& track,
                                              int64_t display_timestamp_ms,
                                              float max_prediction_steps);
    void applyVisualTracking(const cv::Mat& gray_frame, int64_t timestamp_ms);
    void syncTrackStateToRect(const TrackPtr& track) const;
    void updateDisplayRect(const TrackPtr& track, bool force) const;
    static float median(std::vector<float>& values);

    Config config_;
    size_t frame_id_ = 0;
    int next_track_id_ = 1;
    int64_t current_display_timestamp_ms_ = 0;
    cv::Mat previous_gray_frame_;
    int64_t previous_frame_timestamp_ms_ = 0;
    std::vector<TrackPtr> tracked_tracks_;
    std::vector<TrackPtr> lost_tracks_;
    std::vector<TrackPtr> removed_tracks_;
};
