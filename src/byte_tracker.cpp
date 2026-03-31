#include "byte_tracker.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <utility>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#include "byte_tracker_lapjv.h"

// Core tracking flow adapted from the MIT-licensed ByteTrack-cpp project.

ByteTracker::ByteTracker(Config config)
    : config_(config) {}

void ByteTracker::advanceTo(int64_t timestamp_ms) {
    current_display_timestamp_ms_ = timestamp_ms;
}

void ByteTracker::observeFrame(const cv::Mat& frame, int64_t timestamp_ms) {
    current_display_timestamp_ms_ = std::max(current_display_timestamp_ms_, timestamp_ms);
    if (frame.empty()) {
        return;
    }

    cv::Mat gray_frame;
    if (frame.channels() == 1) {
        gray_frame = frame.clone();
    } else {
        cv::cvtColor(frame, gray_frame, cv::COLOR_BGR2GRAY);
    }

    if (!previous_gray_frame_.empty() && timestamp_ms > previous_frame_timestamp_ms_) {
        applyVisualTracking(gray_frame, timestamp_ms);
    }

    previous_gray_frame_ = gray_frame;
    previous_frame_timestamp_ms_ = timestamp_ms;
}

void ByteTracker::updateWithDetections(const std::vector<Detection>& detections, int64_t timestamp_ms) {
    current_display_timestamp_ms_ = std::max(current_display_timestamp_ms_, timestamp_ms);
    ++frame_id_;

    std::vector<TrackPtr> high_detections;
    std::vector<TrackPtr> low_detections;
    high_detections.reserve(detections.size());
    low_detections.reserve(detections.size());

    for (const auto& detection : detections) {
        auto det_track = createDetectionTrack(detection);
        if (!det_track) {
            continue;
        }

        if (det_track->score >= config_.track_threshold) {
            high_detections.push_back(det_track);
        } else {
            low_detections.push_back(det_track);
        }
    }

    std::vector<TrackPtr> active_tracks;
    std::vector<TrackPtr> unconfirmed_tracks;
    for (const auto& track : tracked_tracks_) {
        if (!track->is_activated) {
            unconfirmed_tracks.push_back(track);
        } else {
            active_tracks.push_back(track);
        }
    }

    std::vector<TrackPtr> track_pool = jointTracks(active_tracks, lost_tracks_);
    for (const auto& track : track_pool) {
        predictTrack(track);
    }

    std::vector<TrackPtr> current_tracked;
    std::vector<TrackPtr> remain_tracked;
    std::vector<TrackPtr> remaining_high_detections;
    std::vector<TrackPtr> refound_tracks;

    {
        std::vector<std::vector<int>> matches;
        std::vector<int> unmatched_track_indices;
        std::vector<int> unmatched_detection_indices;

        const auto distances = calcIouDistance(track_pool, high_detections);
        linearAssignment(
            distances,
            static_cast<int>(track_pool.size()),
            static_cast<int>(high_detections.size()),
            config_.match_threshold,
            matches,
            unmatched_track_indices,
            unmatched_detection_indices);

        for (const auto& match : matches) {
            const auto& track = track_pool[match[0]];
            const auto& detection = high_detections[match[1]];
            if (track->state == TrackState::Tracked) {
                updateTrack(track, detection, frame_id_, timestamp_ms);
                current_tracked.push_back(track);
            } else {
                reactivateTrack(track, detection, frame_id_, timestamp_ms, false);
                refound_tracks.push_back(track);
            }
        }

        for (int index : unmatched_detection_indices) {
            remaining_high_detections.push_back(high_detections[index]);
        }

        for (int index : unmatched_track_indices) {
            if (track_pool[index]->state == TrackState::Tracked) {
                remain_tracked.push_back(track_pool[index]);
            }
        }
    }

    std::vector<TrackPtr> current_lost;
    {
        std::vector<std::vector<int>> matches;
        std::vector<int> unmatched_track_indices;
        std::vector<int> unmatched_detection_indices;

        const auto distances = calcIouDistance(remain_tracked, low_detections);
        linearAssignment(
            distances,
            static_cast<int>(remain_tracked.size()),
            static_cast<int>(low_detections.size()),
            config_.low_match_threshold,
            matches,
            unmatched_track_indices,
            unmatched_detection_indices);

        for (const auto& match : matches) {
            const auto& track = remain_tracked[match[0]];
            const auto& detection = low_detections[match[1]];
            if (track->state == TrackState::Tracked) {
                updateTrack(track, detection, frame_id_, timestamp_ms);
                current_tracked.push_back(track);
            } else {
                reactivateTrack(track, detection, frame_id_, timestamp_ms, false);
                refound_tracks.push_back(track);
            }
        }

        for (int index : unmatched_track_indices) {
            if (remain_tracked[index]->state != TrackState::Lost) {
                markLost(remain_tracked[index]);
                current_lost.push_back(remain_tracked[index]);
            }
        }
    }

    std::vector<TrackPtr> current_removed;
    {
        std::vector<std::vector<int>> matches;
        std::vector<int> unmatched_unconfirmed_indices;
        std::vector<int> unmatched_detection_indices;

        const auto distances = calcIouDistance(unconfirmed_tracks, remaining_high_detections);
        linearAssignment(
            distances,
            static_cast<int>(unconfirmed_tracks.size()),
            static_cast<int>(remaining_high_detections.size()),
            config_.unconfirmed_match_threshold,
            matches,
            unmatched_unconfirmed_indices,
            unmatched_detection_indices);

        for (const auto& match : matches) {
            updateTrack(
                unconfirmed_tracks[match[0]],
                remaining_high_detections[match[1]],
                frame_id_,
                timestamp_ms);
            current_tracked.push_back(unconfirmed_tracks[match[0]]);
        }

        for (int index : unmatched_unconfirmed_indices) {
            markRemoved(unconfirmed_tracks[index]);
            current_removed.push_back(unconfirmed_tracks[index]);
        }

        for (int index : unmatched_detection_indices) {
            const auto& track = remaining_high_detections[index];
            if (track->score < config_.high_threshold) {
                continue;
            }
            activateTrack(track, frame_id_, timestamp_ms);
            current_tracked.push_back(track);
        }
    }

    for (const auto& track : lost_tracks_) {
        if ((frame_id_ - track->frame_id) > static_cast<size_t>(config_.track_buffer)) {
            markRemoved(track);
            current_removed.push_back(track);
        }
    }

    tracked_tracks_ = jointTracks(current_tracked, refound_tracks);
    lost_tracks_ = subtractTracks(
        jointTracks(subtractTracks(lost_tracks_, tracked_tracks_), current_lost),
        removed_tracks_);
    removed_tracks_ = jointTracks(removed_tracks_, current_removed);

    std::vector<TrackPtr> deduped_tracked;
    std::vector<TrackPtr> deduped_lost;
    removeDuplicateTracks(tracked_tracks_, lost_tracks_, deduped_tracked, deduped_lost);
    tracked_tracks_ = std::move(deduped_tracked);
    lost_tracks_ = std::move(deduped_lost);
}

std::vector<Detection> ByteTracker::getTrackedDetections() const {
    std::vector<Detection> detections;
    detections.reserve(tracked_tracks_.size());

    for (const auto& track : tracked_tracks_) {
        if (!track->is_activated || track->state != TrackState::Tracked) {
            continue;
        }

        const cv::Rect2f predicted_rect = predictedRectForDisplay(
            *track,
            current_display_timestamp_ms_,
            config_.max_display_prediction_steps);
        const cv::Rect2f output_rect = sanitizeRect(track->display_rect);

        if (predicted_rect.width < config_.min_box_area || predicted_rect.height < config_.min_box_area) {
            continue;
        }

        detections.emplace_back(
            track->label,
            track->score,
            BBox(
                static_cast<int>(std::round(output_rect.x)),
                static_cast<int>(std::round(output_rect.y)),
                static_cast<int>(std::round(output_rect.width)),
                static_cast<int>(std::round(output_rect.height))),
            track->id);
    }

    return detections;
}

ByteTracker::TrackPtr ByteTracker::createDetectionTrack(const Detection& detection) const {
    cv::Rect2f rect(
        static_cast<float>(detection.bbox.x),
        static_cast<float>(detection.bbox.y),
        static_cast<float>(detection.bbox.width),
        static_cast<float>(detection.bbox.height));
    rect = sanitizeRect(rect);

    if (rect.width < config_.min_box_area || rect.height < config_.min_box_area) {
        return nullptr;
    }

    const float aspect_ratio = rect.width / std::max(rect.height, 1.0f);
    if (aspect_ratio > config_.max_aspect_ratio) {
        return nullptr;
    }

    auto track = std::make_shared<Track>();
    track->label = detection.label;
    track->score = detection.confidence;
    track->rect = rect;
    track->display_rect = rect;
    track->kalman_filter.init(8, 4, 0, CV_32F);
    configureKalmanFilter(track->kalman_filter);
    return track;
}

void ByteTracker::activateTrack(const TrackPtr& track, size_t frame_id, int64_t timestamp_ms) {
    track->kalman_filter.init(8, 4, 0, CV_32F);
    configureKalmanFilter(track->kalman_filter);

    const cv::Mat measurement = rectToMeasurement(track->rect);
    track->kalman_filter.statePost = cv::Mat::zeros(8, 1, CV_32F);
    measurement.copyTo(track->kalman_filter.statePost.rowRange(0, 4));

    cv::Mat covariance = cv::Mat::zeros(8, 8, CV_32F);
    const float h = std::max(track->rect.height, 1.0f);
    const float std_weight_position = 1.0f / 20.0f;
    const float std_weight_velocity = 1.0f / 160.0f;
    covariance.at<float>(0, 0) = std::pow(2.0f * std_weight_position * h, 2.0f);
    covariance.at<float>(1, 1) = std::pow(2.0f * std_weight_position * h, 2.0f);
    covariance.at<float>(2, 2) = std::pow(1e-2f, 2.0f);
    covariance.at<float>(3, 3) = std::pow(2.0f * std_weight_position * h, 2.0f);
    covariance.at<float>(4, 4) = std::pow(10.0f * std_weight_velocity * h, 2.0f);
    covariance.at<float>(5, 5) = std::pow(10.0f * std_weight_velocity * h, 2.0f);
    covariance.at<float>(6, 6) = std::pow(1e-5f, 2.0f);
    covariance.at<float>(7, 7) = std::pow(10.0f * std_weight_velocity * h, 2.0f);
    track->kalman_filter.errorCovPost = covariance;

    track->rect = measurementToRect(track->kalman_filter.statePost.rowRange(0, 4));
    track->state = TrackState::Tracked;
    // In this service we want a newly accepted high-confidence detection
    // to appear on the detection stream immediately.
    track->is_activated = true;
    track->id = next_track_id_++;
    track->frame_id = frame_id;
    track->start_frame_id = frame_id;
    track->tracklet_length = 0;
    track->last_detection_timestamp_ms = timestamp_ms;
    updateDisplayRect(track, true);
}

void ByteTracker::predictTrack(const TrackPtr& track) const {
    if (track->state != TrackState::Tracked) {
        track->kalman_filter.statePost.at<float>(7, 0) = 0.0f;
    }

    updateProcessNoise(track->kalman_filter, std::max(track->rect.height, 1.0f));
    track->kalman_filter.predict();
    track->kalman_filter.statePost = track->kalman_filter.statePre.clone();
    track->kalman_filter.errorCovPost = track->kalman_filter.errorCovPre.clone();
    track->rect = measurementToRect(track->kalman_filter.statePost.rowRange(0, 4));
}

void ByteTracker::updateTrack(const TrackPtr& track,
                              const TrackPtr& detection,
                              size_t frame_id,
                              int64_t timestamp_ms) {
    updateMeasurementNoise(track->kalman_filter, std::max(detection->rect.height, 1.0f));
    const cv::Mat corrected = track->kalman_filter.correct(rectToMeasurement(detection->rect));
    track->rect = measurementToRect(corrected.rowRange(0, 4));
    track->state = TrackState::Tracked;
    track->is_activated = true;
    track->score = detection->score;
    track->label = detection->label;
    track->frame_id = frame_id;
    track->tracklet_length += 1;
    if (track->last_detection_timestamp_ms > 0 && timestamp_ms > track->last_detection_timestamp_ms) {
        track->average_detection_interval_ms =
            (track->average_detection_interval_ms * 0.8f) +
            (static_cast<float>(timestamp_ms - track->last_detection_timestamp_ms) * 0.2f);
    }
    track->last_detection_timestamp_ms = timestamp_ms;
    updateDisplayRect(track, false);
}

void ByteTracker::reactivateTrack(const TrackPtr& track,
                                  const TrackPtr& detection,
                                  size_t frame_id,
                                  int64_t timestamp_ms,
                                  bool assign_new_id) {
    updateTrack(track, detection, frame_id, timestamp_ms);
    track->tracklet_length = 0;
    if (assign_new_id) {
        track->id = next_track_id_++;
    }
}

void ByteTracker::markLost(const TrackPtr& track) const {
    track->state = TrackState::Lost;
}

void ByteTracker::markRemoved(const TrackPtr& track) const {
    track->state = TrackState::Removed;
}

std::vector<ByteTracker::TrackPtr> ByteTracker::jointTracks(const std::vector<TrackPtr>& lhs,
                                                            const std::vector<TrackPtr>& rhs) const {
    std::map<int, TrackPtr> unique_tracks;
    for (const auto& track : lhs) {
        unique_tracks[track->id] = track;
    }
    for (const auto& track : rhs) {
        unique_tracks.emplace(track->id, track);
    }

    std::vector<TrackPtr> result;
    result.reserve(unique_tracks.size());
    for (const auto& entry : unique_tracks) {
        if (entry.second->state != TrackState::Removed) {
            result.push_back(entry.second);
        }
    }
    return result;
}

std::vector<ByteTracker::TrackPtr> ByteTracker::subtractTracks(const std::vector<TrackPtr>& lhs,
                                                               const std::vector<TrackPtr>& rhs) const {
    std::map<int, TrackPtr> remaining;
    for (const auto& track : lhs) {
        remaining[track->id] = track;
    }
    for (const auto& track : rhs) {
        remaining.erase(track->id);
    }

    std::vector<TrackPtr> result;
    result.reserve(remaining.size());
    for (const auto& entry : remaining) {
        if (entry.second->state != TrackState::Removed) {
            result.push_back(entry.second);
        }
    }
    return result;
}

void ByteTracker::removeDuplicateTracks(const std::vector<TrackPtr>& lhs,
                                        const std::vector<TrackPtr>& rhs,
                                        std::vector<TrackPtr>& lhs_result,
                                        std::vector<TrackPtr>& rhs_result) const {
    const auto distances = calcIouDistance(lhs, rhs);

    std::vector<bool> lhs_overlap(lhs.size(), false);
    std::vector<bool> rhs_overlap(rhs.size(), false);

    for (size_t i = 0; i < distances.size(); ++i) {
        for (size_t j = 0; j < distances[i].size(); ++j) {
            if (distances[i][j] < 0.15f) {
                const size_t lhs_lifetime = lhs[i]->frame_id - lhs[i]->start_frame_id;
                const size_t rhs_lifetime = rhs[j]->frame_id - rhs[j]->start_frame_id;
                if (lhs_lifetime > rhs_lifetime) {
                    rhs_overlap[j] = true;
                } else {
                    lhs_overlap[i] = true;
                }
            }
        }
    }

    for (size_t i = 0; i < lhs.size(); ++i) {
        if (!lhs_overlap[i]) {
            lhs_result.push_back(lhs[i]);
        }
    }
    for (size_t i = 0; i < rhs.size(); ++i) {
        if (!rhs_overlap[i]) {
            rhs_result.push_back(rhs[i]);
        }
    }
}

void ByteTracker::linearAssignment(const std::vector<std::vector<float>>& cost_matrix,
                                   int row_count,
                                   int column_count,
                                   float threshold,
                                   std::vector<std::vector<int>>& matches,
                                   std::vector<int>& unmatched_rows,
                                   std::vector<int>& unmatched_columns) const {
    if (cost_matrix.empty()) {
        for (int i = 0; i < row_count; ++i) {
            unmatched_rows.push_back(i);
        }
        for (int i = 0; i < column_count; ++i) {
            unmatched_columns.push_back(i);
        }
        return;
    }

    std::vector<int> rowsol;
    std::vector<int> colsol;
    byte_tracker_internal::exec_lapjv(cost_matrix, rowsol, colsol, true, threshold);

    for (size_t row = 0; row < rowsol.size(); ++row) {
        if (rowsol[row] >= 0) {
            matches.push_back({static_cast<int>(row), rowsol[row]});
        } else {
            unmatched_rows.push_back(static_cast<int>(row));
        }
    }

    for (size_t column = 0; column < colsol.size(); ++column) {
        if (colsol[column] < 0) {
            unmatched_columns.push_back(static_cast<int>(column));
        }
    }
}

std::vector<std::vector<float>> ByteTracker::calcIouDistance(const std::vector<TrackPtr>& lhs,
                                                             const std::vector<TrackPtr>& rhs) const {
    if (lhs.empty() || rhs.empty()) {
        return {};
    }

    std::vector<std::vector<float>> distances(lhs.size(), std::vector<float>(rhs.size(), 1.0f));
    for (size_t i = 0; i < lhs.size(); ++i) {
        for (size_t j = 0; j < rhs.size(); ++j) {
            if (lhs[i]->label != rhs[j]->label) {
                distances[i][j] = 1.0f + config_.match_threshold;
                continue;
            }
            distances[i][j] = 1.0f - computeIoU(lhs[i]->rect, rhs[j]->rect);
        }
    }
    return distances;
}

float ByteTracker::computeIoU(const cv::Rect2f& lhs, const cv::Rect2f& rhs) {
    const cv::Rect2f clean_lhs = sanitizeRect(lhs);
    const cv::Rect2f clean_rhs = sanitizeRect(rhs);
    const float intersection = (clean_lhs & clean_rhs).area();
    if (intersection <= 0.0f) {
        return 0.0f;
    }
    const float union_area = clean_lhs.area() + clean_rhs.area() - intersection;
    if (union_area <= 0.0f) {
        return 0.0f;
    }
    return intersection / union_area;
}

cv::Rect2f ByteTracker::sanitizeRect(const cv::Rect2f& rect) {
    return {
        std::max(rect.x, 0.0f),
        std::max(rect.y, 0.0f),
        std::max(rect.width, 0.0f),
        std::max(rect.height, 0.0f)
    };
}

cv::Rect2f ByteTracker::measurementToRect(const cv::Mat& measurement) {
    const float center_x = measurement.at<float>(0, 0);
    const float center_y = measurement.at<float>(1, 0);
    const float aspect = measurement.at<float>(2, 0);
    const float height = std::max(measurement.at<float>(3, 0), 1.0f);
    const float width = std::max(aspect * height, 1.0f);

    return sanitizeRect({
        center_x - (width * 0.5f),
        center_y - (height * 0.5f),
        width,
        height
    });
}

cv::Mat ByteTracker::rectToMeasurement(const cv::Rect2f& rect) {
    const cv::Rect2f clean = sanitizeRect(rect);
    cv::Mat measurement = cv::Mat::zeros(4, 1, CV_32F);
    measurement.at<float>(0, 0) = clean.x + (clean.width * 0.5f);
    measurement.at<float>(1, 0) = clean.y + (clean.height * 0.5f);
    measurement.at<float>(2, 0) = clean.width / std::max(clean.height, 1.0f);
    measurement.at<float>(3, 0) = clean.height;
    return measurement;
}

void ByteTracker::configureKalmanFilter(cv::KalmanFilter& filter) {
    filter.transitionMatrix = cv::Mat::eye(8, 8, CV_32F);
    for (int i = 0; i < 4; ++i) {
        filter.transitionMatrix.at<float>(i, i + 4) = 1.0f;
    }

    filter.measurementMatrix = cv::Mat::zeros(4, 8, CV_32F);
    for (int i = 0; i < 4; ++i) {
        filter.measurementMatrix.at<float>(i, i) = 1.0f;
    }

    filter.processNoiseCov = cv::Mat::eye(8, 8, CV_32F);
    filter.measurementNoiseCov = cv::Mat::eye(4, 4, CV_32F);
    filter.errorCovPost = cv::Mat::eye(8, 8, CV_32F);
    filter.statePost = cv::Mat::zeros(8, 1, CV_32F);
}

void ByteTracker::updateProcessNoise(cv::KalmanFilter& filter, float height) {
    const float h = std::max(height, 1.0f);
    const float std_weight_position = 1.0f / 20.0f;
    const float std_weight_velocity = 1.0f / 160.0f;

    filter.processNoiseCov = cv::Mat::zeros(8, 8, CV_32F);
    filter.processNoiseCov.at<float>(0, 0) = std::pow(std_weight_position * h, 2.0f);
    filter.processNoiseCov.at<float>(1, 1) = std::pow(std_weight_position * h, 2.0f);
    filter.processNoiseCov.at<float>(2, 2) = std::pow(1e-2f, 2.0f);
    filter.processNoiseCov.at<float>(3, 3) = std::pow(std_weight_position * h, 2.0f);
    filter.processNoiseCov.at<float>(4, 4) = std::pow(std_weight_velocity * h, 2.0f);
    filter.processNoiseCov.at<float>(5, 5) = std::pow(std_weight_velocity * h, 2.0f);
    filter.processNoiseCov.at<float>(6, 6) = std::pow(1e-5f, 2.0f);
    filter.processNoiseCov.at<float>(7, 7) = std::pow(std_weight_velocity * h, 2.0f);
}

void ByteTracker::updateMeasurementNoise(cv::KalmanFilter& filter, float height) {
    const float h = std::max(height, 1.0f);
    const float std_weight_position = 1.0f / 20.0f;

    filter.measurementNoiseCov = cv::Mat::zeros(4, 4, CV_32F);
    filter.measurementNoiseCov.at<float>(0, 0) = std::pow(std_weight_position * h, 2.0f);
    filter.measurementNoiseCov.at<float>(1, 1) = std::pow(std_weight_position * h, 2.0f);
    filter.measurementNoiseCov.at<float>(2, 2) = std::pow(1e-1f, 2.0f);
    filter.measurementNoiseCov.at<float>(3, 3) = std::pow(std_weight_position * h, 2.0f);
}

cv::Rect2f ByteTracker::predictedRectForDisplay(const Track& track,
                                                int64_t display_timestamp_ms,
                                                float max_prediction_steps) {
    if (display_timestamp_ms <= track.last_detection_timestamp_ms) {
        return sanitizeRect(track.rect);
    }

    const float interval_ms = std::max(track.average_detection_interval_ms, 1.0f);
    const float elapsed_steps = std::min(
        static_cast<float>(display_timestamp_ms - track.last_detection_timestamp_ms) / interval_ms,
        max_prediction_steps);

    cv::Mat state = track.kalman_filter.statePost.clone();
    state.at<float>(0, 0) += state.at<float>(4, 0) * elapsed_steps;
    state.at<float>(1, 0) += state.at<float>(5, 0) * elapsed_steps;

    cv::Rect2f rect = measurementToRect(state.rowRange(0, 4));
    rect.width = track.rect.width;
    rect.height = track.rect.height;
    rect.x = state.at<float>(0, 0) - (rect.width * 0.5f);
    rect.y = state.at<float>(1, 0) - (rect.height * 0.5f);

    return sanitizeRect(rect);
}

void ByteTracker::applyVisualTracking(const cv::Mat& gray_frame, int64_t timestamp_ms) {
    std::vector<cv::Point2f> previous_points;
    cv::goodFeaturesToTrack(
        previous_gray_frame_,
        previous_points,
        config_.optical_flow_max_corners,
        config_.optical_flow_quality_level,
        config_.optical_flow_min_distance);

    if (static_cast<int>(previous_points.size()) < config_.optical_flow_min_points) {
        return;
    }

    std::vector<cv::Point2f> current_points;
    std::vector<unsigned char> status;
    std::vector<float> errors;
    cv::calcOpticalFlowPyrLK(
        previous_gray_frame_,
        gray_frame,
        previous_points,
        current_points,
        status,
        errors);

    std::vector<cv::Point2f> matched_previous_points;
    std::vector<cv::Point2f> matched_current_points;
    matched_previous_points.reserve(previous_points.size());
    matched_current_points.reserve(previous_points.size());
    for (size_t i = 0; i < previous_points.size(); ++i) {
        if (!status[i]) {
            continue;
        }
        if (i < errors.size() && errors[i] > config_.optical_flow_max_error) {
            continue;
        }
        matched_previous_points.push_back(previous_points[i]);
        matched_current_points.push_back(current_points[i]);
    }

    if (static_cast<int>(matched_previous_points.size()) < config_.optical_flow_min_points) {
        return;
    }

    cv::Mat inlier_mask;
    cv::Mat affine = cv::estimateAffinePartial2D(
        matched_previous_points,
        matched_current_points,
        inlier_mask,
        cv::RANSAC,
        config_.global_motion_ransac_threshold);

    if (affine.empty() || affine.rows != 2 || affine.cols != 3) {
        return;
    }

    const float a = static_cast<float>(affine.at<double>(0, 0));
    const float b = static_cast<float>(affine.at<double>(0, 1));
    const float tx = static_cast<float>(affine.at<double>(0, 2));
    const float c = static_cast<float>(affine.at<double>(1, 0));
    const float d = static_cast<float>(affine.at<double>(1, 1));
    const float ty = static_cast<float>(affine.at<double>(1, 2));
    const float scale_x = std::sqrt((a * a) + (c * c));
    const float scale_y = std::sqrt((b * b) + (d * d));

    for (const auto& track : tracked_tracks_) {
        if (!track->is_activated || track->state != TrackState::Tracked) {
            continue;
        }

        const cv::Rect2f predicted_rect = predictedRectForDisplay(
            *track,
            timestamp_ms,
            config_.max_display_prediction_steps);
        const float center_x = predicted_rect.x + (predicted_rect.width * 0.5f);
        const float center_y = predicted_rect.y + (predicted_rect.height * 0.5f);
        const float transformed_center_x = (a * center_x) + (b * center_y) + tx;
        const float transformed_center_y = (c * center_x) + (d * center_y) + ty;
        const float transformed_width = predicted_rect.width * std::clamp(scale_x, 0.85f, 1.15f);
        const float transformed_height = predicted_rect.height * std::clamp(scale_y, 0.85f, 1.15f);

        const float max_step = std::max(predicted_rect.width, predicted_rect.height) * config_.optical_flow_max_step_ratio;
        if (std::abs(transformed_center_x - center_x) > max_step ||
            std::abs(transformed_center_y - center_y) > max_step) {
            continue;
        }

        track->rect = sanitizeRect({
            transformed_center_x - (transformed_width * 0.5f),
            transformed_center_y - (transformed_height * 0.5f),
            transformed_width,
            transformed_height
        });
        syncTrackStateToRect(track);
        updateDisplayRect(track, false);
    }
}

void ByteTracker::syncTrackStateToRect(const TrackPtr& track) const {
    const cv::Mat measurement = rectToMeasurement(track->rect);
    track->kalman_filter.statePost.at<float>(0, 0) = measurement.at<float>(0, 0);
    track->kalman_filter.statePost.at<float>(1, 0) = measurement.at<float>(1, 0);
    track->kalman_filter.statePost.at<float>(2, 0) = measurement.at<float>(2, 0);
    track->kalman_filter.statePost.at<float>(3, 0) = measurement.at<float>(3, 0);
    track->kalman_filter.statePost.at<float>(6, 0) = 0.0f;
    track->kalman_filter.statePost.at<float>(7, 0) = 0.0f;
}

void ByteTracker::updateDisplayRect(const TrackPtr& track, bool force) const {
    if (force) {
        track->display_rect = track->rect;
        return;
    }

    const float alpha = std::clamp(config_.display_smoothing_alpha, 0.0f, 1.0f);
    track->display_rect.x = (track->display_rect.x * (1.0f - alpha)) + (track->rect.x * alpha);
    track->display_rect.y = (track->display_rect.y * (1.0f - alpha)) + (track->rect.y * alpha);
    track->display_rect.width = (track->display_rect.width * (1.0f - alpha)) + (track->rect.width * alpha);
    track->display_rect.height = (track->display_rect.height * (1.0f - alpha)) + (track->rect.height * alpha);
}

float ByteTracker::median(std::vector<float>& values) {
    if (values.empty()) {
        return 0.0f;
    }

    const auto middle = values.begin() + static_cast<std::ptrdiff_t>(values.size() / 2);
    std::nth_element(values.begin(), middle, values.end());
    float result = *middle;

    if ((values.size() % 2) == 0) {
        const auto lower_middle = values.begin() + static_cast<std::ptrdiff_t>((values.size() / 2) - 1);
        std::nth_element(values.begin(), lower_middle, values.end());
        result = (result + *lower_middle) * 0.5f;
    }

    return result;
}
